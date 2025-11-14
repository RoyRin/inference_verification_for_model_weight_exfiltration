"""
Text Generation using GLS-based Sampling

This script implements a custom sampling strategy that selects tokens based on
Gumbel Likelihood Scores (GLS) rather than standard Gumbel-max sampling.

For each position:
1. Get logits and apply temperature/top-k/top-p
2. Select top-N candidates by probability
3. Compute GLS for each candidate (as if it were the sampled token)
4. Select the token with the highest (least negative) GLS
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional
import pickle
from dataclasses import dataclass
import gc
from datetime import datetime
import yaml

from inference_verification.scoring_functions import (
    compute_gumbel_likelihood_score,
)

EPSILON = 1e-12


@dataclass
class GLSGenerationConfig:
    """Configuration for GLS-based generation."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Generation settings
    n_prompts: int = 100
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: float = 0.95
    seed: int = 42

    # GLS sampling settings
    top_n_candidates: int = 100  # Number of top tokens to evaluate with GLS
    gumbel_sigma: float = 1.0  # Gaussian noise scale for GLS computation

    # Dataset
    dataset_name: str = "lmsys/lmsys-chat-1m"
    max_ctx_len: int = 512

    # Save settings
    save_dir: str = "generated_outputs_gls"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GLSGenerationConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract model and generation_params sections
        model_config = config_dict.get("model", {})
        generation_config = config_dict.get("generation_params", {})
        gls_config = config_dict.get("gls_sampling", {})

        # Merge all sections
        merged_config = {**model_config, **generation_config, **gls_config}

        # Create config object
        return cls(**merged_config)


def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Apply top-k mask to logits (from vLLM)."""
    assert len(logits.shape) == 2
    V = logits.shape[1]
    k = torch.minimum(k, torch.tensor([V], device=k.device, dtype=k.dtype))
    no_top_k_mask = k == logits.shape[1]
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = int(k.max().item())
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to logits (from vLLM)."""
    if p is None:
        if k is None:
            return logits
        return apply_top_k_only(logits, k)

    assert len(logits.shape) == 2

    if k is not None:
        V = logits.shape[1]
        k = torch.minimum(k, torch.tensor([V], device=k.device, dtype=k.dtype))

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None and (k > 0).all():
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def load_prompts(cfg: GLSGenerationConfig) -> list[list[int]]:
    """Load and tokenize prompts from dataset."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    ds = load_dataset(cfg.dataset_name, split="train")

    tokenized_prompts = []
    unique_prompts = set()

    count = 0
    pbar = tqdm(total=cfg.n_prompts, desc="Loading prompts")
    while len(tokenized_prompts) < cfg.n_prompts and count < len(ds):
        try:
            raw_prompt = ds[count]["conversation"]
            rendered_prompt = tokenizer.apply_chat_template(raw_prompt, tokenize=False, add_generation_prompt=True)
            tokenized_prompt = tokenizer.encode(rendered_prompt, add_special_tokens=False, return_tensors=None)

            if len(tokenized_prompt) <= cfg.max_ctx_len:
                if tuple(tokenized_prompt) not in unique_prompts:
                    unique_prompts.add(tuple(tokenized_prompt))
                    tokenized_prompts.append(tokenized_prompt)
                    pbar.update(1)
        except Exception as e:
            print(f"Warning: Failed to process prompt {count}: {e}")

        count += 1

    pbar.close()
    del tokenizer
    return tokenized_prompts


def sample_token_with_gls(
    logits_V: torch.Tensor,
    exponential_noise_V: torch.Tensor,
    cfg: GLSGenerationConfig,
    device: torch.device,
) -> int:
    """
    Sample a token using GLS-based selection.

    Args:
        logits_V: [V] raw logits
        exponential_noise_V: [V] exponential noise for Gumbel sampling
        cfg: generation config
        device: torch device

    Returns:
        Selected token index
    """
    # Apply temperature and top-k/top-p to get valid token set
    temp_logits_V = logits_V.clone()
    if cfg.temperature > 0.0:
        temp_logits_V = temp_logits_V / cfg.temperature

    top_k_tensor = torch.tensor([cfg.top_k], device=device) if cfg.top_k is not None else None
    top_p_tensor = torch.tensor([cfg.top_p], device=device)

    filtered_logits_V = apply_top_k_top_p(temp_logits_V[None, :], top_k_tensor, top_p_tensor).squeeze()

    # Get probabilities for top-N selection
    probs_V = torch.nn.functional.softmax(filtered_logits_V, dim=-1)

    # Select top-N candidates by probability
    top_n = min(cfg.top_n_candidates, (probs_V > 0).sum().item())
    top_probs, top_indices = torch.topk(probs_V, top_n)

    # Compute GLS for each candidate token
    gls_scores = []
    for candidate_idx in top_indices:
        gls_score = compute_gumbel_likelihood_score(
            logits_V=logits_V,
            exponential_noise_V=exponential_noise_V,
            temperature=cfg.temperature,
            top_k=top_k_tensor,
            top_p=top_p_tensor,
            gold_idx=candidate_idx,
            noise_sigma=cfg.gumbel_sigma,
            apply_top_k_top_p_fn=apply_top_k_top_p,
            epsilon=EPSILON,
        )
        gls_scores.append(gls_score)

    # Select token with highest (least negative) GLS
    best_gls_idx = torch.tensor(gls_scores).argmax()
    selected_token = top_indices[best_gls_idx].item()

    return selected_token


def generate_with_gls_sampling(
    cfg: GLSGenerationConfig,
    prompts: list[list[int]],
) -> list[dict]:
    """
    Generate sequences using GLS-based sampling.

    Args:
        cfg: generation configuration
        prompts: list of tokenized prompts

    Returns:
        List of generation outputs with token_ids and metadata
    """
    print(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Set pad token if needed
    if not tokenizer.pad_token:
        if "llama" in cfg.model_name.lower():
            tokenizer.pad_token_id = (
                model.config.eos_token_id[0]
                if isinstance(model.config.eos_token_id, list)
                else model.config.eos_token_id
            )
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    device = model.device
    outputs = []

    # Initialize Gumbel noise generator with seed
    gumbel_gen = torch.Generator(device=device)

    print(f"Generating {len(prompts)} sequences with GLS-based sampling...")
    for prompt_idx, prompt_ids in enumerate(tqdm(prompts, desc="Generating")):
        # Reset generator for each prompt
        gumbel_gen.manual_seed(cfg.seed)

        # Start with prompt
        current_ids = prompt_ids.copy()
        generated_ids = []

        # Generate tokens autoregressively
        for step in range(cfg.max_tokens):
            # Prepare input
            input_tensor = torch.tensor([current_ids], dtype=torch.long, device=device)

            # Get logits
            with torch.no_grad():
                logits = model(input_ids=input_tensor).logits

            # Get logits for next token position
            next_token_logits = logits[0, -1, :].float()

            # Draw exponential noise for Gumbel sampling
            exponential_noise = torch.empty_like(next_token_logits)
            exponential_noise.exponential_(generator=gumbel_gen)

            # Sample token using GLS-based selection
            selected_token = sample_token_with_gls(
                logits_V=next_token_logits,
                exponential_noise_V=exponential_noise,
                cfg=cfg,
                device=device,
            )

            # Add to sequence
            generated_ids.append(selected_token)
            current_ids.append(selected_token)

            # Check for EOS
            if selected_token == tokenizer.eos_token_id:
                break

        # Store output
        output = {
            "prompt_token_ids": prompt_ids,
            "token_ids": generated_ids,
            "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        }
        outputs.append(output)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return outputs


def save_outputs(outputs: list[dict], save_dir: str) -> str:
    """Save generated outputs to pickle file."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    output_file = save_path / "generated_outputs_gls.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(outputs, f)

    print(f"Saved {len(outputs)} generated outputs to {output_file}")
    return str(output_file)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate text using GLS-based sampling")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Optional overrides
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--n-prompts", type=int, default=None, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--top-n-candidates", type=int, default=None, help="Number of top candidates for GLS evaluation")
    parser.add_argument("--gumbel-sigma", type=float, default=None, help="Gaussian noise scale for GLS")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Load config from YAML or use defaults
    if args.config is not None:
        print(f"Loading configuration from {args.config}")
        cfg = GLSGenerationConfig.from_yaml(args.config)
    else:
        cfg = GLSGenerationConfig()

        # Override config with command-line arguments
        if args.model is not None:
            cfg.model_name = args.model
        if args.n_prompts is not None:
            cfg.n_prompts = args.n_prompts
        if args.max_tokens is not None:
            cfg.max_tokens = args.max_tokens
        if args.temperature is not None:
            cfg.temperature = args.temperature
        if args.top_k is not None:
            cfg.top_k = args.top_k
        if args.top_p is not None:
            cfg.top_p = args.top_p
        if args.seed is not None:
            cfg.seed = args.seed
        if args.top_n_candidates is not None:
            cfg.top_n_candidates = args.top_n_candidates
        if args.gumbel_sigma is not None:
            cfg.gumbel_sigma = args.gumbel_sigma

    # Save dir override
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    elif cfg.save_dir == "generated_outputs_gls":
        # Create timestamped directory if using default
        datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.save_dir = f"generated_outputs_gls/{datestr}"

    print("=" * 80)
    print("TEXT GENERATION WITH GLS-BASED SAMPLING")
    print("=" * 80)
    print(f"Model: {cfg.model_name}")
    print(f"Prompts: {cfg.n_prompts}")
    print(f"Max tokens: {cfg.max_tokens}")
    print(f"Temperature: {cfg.temperature}")
    print(f"Top-k: {cfg.top_k}")
    print(f"Top-p: {cfg.top_p}")
    print(f"Seed: {cfg.seed}")
    print(f"Top-N candidates for GLS: {cfg.top_n_candidates}")
    print(f"Gumbel sigma: {cfg.gumbel_sigma}")
    print(f"Save dir: {cfg.save_dir}")
    print("=" * 80)

    # Load prompts
    prompts = load_prompts(cfg)
    print(f"Loaded {len(prompts)} prompts")

    # Generate with GLS-based sampling
    outputs = generate_with_gls_sampling(cfg, prompts)
    print(f"Generated {len(outputs)} outputs")

    # Save
    output_file = save_outputs(outputs, cfg.save_dir)

    print("\nDone! Generated outputs saved to:", output_file)


if __name__ == "__main__":
    main()
