"""
Vocab-wide Gumbel-max and CGS verification score analysis.

For each generated token, compute:
1. Gumbel-max verification scores for EVERY token in vocabulary
2. CGS (Circuit Genuineness Score) for EVERY token in vocabulary
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path to import inference_verification module
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import vllm
from vllm import LLM, SamplingParams, RequestOutput
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional
import pickle
from dataclasses import asdict, dataclass, field
import gc
import numpy as np
from datetime import datetime

# Import GLS and CGS scoring functions from inference_verification.scoring_functions module
from inference_verification.scoring_functions import (
    compute_gumbel_likelihood_score,
    compute_gumbel_likelihood_score_batch,
    compute_convolved_gaussian_score,
    get_seed,
    draw_u,
)

EPSILON = 1e-12
@dataclass
class GumbelCGSAnalysisConfig:
    """Configuration for vocab-wide Gumbel and CGS analysis."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Generation settings
    n_prompts: int = 100
    max_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: float = 0.95
    seed: int = 42

    # Dataset
    dataset_name: str = "lmsys/lmsys-chat-1m"
    max_ctx_len: int = 512

    # Save settings
    save_dir: str = "gumbel_cgs_analysis_results"

    # vLLM settings
    gpu_memory_utilization: float = 0.7

    # CGS settings
    cgs_sigma: float = 0.01  # Gaussian std for CGS

    # Gumbel analytical likelihood estimator settings
    gumbel_sigmas: list[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    )  # Gaussian noise scale for Gumbel verification

    # Support size for GLS scoring
    support_size: int = 1000  # Number of top tokens to score (BEFORE top-k/top-p filtering)


# GLS and CGS functions now imported from separate modules


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.

    NOTE: this is directly copy pasted from vllm: https://github.com/vllm-project/vllm/blob/10d765482d19abfab6c66b5f815720a66aa9de42/vllm/v1/sample/ops/topk_topp_sampler.py#L164
    """
    # probably not necessary, keeping it for now.
    assert len(logits.shape) == 2

    # Guard: if k > vocab_size, clamp to vocab_size (select all tokens)
    V = logits.shape[1]
    k = torch.minimum(k, torch.tensor([V], device=k.device, dtype=k.dtype))

    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = int(k.max().item())
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.

    NOTE: this is directly copy pasted from vllm: https://github.com/vllm-project/vllm/blob/10d765482d19abfab6c66b5f815720a66aa9de42/vllm/v1/sample/ops/topk_topp_sampler.py#L164
    They use 2D, we use 3d. So we do a quick reshape to match them.
    """
    if p is None:
        if k is None:
            return logits

        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k)

    # probably not necessary, keeping it for now.
    assert len(logits.shape) == 2

    # Guard: if k > vocab_size, clamp to vocab_size (select all tokens)
    if k is not None:
        V = logits.shape[1]
        k = torch.minimum(k, torch.tensor([V], device=k.device, dtype=k.dtype))

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None and (k > 0).all():
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def keep_one_token(scores: torch.Tensor, tok_idx: torch.Tensor) -> torch.Tensor:
    """
    Keep exactly one token per row along the last dimension.

    Args:
        scores: shape (..., V) - logits/scores tensor
        tok_idx: shape (...) - must match scores.shape[:-1]

    Returns:
        shape (..., V) with all -inf except at chosen indices
    """
    # Simple rule: tok_idx shape must match all dims except last
    assert tok_idx.shape == scores.shape[:-1], (
        f"tok_idx.shape {tok_idx.shape} must match scores.shape[:-1] {scores.shape[:-1]}"
    )
    out = torch.full_like(scores, float("-inf"))

    idx = tok_idx.unsqueeze(-1)

    values = torch.gather(scores, dim=-1, index=idx)
    out.scatter_(dim=-1, index=idx, src=values)

    return out


def get_probs(logits: torch.Tensor, temperature: float, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """
    logits: shape [..., V]
    returns: probabilities with same shape, normalized along the last dim
    """

    assert len(logits.shape) == 2, print(f"Expected 2D logits, got shape {logits.shape}")

    if temperature > 0.0:
        x = logits / max(temperature, 1e-10)
    else:
        # greedy: pick argmax per row
        idx = torch.argmax(logits, dim=-1)
        x = keep_one_token(logits, idx)

    x = apply_top_k_top_p(x, top_k, top_p)
    probs = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
    return probs


# compute_convolved_gaussian_score now imported from convolved_gaussian_score module


def set_tokenizer_pad_token(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, model_name: str) -> AutoTokenizer:
    """Set pad token for tokenizer if not already set."""
    if not tokenizer.pad_token and "llama" in model_name.lower():
        tokenizer.pad_token_id = (
            model.config.eos_token_id[0] if isinstance(model.config.eos_token_id, list) else model.config.eos_token_id
        )
    elif not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _as_list(x):
    """Convert tensor/tuple to list."""
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    return list(x)


def load_prompts(cfg: GumbelCGSAnalysisConfig) -> list[list[int]]:
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


def generate_with_vllm(cfg: GumbelCGSAnalysisConfig, prompts: list[list[int]], max_model_len: Optional[int] = None) -> list[RequestOutput]:
    """Generate sequences using vLLM."""
    print(f"Loading vLLM model: {cfg.model_name}")
    llm_kwargs = {
        "model": cfg.model_name,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    model = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        seed=cfg.seed,
    )

    print(f"Generating {len(prompts)} sequences...")
    outputs = model.generate(prompt_token_ids=prompts, sampling_params=sampling_params)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return outputs


def verify_and_save(cfg: GumbelCGSAnalysisConfig, outputs: list[RequestOutput]) -> None:
    """
    Verify each output and save vocab-wide Gumbel + CGS scores to individual files.

    Each prompt gets its own pickle file containing:
    - top_k_gumbel_scores: dict[sigma -> array] - Gumbel scores for top-k tokens
    - sampled_gumbel_scores: dict[sigma -> float] - Gumbel scores of sampled token
    - sampled_support_idx: int - rank of sampled token in support set (0-indexed)
    - logit_rank: int - rank of sampled token in raw logits (0=highest, before temp/top-k/top-p)
    """
    print(f"Loading verification model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer = set_tokenizer_pad_token(tokenizer, model, cfg.model_name)

    device = model.device

    # Create save directory
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"Verifying and saving {len(outputs)} outputs...")
    for prompt_idx, output in enumerate(tqdm(outputs, desc="Verifying")):
        prompt_ids = _as_list(output.prompt_token_ids)
        gen_ids = _as_list(output.outputs[0].token_ids)

        # Concatenate for single forward pass
        full_sequence = prompt_ids + gen_ids
        input_ids = torch.tensor([full_sequence], dtype=torch.long, device=device)

        # Get logits for entire sequence
        with torch.no_grad():
            logits_BLV = model(input_ids=input_ids).logits  # [1, L, V]

        logits_LV = logits_BLV.squeeze().float()  # [L, V]
        top_k_tensor = torch.tensor([cfg.top_k], device=logits_LV.device)
        top_p_tensor = torch.tensor([cfg.top_p], device=logits_LV.device)
        probs_LV = get_probs(logits_LV, cfg.temperature, top_k_tensor, top_p_tensor)  # [L, V] - filtered probs

        # Compute unfiltered probabilities (only temperature, no top-k/top-p) for support selection
        if cfg.temperature > 0.0:
            unfiltered_logits_LV = logits_LV / max(cfg.temperature, 1e-10)
        else:
            unfiltered_logits_LV = logits_LV
        unfiltered_probs_LV = torch.nn.functional.softmax(unfiltered_logits_LV, dim=-1, dtype=torch.float32)  # [L, V]

        # Initialize RNGs
        gumbel_gen = torch.Generator(device=device)
        gumbel_gen.manual_seed(cfg.seed)

        cgs_gen = torch.Generator(device=device)

        past_tokens = []

        # Process each generated token
        for j, sampled_token in enumerate(gen_ids):
            # Logit position that predicted this token
            pos = len(prompt_ids) + j - 1
            logits_V = logits_LV[pos]

            # === LOGIT RANK ===
            # Compute rank of sampled token in raw logits (before temp/top-k/top-p)
            sorted_indices = torch.argsort(logits_V, descending=True)
            logit_rank = (sorted_indices == sampled_token).nonzero(as_tuple=True)[0].item()

            # === GUMBEL SCORES ===
            # Draw exponential noise for FULL vocabulary (for Gumbel-max sampling)
            noise_V = torch.empty_like(logits_V)
            noise_V.exponential_(generator=gumbel_gen)

            probs_V = probs_LV[pos]

            probs_V = probs_V / noise_V

            # Compute CDF for CGS
            cdf_V = probs_V.cumsum(dim=-1)
            cdf_V[-1] = 1.0  # Ensure CDF ends at 1

            # Select support tokens from UNFILTERED probabilities (before top-k/top-p)
            unfiltered_probs_V = unfiltered_probs_LV[pos]
            support_indices = unfiltered_probs_V.topk(k=cfg.support_size).indices  # Top-k tokens by unfiltered prob

            matches = torch.where(support_indices == sampled_token)[0]
            sampled_support_idx = matches[0].item() if len(matches) > 0 else -1

            top_k_tensor = torch.tensor([cfg.top_k], device=device) if cfg.top_k is not None else None
            top_p_tensor = torch.tensor([cfg.top_p], device=device)

            pairwise_gumbel_scores = {}
            support_gumbel_scores = {}

            for sigma in cfg.gumbel_sigmas:
                claimed_token_score = compute_gumbel_likelihood_score(
                    logits_V=logits_V,
                    exponential_noise_V=noise_V,
                    temperature=cfg.temperature,
                    top_k=top_k_tensor,
                    top_p=top_p_tensor,
                    gold_idx=torch.tensor(sampled_token, device=device),
                    noise_sigma=sigma,
                    apply_top_k_top_p_fn=apply_top_k_top_p,
                    epsilon=EPSILON,
                )

                support_scores = compute_gumbel_likelihood_score_batch(
                    logits_V=logits_V,
                    exponential_noise_V=noise_V,
                    temperature=cfg.temperature,
                    top_k=top_k_tensor,
                    top_p=top_p_tensor,
                    gold_idx_list=support_indices,
                    noise_sigma=sigma,
                    apply_top_k_top_p_fn=apply_top_k_top_p,
                    epsilon=EPSILON,
                )

                pairwise_gumbel_scores[sigma] = claimed_token_score
                support_gumbel_scores[sigma] = support_scores.cpu().numpy()

            # === CGS SCORES ===
            # Deterministically draw u from seed + past tokens
            seed = get_seed(cfg.seed, past_tokens)
            u = draw_u(seed, cgs_gen)

            # Compute CGS scores for ALL tokens in vocabulary
            # cgs_scores_V = compute_vocab_wide_cgs_scores(cdf_V, u, cfg.cgs_sigma)

            # cgs_score = cgs_scores_V[sampled_token].item()

            result_dict = {
                "top_k_gumbel_scores": support_gumbel_scores,  # [support_size] - GLS scores for top tokens
                "sampled_gumbel_scores": pairwise_gumbel_scores,  # scalar - GLS score for sampled token
                "sampled_support_idx": sampled_support_idx,  # int - rank of sampled token (0-indexed)
                "logit_rank": logit_rank,  # int - rank of sampled token in raw logits (0=highest, before temp/top-k/top-p)
            }

            results.append(result_dict)

            # Update past tokens for next iteration
            past_tokens.append(sampled_token)

    save_path = save_dir / f"all_prompts.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved results to {save_dir}/")

    del model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Gumbel + CGS verification experiments")
    parser.add_argument("--model", type=str, default=None, help="Model name (default: meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--gumbel-sigma", type=float, default=None, help="Gumbel noise scale (default: 0.02)")
    parser.add_argument("--n-prompts", type=int, default=None, help="Number of prompts (default: 100)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate (default: 100)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPU memory utilization (default: 0.7)")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model sequence length for KV cache")
    parser.add_argument("--gumbel-sigmas", type=str, default=None, help="Comma-separated list of sigma values (e.g., '0.01,0.05')")
    parser.add_argument("--support-size", type=int, default=None, help="Number of top tokens to score (default: 500)")
    parser.add_argument("--sweep-dir", type=str, default=None, help="Parent directory for sweep (all runs save here)")
    args = parser.parse_args()

    cfg = GumbelCGSAnalysisConfig()
    max_model_len = args.max_model_len

    # Override config with command-line arguments if provided
    if args.model is not None:
        cfg.model_name = args.model
    if args.n_prompts is not None:
        cfg.n_prompts = args.n_prompts
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.gpu_memory_utilization is not None:
        cfg.gpu_memory_utilization = args.gpu_memory_utilization
    if args.gumbel_sigmas is not None:
        cfg.gumbel_sigmas = [float(s.strip()) for s in args.gumbel_sigmas.split(',')]
    if args.support_size is not None:
        cfg.support_size = args.support_size

    # Create timestamped subdirectory with sigma and n_samples in name
    if args.sweep_dir is not None:
        # Save under the provided sweep directory
        cfg.save_dir = f"{args.sweep_dir}/results"
    else:
        # Default: create new timestamped directory
        datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.save_dir = f"{cfg.save_dir}/{datestr}"

    print("=" * 80)
    print("Gumbel + CGS Vocab-wide Analysis")
    print("=" * 80)
    print(f"Model: {cfg.model_name}")
    print(f"Prompts: {cfg.n_prompts}")
    print(f"Max tokens: {cfg.max_tokens}")
    print(f"Temperature: {cfg.temperature}")
    print(f"Top-k: {cfg.top_k}")
    print(f"Top-p: {cfg.top_p}")
    print(f"Seed: {cfg.seed}")
    print(f"Support size (tokens to score): {cfg.support_size}")
    print(f"Gumbel sigma: {cfg.gumbel_sigmas}")
    print(f"CGS sigma: {cfg.cgs_sigma}")
    print(f"Save dir: {cfg.save_dir}")
    print("=" * 80)

    # Step 1: Load prompts
    prompts = load_prompts(cfg)
    print(f"Loaded {len(prompts)} prompts")

    # Step 2: Generate with vLLM
    outputs = generate_with_vllm(cfg, prompts, max_model_len)
    print(f"Generated {len(outputs)} outputs")

    # Step 3: Verify and save individual prompt results
    verify_and_save(cfg, outputs)

    print("\nDone! Results saved to:", cfg.save_dir)


if __name__ == "__main__":
    main()
