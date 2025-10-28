"""
Token Verification using GLS and CGS

This script verifies generated tokens by computing Gumbel Likelihood Scores (GLS)
and optionally Convolved Gaussian Scores (CGS) for each token.
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from vllm import RequestOutput
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Optional
import pickle
from dataclasses import dataclass, field
import gc
import numpy as np
from datetime import datetime

# Import GLS and CGS scoring functions from scoring_functions module
from inference_verification.scoring_functions import (
    compute_gumbel_likelihood_score,
    compute_gumbel_likelihood_score_batch,
    compute_convolved_gaussian_score,
    get_seed,
    draw_u,
)

EPSILON = 1e-12


@dataclass
class VerificationConfig:
    """Configuration for token verification."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Sampling parameters (must match generation)
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: float = 0.95
    seed: int = 42

    # Verification settings
    gumbel_sigmas: list[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    )  # Gaussian noise scales for GLS
    cgs_sigma: float = 0.01  # Gaussian std for CGS
    support_size: int = 1000  # Number of top tokens to score

    # Save settings
    save_dir: str = "verification_results"


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


def keep_one_token(scores: torch.Tensor, tok_idx: torch.Tensor) -> torch.Tensor:
    """Keep exactly one token (for greedy sampling)."""
    assert tok_idx.shape == scores.shape[:-1]
    out = torch.full_like(scores, float("-inf"))
    idx = tok_idx.unsqueeze(-1)
    values = torch.gather(scores, dim=-1, index=idx)
    out.scatter_(dim=-1, index=idx, src=values)
    return out


def get_probs(logits: torch.Tensor, temperature: float, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Compute probabilities from logits with temperature and top-k/top-p."""
    assert len(logits.shape) == 2

    if temperature > 0.0:
        x = logits / max(temperature, 1e-10)
    else:
        idx = torch.argmax(logits, dim=-1)
        x = keep_one_token(logits, idx)

    x = apply_top_k_top_p(x, top_k, top_p)
    probs = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
    return probs


def set_tokenizer_pad_token(tokenizer, model, model_name):
    """Set pad token if not already set."""
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


def verify_outputs(cfg: VerificationConfig, outputs: list[RequestOutput]) -> list[dict]:
    """
    Verify generated outputs and compute GLS/CGS scores.

    Returns list of dictionaries, one per token:
    - top_k_gumbel_scores: dict[sigma -> array] - GLS scores for support tokens
    - sampled_gumbel_scores: dict[sigma -> float] - GLS score for sampled token
    - sampled_support_idx: int - rank in support set
    - logit_rank: int - rank in raw logits
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
    results = []

    print(f"Verifying {len(outputs)} outputs...")
    for prompt_idx, output in enumerate(tqdm(outputs, desc="Verifying")):
        prompt_ids = _as_list(output.prompt_token_ids)
        gen_ids = _as_list(output.outputs[0].token_ids)

        # Single forward pass for entire sequence
        full_sequence = prompt_ids + gen_ids
        input_ids = torch.tensor([full_sequence], dtype=torch.long, device=device)

        with torch.no_grad():
            logits_BLV = model(input_ids=input_ids).logits

        logits_LV = logits_BLV.squeeze().float()
        top_k_tensor = torch.tensor([cfg.top_k], device=device)
        top_p_tensor = torch.tensor([cfg.top_p], device=device)
        probs_LV = get_probs(logits_LV, cfg.temperature, top_k_tensor, top_p_tensor)

        # Unfiltered probabilities for support selection
        if cfg.temperature > 0.0:
            unfiltered_logits_LV = logits_LV / max(cfg.temperature, 1e-10)
        else:
            unfiltered_logits_LV = logits_LV
        unfiltered_probs_LV = torch.nn.functional.softmax(unfiltered_logits_LV, dim=-1, dtype=torch.float32)

        # Initialize RNGs
        gumbel_gen = torch.Generator(device=device)
        gumbel_gen.manual_seed(cfg.seed)
        cgs_gen = torch.Generator(device=device)
        past_tokens = []

        # Process each generated token
        for j, sampled_token in enumerate(gen_ids):
            pos = len(prompt_ids) + j - 1
            logits_V = logits_LV[pos]

            # Logit rank
            sorted_indices = torch.argsort(logits_V, descending=True)
            logit_rank = (sorted_indices == sampled_token).nonzero(as_tuple=True)[0].item()

            # Draw Gumbel noise
            noise_V = torch.empty_like(logits_V)
            noise_V.exponential_(generator=gumbel_gen)

            probs_V = probs_LV[pos]
            probs_V = probs_V / noise_V
            cdf_V = probs_V.cumsum(dim=-1)
            cdf_V[-1] = 1.0

            # Select support tokens
            unfiltered_probs_V = unfiltered_probs_LV[pos]
            support_indices = unfiltered_probs_V.topk(k=cfg.support_size).indices

            matches = torch.where(support_indices == sampled_token)[0]
            sampled_support_idx = matches[0].item() if len(matches) > 0 else -1

            top_k_tensor = torch.tensor([cfg.top_k], device=device) if cfg.top_k is not None else None
            top_p_tensor = torch.tensor([cfg.top_p], device=device)

            # Compute GLS scores
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

            # CGS (deterministic from seed + past tokens)
            seed = get_seed(cfg.seed, past_tokens)
            u = draw_u(seed, cgs_gen)

            result_dict = {
                "top_k_gumbel_scores": support_gumbel_scores,
                "sampled_gumbel_scores": pairwise_gumbel_scores,
                "sampled_support_idx": sampled_support_idx,
                "logit_rank": logit_rank,
            }

            results.append(result_dict)
            past_tokens.append(sampled_token)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def save_verification_results(results: list[dict], save_dir: str) -> str:
    """Save verification results to pickle file."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    output_file = save_path / "all_prompts.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved verification results to {output_file}")
    return str(output_file)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify generated tokens using GLS/CGS")
    parser.add_argument("--input", type=str, required=True, help="Path to generated_outputs.pkl")
    parser.add_argument("--model", type=str, default=None, help="Model name (must match generation)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (must match generation)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k (must match generation)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (must match generation)")
    parser.add_argument("--seed", type=int, default=None, help="Seed (must match generation)")
    parser.add_argument("--gumbel-sigmas", type=str, default=None, help="Comma-separated sigma values")
    parser.add_argument("--support-size", type=int, default=None, help="Number of support tokens")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = VerificationConfig()

    # Override config
    if args.model is not None:
        cfg.model_name = args.model
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.seed is not None:
        cfg.seed = args.seed
    if args.gumbel_sigmas is not None:
        cfg.gumbel_sigmas = [float(s.strip()) for s in args.gumbel_sigmas.split(',')]
    if args.support_size is not None:
        cfg.support_size = args.support_size
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    else:
        # Use same directory as input
        cfg.save_dir = str(Path(args.input).parent)

    print("=" * 80)
    print("TOKEN VERIFICATION (GLS/CGS)")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Model: {cfg.model_name}")
    print(f"Temperature: {cfg.temperature}")
    print(f"Top-k: {cfg.top_k}")
    print(f"Top-p: {cfg.top_p}")
    print(f"Seed: {cfg.seed}")
    print(f"Gumbel sigmas: {cfg.gumbel_sigmas}")
    print(f"Support size: {cfg.support_size}")
    print(f"Save dir: {cfg.save_dir}")
    print("=" * 80)

    # Load generated outputs
    print(f"\nLoading generated outputs from {args.input}...")
    with open(args.input, 'rb') as f:
        outputs = pickle.load(f)
    print(f"Loaded {len(outputs)} generated outputs")

    # Verify
    results = verify_outputs(cfg, outputs)
    print(f"Verified {len(results)} tokens")

    # Save
    output_file = save_verification_results(results, cfg.save_dir)

    print("\nDone! Verification results saved to:", output_file)


if __name__ == "__main__":
    main()
