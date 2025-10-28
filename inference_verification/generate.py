"""
Text Generation using vLLM

This script handles generating text from prompts using vLLM with Gumbel-max sampling.
The generated sequences are saved for later verification.
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import vllm
from vllm import LLM, SamplingParams, RequestOutput
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional
import pickle
from dataclasses import dataclass, field
import gc
from datetime import datetime


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

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
    save_dir: str = "generated_outputs"

    # vLLM settings
    gpu_memory_utilization: float = 0.7


def load_prompts(cfg: GenerationConfig) -> list[list[int]]:
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


def generate_with_vllm(cfg: GenerationConfig, prompts: list[list[int]], max_model_len: Optional[int] = None) -> list[RequestOutput]:
    """Generate sequences using vLLM with Gumbel-max sampling."""
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


def save_outputs(outputs: list[RequestOutput], save_dir: str) -> str:
    """Save generated outputs to pickle file."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    output_file = save_path / "generated_outputs.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(outputs, f)

    print(f"Saved {len(outputs)} generated outputs to {output_file}")
    return str(output_file)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate text using vLLM")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--n-prompts", type=int, default=None, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max model sequence length")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = GenerationConfig()
    max_model_len = args.max_model_len

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
    if args.gpu_memory_utilization is not None:
        cfg.gpu_memory_utilization = args.gpu_memory_utilization
    if args.save_dir is not None:
        cfg.save_dir = args.save_dir
    else:
        # Create timestamped directory
        datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.save_dir = f"generated_outputs/{datestr}"

    print("=" * 80)
    print("TEXT GENERATION WITH vLLM")
    print("=" * 80)
    print(f"Model: {cfg.model_name}")
    print(f"Prompts: {cfg.n_prompts}")
    print(f"Max tokens: {cfg.max_tokens}")
    print(f"Temperature: {cfg.temperature}")
    print(f"Top-k: {cfg.top_k}")
    print(f"Top-p: {cfg.top_p}")
    print(f"Seed: {cfg.seed}")
    print(f"Save dir: {cfg.save_dir}")
    print("=" * 80)

    # Load prompts
    prompts = load_prompts(cfg)
    print(f"Loaded {len(prompts)} prompts")

    # Generate
    outputs = generate_with_vllm(cfg, prompts, max_model_len)
    print(f"Generated {len(outputs)} outputs")

    # Save
    output_file = save_outputs(outputs, cfg.save_dir)

    print("\nDone! Generated outputs saved to:", output_file)


if __name__ == "__main__":
    main()
