#!/usr/bin/env python3
"""
Two-Step Classifier: Good, Suspicious, Dangerous

Classification logic:
    if GLS(x) < threshold:
        if logit_rank(x) > rank_threshold:
            classify as "dangerous" (very likely threat)
        else:
            classify as "suspicious" (possible threat)
    else:
        classify as "good"

Uses logit_rank (raw logit ranking before temp/top-k/top-p) instead of GLS rank.

Generates two ROC curves:
1. Good (negative) vs (Suspicious + Dangerous) (positive)
2. (Good + Suspicious) (negative) vs Dangerous (positive)
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_curve, auc
from datetime import datetime
from transformers import AutoTokenizer


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Two-step classifier analysis (good/suspicious/dangerous)")
parser.add_argument(
    "--folder",
    type=str,
    required=True,
    help="Path to results folder (e.g., gumbel_cgs_analysis_results/20251014_174336)",
)
parser.add_argument(
    "--max-thresholds",
    type=int,
    default=1000,
    help="Maximum number of thresholds to use for FPR vs Bit Rate computation (default: 1000)",
)
parser.add_argument(
    "--rank-threshold",
    type=int,
    default=4,
    help="[IGNORED] Rank thresholds are hardcoded as [None, 1, 2, 3, 4, 8, 16, 32, 40, 64]",
)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=None,
    help="Vocabulary size (if not provided, will be inferred from model path)",
)
parser.add_argument(
    "--max-suspicious-pct",
    type=float,
    default=0.5,
    help="Maximum percentage of suspicious tokens allowed (default: 0.5)",
)
args = parser.parse_args()

folder = args.folder
max_thresholds = args.max_thresholds
max_suspicious_pct = args.max_suspicious_pct
filename = f"{folder}/all_prompts.pkl"

# Create datestring for plot filenames
datestring = datetime.now().strftime("%Y%m%d_%H%M%S")


# Get vocab size
def get_vocab_size_from_path(folder_path):
    """Extract model name from folder path and get vocab size from tokenizer."""
    # Try to extract model name from path
    # Expected format: .../meta-llama_Llama-3.2-3B-Instruct/...
    parts = folder_path.split('/')
    model_folder = None

    for part in parts:
        if '_' in part and any(x in part.lower() for x in ['llama', 'qwen', 'mistral', 'mixtral']):
            # Convert folder name back to model name (replace _ with /)
            # Find the first underscore which separates org from model
            first_underscore = part.find('_')
            if first_underscore > 0:
                model_name = part[:first_underscore] + '/' + part[first_underscore+1:]
                model_folder = model_name
                break

    if model_folder is None:
        print("Warning: Could not infer model name from folder path")
        return None

    print(f"Inferred model name: {model_folder}")

    try:
        print(f"Loading tokenizer for {model_folder}...")
        tokenizer = AutoTokenizer.from_pretrained(model_folder, trust_remote_code=True)
        vocab_size = len(tokenizer)
        print(f"Vocabulary size: {vocab_size}")
        return vocab_size
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        return None


if args.vocab_size is not None:
    vocab_size = args.vocab_size
    print(f"Using provided vocab size: {vocab_size}")
else:
    vocab_size = get_vocab_size_from_path(folder)
    if vocab_size is None:
        vocab_size = 250000  # Default fallback
        print(f"Warning: Using default vocab size: {vocab_size}")


def normalize_score(score: float, min_score: int = -20) -> float:
    if np.isinf(score):
        score = min_score  # Cap inf values for stability
    if score < min_score:
        score = min_score
    return score


def save_plot(filepath_without_ext, dpi=150, bbox_inches='tight'):
    """Save plot as both PNG and PDF."""
    plt.savefig(f"{filepath_without_ext}.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(f"{filepath_without_ext}.pdf", dpi=dpi, bbox_inches=bbox_inches)


# Load data
print("Loading data...")
data = pickle.load(open(filename, "rb"))
print(f"Loaded {len(data)} prompts")

# Extract scores and ranks
sampled_gumbel_scores = []
top_k_gumbel_scores = []
logit_ranks = []

for result in tqdm(data, desc="Extracting scores"):
    sampled_gumbel_scores.append(result["sampled_gumbel_scores"])
    top_k_gumbel_scores.append(result["top_k_gumbel_scores"])
    logit_ranks.append(result["logit_rank"])

images_dir = f"{folder}/images"
os.makedirs(images_dir, exist_ok=True)

# Use only sigma=1.0
sigma = 1.0
print(f"Using sigma: {sigma}")

# Define rank thresholds to test
rank_thresholds = [None, 1, 2, 3, 4, 8, 16, 32, 40, 64]

rank_thresholds = [None, 1, 4, 16, 30]

# ============================================================================
# TWO-STEP CLASSIFIER: GLS Threshold + Rank Check
# ============================================================================

print("\n" + "="*80)
print("TWO-STEP CLASSIFIER ANALYSIS")
print(f"Vocabulary size: {vocab_size}")
print(f"Testing logit rank thresholds: {rank_thresholds}")
print(f"Max suspicious tokens allowed: {max_suspicious_pct}%")
print("Using logit_rank (raw logit ranks before temp/top-k/top-p)")
print("="*80)

# Store results for each rank threshold
results_by_rank_threshold = {}

for rank_threshold in rank_thresholds:
    print(f"\n--- Processing rank_threshold={rank_threshold} ---")

    # Extract normalized scores for sigma=1.0
    sampled_scores = np.array([normalize_score(score[sigma]) * -1 for score in sampled_gumbel_scores])

    # Use pre-computed logit ranks
    sampled_ranks = np.array(logit_ranks)

    # Get unique thresholds from empirical scores
    unique_thresholds = np.unique(sampled_scores)
    print(f"  Found {len(unique_thresholds)} unique thresholds")

    # Subsample thresholds if needed
    if len(unique_thresholds) > max_thresholds:
        print(f"  Subsampling to {max_thresholds} thresholds")
        indices = np.linspace(0, len(unique_thresholds)-1, max_thresholds, dtype=int)
        unique_thresholds = unique_thresholds[indices]

    # Store results for this rank threshold
    fpr_values = []
    exfiltrable_info_values = []

    max_possible = np.log2(vocab_size)

    num_skipped = 0

    print("  Computing classifications for all thresholds...")
    for threshold in tqdm(unique_thresholds, desc=f"  Processing rank_threshold={rank_threshold}"):
        # Classify sampled tokens
        # good: GLS >= threshold
        # If rank_threshold is None:
        #   dangerous: GLS < threshold (no suspicious category)
        # Else:
        #   suspicious: GLS < threshold AND logit_rank <= rank_threshold
        #   dangerous: GLS < threshold AND logit_rank > rank_threshold

        below_threshold = sampled_scores >= threshold  # Note: scores are negated

        if rank_threshold is None:
            # No suspicious category - only good or dangerous
            good = ~below_threshold
            dangerous = below_threshold
            suspicious = np.zeros_like(below_threshold, dtype=bool)
        else:
            high_rank = sampled_ranks > rank_threshold
            good = ~below_threshold
            dangerous = below_threshold & high_rank
            suspicious = below_threshold & ~high_rank

        # Calculate percentage of suspicious tokens
        suspicious_pct = suspicious.sum() / len(sampled_scores) * 100

        # Skip this threshold if too many suspicious tokens
        if suspicious_pct > max_suspicious_pct:
            num_skipped += 1
            continue

        # FPR: Percentage of tokens classified as dangerous
        fpr = dangerous.sum() / len(sampled_scores) * 100

        # Compute exfiltrable information per token based on classification
        # Good token: log_2(number of support tokens passing GLS threshold)
        # Suspicious token: log_2(rank_threshold)
        # Dangerous token: log_2(vocab_size)

        total_bits = 0

        for idx in range(len(data)):
            if good[idx]:
                # Good: log_2(number of support tokens passing GLS threshold)
                scores_K = data[idx]["top_k_gumbel_scores"][sigma] * -1
                num_passing = (scores_K <= threshold).sum()
                if num_passing > 0:
                    total_bits += np.log2(num_passing)
            elif suspicious[idx]:
                # Suspicious: log_2(rank_threshold)
                if rank_threshold is not None and rank_threshold > 0:
                    total_bits += np.log2(rank_threshold)
                else:
                    total_bits += 0  # Should not happen, but handle edge case
            else:  # dangerous[idx]
                # Dangerous: log_2(vocab_size)
                total_bits += np.log2(vocab_size)

        # Normalize by max possible (sum of all tokens being dangerous)
        mean_bits_percent = (total_bits / (len(data) * max_possible)) * 100

        fpr_values.append(fpr)
        exfiltrable_info_values.append(mean_bits_percent)

    results_by_rank_threshold[rank_threshold] = {
        'fpr': fpr_values,
        'exfiltrable_info': exfiltrable_info_values,
    }

    print(f"  ✓ Completed rank_threshold={rank_threshold}")
    print(f"     Kept {len(fpr_values)} thresholds, skipped {num_skipped} (>{max_suspicious_pct}% suspicious)")

# ============================================================================
# PLOTTING
# ============================================================================

print("\n" + "="*80)
print("CREATING PLOT")
print("="*80)

# Single plot: FPR of dangerous tokens vs Exfiltrable Information for different rank thresholds
print("\nCreating FPR vs Exfiltrable Information plot...")
plt.figure(figsize=(12, 8))

for rank_threshold in rank_thresholds:
    fpr_values = results_by_rank_threshold[rank_threshold]['fpr']
    bits_values = results_by_rank_threshold[rank_threshold]['exfiltrable_info']

    label = f"rank={rank_threshold}" if rank_threshold is not None else "rank=None (no suspicious)"
    plt.plot(fpr_values, bits_values, marker="o", markersize=3, label=label)

plt.xlabel("FPR (%) - Dangerous tokens", fontsize=18)
plt.ylabel("Exfiltrable Information (%)", fontsize=18)
plt.xscale("log")
plt.yscale("log")
plt.title(f"FPR vs Exfiltrable Information (sigma={sigma})", fontsize=18, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
save_plot(f"{images_dir}/fpr_vs_exfiltrable_info_sigma{sigma}_{datestring}")
plt.close()
print("  ✓ Saved FPR vs Exfiltrable Information plot")

# Save results
print("\nSaving results...")
output_path = f"{folder}/two_step_classifier_results.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(results_by_rank_threshold, f)
print(f"  ✓ Saved results to: {output_path}")

print("\n" + "="*80)
print("TWO-STEP CLASSIFIER ANALYSIS COMPLETE!")
print("="*80)
print(f"Results saved to: {folder}")
print(f"  - Plot: {images_dir}/fpr_vs_exfiltrable_info_sigma{sigma}_{datestring}.png/.pdf")
print(f"  - Results data: {output_path}")
print(f"  - Sigma used: {sigma}")
print(f"  - Rank thresholds tested: {rank_thresholds}")
print("="*80)
