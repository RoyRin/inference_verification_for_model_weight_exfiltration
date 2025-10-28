# %%
import os
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from datetime import datetime


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Analyze Gumbel thresholds from experiment results")
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
    "--skip-logistic-regression",
    action="store_true",
    help="Skip logistic regression training and analysis (makes script much faster)",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=None,
    help="Only process this sigma value (default: process all sigmas)",
)
args = parser.parse_args()

folder = args.folder
max_thresholds = args.max_thresholds
skip_lr = args.skip_logistic_regression
sigma_filter = args.sigma
filename = f"{folder}/all_prompts.pkl"

# Create datestring for plot filenames
datestring = datetime.now().strftime("%Y%m%d_%H%M%S")

# Determine vocab size based on model name in folder path
def get_vocab_size_from_path(folder_path):
    """Extract model name from path and return appropriate vocab size."""
    folder_lower = folder_path.lower()
    if 'llama' in folder_lower:
        return 128256
    elif 'qwen' in folder_lower:
        return 151646
    else:
        # Default fallback
        return 250000

VOCAB_SIZE = get_vocab_size_from_path(folder)
VOCAB_SIZE_BITS = np.log2(np.array(VOCAB_SIZE))

print(f"Using vocab size: {VOCAB_SIZE:,} (detected from folder path: {folder})")

# Helper function to save plots in both PNG and PDF formats
def save_plot(filepath_without_ext, dpi=150, bbox_inches='tight'):
    """Save plot as both PNG and PDF."""
    plt.savefig(f"{filepath_without_ext}.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(f"{filepath_without_ext}.pdf", dpi=dpi, bbox_inches=bbox_inches)

data = pickle.load(open(filename, "rb"))
# %%
print(data[0].keys())

sampled_gumbel_scores = []
top_k_gumbel_scores = []
ranks = []


def normalize_score(score: float, min_score: int = -20) -> float:
    if np.isinf(score):
        score = min_score  # Cap inf values for stability
    if score < min_score:
        score = min_score
    return score


for result in tqdm(data):
    sampled_gumbel_scores.append(result["sampled_gumbel_scores"])
    top_k_gumbel_scores.append(result["top_k_gumbel_scores"])
    ranks.append(result["sampled_support_idx"])

print(sampled_gumbel_scores[0])

# %%

plt.figure()
plt.yscale("log")
plt.title("Histogram of Ranks of Sampled Token")
plt.hist(ranks, bins=50)
plt.close()

# %%

old_ranks = []
sigma = 0.01

for result in tqdm(data):
    true_score = normalize_score(result["sampled_gumbel_scores"][sigma])
    rank = (result["top_k_gumbel_scores"][sigma] > true_score).sum().item()
    old_ranks.append(rank)

plt.figure()
plt.yscale("log")
plt.title("Histogram of Ranks of Sampled Token Using sum() > true_score")
plt.hist(old_ranks, bins=50)
plt.close()

# %%


images_dir = f"{folder}/images"
os.makedirs(images_dir, exist_ok=True)

sigmas = list(sampled_gumbel_scores[0].keys())

# Filter to specific sigma if requested
if sigma_filter is not None:
    if sigma_filter in sigmas:
        sigmas = [sigma_filter]
        print(f"Processing only sigma={sigma_filter}")
    else:
        print(f"Warning: Requested sigma={sigma_filter} not found in data. Available: {sigmas}")
        print("Processing all sigmas instead.")
else:
    print(f"Processing all sigmas: {sigmas}")

print("\n" + "="*80)
print("CREATING HISTOGRAM PLOTS")
print("="*80)
for sigma in sigmas:
    print(f"Creating histogram for sigma={sigma}...")
    sampled_scores = [normalize_score(score[sigma]) for score in sampled_gumbel_scores]

    plt.figure(figsize=(10, 6))
    plt.yscale("log")
    plt.hist(sampled_scores, bins=50, alpha=0.5, label="Sampled Gumbel")
    plt.title(f"Gumbel Score Distributions (sigma={sigma})", fontsize=18, fontweight='bold')
    plt.xlabel("Gumbel Score", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    save_plot(f"{images_dir}/gumbel_score_histogram_sigma{sigma}_{datestring}")
    plt.close()  # Close figure to free memory
    print(f"  ✓ Saved histogram for sigma={sigma}")

# %%
percentiles = [90, 95, 98, 99, 99.5, 99.8, 99.9]

# Store FPR values for all sigmas to plot together
mean_bits_by_sigma = {}

print("\n" + "="*80)
print("COMPUTING FPR VS BIT RATE (GLS THRESHOLD)")
print("="*80)

# Store results for both versions
mean_bits_by_sigma_pass_only = {}  # Only count passing tokens
mean_bits_by_sigma_with_fp = {}    # Count passing + FP penalty

for sigma in sigmas:
    print(f"Processing sigma={sigma}...")
    sampled_scores = np.array([normalize_score(score[sigma]) * -1 for score in sampled_gumbel_scores])

    # Use unique empirical scores as thresholds
    unique_thresholds = np.unique(sampled_scores)
    print(f"  Found {len(unique_thresholds)} unique thresholds")

    # Subsample thresholds if there are too many
    if len(unique_thresholds) > max_thresholds:
        print(f"  Subsampling to {max_thresholds} thresholds (from {len(unique_thresholds)})")
        indices = np.linspace(0, len(unique_thresholds)-1, max_thresholds, dtype=int)
        unique_thresholds = unique_thresholds[indices]

    percentile_mean_bits_pass_only = {}
    percentile_mean_bits_with_fp = {}

    max_possible = np.log2(np.array(VOCAB_SIZE))

    for i, threshold in enumerate(tqdm(unique_thresholds, desc=f"  Thresholds for sigma={sigma}")):
        # FPR: % of sampled tokens that fail verification
        fpr = (sampled_scores >= threshold).sum() / len(sampled_scores)
        fpr *= 100

        # Compute bits per-token (two versions)
        total_bits_pass_only = 0
        total_bits_with_fp = 0

        for idx in range(len(data)):
            sampled_score = sampled_scores[idx]
            scores_K = top_k_gumbel_scores[idx][sigma] * -1

            # Check if THIS sampled token passes or fails verification
            if sampled_score < threshold:  # PASSES verification
                num_passing = (scores_K <= threshold).sum()
                bits = np.log2(num_passing) if num_passing > 0 else 0

                total_bits_pass_only += (bits / max_possible) * 100
                total_bits_with_fp += (bits / max_possible) * 100
            else:  # FAILS verification
                # Pass-only version: don't count failed tokens
                total_bits_pass_only += 0
                # With FP version: count full vocabulary
                total_bits_with_fp += (np.log2(VOCAB_SIZE) / max_possible) * 100

        percentile_mean_bits_pass_only[fpr] = total_bits_pass_only / len(data)
        percentile_mean_bits_with_fp[fpr] = total_bits_with_fp / len(data)

    mean_bits_by_sigma_pass_only[sigma] = percentile_mean_bits_pass_only
    mean_bits_by_sigma_with_fp[sigma] = percentile_mean_bits_with_fp
    print(f"  ✓ Completed sigma={sigma}")

# Use the with_fp version for backward compatibility
mean_bits_by_sigma = mean_bits_by_sigma_with_fp

print("\n  ✓ FPR vs Bit Rate computation complete")

# Save the computed FPR vs Bit Rate data
fpr_bitrate_output = f"{folder}/fpr_vs_bitrate.pkl"
with open(fpr_bitrate_output, 'wb') as f:
    pickle.dump(mean_bits_by_sigma, f)
print(f"  ✓ Saved FPR vs Bit Rate data to {fpr_bitrate_output}")

print("\nCreating FPR vs Bit Rate plots...")

# Plot 1: Pass-only (no FP penalty)
plt.figure(figsize=(12, 7))
for sigma in sigmas:
    fpr_values = sorted(mean_bits_by_sigma_pass_only[sigma].keys())
    mean_bits_values = [mean_bits_by_sigma_pass_only[sigma][fpr] for fpr in fpr_values]

    plt.plot(fpr_values, mean_bits_values, marker="o", markersize=3, label=f"sigma={sigma}")

plt.xlabel("False Positive Rate (%)", fontsize=18)
plt.ylabel("Extractable Information (%)", fontsize=18)
plt.xscale("log")
plt.yscale("log")
plt.title("Pass-Only: Information from tokens passing verification", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
save_plot(f"{images_dir}/mean_bits_vs_fpr_pass_only_{datestring}")
plt.close()
print("  ✓ Saved FPR vs Bit Rate plot (pass-only)")

# Plot 2: With FP penalty
plt.figure(figsize=(12, 7))
for sigma in sigmas:
    fpr_values = sorted(mean_bits_by_sigma_with_fp[sigma].keys())
    mean_bits_values = [mean_bits_by_sigma_with_fp[sigma][fpr] for fpr in fpr_values]

    plt.plot(fpr_values, mean_bits_values, marker="o", markersize=3, label=f"sigma={sigma}")

plt.xlabel("False Positive Rate (%)", fontsize=18)
plt.ylabel("Extractable Information (%)", fontsize=18)
plt.xscale("log")
plt.yscale("log")
plt.title("With FP Penalty: Pass tokens + failed tokens = full vocab", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
save_plot(f"{images_dir}/mean_bits_vs_fpr_with_fp_{datestring}")
plt.close()
print("  ✓ Saved FPR vs Bit Rate plot (with FP penalty)")

print(f"\n{'='*80}")
print(f"All plots saved to: {images_dir}/")
print(f"  - Histogram plots for each sigma: gumbel_score_histogram_sigma*_{datestring}.png/.pdf")
print(f"  - FPR vs Bit Rate (pass-only): mean_bits_vs_fpr_pass_only_{datestring}.png/.pdf")
print(f"  - FPR vs Bit Rate (with FP penalty): mean_bits_vs_fpr_with_fp_{datestring}.png/.pdf")
print(f"{'='*80}")

# %%
# ============================================================================
# LOGISTIC REGRESSION: Train classifier on (score, rank) features
# ============================================================================

if not skip_lr:
    print(f"\n{'='*80}")
    print("LOGISTIC REGRESSION TRAINING")
    print(f"{'='*80}\n")

    classifier_results = {}

    for idx, sigma in enumerate(sigmas):
        print(f"\n--- Training classifier for sigma={sigma} ({idx+1}/{len(sigmas)}) ---")
    
        # Extract features and labels
        features_list = []
        labels_list = []
    
        print(f"  Extracting features...")
        for result in tqdm(data, desc=f"  Extracting features (sigma={sigma})"):
            sampled_score = normalize_score(result["sampled_gumbel_scores"][sigma])
            top_k_scores = result["top_k_gumbel_scores"][sigma]
    
            # Compute rank for sampled token: number of tokens with higher score
            sampled_rank = (top_k_scores > sampled_score).sum().item()
    
            # Positive example: sampled token
            features_list.append([sampled_score, sampled_rank])
            labels_list.append(1)
    
            # Negative examples: all other tokens in top-K
            for i, score in enumerate(top_k_scores):
                score = normalize_score(score.item())
                # Compute rank for this token
                rank = (top_k_scores > score).sum().item()
    
                features_list.append([score, rank])
                labels_list.append(0)
    
        features = np.array(features_list)
        labels = np.array(labels_list)
    
        print(f"Total samples: {len(labels):,}")
        print(f"  Positive (sampled): {np.sum(labels):,} ({100*np.mean(labels):.2f}%)")
        print(f"  Negative (other): {len(labels) - np.sum(labels):,} ({100*(1-np.mean(labels)):.2f}%)")
    
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
    
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # Train logistic regression
        clf = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        clf.fit(X_train_scaled, y_train)
    
        print(f"Converged: {clf.n_iter_[0] < clf.max_iter}")
        print(f"Coefficients (score, rank): {clf.coef_[0]}")
        print(f"Intercept: {clf.intercept_[0]:.3f}")
    
        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
        print("\nTest Set Performance:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(f"                 Predicted")
        print(f"               Negative  Positive")
        print(f"Actual Negative  {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Positive  {cm[1,0]:6d}  {cm[1,1]:6d}")
    
        # ROC curve
        fpr_roc, tpr_roc, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr_roc, tpr_roc)
        print(f"\nROC AUC: {roc_auc:.4f}")
    
        # Store results
        classifier_results[sigma] = {
            'classifier': clf,
            'scaler': scaler,
            'roc_auc': roc_auc,
            'fpr': fpr_roc,
            'tpr': tpr_roc,
            'cm': cm,
        }
    
    # %%
    # Plot ROC curves for all sigmas
    print("\nCreating ROC curves plot...")
    plt.figure(figsize=(10, 8))
    for sigma in sigmas:
        result = classifier_results[sigma]
        plt.plot(result['fpr'], result['tpr'], linewidth=2,
                 label=f'sigma={sigma} (AUC={result["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC Curves: Logistic Regression (Score + Rank)', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(alpha=0.3)
    save_plot(f"{images_dir}/logistic_regression_roc_curves_{datestring}")
    plt.close()
    print("  ✓ Saved ROC curves plot")
    
    # %%
    # ============================================================================
    # LOGISTIC REGRESSION: Compute FPR vs Bit Rate using classifier probabilities
    # ============================================================================
    
    print(f"\n{'='*80}")
    print("COMPUTING FPR VS BIT RATE FOR LOGISTIC REGRESSION")
    print(f"{'='*80}\n")
    
    lr_mean_bits_by_sigma = {}
    
    for idx, sigma in enumerate(sigmas):
        print(f"Processing sigma={sigma} ({idx+1}/{len(sigmas)})...")
    
        clf = classifier_results[sigma]['classifier']
        scaler = classifier_results[sigma]['scaler']
    
        # Compute probabilities for sampled tokens
        sampled_probs = []
        for result in data:
            sampled_score = normalize_score(result["sampled_gumbel_scores"][sigma])
            top_k_scores = result["top_k_gumbel_scores"][sigma]
            sampled_rank = (top_k_scores > sampled_score).sum().item()
    
            features = np.array([[sampled_score, sampled_rank]])
            features_scaled = scaler.transform(features)
            prob = clf.predict_proba(features_scaled)[0, 1]  # Probability of being valid
            sampled_probs.append(prob)
    
        sampled_probs = np.array(sampled_probs)
    
        # Compute probabilities for all top-K tokens at each position
        topk_probs_by_position = []
        for result in tqdm(data, desc=f"Computing top-K probs (sigma={sigma})"):
            top_k_scores = result["top_k_gumbel_scores"][sigma]
    
            position_probs = []
            for score in top_k_scores:
                score = normalize_score(score.item())
                rank = (top_k_scores > score).sum().item()
    
                features = np.array([[score, rank]])
                features_scaled = scaler.transform(features)
                prob = clf.predict_proba(features_scaled)[0, 1]
                position_probs.append(prob)
    
            topk_probs_by_position.append(np.array(position_probs))
    
        # Sweep over probability thresholds
        unique_thresholds = np.unique(sampled_probs)

        percentile_mean_bits = {}
        max_possible = np.log2(np.array(VOCAB_SIZE))
    
        for threshold in tqdm(unique_thresholds, desc=f"Sweeping thresholds (sigma={sigma})"):
            # FPR: % of sampled tokens with prob < threshold (rejected as invalid)
            fpr = (sampled_probs < threshold).sum() / len(sampled_probs)
            fpr *= 100
    
            # Bit rate: For each position, count tokens with prob >= threshold (pass verification)
            total_bits = 0
            for position_probs in topk_probs_by_position:
                num_valid = (position_probs >= threshold).sum()
                if num_valid > 0:
                    total_bits += ((np.log2(num_valid)) / max_possible) * 100
                    
                # If num_valid == 0, contributes 0 bits (no valid tokens)
    
            percentile_mean_bits[fpr] = total_bits / len(topk_probs_by_position)
    
        lr_mean_bits_by_sigma[sigma] = percentile_mean_bits
        print(f"  ✓ Completed sigma={sigma}")
    
    # %%
    # Plot comparison: GLS threshold vs Logistic Regression
    print("\nCreating GLS vs LR comparison plot...")
    plt.figure(figsize=(12, 7))
    
    # Plot GLS threshold curves
    for sigma in sigmas:
        fpr_values = sorted(mean_bits_by_sigma[sigma].keys())
        mean_bits_values = [mean_bits_by_sigma[sigma][fpr] for fpr in fpr_values]
        plt.plot(fpr_values, mean_bits_values, marker="o", markersize=3,
                 label=f"GLS sigma={sigma}", linestyle='--', alpha=0.6)
    
    # Plot Logistic Regression curves
    for sigma in sigmas:
        fpr_values = sorted(lr_mean_bits_by_sigma[sigma].keys())
        mean_bits_values = [lr_mean_bits_by_sigma[sigma][fpr] for fpr in fpr_values]
        plt.plot(fpr_values, mean_bits_values, marker="x", markersize=4,
                 label=f"LR sigma={sigma}", linewidth=2)
    
    plt.xlabel("False Positive Rate (FPR %)", fontsize=18)
    plt.ylabel("Bit Rate (%)", fontsize=18)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Mean Bit Rate vs FPR: GLS Threshold vs Logistic Regression", fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(f"{images_dir}/gls_vs_lr_comparison_{datestring}")
    plt.close()
    print("  ✓ Saved GLS vs LR comparison plot")
    
    # %%
    # Save trained classifiers
    print("\nSaving trained classifiers...")
    classifier_output_path = f"{folder}/trained_classifiers.pkl"
    with open(classifier_output_path, 'wb') as f:
        pickle.dump(classifier_results, f)
    
    print(f"\n{'='*80}")
    print(f"Trained classifiers saved to: {classifier_output_path}")
    print(f"ROC curves saved to: {images_dir}/logistic_regression_roc_curves_{datestring}.png/.pdf")
    print(f"GLS vs LR comparison saved to: {images_dir}/gls_vs_lr_comparison_{datestring}.png/.pdf")
    print(f"FP adjusted plot saved to: {images_dir}/mean_bits_vs_fpr_fp_adjusted_{datestring}.png/.pdf")
    print(f"{'='*80}")
# %%

print("\n" + "="*80)
print("COMPUTING FP ADJUSTED BIT RATE")
print("="*80)

percentiles = [90, 95, 98, 99, 99.5, 99.8, 99.9]

# Store FPR values for all sigmas to plot together
mean_bits_by_sigma = {}

best_possible = 100
max_possible = np.log2(np.array(VOCAB_SIZE))
fp_bit_rate = max_possible
# fp_bit_rate = np.log2(8)

for sigma in sigmas:
    print(f"Processing sigma={sigma}...")
    sampled_scores = np.array([normalize_score(score[sigma]) * -1 for score in sampled_gumbel_scores])

    # Use unique empirical scores as thresholds
    unique_thresholds = np.unique(sampled_scores)

    # skip the first score, which should be exact match score
    unique_thresholds = np.sort(unique_thresholds)[1:]

    # Subsample thresholds if there are too many
    if len(unique_thresholds) > max_thresholds:
        print(f"  Subsampling to {max_thresholds} thresholds (from {len(unique_thresholds)})")
        indices = np.linspace(0, len(unique_thresholds)-1, max_thresholds, dtype=int)
        unique_thresholds = unique_thresholds[indices]

    percentile_mean_bits = {}

    for threshold in tqdm(unique_thresholds, desc=f"  Thresholds for sigma={sigma}"):
        fpr = (sampled_scores >= threshold).sum() / len(sampled_scores)
        fpr *= 100

        # Compute bits per-token with FP penalty
        total_bits = 0
        for idx in range(len(data)):
            sampled_score = sampled_scores[idx]
            scores_K = top_k_gumbel_scores[idx][sigma] * -1

            # Check if THIS sampled token passes or fails verification
            if sampled_score < threshold:  # PASSES verification
                num_passing = (scores_K <= threshold).sum()
                bits = np.log2(num_passing) if num_passing > 0 else 0
            else:  # FAILS verification - use FP bit rate penalty
                bits = fp_bit_rate

            total_bits += (bits / max_possible) * 100

        bit_rate = total_bits / len(data)
        percentile_mean_bits[fpr] = bit_rate
        best_possible = min(best_possible, bit_rate)

    mean_bits_by_sigma[sigma] = percentile_mean_bits
    print(f"  ✓ Completed sigma={sigma}")

print("\nCreating FP adjusted plot...")
plt.figure(figsize=(12, 7))
for sigma in sigmas:
    fpr_values = sorted(mean_bits_by_sigma[sigma].keys())
    #
    mean_bits_values = [mean_bits_by_sigma[sigma][fpr] for fpr in fpr_values]

    plt.plot(fpr_values, mean_bits_values, marker="o", markersize=3, label=f"sigma={sigma}")

plt.xlabel("False Positive Rate (FPR)", fontsize=18)
plt.ylabel("Bit Rate (%)", fontsize=18)
# plt.xlim((0.0, 0.05))
plt.xscale("log")
plt.yscale("log")
plt.title(
    f"Mean Bit Rate (%) vs FPR for Different Sigma Values\nBest Possible at FP bit rate {fp_bit_rate:.4f}: {best_possible:.4f} %",
    fontsize=18, fontweight='bold'
)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
save_plot(f"{images_dir}/mean_bits_vs_fpr_fp_adjusted_{datestring}")
plt.close()
print("  ✓ Saved FP adjusted plot")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
# %%
