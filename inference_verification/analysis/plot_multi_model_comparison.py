#!/usr/bin/env python
"""
Plot FPR vs Bit Rate comparison across multiple models.
"""

import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def load_fpr_bitrate_data(results_dir):
    """Load precomputed FPR vs Bit Rate data from results directory."""
    data_file = Path(results_dir) / "fpr_vs_bitrate.pkl"
    if not data_file.exists():
        print(f"Warning: {data_file} not found")
        return None

    with open(data_file, 'rb') as f:
        mean_bits_by_sigma = pickle.load(f)

    return mean_bits_by_sigma


def plot_multi_model_comparison(sweep_dir, sigmas=[0.01, 0.05]):
    """Create comparison plots for all models in sweep directory."""
    sweep_dir = Path(sweep_dir)

    # Find all model directories
    model_dirs = []
    for subdir in sweep_dir.iterdir():
        if subdir.is_dir():
            # Look for results folder
            results_dir = None
            if (subdir / "results").exists():
                results_dir = subdir / "results"
            else:
                # Try to find gumbel_cgs_analysis_results subfolder
                gumbel_dirs = list((subdir / "gumbel_cgs_analysis_results").glob("*"))
                if gumbel_dirs:
                    results_dir = sorted(gumbel_dirs)[-1]  # Most recent

            if results_dir and (results_dir / "all_prompts.pkl").exists():
                model_name = subdir.name.replace('_', '/')
                model_dirs.append((model_name, results_dir))

    if not model_dirs:
        print(f"Error: No valid model results found in {sweep_dir}")
        return

    print(f"Found {len(model_dirs)} models:")
    for model_name, _ in model_dirs:
        print(f"  - {model_name}")

    # Define colors and markers for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'v', 'D', 'P']

    # Create plots for each sigma
    for sigma in sigmas:
        plt.figure(figsize=(12, 8))

        for idx, (model_name, results_dir) in enumerate(model_dirs):
            print(f"\nProcessing {model_name} for sigma={sigma}...")
            mean_bits_by_sigma = load_fpr_bitrate_data(results_dir)
            if mean_bits_by_sigma is None:
                continue

            # Check if this sigma was computed
            if sigma not in mean_bits_by_sigma:
                print(f"  ✗ Sigma {sigma} not found in precomputed data")
                continue

            try:
                # Extract FPR and bit rate values for this sigma
                fpr_bitrate_dict = mean_bits_by_sigma[sigma]
                fpr_values = sorted(fpr_bitrate_dict.keys())
                bitrate_values = [fpr_bitrate_dict[fpr] for fpr in fpr_values]

                # Plot
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]

                # Simplify model name for legend
                if 'Llama' in model_name:
                    label = 'Llama-' + model_name.split('-')[3]  # e.g., "Llama-8B" or "Llama-3B"
                elif 'Qwen' in model_name:
                    label = 'Qwen-' + model_name.split('-')[1]  # e.g., "Qwen-30B"
                elif 'Mixtral' in model_name:
                    label = 'Mixtral-8x7B'
                else:
                    label = model_name

                plt.plot(fpr_values, bitrate_values,
                        marker=marker, markersize=4, markevery=10,
                        color=color, linewidth=2, label=label, alpha=0.8)

                print(f"  ✓ Plotted {model_name}")

            except Exception as e:
                print(f"  ✗ Failed to process {model_name}: {e}")

        plt.xlabel("False Positive Rate (%)", fontsize=18)
        plt.ylabel("Extractable Information (%)", fontsize=18)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"FPR vs Bit Rate Comparison (σ={sigma})", fontsize=20, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=14, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        datestring = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = sweep_dir / f"multi_model_comparison_sigma{sigma}_{datestring}"
        plt.savefig(f"{output_file}.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{output_file}.pdf", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Saved comparison plot to {output_file}.png/.pdf")

    print(f"\n{'='*80}")
    print("All comparison plots created successfully!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Create multi-model comparison plots")
    parser.add_argument("--sweep-dir", type=str, required=True,
                       help="Directory containing model experiment results")
    parser.add_argument("--sigmas", type=str, default="0.01,0.05",
                       help="Comma-separated list of sigma values to plot (default: 0.01,0.05)")

    args = parser.parse_args()

    sigmas = [float(s.strip()) for s in args.sigmas.split(',')]

    plot_multi_model_comparison(args.sweep_dir, sigmas)


if __name__ == "__main__":
    main()
