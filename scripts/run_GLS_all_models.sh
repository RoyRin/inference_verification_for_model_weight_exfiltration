#!/bin/bash
# Run Gumbel experiments for multiple models

# Configuration
N_PROMPTS=2000
MAX_TOKENS=1000
GPU_MEM=0.85
SIGMAS="0.001,0.01,0.1,1.0"
SUPPORT_SIZE=500

MAX_THRESHOLDS=500

# Create a timestamped directory for this sweep
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="logit_rank__multi_model_sweep_${TIMESTAMP}"
mkdir -p "$SWEEP_DIR"

echo "========================================================================"
echo "Running multi-model Gumbel experiments"
echo "Sweep directory: $SWEEP_DIR"
echo "Sigmas: $SIGMAS"
echo "N_prompts: $N_PROMPTS, Max_tokens: $MAX_TOKENS"
echo "========================================================================"

# Define models
declare -a MODELS=(
    #"openai/gpt-oss-120b"
    "Qwen/Qwen1.5-MoE-A2.7B"

    "Qwen/Qwen3-30B-A3B"
    #"mistralai/Mixtral-8x7B-Instruct-v0.1"
    #"meta-llama/Llama-3.1-70B-Instruct"
    #"Qwen/Qwen2.5-72B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Run experiments for each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Processing model: $MODEL"
    echo "========================================================================"

    # Create model-specific directory
    MODEL_NAME=$(echo "$MODEL" | tr '/' '_')
    MODEL_DIR="${SWEEP_DIR}/${MODEL_NAME}"

    # Run experiment
    python ../inference_verification/run_generate_and_verify.py \
        --model "$MODEL" \
        --n-prompts $N_PROMPTS \
        --max-tokens $MAX_TOKENS \
        --gpu-memory-utilization $GPU_MEM \
        --gumbel-sigmas "$SIGMAS" \
        --sweep-dir "$MODEL_DIR" \
        --support-size $SUPPORT_SIZE \
        --max-model-len 8192

    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed experiment for $MODEL"

        # Run analysis
        echo "Running analysis..."
        RESULTS_DIR=$(ls -td ${MODEL_DIR}/gumbel_cgs_analysis_results/* 2>/dev/null | head -1)
        if [ -z "$RESULTS_DIR" ]; then
            RESULTS_DIR="${MODEL_DIR}/results"
        fi

        python ../inference_verification/analysis/analyze_thresholds.py \
            --folder "$RESULTS_DIR" \
            --max-thresholds $MAX_THRESHOLDS \
            --skip-logistic-regression

        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed analysis for $MODEL"

            # Run two-step classifier analysis
            echo "Running two-step classifier analysis..."
            python ../inference_verification/analysis/analyze_two_step_classifier.py \
                --folder "$RESULTS_DIR" \
                --max-thresholds $MAX_THRESHOLDS \
                --rank-threshold 4

            if [ $? -eq 0 ]; then
                echo "✓ Successfully completed two-step classifier analysis for $MODEL"
            else
                echo "✗ Two-step classifier analysis failed for $MODEL"
            fi
        else
            echo "✗ Analysis failed for $MODEL"
        fi
    else
        echo "✗ Experiment failed for $MODEL"
    fi
done

echo ""
echo "========================================================================"
echo "All experiments complete!"
echo "Results saved to: $SWEEP_DIR"
echo "========================================================================"
echo ""
echo "To create combined plots, run:"
echo "python ../inference_verification/analysis/plot_multi_model_comparison.py --sweep-dir $SWEEP_DIR"
