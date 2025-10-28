#!/bin/bash
set -e  # Exit on error

# Configuration
MODEL="meta-llama/Llama-3.2-3B-Instruct"
N_PROMPTS=100
MAX_TOKENS=100
SIGMAS="0.01,1.0"
SUPPORT_SIZE=100
MAX_MODEL_LEN=2048

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="test_run_${TIMESTAMP}"

echo "================================================================================"
echo "MINIMAL TEST: SEPARATED GENERATION AND VERIFICATION"
echo "================================================================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

echo "Step 1: Generating text with vLLM..."
python ../inference_verification/generate.py \
  --model "$MODEL" \
  --n-prompts $N_PROMPTS \
  --max-tokens $MAX_TOKENS \
  --save-dir "$OUTPUT_DIR" \
  --max-model-len $MAX_MODEL_LEN

echo ""
echo "Step 2: Verifying generated tokens (computing GLS/CGS scores)..."
python ../inference_verification/verify.py \
  --input "$OUTPUT_DIR/generated_outputs.pkl" \
  --model "$MODEL" \
  --gumbel-sigmas "$SIGMAS" \
  --support-size $SUPPORT_SIZE

echo ""
echo "Step 3: Running threshold analysis (Pareto plots)..."
python ../inference_verification/analysis/analyze_thresholds.py \
  --folder "$OUTPUT_DIR" \
  --max-thresholds 100 \
  --skip-logistic-regression

echo ""
echo "Step 4: Running two-step classifier analysis..."
python ../inference_verification/analysis/analyze_two_step_classifier.py \
  --folder "$OUTPUT_DIR" \
  --max-thresholds 100

echo ""
echo "================================================================================"
echo "✓ ALL TESTS COMPLETE!"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "  - Generated outputs: $OUTPUT_DIR/generated_outputs.pkl"
echo "  - Verification results: $OUTPUT_DIR/all_prompts.pkl"
echo "  - Plots: $OUTPUT_DIR/images/"
echo ""
echo "Generated Pareto plots:"
ls -1 $OUTPUT_DIR/images/
echo "================================================================================"


