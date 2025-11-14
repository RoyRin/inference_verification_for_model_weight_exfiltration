#!/bin/bash
# Test script for GLS-based sampling generation

set -e

echo "Testing GLS-based Sampling Generation"
echo "======================================"
echo ""

# Test with minimal settings for quick verification
python -m inference_verification.generate_gls_sampling \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --n-prompts 2 \
    --max-tokens 20 \
    --temperature 1.0 \
    --top-k 50 \
    --top-p 0.95 \
    --seed 42 \
    --top-n-candidates 100 \
    --gumbel-sigma 1.0 \
    --save-dir "test_outputs_gls"

echo ""
echo "Test completed! Check test_outputs_gls/ for results."
