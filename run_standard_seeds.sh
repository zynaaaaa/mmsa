#!/bin/bash
# Run standard random seed experiments
# Plan A: Main models with seeds [0,1,2,3,4], Early Fusion with seeds [0-9]

set -e  # Exit on error

VENV_PYTHON="/Users/zhangyani/工作/Multimodal Sentiment Analysis/mmsa/.venv/bin/python"
WORK_DIR="/Users/zhangyani/工作/Multimodal Sentiment Analysis/mmsa"
RESULTS_DIR="$WORK_DIR/experiment_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

cd "$WORK_DIR"

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Starting Standard Random Seed Experiments"
echo "Plan A: Main models seeds [0,1,2,3,4]"
echo "        Early Fusion seeds [0-9]"
echo "=========================================="
echo "Results will be saved to: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Function to run experiment and capture output
run_experiment() {
    local model_name=$1
    local script_name=$2
    shift 2
    local seeds=("$@")
    
    local log_file="$RESULTS_DIR/${model_name}_${TIMESTAMP}.log"
    local result_file="$RESULTS_DIR/${model_name}_${TIMESTAMP}_results.txt"
    
    echo "Running $model_name with seeds: ${seeds[*]}"
    echo "Log: $log_file"
    
    # Run and capture both stdout and stderr
    "$VENV_PYTHON" "$script_name" --seeds "${seeds[@]}" 2>&1 | tee "$log_file"
    
    # Extract final results from log
    echo "Extracting results for $model_name..."
    grep -E "(Result for seed|Final mean|Non0_acc_2|Non0_F1_score|MAE|Corr)" "$log_file" > "$result_file" || true
    
    echo "✓ $model_name completed"
    echo ""
}

# Stage 1: Fast models
echo "=========================================="
echo "[Stage 1/4] Fast Models (~50 minutes)"
echo "=========================================="

run_experiment "text_only_mlp" "run_text_only_multiseed.py" 0 1 2 3 4

run_experiment "text_bilstm_attention" "run_text_attention_multiseed.py" 0 1 2 3 4

echo "Stage 1 completed!"
echo ""

# Stage 2: Medium speed models
echo "=========================================="
echo "[Stage 2/4] Medium Speed Models (~2 hours)"
echo "=========================================="

run_experiment "late_fusion" "run_late_fusion_multiseed.py" 0 1 2 3 4

run_experiment "lmf" "run_lmf_multiseed.py" 0 1 2 3 4

run_experiment "self_mm" "run_self_mm_multiseed.py" 0 1 2 3 4

echo "Stage 2 completed!"
echo ""

# Stage 3: Slow models
echo "=========================================="
echo "[Stage 3/4] Slow Models (~2.5 hours)"
echo "=========================================="

run_experiment "mult" "run_mult_multiseed.py" 0 1 2 3 4

echo "Stage 3 completed!"
echo ""

# Stage 4: Early Fusion (10 seeds)
echo "=========================================="
echo "[Stage 4/4] Early Fusion 10 Seeds (~2.5 hours)"
echo "=========================================="

run_experiment "early_fusion" "run_early_fusion_multiseed.py" 0 1 2 3 4 5 6 7 8 9

echo "Stage 4 completed!"
echo ""

# Generate summary
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary of result files:"
ls -lh "$RESULTS_DIR"/*_${TIMESTAMP}_results.txt
echo ""
echo "Experiment configuration:"
echo "  Main models: seeds [0, 1, 2, 3, 4]"
echo "  Early Fusion: seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR"
echo "  2. Calculate 5-seed averages (main models)"
echo "  3. Calculate 10-seed averages (Early Fusion)"
echo "  4. Analyze Early Fusion failure rate"
echo ""
echo "Note: Previous results with seeds [1111, 1112, 1113] can be"
echo "      kept as reference, but use standard seeds [0-4] in paper."
