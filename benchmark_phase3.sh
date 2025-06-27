#!/bin/bash

# MatterGen Phase 3 Performance Benchmark Script
# Tests multi-GPU support, caching, and advanced optimizations

set -e

echo "=================================="
echo "MATTERGEN PHASE 3 BENCHMARK"
echo "=================================="

# Configuration
CHECKPOINT="mattergen_base"
OUTPUT_BASE="results/phase3_benchmark"
NUM_SAMPLES=100
BATCH_SIZE=32
TEST_CONFIGS=("default" "optimized_compatible" "phase3_optimized")

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to run benchmark
run_benchmark() {
    local config_name=$1
    local test_name=$2
    local extra_flags=$3
    
    echo ""
    echo "Running benchmark: $test_name"
    echo "Configuration: $config_name"
    echo "Extra flags: $extra_flags"
    echo "----------------------------------------"
    
    local output_dir="${OUTPUT_BASE}/${test_name}"
    local start_time=$(date +%s)
    
    # Run generation with timing
    if timeout 600 python -m mattergen.scripts.generate \
        "$output_dir" \
        --pretrained-name="$CHECKPOINT" \
        --batch_size="$BATCH_SIZE" \
        --num_batches=$((NUM_SAMPLES / BATCH_SIZE)) \
        --sampling_config_name="$config_name" \
        --record_trajectories=False \
        --print_optimization_info=True \
        $extra_flags; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Count generated structures
        local structures_count=0
        if [ -f "${output_dir}/structures.extxyz" ]; then
            structures_count=$(grep -c "Properties=" "${output_dir}/structures.extxyz" || echo "0")
        fi
        
        echo "✓ Benchmark completed successfully"
        echo "  Duration: ${duration}s"
        echo "  Structures: ${structures_count}"
        echo "  Speed: $(echo "scale=2; $structures_count / $duration" | bc -l) structures/second"
        
        # Save results
        echo "$test_name,$config_name,$duration,$structures_count,$(echo "scale=2; $structures_count / $duration" | bc -l)" >> "$OUTPUT_BASE/benchmark_results.csv"
        
    else
        echo "✗ Benchmark failed or timed out"
        echo "$test_name,$config_name,timeout,0,0" >> "$OUTPUT_BASE/benchmark_results.csv"
    fi
    
    echo "----------------------------------------"
}

# Initialize results file
echo "test_name,config,duration_seconds,structures_generated,structures_per_second" > "$OUTPUT_BASE/benchmark_results.csv"

# Print system information
echo ""
echo "SYSTEM INFORMATION:"
echo "==================="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')

import psutil
print(f'CPU cores: {psutil.cpu_count()}')
print(f'Memory: {psutil.virtual_memory().total / 1e9:.1f} GB')
"

# Test 1: Baseline (Phase 1) - default config, no optimizations
echo ""
echo "TEST 1: Baseline Performance (No Optimizations)"
run_benchmark "default" "baseline" "--enable_optimizations=False"

# Test 2: Phase 2 optimizations only
echo ""
echo "TEST 2: Phase 2 Optimizations (Mixed Precision + Compilation)"
run_benchmark "optimized_compatible" "phase2" "--enable_multi_gpu=False --enable_graph_caching=False"

# Test 3: Phase 3 with single GPU
echo ""
echo "TEST 3: Phase 3 Single GPU (All optimizations, single GPU)"
run_benchmark "phase3_optimized" "phase3_single" "--enable_multi_gpu=False"

# Test 4: Phase 3 with multi-GPU (if available)
if python -c "import torch; exit(0 if torch.cuda.device_count() > 1 else 1)" 2>/dev/null; then
    echo ""
    echo "TEST 4: Phase 3 Multi-GPU (DataParallel)"
    run_benchmark "phase3_optimized" "phase3_multi_dp" "--multi_gpu_strategy=dp"
    
    echo ""
    echo "TEST 5: Phase 3 Multi-GPU (Auto Strategy)"
    run_benchmark "phase3_optimized" "phase3_multi_auto" "--multi_gpu_strategy=auto"
else
    echo ""
    echo "TEST 4-5: Skipping multi-GPU tests (insufficient GPUs)"
    echo "phase3_multi_dp,phase3_optimized,skipped,0,0" >> "$OUTPUT_BASE/benchmark_results.csv"
    echo "phase3_multi_auto,phase3_optimized,skipped,0,0" >> "$OUTPUT_BASE/benchmark_results.csv"
fi

# Test 6: Memory intensive test (if sufficient memory)
echo ""
echo "TEST 6: Large Batch Size Test"
LARGE_BATCH=$((BATCH_SIZE * 4))
if timeout 300 python -c "
import torch
import psutil
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
system_memory = psutil.virtual_memory().total / 1e9
print(f'GPU Memory: {gpu_memory:.1f} GB, System Memory: {system_memory:.1f} GB')
exit(0 if gpu_memory > 12 or system_memory > 32 else 1)
" 2>/dev/null; then
    run_benchmark "phase3_optimized" "large_batch" "--batch_size=$LARGE_BATCH --num_batches=$((NUM_SAMPLES / LARGE_BATCH))"
else
    echo "Skipping large batch test (insufficient memory)"
    echo "large_batch,phase3_optimized,skipped,0,0" >> "$OUTPUT_BASE/benchmark_results.csv"
fi

# Generate summary report
echo ""
echo "=================================="
echo "BENCHMARK RESULTS SUMMARY"
echo "=================================="

if [ -f "$OUTPUT_BASE/benchmark_results.csv" ]; then
    python3 - << 'EOF'
import pandas as pd
import sys

try:
    df = pd.read_csv('results/phase3_benchmark/benchmark_results.csv')
    
    print("Performance Summary:")
    print("=" * 50)
    
    for _, row in df.iterrows():
        if row['duration_seconds'] != 'timeout' and row['duration_seconds'] != 'skipped':
            print(f"{row['test_name']:20} | {row['structures_per_second']:>8.2f} structures/sec")
        else:
            print(f"{row['test_name']:20} | {row['duration_seconds']:>15}")
    
    # Calculate speedups if baseline exists
    baseline_speed = None
    for _, row in df.iterrows():
        if row['test_name'] == 'baseline' and row['duration_seconds'] not in ['timeout', 'skipped']:
            baseline_speed = float(row['structures_per_second'])
            break
    
    if baseline_speed and baseline_speed > 0:
        print("\nSpeedup vs Baseline:")
        print("=" * 30)
        for _, row in df.iterrows():
            if row['duration_seconds'] not in ['timeout', 'skipped'] and row['test_name'] != 'baseline':
                speedup = float(row['structures_per_second']) / baseline_speed
                print(f"{row['test_name']:20} | {speedup:>6.2f}x")
    
    print(f"\nDetailed results saved to: results/phase3_benchmark/benchmark_results.csv")
    
except Exception as e:
    print(f"Error generating summary: {e}")
    print("Raw results:")
    with open('results/phase3_benchmark/benchmark_results.csv', 'r') as f:
        print(f.read())
EOF
else
    echo "No benchmark results found!"
fi

echo ""
echo "Phase 3 benchmark complete!"
echo "Check results/phase3_benchmark/ for detailed outputs"
