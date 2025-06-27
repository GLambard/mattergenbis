#!/bin/bash

# MatterGen Phase 2 Performance Benchmark
# Usage: bash benchmark_phase2.sh [MODEL_NAME] [BATCH_SIZE]

set -e

MODEL_NAME=${1:-"mattergen_base"}
BATCH_SIZE=${2:-32}
RESULTS_BASE="results/phase2_benchmark"

echo "🚀 MatterGen Phase 2 Performance Benchmark"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Results directory: $RESULTS_BASE"
echo ""

# Create results directory
mkdir -p "$RESULTS_BASE"

# Function to run a benchmark test
benchmark_test() {
    local test_name="$1"
    local config="$2"
    local optimizations="$3"
    local description="$4"
    local results_path="$RESULTS_BASE/$test_name"
    
    echo "🔄 Testing: $description"
    echo "   Config: $config"
    echo "   Optimizations: $optimizations"
    
    # Clean up previous results
    rm -rf "$results_path"
    mkdir -p "$results_path"
    
    # Run the generation and measure time
    start_time=$(date +%s.%N)
    
    # Build command
    cmd="mattergen-generate \"$results_path\" --pretrained-name=\"$MODEL_NAME\" --batch_size=\"$BATCH_SIZE\" --num_batches=1 --record_trajectories=False --sampling_config_name=\"$config\""
    
    if [ "$optimizations" = "enabled" ]; then
        cmd="$cmd --enable_optimizations=True --enable_mixed_precision=True --enable_model_compilation=True"
    elif [ "$optimizations" = "disabled" ]; then
        cmd="$cmd --enable_optimizations=False"
    fi
    
    echo "   Command: $cmd"
    
    # Execute command
    if eval "$cmd > \"$results_path/log.txt\" 2>&1"; then
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l)
        
        echo "   ✅ Completed in: ${duration}s"
        echo "$duration" > "$results_path/time.txt"
        
        # Count generated structures
        if [ -f "$results_path/generated_crystals_cif.zip" ]; then
            structure_count=$(unzip -l "$results_path/generated_crystals_cif.zip" | grep "\.cif$" | wc -l)
            echo "   📊 Generated structures: $structure_count"
            echo "$structure_count" > "$results_path/count.txt"
            
            # Calculate structures per second
            if (( $(echo "$duration > 0" | bc -l) )); then
                structures_per_sec=$(echo "scale=2; $structure_count / $duration" | bc -l)
                echo "   ⚡ Structures/sec: $structures_per_sec"
                echo "$structures_per_sec" > "$results_path/rate.txt"
            fi
        fi
    else
        echo "   ❌ Test failed - check $results_path/log.txt for details"
        return 1
    fi
    
    echo ""
}

# Test 1: Baseline (default config, no optimizations)
benchmark_test "baseline" "default" "disabled" "Baseline (default config, no optimizations)"

# Test 2: Optimized sampling only
benchmark_test "optimized_sampling" "optimized" "disabled" "Optimized sampling (250 steps, no model optimizations)"

# Test 3: Fast sampling only
benchmark_test "fast_sampling" "fast" "disabled" "Fast sampling (100 steps, no model optimizations)"

# Test 4: Optimized + All Performance Features
benchmark_test "optimized_full" "optimized" "enabled" "Optimized + FP16 + Compilation"

# Test 5: Ultra-fast (all optimizations)
benchmark_test "ultra_fast" "fast" "enabled" "Ultra-fast (100 steps + all optimizations)"

# Generate benchmark report
echo "📈 Generating Performance Report"
echo "================================="

report_file="$RESULTS_BASE/benchmark_report.txt"
cat > "$report_file" << EOF
MatterGen Phase 2 Performance Benchmark Results
===============================================

Model: $MODEL_NAME
Batch Size: $BATCH_SIZE
Date: $(date)

Test Results:
EOF

# Add results to report
for test_dir in "$RESULTS_BASE"/*; do
    if [ -d "$test_dir" ] && [ "$(basename "$test_dir")" != "benchmark_report.txt" ]; then
        test_name=$(basename "$test_dir")
        
        if [ -f "$test_dir/time.txt" ]; then
            time_val=$(cat "$test_dir/time.txt")
            rate_val=$(cat "$test_dir/rate.txt" 2>/dev/null || echo "N/A")
            count_val=$(cat "$test_dir/count.txt" 2>/dev/null || echo "N/A")
            
            echo "  $test_name:" >> "$report_file"
            echo "    Duration: ${time_val}s" >> "$report_file"
            echo "    Structures: $count_val" >> "$report_file"
            echo "    Rate: $rate_val structures/sec" >> "$report_file"
            echo "" >> "$report_file"
        fi
    fi
done

# Calculate speedups
if [ -f "$RESULTS_BASE/baseline/time.txt" ]; then
    baseline_time=$(cat "$RESULTS_BASE/baseline/time.txt")
    
    echo "Speedup Analysis:" >> "$report_file"
    for test_dir in "$RESULTS_BASE"/*; do
        if [ -d "$test_dir" ] && [ "$(basename "$test_dir")" != "baseline" ]; then
            test_name=$(basename "$test_dir")
            
            if [ -f "$test_dir/time.txt" ]; then
                test_time=$(cat "$test_dir/time.txt")
                speedup=$(echo "scale=2; $baseline_time / $test_time" | bc -l)
                echo "  $test_name: ${speedup}x speedup" >> "$report_file"
            fi
        fi
    done
fi

echo ""
echo "📊 Benchmark Complete!"
echo "Report saved to: $report_file"
echo ""
cat "$report_file"
