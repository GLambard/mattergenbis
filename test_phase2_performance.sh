#!/bin/bash

# MatterGen Phase 2 Performance Testing Script
# Tests different optimization configurations and measures performance

set -e

# Configuration
MODEL_NAME=${MODEL_NAME:-"mattergen_base"}
RESULTS_BASE=${RESULTS_BASE:-"results/performance_test"}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_BATCHES=${NUM_BATCHES:-1}

echo "🚀 MatterGen Phase 2 Performance Testing"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Num batches: $NUM_BATCHES"
echo ""

# Create results directory
mkdir -p "$RESULTS_BASE"

# Function to run a test and measure time
run_test() {
    local test_name="$1"
    local config="$2"
    local extra_args="$3"
    local results_path="$RESULTS_BASE/$test_name"
    
    echo "🔄 Running test: $test_name"
    echo "   Config: $config"
    echo "   Extra args: $extra_args"
    
    # Clean up previous results
    rm -rf "$results_path"
    mkdir -p "$results_path"
    
    # Run the generation and measure time
    start_time=$(date +%s.%N)
    
    mattergen-generate "$results_path" \
        --pretrained-name="$MODEL_NAME" \
        --batch_size="$BATCH_SIZE" \
        --num_batches="$NUM_BATCHES" \
        --record_trajectories=False \
        --sampling_config_name="$config" \
        $extra_args \
        > "$results_path/log.txt" 2>&1
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    echo "   ✅ Completed in: ${duration}s"
    echo "$duration" > "$results_path/time.txt"
    
    # Count generated structures
    if [ -f "$results_path/generated_crystals_cif.zip" ]; then
        structure_count=$(unzip -l "$results_path/generated_crystals_cif.zip" | grep "\.cif$" | wc -l)
        echo "   📊 Generated structures: $structure_count"
        echo "$structure_count" > "$results_path/count.txt"
        
        # Calculate structures per second
        if (( $(echo "$duration > 0" | bc -l) )); then
            structures_per_sec=$(echo "scale=2; $structure_count / $duration" | bc)
            echo "   ⚡ Structures/sec: $structures_per_sec"
            echo "$structures_per_sec" > "$results_path/rate.txt"
        fi
    fi
    
    echo ""
}

# Test 1: Baseline (original settings)
run_test "baseline" "default" ""

# Test 2: Optimized sampling only
run_test "optimized_sampling" "optimized" ""

# Test 3: Fast sampling only  
run_test "fast_sampling" "fast" ""

# Test 4: Optimized + Mixed Precision
run_test "optimized_fp16" "optimized" "--enable_mixed_precision=True"

# Test 5: Optimized + Model Compilation
run_test "optimized_compiled" "optimized" "--enable_model_compilation=True"

# Test 6: All optimizations
run_test "all_optimizations" "optimized" "--enable_optimizations=True --enable_mixed_precision=True --enable_model_compilation=True"

# Test 7: Ultra-fast configuration
run_test "ultra_fast" "fast" "--enable_optimizations=True --enable_mixed_precision=True --enable_model_compilation=True"

# Generate performance report
echo "📈 Generating Performance Report"
echo "================================="

report_file="$RESULTS_BASE/performance_report.txt"
cat > "$report_file" << EOF
MatterGen Phase 2 Performance Test Results
==========================================
Test Date: $(date)
Model: $MODEL_NAME
Batch Size: $BATCH_SIZE
Number of Batches: $NUM_BATCHES

Configuration                 | Time (s) | Structures | Rate (struct/s) | Speedup
------------------------------|----------|------------|-----------------|--------
EOF

# Calculate speedups relative to baseline
baseline_time=$(cat "$RESULTS_BASE/baseline/time.txt" 2>/dev/null || echo "1")

for test_dir in "$RESULTS_BASE"/*; do
    if [ -d "$test_dir" ] && [ "$(basename "$test_dir")" != "performance_test" ]; then
        test_name=$(basename "$test_dir")
        
        if [ -f "$test_dir/time.txt" ]; then
            time=$(cat "$test_dir/time.txt")
            count=$(cat "$test_dir/count.txt" 2>/dev/null || echo "0")
            rate=$(cat "$test_dir/rate.txt" 2>/dev/null || echo "0")
            
            # Calculate speedup
            if (( $(echo "$time > 0 && $baseline_time > 0" | bc -l) )); then
                speedup=$(echo "scale=2; $baseline_time / $time" | bc)
            else
                speedup="N/A"
            fi
            
            printf "%-29s | %8.2f | %10s | %15s | %6s\n" \
                "$test_name" "$time" "$count" "$rate" "${speedup}x" >> "$report_file"
        fi
    fi
done

# Display the report
cat "$report_file"
echo ""
echo "📁 Full results available in: $RESULTS_BASE"
echo "📄 Report saved to: $report_file"

# Check for GPU memory usage if available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️  Current GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
fi

echo ""
echo "✅ Performance testing completed!"
echo ""
echo "💡 Quick recommendations based on results:"
echo "   - Use 'optimized' config for balanced speed/quality"
echo "   - Use 'fast' config for maximum speed when quality is less critical"
echo "   - Enable all optimizations for best performance"
echo "   - Consider increasing batch size if GPU memory allows"
