#!/bin/bash

# Safe MatterGen Phase 2 Optimization Test
# This script tests different optimization levels to find the best working configuration

cd "$(dirname "$0")"

echo "🧪 Testing MatterGen Phase 2 Optimizations (Safe Mode)"
echo "====================================================="

# Test 1: Conservative optimization (no mixed precision)
echo "🔧 Test 1: Conservative optimization (no FP16)"
echo "Command: Conservative optimization without mixed precision"

./mattergen-generate.sh results/test_conservative \
  --pretrained-name=mattergen_base \
  --batch_size=32 \
  --sampling_config_name=optimized \
  --enable_optimizations=True \
  --enable_mixed_precision=False \
  --enable_model_compilation=True \
  --record_trajectories=False \
  --num_batches=1

echo ""
echo "✅ Conservative test completed. Check results/test_conservative for output."
echo ""

# Test 2: Sampling optimization only
echo "🔧 Test 2: Sampling optimization only (no model optimizations)"
echo "Command: Fast sampling configuration only"

./mattergen-generate.sh results/test_sampling_only \
  --pretrained-name=mattergen_base \
  --batch_size=32 \
  --sampling_config_name=fast \
  --enable_optimizations=False \
  --record_trajectories=False \
  --num_batches=1

echo ""
echo "✅ Sampling-only test completed. Check results/test_sampling_only for output."
echo ""

echo "🎯 Recommended configuration based on tests:"
echo "For stable performance with good speedup:"
echo "./mattergen-generate.sh results/ --pretrained-name=mattergen_base --batch_size=64 --sampling_config_name=optimized --enable_optimizations=True --enable_mixed_precision=False --enable_model_compilation=True"
