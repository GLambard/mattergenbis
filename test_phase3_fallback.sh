#!/bin/bash

# Simple Phase 3 test that works around import issues
echo "Testing Phase 3 with fallback configuration..."

cd /home/guillaume/Documents/Projects/LINK/Internships/Auguste/mattergenbis

# Test 1: Try with Phase 3 disabled to confirm basic functionality
echo "1. Testing with Phase 3 disabled..."
timeout 60 python -m mattergen.scripts.generate results/phase3_fallback_test \
    --pretrained-name=mattergen_base \
    --batch_size=8 \
    --num_batches=1 \
    --sampling_config_name=optimized_compatible \
    --enable_multi_gpu=False \
    --enable_graph_caching=False \
    --record_trajectories=False \
    --print_optimization_info=False || echo "Basic test failed"

# Test 2: Try with Phase 3 features enabled but conservative settings
echo ""
echo "2. Testing with Phase 3 enabled (conservative)..."
timeout 60 python -m mattergen.scripts.generate results/phase3_conservative_test \
    --pretrained-name=mattergen_base \
    --batch_size=4 \
    --num_batches=1 \
    --sampling_config_name=optimized_compatible \
    --enable_multi_gpu=True \
    --multi_gpu_strategy=single \
    --enable_graph_caching=True \
    --record_trajectories=False \
    --print_optimization_info=True || echo "Conservative Phase 3 test failed"

echo ""
echo "Phase 3 fallback tests complete!"
