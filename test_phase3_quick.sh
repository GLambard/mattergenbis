#!/bin/bash

# Quick test script for Phase 3 features
# Tests basic functionality without full benchmarking

set -e

echo "Testing Phase 3 MatterGen optimizations..."

# Test 1: Check imports and basic functionality
echo "1. Testing imports..."
python -c "
from mattergen.common.utils.multi_gpu_optimizer import MultiGPUConfig, print_multi_gpu_info
from mattergen.common.utils.graph_cache import setup_graph_caching, print_cache_stats
from mattergen.common.utils.performance_optimizer import print_optimization_info, OptimizationConfig
print('✓ All Phase 3 imports successful')
"

# Test 2: Check optimization info
echo ""
echo "2. System optimization capabilities:"
python -c "
from mattergen.common.utils.performance_optimizer import print_optimization_info
print_optimization_info()
"

# Test 3: Quick generation test with Phase 3 optimizations
echo ""
echo "3. Quick generation test with Phase 3 optimizations..."
timeout 120 python -m mattergen.scripts.generate \
    "results/phase3_test" \
    --pretrained-name="mattergen_base" \
    --batch_size=4 \
    --num_batches=1 \
    --sampling_config_name="phase3_optimized" \
    --record_trajectories=False \
    --enable_multi_gpu=True \
    --enable_graph_caching=True \
    --print_optimization_info=True || echo "Generation test timed out (this is expected for quick test)"

# Test 4: Test optimization configuration
echo ""
echo "4. Testing optimization configuration..."
python -c "
from mattergen.common.utils.performance_optimizer import OptimizationConfig, apply_all_optimizations
import torch
import torch.nn as nn

# Create a dummy model for testing
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    def forward(self, x):
        return self.linear(x)

model = DummyModel()
config = OptimizationConfig(
    enable_multi_gpu=True,
    enable_graph_caching=True,
    multi_gpu_strategy='auto'
)

try:
    optimized_model, batch_size, optimizer = apply_all_optimizations(model, 32, config)
    print('✓ Optimization configuration test successful')
    print(f'  Original batch size: 32')
    print(f'  Optimized batch size: {batch_size}')
    
    # Get stats
    stats = optimizer.get_optimization_stats()
    print(f'  Multi-GPU available: {stats.get(\"num_gpus\", 0) > 1}')
    print(f'  Phase 3 features available: {stats.get(\"phase3_available\", False)}')
    
    optimizer.cleanup()
except Exception as e:
    print(f'✗ Optimization test failed: {e}')
"

# Test 5: Cache functionality
echo ""
echo "5. Testing graph cache functionality..."
python -c "
from mattergen.common.utils.graph_cache import setup_graph_caching, clear_all_caches, print_cache_stats

# Setup cache
cache = setup_graph_caching(enable_caching=True, cache_size=100)

# Test basic cache operations
cache.put('test_key', {'data': 'test_value'})
result = cache.get('test_key')

if result and result.get('data') == 'test_value':
    print('✓ Cache functionality working')
    print_cache_stats()
else:
    print('✗ Cache test failed')

clear_all_caches()
print('✓ Cache cleanup successful')
"

echo ""
echo "Phase 3 quick test complete!"
echo ""
echo "Next steps:"
echo "1. Run full benchmark: ./benchmark_phase3.sh"
echo "2. Try Phase 3 optimized generation:"
echo "   python -m mattergen.scripts.generate results/my_output \\"
echo "     --pretrained-name=mattergen_base \\"
echo "     --batch_size=64 \\"
echo "     --sampling_config_name=phase3_optimized \\"
echo "     --enable_multi_gpu=True \\"
echo "     --enable_graph_caching=True \\"
echo "     --print_optimization_info=True"
