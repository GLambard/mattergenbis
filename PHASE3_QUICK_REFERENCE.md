# MatterGen Phase 3 Quick Reference

## TL;DR - Phase 3 Commands

### Quick Start (Auto-Optimized)
```bash
# Best performance with auto-optimization
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --sampling_config_name=phase3_optimized \
    --batch_size=64 \
    --print_optimization_info=True
```

### Multi-GPU Generation
```bash
# Use all available GPUs
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --enable_multi_gpu=True \
    --multi_gpu_strategy=auto \
    --batch_size=128

# Force specific strategy
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --multi_gpu_strategy=dp \
    --max_gpus=4
```

### Memory-Optimized Generation
```bash
# For limited memory systems
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --max_memory_usage_gb=16 \
    --enable_gradient_checkpointing=True \
    --batch_size=32
```

## Key Phase 3 Features

| Feature | CLI Flag | Default | Description |
|---------|----------|---------|-------------|
| Multi-GPU | `--enable_multi_gpu` | `True` | Enable multi-GPU support |
| GPU Strategy | `--multi_gpu_strategy` | `"auto"` | `"auto"`, `"dp"`, `"ddp"`, `"single"` |
| Max GPUs | `--max_gpus` | `None` | Limit number of GPUs used |
| Graph Cache | `--enable_graph_caching` | `True` | Cache expensive graph ops |
| Memory Limit | `--max_memory_usage_gb` | `None` | Set memory usage limit |
| Gradient Checkpoint | `--enable_gradient_checkpointing` | `False` | Save memory with checkpointing |
| Optimization Info | `--print_optimization_info` | `False` | Show system capabilities |

## Performance Configurations

### Maximum Speed (Multi-GPU)
```bash
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --sampling_config_name=phase3_optimized \
    --batch_size=128 \
    --enable_multi_gpu=True \
    --enable_graph_caching=True \
    --record_trajectories=False
```

### Maximum Memory Efficiency
```bash
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --batch_size=16 \
    --enable_gradient_checkpointing=True \
    --max_memory_usage_gb=8 \
    --multi_gpu_strategy=single
```

### Balanced Performance
```bash
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --sampling_config_name=optimized_compatible \
    --batch_size=64 \
    --enable_multi_gpu=True \
    --enable_graph_caching=True
```

## Testing Commands

### Quick Functionality Test
```bash
./test_phase3_quick.sh
```

### Full Performance Benchmark
```bash
./benchmark_phase3.sh
```

### Check System Capabilities
```bash
python -c "from mattergen.common.utils.performance_optimizer import print_optimization_info; print_optimization_info()"
```

## Configuration Files

| Config | Use Case | Speed | Memory |
|--------|----------|-------|--------|
| `default` | Baseline | 1x | Moderate |
| `optimized_compatible` | Phase 2 | 2-3x | Moderate |
| `phase3_optimized` | Phase 3 | 3-6x | Higher |

## Multi-GPU Strategies

| Strategy | Best For | GPU Count | Performance |
|----------|----------|-----------|-------------|
| `auto` | General use | Any | Automatic selection |
| `single` | Single GPU/CPU | 1 | No multi-GPU overhead |
| `dp` | Small clusters | 2-4 | Good for shared memory |
| `ddp` | Large clusters | 4+ | Best scaling |

## Troubleshooting

### Common Issues & Solutions

**CUDA Out of Memory:**
```bash
# Reduce batch size
--batch_size=16

# Enable gradient checkpointing
--enable_gradient_checkpointing=True

# Set memory limit
--max_memory_usage_gb=12
```

**Multi-GPU Errors:**
```bash
# Fall back to single GPU
--multi_gpu_strategy=single

# Use DataParallel instead of DDP
--multi_gpu_strategy=dp
```

**Slow Performance:**
```bash
# Check if optimizations are enabled
--print_optimization_info=True

# Try different sampling config
--sampling_config_name=phase3_optimized

# Enable all optimizations
--enable_multi_gpu=True --enable_graph_caching=True
```

## Python API Quick Start

### Basic Usage
```python
from mattergen.generator import CrystalGenerator
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo

# Phase 3 optimized generator
generator = CrystalGenerator(
    checkpoint_info=MatterGenCheckpointInfo.from_hf_hub("mattergen_base"),
    batch_size=64,
    num_batches=10,
    # Phase 3 features
    enable_multi_gpu=True,
    enable_graph_caching=True,
    multi_gpu_strategy="auto",
    sampling_config_name="phase3_optimized"
)

try:
    structures = generator.generate(output_dir="results")
    print(f"Generated {len(structures)} structures")
finally:
    generator.cleanup()  # Important!
```

### Advanced Configuration
```python
from mattergen.common.utils.performance_optimizer import OptimizationConfig

config = OptimizationConfig(
    enable_multi_gpu=True,
    multi_gpu_strategy="ddp",
    max_gpus=4,
    graph_cache_size=2000,
    enable_memory_monitoring=True
)

generator = CrystalGenerator(
    checkpoint_info=checkpoint_info,
    batch_size=128,
    **{k: v for k, v in config.__dict__.items() if hasattr(CrystalGenerator, k)}
)
```

## Performance Expectations

### Speedup vs Phase 2
- **Single GPU**: 1.2-1.5x (improved caching/memory)
- **Dual GPU**: 1.8-2.2x 
- **Quad GPU**: 3.0-4.0x
- **8+ GPU**: 4.0-6.0x

### Memory Usage
- **Base**: ~6GB GPU memory
- **Multi-GPU**: ~4GB per GPU + overhead
- **Large Batch**: 12GB+ per GPU

## Quick Verification

After installation, verify Phase 3 is working:

```bash
# 1. Check imports
python -c "from mattergen.common.utils.multi_gpu_optimizer import print_multi_gpu_info; print_multi_gpu_info()"

# 2. Quick test
./test_phase3_quick.sh

# 3. Small generation test
python -m mattergen.scripts.generate results/test \
    --pretrained-name=mattergen_base \
    --batch_size=4 --num_batches=1 \
    --sampling_config_name=phase3_optimized \
    --print_optimization_info=True
```

Success indicators:
- ✓ No import errors
- ✓ Multi-GPU info shows available GPUs
- ✓ Generation completes without errors
- ✓ Optimization info shows Phase 3 features enabled
