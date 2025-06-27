# MatterGen Phase 3 Implementation Guide
## Multi-GPU Support, Advanced Caching, and Hardware Optimizations

### Overview

Phase 3 represents the most advanced optimization tier for MatterGen, building upon Phase 2's foundation with:

- **Multi-GPU Support**: DataParallel and DistributedDataParallel for scale-out performance
- **Advanced Graph Caching**: Intelligent caching of expensive graph operations
- **Memory Management**: Smart memory monitoring and cleanup
- **Hardware-Specific Optimizations**: Tensor cores, Flash Attention, and more

### New Components

#### 1. Multi-GPU Optimizer (`multi_gpu_optimizer.py`)

**Key Features:**
- Auto-detection of optimal multi-GPU strategy
- Support for DataParallel (DP) and DistributedDataParallel (DDP)
- Intelligent batch size scaling
- Memory management across devices

**Classes:**
- `MultiGPUConfig`: Configuration for multi-GPU operations
- `MultiGPUWrapper`: Model wrapper for multi-GPU execution
- `MultiGPUMemoryManager`: Memory management across devices

#### 2. Graph Operation Cache (`graph_cache.py`)

**Key Features:**
- Thread-safe caching of expensive operations
- Memory and disk-based caching
- TTL (Time-To-Live) support
- Adaptive cache sizing

**Classes:**
- `GraphOperationCache`: Main cache implementation
- `NeighborSearchCache`: Specialized for graph neighbor operations
- `AdaptiveCache`: Self-adjusting cache behavior

#### 3. Enhanced Performance Optimizer

**New Features:**
- Memory monitoring with automatic cleanup
- Gradient checkpointing support
- Hardware-specific optimizations (Tensor cores, Flash Attention)
- Integration with multi-GPU and caching systems

### Configuration

#### Phase 3 Optimization Config

```python
from mattergen.common.utils.performance_optimizer import OptimizationConfig

config = OptimizationConfig(
    # Phase 2 features
    enable_mixed_precision=True,
    enable_model_compilation=True,
    enable_memory_optimization=True,
    
    # Phase 3 features
    enable_multi_gpu=True,
    enable_graph_caching=True,
    enable_gradient_checkpointing=False,
    enable_memory_monitoring=True,
    
    # Multi-GPU settings
    multi_gpu_strategy="auto",  # "auto", "dp", "ddp", "single"
    max_gpus=None,  # Use all available
    
    # Memory settings
    memory_cleanup_frequency=10,
    max_memory_usage_gb=None,  # Auto-detect
    
    # Caching settings
    graph_cache_size=1000,
    enable_disk_cache=True,
)
```

#### Generator Configuration

```python
from mattergen.generator import CrystalGenerator

generator = CrystalGenerator(
    checkpoint_info=checkpoint_info,
    batch_size=64,
    num_batches=10,
    
    # Phase 3 optimizations
    enable_multi_gpu=True,
    enable_graph_caching=True,
    enable_gradient_checkpointing=False,
    multi_gpu_strategy="auto",
    max_gpus=None,
    max_memory_usage_gb=None,
)
```

### CLI Usage

#### Basic Phase 3 Generation

```bash
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --batch_size=64 \
    --sampling_config_name=phase3_optimized \
    --enable_multi_gpu=True \
    --enable_graph_caching=True \
    --print_optimization_info=True
```

#### Advanced Multi-GPU Configuration

```bash
# Force DataParallel strategy
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --multi_gpu_strategy=dp \
    --max_gpus=4 \
    --batch_size=128

# Force single GPU (disable multi-GPU)
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --multi_gpu_strategy=single \
    --enable_multi_gpu=False
```

#### Memory-Optimized Configuration

```bash
python -m mattergen.scripts.generate results/output \
    --pretrained-name=mattergen_base \
    --max_memory_usage_gb=20 \
    --enable_gradient_checkpointing=True \
    --memory_cleanup_frequency=5
```

### Performance Expectations

#### Multi-GPU Scaling

| GPUs | Expected Speedup | Best Strategy |
|------|------------------|---------------|
| 1    | 1.0x (baseline)  | single        |
| 2    | 1.6-1.8x        | dp/auto       |
| 4    | 2.8-3.5x        | dp/ddp        |
| 8+   | 4.0-6.0x        | ddp           |

#### Memory Requirements

| Configuration | GPU Memory | System Memory |
|---------------|------------|---------------|
| Basic Phase 3 | 8GB+       | 16GB+         |
| Multi-GPU     | 6GB+ per GPU| 32GB+        |
| Large Batch   | 12GB+      | 64GB+         |

### Monitoring and Debugging

#### System Information

```bash
# Check optimization capabilities
python -c "from mattergen.common.utils.performance_optimizer import print_optimization_info; print_optimization_info()"

# Check multi-GPU status
python -c "from mattergen.common.utils.multi_gpu_optimizer import print_multi_gpu_info; print_multi_gpu_info()"
```

#### Cache Statistics

```python
from mattergen.common.utils.graph_cache import print_cache_stats

# During generation...
print_cache_stats()
```

#### Optimization Statistics

```python
# After generation
opt_info = generator.get_optimization_info()
print(opt_info)
```

### Troubleshooting

#### Common Issues

1. **CUDA OOM with Multi-GPU**
   - Reduce batch size per GPU
   - Enable gradient checkpointing
   - Set max_memory_usage_gb

2. **DDP Initialization Errors**
   - Check network/firewall settings
   - Try DataParallel strategy instead
   - Ensure NCCL is properly installed

3. **Cache Performance Issues**
   - Adjust cache size based on available memory
   - Enable/disable disk caching
   - Monitor cache hit rates

4. **Compilation Failures**
   - Disable compilation for debugging: `--enable_model_compilation=False`
   - Check PyTorch version compatibility
   - Try different compilation modes

#### Performance Debugging

```bash
# Run with detailed logging
CUDA_LAUNCH_BLOCKING=1 python -m mattergen.scripts.generate ... --print_optimization_info=True

# Profile memory usage
python -m memory_profiler -m mattergen.scripts.generate ...

# Check GPU utilization
nvidia-smi -l 1
```

### Best Practices

#### Multi-GPU Setup

1. **Strategy Selection**:
   - Use "auto" for most cases
   - Use "dp" for 2-4 GPUs with high interconnect
   - Use "ddp" for 4+ GPUs or distributed setups

2. **Batch Size Optimization**:
   - Start with base batch size × num_GPUs
   - Monitor memory usage and adjust
   - Consider memory bandwidth vs compute balance

3. **Memory Management**:
   - Enable memory monitoring for long runs
   - Set cleanup frequency based on batch size
   - Use gradient checkpointing for large models

#### Caching Strategy

1. **Cache Size**:
   - Start with 1000 items for memory cache
   - Enable disk cache for persistence
   - Monitor hit rates and adjust

2. **Cache Warmup**:
   - Run smaller generation first to populate cache
   - Use consistent graph parameters
   - Consider precomputing common operations

#### Hardware Optimization

1. **GPU Selection**:
   - Use V100/A100 for Tensor Core benefits
   - Ensure sufficient memory per GPU
   - Consider memory bandwidth for large models

2. **System Configuration**:
   - Use high-speed interconnects (NVLink)
   - Ensure adequate CPU cores per GPU
   - Monitor system memory usage

### Integration Examples

#### Custom Optimization Pipeline

```python
from mattergen.common.utils.performance_optimizer import (
    OptimizationConfig, 
    apply_all_optimizations
)
from mattergen.generator import CrystalGenerator

# Custom configuration
config = OptimizationConfig(
    enable_multi_gpu=True,
    multi_gpu_strategy="ddp",
    max_gpus=4,
    graph_cache_size=2000,
    memory_cleanup_frequency=5,
)

# Create and optimize generator
generator = CrystalGenerator(
    checkpoint_info=checkpoint_info,
    batch_size=32,
    **config.__dict__
)

try:
    structures = generator.generate()
    print(f"Generated {len(structures)} structures")
finally:
    generator.cleanup()
```

#### Batch Processing with Phase 3

```python
from mattergen.common.utils.multi_gpu_optimizer import MultiGPUConfig
from mattergen.common.utils.graph_cache import setup_graph_caching

# Setup caching
cache = setup_graph_caching(
    cache_size=5000,
    enable_disk_cache=True
)

# Process multiple batches
for batch_id in range(num_batches):
    generator = CrystalGenerator(...)
    
    try:
        structures = generator.generate()
        save_structures(structures, f"batch_{batch_id}")
    finally:
        generator.cleanup()
        
    # Periodic cache maintenance
    if batch_id % 10 == 0:
        from mattergen.common.utils.graph_cache import print_cache_stats
        print_cache_stats()
```

### Migration from Phase 2

Existing Phase 2 code will continue to work with Phase 3. To enable Phase 3 features:

1. **Update Generator Creation**:
   ```python
   # Old Phase 2
   generator = CrystalGenerator(
       enable_performance_optimizations=True,
       enable_mixed_precision=True,
   )
   
   # New Phase 3
   generator = CrystalGenerator(
       enable_performance_optimizations=True,
       enable_mixed_precision=True,
       enable_multi_gpu=True,        # New
       enable_graph_caching=True,    # New
   )
   ```

2. **Update CLI Commands**:
   ```bash
   # Add Phase 3 flags to existing commands
   python -m mattergen.scripts.generate ... \
       --enable_multi_gpu=True \
       --enable_graph_caching=True
   ```

3. **Add Cleanup**:
   ```python
   try:
       structures = generator.generate()
   finally:
       generator.cleanup()  # New - important for resource cleanup
   ```

Phase 3 is designed to be backward compatible while providing significant performance improvements for users with suitable hardware.
