# MatterGen Phase 2 Optimization Guide

## Performance Optimizations Implemented

### 🚀 **Quick Usage**

Use the optimized generation with:

```bash
# Fast generation with all Phase 2 optimizations
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=64 \
  --num_batches=1 \
  --record_trajectories=False \
  --sampling_config_name=optimized \
  --enable_optimizations=True \
  --enable_mixed_precision=True \
  --enable_model_compilation=True

# Ultra-fast generation (lower quality but maximum speed)
mattergen-generate $RESULTS_PATH \
  --pretrained-name=$MODEL_NAME \
  --batch_size=128 \
  --num_batches=1 \
  --record_trajectories=False \
  --sampling_config_name=fast \
  --enable_optimizations=True \
  --enable_mixed_precision=True \
  --enable_model_compilation=True
```

### 📊 **Performance Configurations**

We've created three sampling configurations:

1. **`default.yaml`** - Original (N=1000, with correctors)
2. **`optimized.yaml`** - Balanced speed/quality (N=250, no correctors) 
3. **`fast.yaml`** - Maximum speed (N=100, no correctors)

### ⚡ **Key Optimizations**

#### 2.1 Mixed Precision (FP16)
- **Speedup**: 1.5-2x
- **Memory**: ~50% reduction
- **Quality**: Minimal impact
- **Enable**: `--enable_mixed_precision=True`

#### 2.2 Model Compilation (PyTorch 2.0+)
- **Speedup**: 1.2-1.8x
- **Memory**: Similar or slight reduction
- **Quality**: No impact
- **Enable**: `--enable_model_compilation=True`

#### 2.3 Reduced Diffusion Steps
- **optimized.yaml**: N=250 (4x speedup vs N=1000)
- **fast.yaml**: N=100 (10x speedup vs N=1000)
- **Quality**: Moderate reduction for fast.yaml

#### 2.4 Disabled Corrector Steps
- **Speedup**: 1.5-2x additional
- **Quality**: Minor impact
- **Implementation**: `n_steps_corrector: 0`

#### 2.5 Optimized GemNet Parameters (future)
- Reduced model complexity for inference
- Located in `mattergen_optimized.yaml`

### 🔧 **Configuration Details**

#### Memory Management
- Automatic CUDA cache clearing
- Memory-efficient attention when available
- Optimized memory fractions for large GPUs

#### Hardware Optimizations
- CUDNN benchmark mode enabled
- Tensor Core utilization (FP16)
- TF32 enabled for compatible hardware

### 📈 **Expected Performance Gains**

| Configuration | Expected Speedup | Quality Impact |
|---------------|------------------|----------------|
| optimized.yaml + FP16 + compilation | 6-8x | Minimal |
| fast.yaml + FP16 + compilation | 15-20x | Moderate |
| Custom tuned | Up to 25x | Variable |

### 🔍 **Performance Monitoring**

The optimizations include automatic performance monitoring:
- GPU memory usage tracking
- Inference time measurement
- Batch processing efficiency metrics

### ⚠️ **Important Notes**

1. **First Run Slower**: Model compilation adds overhead on first run
2. **GPU Memory**: FP16 reduces memory by ~50%, allowing larger batches
3. **Quality Trade-offs**: fast.yaml trades some quality for speed
4. **Hardware Dependent**: Gains vary by GPU generation

### 🔧 **Troubleshooting**

If you encounter issues:

```bash
# Disable optimizations
--enable_optimizations=False

# Disable specific optimizations
--enable_mixed_precision=False
--enable_model_compilation=False

# Use more conservative settings
--sampling_config_name=optimized  # instead of fast
```

### 📝 **Next Steps for Phase 3**

- Multi-GPU support
- Advanced caching mechanisms  
- Custom CUDA kernels
- Dynamic batch sizing
- Progressive sampling algorithms
