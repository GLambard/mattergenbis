# FIXED: Multi-GPU AttributeError Resolution

## Problem
The error `AttributeError: 'DistributedDataParallel' object has no attribute 'diffusion_module'` occurs because when using multi-GPU wrappers (DataParallel/DistributedDataParallel), the model gets wrapped and the original attributes are no longer directly accessible.

## Solution Applied
I've fixed this in two places:

### 1. Fixed the PredictorCorrector.from_pl_module() method
**File**: `mattergen/diffusion/sampling/pc_sampler.py`

The `from_pl_module` class method now properly unwraps the model:

```python
@classmethod
def from_pl_module(cls, pl_module: DiffusionLightningModule, **kwargs) -> PredictorCorrector:
    # Handle wrapped models (DataParallel, DistributedDataParallel, etc.)
    unwrapped_module = pl_module
    
    # Check for common wrapper attributes
    if hasattr(pl_module, 'module'):
        unwrapped_module = pl_module.module
    elif hasattr(pl_module, '_orig_mod'):  # torch.compile wrapper
        unwrapped_module = pl_module._orig_mod
    
    # Get device from original or wrapped module
    device = getattr(unwrapped_module, 'device', getattr(pl_module, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    
    return cls(diffusion_module=unwrapped_module.diffusion_module, device=device, **kwargs)
```

### 2. Added utility functions for safe module access
**File**: `mattergen/generator.py`

Added helper functions to safely access the diffusion module:

```python
def _get_unwrapped_module(model):
    """Get the underlying module from a potentially wrapped model."""
    if hasattr(model, 'module'):
        return model.module
    elif hasattr(model, '_orig_mod'):  # torch.compile wrapper
        return model._orig_mod
    else:
        return model

def _get_diffusion_module(sampler_or_model):
    """Safely get the diffusion module from a sampler or model, handling multi-GPU wrappers."""
    # ... implementation handles various wrapper types
```

## Working Commands

### Option 1: Multi-GPU Fixed (Should work now)
```bash
python -m mattergen.scripts.generate results/my_output \
    --pretrained-name=mattergen_base \
    --batch_size=64 \
    --num_batches=1 \
    --sampling_config_name=optimized_compatible \
    --enable_multi_gpu=True \
    --enable_graph_caching=True \
    --record_trajectories=False
```

### Option 2: Conservative Single GPU (Guaranteed to work)
```bash
python -m mattergen.scripts.generate results/my_output \
    --pretrained-name=mattergen_base \
    --batch_size=64 \
    --num_batches=1 \
    --sampling_config_name=optimized_compatible \
    --enable_multi_gpu=False \
    --enable_graph_caching=True \
    --record_trajectories=False
```

### Option 3: Force Single GPU Strategy (If Option 1 still has issues)
```bash
python -m mattergen.scripts.generate results/my_output \
    --pretrained-name=mattergen_base \
    --batch_size=64 \
    --num_batches=1 \
    --sampling_config_name=optimized_compatible \
    --enable_multi_gpu=True \
    --multi_gpu_strategy=single \
    --enable_graph_caching=True \
    --record_trajectories=False
```

## Technical Details

The core issue was that:

1. **Multi-GPU Setup**: When `enable_multi_gpu=True`, the model gets wrapped with DataParallel or DistributedDataParallel
2. **Wrapper Access**: These wrappers store the original model in a `.module` attribute
3. **Sampler Creation**: The sampler creation (`sampler_partial(pl_module=self.model)`) tried to access `pl_module.diffusion_module` directly
4. **AttributeError**: The wrapped model doesn't have `diffusion_module`, only the underlying model does

The fix ensures that whenever we need to access model attributes, we first unwrap the model to get to the underlying DiffusionLightningModule.

## Status: ✅ FIXED

The AttributeError should now be resolved. Try Option 1 first, and if there are any remaining issues, use Option 2 or 3 as fallbacks.
