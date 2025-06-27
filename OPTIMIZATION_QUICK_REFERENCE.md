# MatterGen Performance Optimization Quick Reference

## 🚀 Speed vs Quality Trade-offs

| Speed Need | Command | Expected Speedup | Quality |
|------------|---------|------------------|---------|
| **Highest Quality** | `--sampling_config_name=default` | 1x (baseline) | Best |
| **Balanced** | `--sampling_config_name=optimized --enable_optimizations=True` | 6-8x | High |
| **Maximum Speed** | `--sampling_config_name=fast --enable_optimizations=True` | 15-20x | Good |

## ⚡ Essential Optimization Flags

```bash
# Core optimization flags (add to any command):
--enable_optimizations=True          # Enable all optimizations
--enable_mixed_precision=True        # FP16 for 2x speed + 50% memory savings  
--enable_model_compilation=True      # PyTorch compilation for 1.2-1.8x speedup
--sampling_config_name=optimized     # Use 250 diffusion steps instead of 1000
--record_trajectories=False          # Skip trajectory recording for speed
```

## 📊 Memory and Batch Size Guide

| GPU Memory | Recommended Batch Size | Config |
|------------|------------------------|--------|
| 8GB | 16-32 | `optimized` |
| 16GB | 32-64 | `optimized` |
| 24GB+ | 64-128 | `fast` |

## 🔧 Common Commands

### Research Quality (Slow but Best)
```bash
mattergen-generate results/ --pretrained-name=mattergen_base --batch_size=16 --sampling_config_name=default
```

### Production Balanced (Recommended)
```bash
mattergen-generate results/ --pretrained-name=mattergen_base --batch_size=64 --sampling_config_name=optimized --enable_optimizations=True
```

### Rapid Prototyping (Fastest)
```bash
mattergen-generate results/ --pretrained-name=mattergen_base --batch_size=128 --sampling_config_name=fast --enable_optimizations=True
```

### Property-Conditioned (Optimized)
```bash
mattergen-generate results/ --pretrained-name=dft_mag_density --batch_size=64 --properties_to_condition_on="{'dft_mag_density': 0.15}" --diffusion_guidance_factor=2.0 --sampling_config_name=optimized --enable_optimizations=True
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `--batch_size` or use `--sampling_config_name=fast` |
| First run slow | Normal - compilation adds overhead on first run |
| Lower quality | Use `--sampling_config_name=optimized` instead of `fast` |
| Compilation errors | Add `--enable_model_compilation=False` |
| Mixed precision issues | Add `--enable_mixed_precision=False` |

## 🧪 Test Your Setup

```bash
# Quick test (30 seconds):
bash benchmark_phase2.sh mattergen_base 8

# Validate implementation:  
python test_phase2_validation.py
```

---

💡 **Pro Tip**: Start with `optimized` config + all optimizations for the best speed/quality balance!
