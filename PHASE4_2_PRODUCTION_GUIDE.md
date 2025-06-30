
# Phase 4.2 Advanced Quality Enhancement - Production Guide

## Overview
Phase 4.2 introduces advanced quality assessment and ML-based prediction capabilities to MatterGen.

## New Features
- Advanced crystallographic quality metrics
- ML-based structure quality prediction
- Comprehensive quality trend analysis  
- Rich quality reporting with visualizations
- Enhanced quality filtering and optimization

## CLI Options

### Multi-GPU Inference (multi_gpu_inference.py)
- `--enable_advanced_quality`: Enable advanced quality analysis
- `--enable_quality_prediction`: Enable ML-based quality prediction
- `--enable_quality_reporting`: Enable comprehensive quality reporting
- `--enable_trend_analysis`: Enable quality trend analysis
- `--quality_model_path`: Path to pre-trained quality prediction model
- `--quality_report_format`: Report format (json, html, pdf, csv)

### Single-GPU Generation (generate.py)
Same options as above, accessed via Fire CLI.

## Example Usage

### Basic Phase 4.2 Generation
```bash
python multi_gpu_inference.py \
  --output_base_path "results/phase4_2_test" \
  --pretrained_name "mattergen_base" \
  --total_structures 128 \
  --num_gpus 4 \
  --enable_phase4_features true \
  --enable_advanced_quality true \
  --enable_quality_reporting true \
  --quality_report_format json
```

### Advanced Production Run with All Features
```bash
python multi_gpu_inference.py \
  --output_base_path "results/production_phase4_2" \
  --pretrained_name "mattergen_base" \
  --total_structures 512 \
  --num_gpus 8 \
  --base_batch_size 32 \
  --enable_phase4_features true \
  --enable_adaptive_sampling true \
  --enable_quality_metrics true \
  --enable_advanced_quality true \
  --enable_quality_reporting true \
  --enable_trend_analysis true \
  --quality_threshold 0.8 \
  --max_adaptation_iterations 5 \
  --quality_report_format json
```

## Output Files
- Quality analysis reports in specified format
- Trend analysis data and visualizations  
- Advanced crystallographic metrics
- ML prediction results (if enabled)
- Comprehensive quality summaries

## Dependencies
- All Phase 4.1 dependencies
- Optional: scikit-learn (for ML prediction)
- Optional: matplotlib (for visualizations)

## Status
Phase 4.2 is production-ready and fully integrated into the main generation pipeline.
All features are backward compatible with existing configurations.
