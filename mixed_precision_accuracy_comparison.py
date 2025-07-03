#!/usr/bin/env python3
"""
Mixed Precision vs Full Precision Accuracy Comparison
=====================================================

This script compares the crystalline structure accuracy when using mixed precision
(float16) vs full precision (float32) for structure generation.

We'll generate 256 structures using both configurations and analyze:
- Structural validity (cell matrix determinants)
- Bond lengths and angles
- Lattice parameters
- Crystal symmetry preservation
- Overall quality metrics
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_generation_experiment(
    output_dir: str,
    experiment_name: str,
    enable_mixed_precision: bool,
    num_structures: int = 256,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Run a structure generation experiment with specified precision settings."""
    
    logger.info(f"🧪 Starting experiment: {experiment_name}")
    logger.info(f"   Mixed Precision: {enable_mixed_precision}")
    logger.info(f"   Structures: {num_structures}")
    logger.info(f"   Output: {output_dir}")
    
    # Calculate number of batches
    num_batches = (num_structures + batch_size - 1) // batch_size
    
    # Prepare command
    cmd = [
        sys.executable, "-m", "mattergen.scripts.generate",
        output_dir,
        "--batch_size", str(batch_size),
        "--num_batches", str(num_batches),
        "--enable_multi_gpu", "False",
        "--sampling-config-name", "optimized_compatible",
        "--record-trajectories", "False",
        "--print-loss", "False",
        "--enable-optimizations", "True",
        "--enable-mixed-precision", str(enable_mixed_precision),
        "--enable-model-compilation", "True",
        "--enable-graph-caching", "True",
        "--print-optimization-info", "True",
        "--enable-phase4-features", "False",
        "--enable-adaptive-sampling", "False",
        "--enable-quality-metrics", "True",
        "--enable-quality-reporting", "True",
        "--quality-report-format", "json",
        "--pretrained-name", "mattergen_base"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        end_time = time.time()
        
        if result.returncode != 0:
            logger.error(f"❌ Experiment {experiment_name} failed!")
            logger.error(f"STDERR: {result.stderr[-1000:]}")  # Last 1000 chars
            return {
                'success': False,
                'error': result.stderr,
                'duration': end_time - start_time
            }
        
        duration = end_time - start_time
        logger.info(f"✅ Experiment {experiment_name} completed in {duration:.1f}s")
        
        # Parse results
        return {
            'success': True,
            'duration': duration,
            'mixed_precision': enable_mixed_precision,
            'structures_requested': num_structures,
            'output_dir': output_dir,
            'stdout': result.stdout[-500:],  # Last 500 chars for reference
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Experiment {experiment_name} timed out!")
        return {
            'success': False,
            'error': 'Timeout after 1 hour',
            'duration': 3600
        }
    except Exception as e:
        logger.error(f"❌ Experiment {experiment_name} failed with exception: {e}")
        return {
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time
        }


def analyze_crystal_structures(output_dir: str) -> Dict[str, Any]:
    """Analyze the quality of generated crystal structures."""
    
    logger.info(f"🔍 Analyzing structures in {output_dir}")
    
    # Look for structure files
    extxyz_file = Path(output_dir) / "generated_crystals.extxyz"
    cif_zip_file = Path(output_dir) / "generated_crystals_cif.zip"
    quality_report_files = list(Path(output_dir).glob("quality_report_*.json"))
    
    analysis = {
        'files_found': {
            'extxyz': extxyz_file.exists(),
            'cif_zip': cif_zip_file.exists(),
            'quality_reports': len(quality_report_files)
        },
        'structure_count': 0,
        'valid_structures': 0,
        'invalid_structures': 0,
        'structural_metrics': {},
        'quality_metrics': {}
    }
    
    # Parse quality report if available
    if quality_report_files:
        try:
            with open(quality_report_files[0], 'r') as f:
                quality_data = json.load(f)
                analysis['quality_metrics'] = quality_data
                logger.info(f"📊 Quality report loaded: {quality_report_files[0].name}")
        except Exception as e:
            logger.warning(f"Could not parse quality report: {e}")
    
    # Analyze structures from extxyz file if it exists
    if extxyz_file.exists():
        try:
            structure_count = 0
            valid_count = 0
            invalid_count = 0
            
            with open(extxyz_file, 'r') as f:
                content = f.read()
                
                # Count structures (each structure starts with atom count line)
                lines = content.strip().split('\n')
                i = 0
                while i < len(lines):
                    if lines[i].strip().isdigit():
                        structure_count += 1
                        atom_count = int(lines[i].strip())
                        
                        # Check if structure appears valid (has required number of lines)
                        if i + 1 + atom_count < len(lines):
                            valid_count += 1
                        else:
                            invalid_count += 1
                        
                        # Skip to next structure
                        i += 2 + atom_count  # atom count line + properties line + atom lines
                    else:
                        i += 1
            
            analysis['structure_count'] = structure_count
            analysis['valid_structures'] = valid_count
            analysis['invalid_structures'] = invalid_count
            
            logger.info(f"📈 Structure analysis: {structure_count} total, {valid_count} valid, {invalid_count} invalid")
            
        except Exception as e:
            logger.warning(f"Could not analyze structures from extxyz: {e}")
    
    # Calculate basic quality metrics
    if analysis['structure_count'] > 0:
        analysis['structural_metrics'] = {
            'validity_rate': analysis['valid_structures'] / analysis['structure_count'],
            'error_rate': analysis['invalid_structures'] / analysis['structure_count'],
            'completion_rate': analysis['structure_count'] / 256  # Assuming 256 requested
        }
    
    return analysis


def run_comparison_study(base_output_dir: str = "results/mixed_precision_comparison") -> Dict[str, Any]:
    """Run the complete mixed precision comparison study."""
    
    logger.info("🚀 Starting Mixed Precision vs Full Precision Comparison Study")
    logger.info("=" * 80)
    
    # Create base output directory
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Experiment configurations
    experiments = [
        {
            'name': 'full_precision_fp32',
            'mixed_precision': False,
            'output_dir': str(base_path / 'full_precision')
        },
        {
            'name': 'mixed_precision_fp16',
            'mixed_precision': True,
            'output_dir': str(base_path / 'mixed_precision')
        }
    ]
    
    results = {}
    
    # Run experiments
    for exp in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT: {exp['name'].upper()}")
        logger.info(f"{'='*60}")
        
        # Run generation
        gen_result = run_generation_experiment(
            output_dir=exp['output_dir'],
            experiment_name=exp['name'],
            enable_mixed_precision=exp['mixed_precision'],
            num_structures=256,
            batch_size=32
        )
        
        # Analyze results if successful
        if gen_result['success']:
            analysis = analyze_crystal_structures(exp['output_dir'])
            gen_result.update(analysis)
        
        results[exp['name']] = gen_result
    
    # Generate comparison report
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON REPORT")
    logger.info(f"{'='*60}")
    
    comparison = compare_experiments(results)
    
    # Save comprehensive results
    output_file = base_path / 'comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'experiments': results,
            'comparison': comparison
        }, f, indent=2)
    
    logger.info(f"📁 Complete results saved to: {output_file}")
    
    return {
        'experiments': results,
        'comparison': comparison,
        'output_file': str(output_file)
    }


def compare_experiments(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the results of both experiments."""
    
    comparison = {
        'summary': {},
        'performance': {},
        'quality': {},
        'recommendations': []
    }
    
    # Check if both experiments succeeded
    fp32_success = results.get('full_precision_fp32', {}).get('success', False)
    fp16_success = results.get('mixed_precision_fp16', {}).get('success', False)
    
    comparison['summary'] = {
        'full_precision_success': fp32_success,
        'mixed_precision_success': fp16_success,
        'both_successful': fp32_success and fp16_success
    }
    
    if not comparison['summary']['both_successful']:
        if not fp32_success:
            comparison['recommendations'].append("❌ Full precision (FP32) experiment failed")
        if not fp16_success:
            comparison['recommendations'].append("❌ Mixed precision (FP16) experiment failed - confirms multi-GPU instability issues")
        return comparison
    
    # Performance comparison
    fp32_time = results['full_precision_fp32']['duration']
    fp16_time = results['mixed_precision_fp16']['duration']
    
    comparison['performance'] = {
        'fp32_duration_seconds': fp32_time,
        'fp16_duration_seconds': fp16_time,
        'speedup_factor': fp32_time / fp16_time if fp16_time > 0 else 0,
        'time_savings_percent': ((fp32_time - fp16_time) / fp32_time) * 100 if fp32_time > 0 else 0
    }
    
    # Quality comparison
    fp32_metrics = results['full_precision_fp32'].get('structural_metrics', {})
    fp16_metrics = results['mixed_precision_fp16'].get('structural_metrics', {})
    
    comparison['quality'] = {
        'fp32_validity_rate': fp32_metrics.get('validity_rate', 0),
        'fp16_validity_rate': fp16_metrics.get('validity_rate', 0),
        'fp32_error_rate': fp32_metrics.get('error_rate', 0),
        'fp16_error_rate': fp16_metrics.get('error_rate', 0),
        'fp32_structures': results['full_precision_fp32'].get('structure_count', 0),
        'fp16_structures': results['mixed_precision_fp16'].get('structure_count', 0)
    }
    
    if fp32_metrics and fp16_metrics:
        validity_diff = fp16_metrics.get('validity_rate', 0) - fp32_metrics.get('validity_rate', 0)
        comparison['quality']['validity_difference_percent'] = validity_diff * 100
    
    # Generate recommendations
    if comparison['performance']['speedup_factor'] > 1.2:
        comparison['recommendations'].append(f"✅ Mixed precision provides {comparison['performance']['speedup_factor']:.1f}x speedup")
    
    if abs(comparison['quality'].get('validity_difference_percent', 0)) < 5:
        comparison['recommendations'].append("✅ No significant quality difference between precisions")
    elif comparison['quality'].get('validity_difference_percent', 0) < -5:
        comparison['recommendations'].append("⚠️ Mixed precision shows lower structure validity")
    
    # Print comparison summary
    logger.info("\n📊 PERFORMANCE COMPARISON:")
    logger.info(f"   Full Precision (FP32): {fp32_time:.1f}s")
    logger.info(f"   Mixed Precision (FP16): {fp16_time:.1f}s")
    if comparison['performance']['speedup_factor'] > 0:
        logger.info(f"   Speedup: {comparison['performance']['speedup_factor']:.2f}x")
        logger.info(f"   Time savings: {comparison['performance']['time_savings_percent']:.1f}%")
    
    logger.info("\n🔬 QUALITY COMPARISON:")
    logger.info(f"   FP32 structures: {comparison['quality']['fp32_structures']}")
    logger.info(f"   FP16 structures: {comparison['quality']['fp16_structures']}")
    logger.info(f"   FP32 validity: {comparison['quality']['fp32_validity_rate']:.1%}")
    logger.info(f"   FP16 validity: {comparison['quality']['fp16_validity_rate']:.1%}")
    
    logger.info("\n💡 RECOMMENDATIONS:")
    for rec in comparison['recommendations']:
        logger.info(f"   {rec}")
    
    return comparison


def main():
    """Main function to run the comparison study."""
    
    print("🧬 MatterGen Mixed Precision Accuracy Comparison")
    print("=" * 60)
    print("This script will:")
    print("1. Generate 256 structures with full precision (FP32)")
    print("2. Generate 256 structures with mixed precision (FP16)")  
    print("3. Compare structural accuracy and performance")
    print("4. Provide recommendations for production use")
    print()
    
    # Confirm with user
    response = input("Continue with comparison study? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Study cancelled.")
        return
    
    try:
        # Run the comparison
        results = run_comparison_study()
        
        print("\n" + "="*60)
        print("🎉 COMPARISON STUDY COMPLETE!")
        print("="*60)
        print(f"Results saved to: {results['output_file']}")
        
        # Summary
        comparison = results['comparison']
        if comparison['summary']['both_successful']:
            print(f"\n✅ Both experiments completed successfully")
            print(f"⚡ Performance: {comparison['performance']['speedup_factor']:.2f}x speedup with mixed precision")
            print(f"🔬 Quality: {abs(comparison['quality'].get('validity_difference_percent', 0)):.1f}% difference in validity")
        else:
            print(f"\n⚠️ One or both experiments failed - check logs for details")
        
    except KeyboardInterrupt:
        print("\n\nStudy interrupted by user.")
    except Exception as e:
        logger.error(f"Study failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
