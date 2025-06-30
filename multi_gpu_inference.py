#!/usr/bin/env python3
"""
Multi-GPU Inference Launcher for MatterGen
==========================================

Production-ready multi-GPU inference launcher with consistent CLI interface.
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_gpu_inference.log')
    ]
)
logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        gpu_ids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
        logger.info(f"Found {len(gpu_ids)} available GPUs: {gpu_ids}")
        return gpu_ids
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not detect GPUs via nvidia-smi, assuming single GPU")
        return [0]


def extract_structure_count(stdout_text: str) -> int:
    """Extract structure count from job stdout."""
    import re
    # Look for "Generated X structures" pattern
    match = re.search(r'Generated (\d+) structures', stdout_text)
    if match:
        return int(match.group(1))
    return 0


def run_single_gpu_job(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single GPU inference job."""
    gpu_id = args['gpu_id']
    job_id = args['job_id']
    output_path = args['output_path']
    structures_for_this_gpu = args['structures_for_this_gpu']
    batch_size = args['batch_size']
    mattergen_args = args['mattergen_args']
    
    start_time = time.time()
    
    # Calculate batches needed for this GPU
    batches_needed = max(1, (structures_for_this_gpu + batch_size - 1) // batch_size)
    
    # Set CUDA_VISIBLE_DEVICES to restrict to this GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Build the command with consistent parameter names
    cmd = [
        sys.executable, '-m', 'mattergen.scripts.generate',
        output_path,
        '--batch_size', str(batch_size),
        '--num_batches', str(batches_needed),
        '--enable_multi_gpu', 'False',  # Disable multi-GPU for single-GPU jobs
    ]
    
    # Add all other mattergen arguments with consistent naming
    for key, value in mattergen_args.items():
        if key in ['output_path', 'batch_size', 'num_batches', 'enable_multi_gpu']:
            continue  # Skip already handled args
        
        # Convert underscores to hyphens for CLI consistency
        cli_key = key.replace('_', '-')
        cmd.extend([f'--{cli_key}', str(value)])
    
    logger.info(f"GPU {gpu_id} Job {job_id}: Starting {structures_for_this_gpu} structures ({batches_needed} batches)")
    logger.debug(f"GPU {gpu_id} Job {job_id}: Command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract actual structure count from output
        structures_generated = extract_structure_count(result.stdout)
        
        logger.info(f"GPU {gpu_id} Job {job_id}: Completed in {duration:.2f}s - Generated {structures_generated} structures")
        
        return {
            'job_id': job_id,
            'gpu_id': gpu_id,
            'success': True,
            'duration': duration,
            'structures_requested': structures_for_this_gpu,
            'structures_generated': structures_generated,
            'batches': batches_needed,
            'output_path': output_path,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"GPU {gpu_id} Job {job_id}: Failed after {duration:.2f}s")
        logger.error(f"GPU {gpu_id} Job {job_id}: Error: {e}")
        logger.error(f"GPU {gpu_id} Job {job_id}: Stderr: {e.stderr}")
        
        return {
            'job_id': job_id,
            'gpu_id': gpu_id,
            'success': False,
            'duration': duration,
            'structures_requested': structures_for_this_gpu,
            'structures_generated': 0,
            'batches': batches_needed,
            'output_path': output_path,
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }


def distribute_structures(total_structures: int, num_gpus: int) -> List[int]:
    """Distribute structures across GPUs as evenly as possible."""
    base_structures = total_structures // num_gpus
    extra_structures = total_structures % num_gpus
    
    distribution = []
    for i in range(num_gpus):
        structures_for_gpu = base_structures + (1 if i < extra_structures else 0)
        distribution.append(structures_for_gpu)
    
    return distribution


def consolidate_results(job_results: List[Dict], output_base_path: str) -> None:
    """Consolidate results from all GPU jobs."""
    logger.info("Consolidating results...")
    
    # Calculate totals
    total_structures_generated = sum(r.get('structures_generated', 0) for r in job_results)
    total_structures_requested = sum(r.get('structures_requested', 0) for r in job_results)
    total_time_seconds = max(r['duration'] for r in job_results) if job_results else 0
    
    # Create comprehensive summary
    summary = {
        'total_jobs': len(job_results),
        'successful_jobs': sum(1 for r in job_results if r['success']),
        'failed_jobs': sum(1 for r in job_results if not r['success']),
        'total_structures_requested': total_structures_requested,
        'total_structures_generated': total_structures_generated,
        'total_time_seconds': total_time_seconds,
        'throughput_structures_per_second': total_structures_generated / total_time_seconds if total_time_seconds > 0 else 0,
        'total_batches': sum(r['batches'] for r in job_results),
        'gpu_utilization': {},
        'job_details': job_results
    }
    
    # Calculate per-GPU stats
    for result in job_results:
        gpu_id = result['gpu_id']
        if gpu_id not in summary['gpu_utilization']:
            summary['gpu_utilization'][gpu_id] = {
                'jobs': 0,
                'structures_generated': 0,
                'total_duration': 0,
                'successful_jobs': 0
            }
        
        summary['gpu_utilization'][gpu_id]['jobs'] += 1
        summary['gpu_utilization'][gpu_id]['structures_generated'] += result.get('structures_generated', 0)
        summary['gpu_utilization'][gpu_id]['total_duration'] += result['duration']
        if result['success']:
            summary['gpu_utilization'][gpu_id]['successful_jobs'] += 1
    
    # Save summary
    summary_path = Path(output_base_path) / 'multi_gpu_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results summary saved to: {summary_path}")
    logger.info(f"Successful jobs: {summary['successful_jobs']}/{summary['total_jobs']}")
    logger.info(f"Total structures generated: {total_structures_generated}/{total_structures_requested}")
    logger.info(f"Total wall time: {total_time_seconds:.2f}s")
    logger.info(f"Throughput: {summary['throughput_structures_per_second']:.3f} structures/second")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Inference Launcher for MatterGen - Production Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Production Examples:

  # Generate 256 structures on 4 GPUs with all optimizations
  python multi_gpu_inference.py \\
    --output_base_path "results/production_256_4gpu" \\
    --pretrained_name "mattergen_base" \\
    --total_structures 256 \\
    --num_gpus 4 \\
    --base_batch_size 16 \\
    --sampling_config_name "optimized_compatible" \\
    --enable_optimizations true \\
    --enable_model_compilation true \\
    --enable_graph_caching true \\
    --record_trajectories false

  # Large-scale production run with 1024 structures
  python multi_gpu_inference.py \\
    --output_base_path "results/large_scale_1024" \\
    --pretrained_name "mattergen_base" \\
    --total_structures 1024 \\
    --num_gpus 8 \\
    --base_batch_size 32 \\
    --sampling_config_name "phase3_optimized" \\
    --enable_optimizations true
        """
    )
    
    # Required arguments
    parser.add_argument('--output_base_path', required=True,
                        help='Base output directory (subdirs will be created for each GPU)')
    parser.add_argument('--total_structures', type=int, required=True,
                        help='Total number of structures to generate across all GPUs')
    
    # Model selection (one required)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--pretrained_name', 
                           help='Name of pretrained model (e.g., "mattergen_base")')
    model_group.add_argument('--model_path',
                           help='Path to custom model checkpoint')
    
    # Multi-GPU settings
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: auto-detect all)')
    parser.add_argument('--base_batch_size', type=int, default=16,
                        help='Batch size per GPU (default: 16)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Max concurrent processes (default: num_gpus)')
    
    # Sampling configuration
    parser.add_argument('--sampling_config_name', default='optimized_compatible',
                        help='Sampling config name (default: optimized_compatible)')
    parser.add_argument('--sampling_config_path', default=None,
                        help='Path to sampling config directory')
    
    # Conditioning and guidance
    parser.add_argument('--properties_to_condition_on', type=str, default=None,
                        help='Properties to condition on (JSON string)')
    parser.add_argument('--guidance', type=str, default=None,
                        help='Guidance settings (JSON string)')
    parser.add_argument('--diffusion_guidance_factor', type=float, default=None,
                        help='Diffusion guidance factor')
    
    # Performance optimization flags
    parser.add_argument('--enable_optimizations', type=str, default='true',
                        choices=['true', 'false'],
                        help='Enable performance optimizations (default: true)')
    parser.add_argument('--enable_mixed_precision', type=str, default='false',
                        choices=['true', 'false'],
                        help='Enable mixed precision (default: false - disabled due to stability)')
    parser.add_argument('--enable_model_compilation', type=str, default='true',
                        choices=['true', 'false'],
                        help='Enable model compilation (default: true)')
    parser.add_argument('--enable_graph_caching', type=str, default='true',
                        choices=['true', 'false'],
                        help='Enable graph caching (default: true)')
    
    # Output options
    parser.add_argument('--record_trajectories', type=str, default='false',
                        choices=['true', 'false'],
                        help='Whether to record trajectories (default: false)')
    parser.add_argument('--print_loss', type=str, default='false',
                        choices=['true', 'false'],
                        help='Whether to print loss information (default: false)')
    parser.add_argument('--print_optimization_info', type=str, default='false',
                        choices=['true', 'false'],
                        help='Print optimization information (default: false)')
    
    # Advanced options
    parser.add_argument('--diffusion_loss_weight', type=float, default=1.0,
                        help='Diffusion loss weight (default: 1.0)')
    
    # Phase 4: Adaptive sampling and quality metrics
    parser.add_argument('--enable_phase4_features', type=str, default='false',
                        choices=['true', 'false'],
                        help='Enable Phase 4 adaptive features (default: false)')
    parser.add_argument('--enable_adaptive_sampling', type=str, default='false',
                        choices=['true', 'false'],
                        help='Enable adaptive sampling (default: false)')
    parser.add_argument('--enable_quality_metrics', type=str, default='false',
                        choices=['true', 'false'],
                        help='Enable quality metrics assessment (default: false)')
    parser.add_argument('--adaptive_config_path', default=None,
                        help='Path to adaptive sampling config file')
    parser.add_argument('--quality_threshold', type=float, default=0.7,
                        help='Quality threshold for structure filtering (default: 0.7)')
    parser.add_argument('--max_adaptation_iterations', type=int, default=5,
                        help='Maximum adaptation iterations (default: 5)')
    
    args = parser.parse_args()
    
    # Convert string booleans to actual booleans
    bool_args = ['enable_optimizations', 'enable_mixed_precision', 'enable_model_compilation', 
                 'enable_graph_caching', 'record_trajectories', 'print_loss', 'print_optimization_info',
                 'enable_phase4_features', 'enable_adaptive_sampling', 'enable_quality_metrics']
    for arg_name in bool_args:
        setattr(args, arg_name, getattr(args, arg_name).lower() == 'true')
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    num_gpus = args.num_gpus if args.num_gpus is not None else len(available_gpus)
    num_gpus = min(num_gpus, len(available_gpus))
    
    if num_gpus <= 0:
        logger.error("No GPUs available or specified")
        sys.exit(1)
    
    gpus_to_use = available_gpus[:num_gpus]
    logger.info(f"Using {num_gpus} GPUs: {gpus_to_use}")
    
    # Distribute structures across GPUs
    structure_distribution = distribute_structures(args.total_structures, num_gpus)
    logger.info(f"Structure distribution: {structure_distribution} (total: {sum(structure_distribution)})")
    
    # Create output directories and job configurations
    output_base = Path(args.output_base_path)
    output_base.mkdir(parents=True, exist_ok=True)
    
    jobs = []
    for i, (gpu_id, structures) in enumerate(zip(gpus_to_use, structure_distribution)):
        if structures == 0:
            continue
        
        job_output_path = str(output_base / f"gpu_{gpu_id}")
        
        # Prepare mattergen arguments with consistent naming
        mattergen_args = {
            'sampling_config_name': args.sampling_config_name,
            'record_trajectories': args.record_trajectories,
            'print_loss': args.print_loss,
            'diffusion_loss_weight': args.diffusion_loss_weight,
            'enable_optimizations': args.enable_optimizations,
            'enable_mixed_precision': args.enable_mixed_precision,
            'enable_model_compilation': args.enable_model_compilation,
            'enable_graph_caching': args.enable_graph_caching,
            'print_optimization_info': args.print_optimization_info,
            # Phase 4 options
            'enable_phase4_features': args.enable_phase4_features,
            'enable_adaptive_sampling': args.enable_adaptive_sampling,
            'enable_quality_metrics': args.enable_quality_metrics,
            'quality_threshold': args.quality_threshold,
            'max_adaptation_iterations': args.max_adaptation_iterations,
        }
        
        # Add model selection
        if args.pretrained_name:
            mattergen_args['pretrained_name'] = args.pretrained_name
        else:
            mattergen_args['model_path'] = args.model_path
        
        # Add optional arguments
        if args.properties_to_condition_on:
            mattergen_args['properties_to_condition_on'] = args.properties_to_condition_on
        if args.guidance:
            mattergen_args['guidance'] = args.guidance
        if args.diffusion_guidance_factor is not None:
            mattergen_args['diffusion_guidance_factor'] = args.diffusion_guidance_factor
        if args.sampling_config_path:
            mattergen_args['sampling_config_path'] = args.sampling_config_path
        if args.adaptive_config_path:
            mattergen_args['adaptive_config_path'] = args.adaptive_config_path
        
        job_config = {
            'job_id': i,
            'gpu_id': gpu_id,
            'output_path': job_output_path,
            'structures_for_this_gpu': structures,
            'batch_size': args.base_batch_size,
            'mattergen_args': mattergen_args
        }
        
        jobs.append(job_config)
    
    if not jobs:
        logger.error("No jobs to run")
        sys.exit(1)
    
    logger.info(f"Prepared {len(jobs)} jobs")
    
    # Run jobs in parallel
    max_workers = args.max_workers if args.max_workers is not None else num_gpus
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info(f"STARTING MULTI-GPU INFERENCE: {args.total_structures} structures on {num_gpus} GPUs")
    logger.info("="*60)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(run_single_gpu_job, job): job for job in jobs}
        
        # Collect results as they complete
        results = []
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            
            if result['success']:
                logger.info(f"✅ Job {result['job_id']} (GPU {result['gpu_id']}) completed successfully")
            else:
                logger.error(f"❌ Job {result['job_id']} (GPU {result['gpu_id']}) failed")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Consolidate results
    consolidate_results(results, args.output_base_path)
    
    # Final summary
    successful_jobs = sum(1 for r in results if r['success'])
    total_structures_generated = sum(r.get('structures_generated', 0) for r in results if r['success'])
    
    logger.info("\n" + "="*60)
    logger.info("🎉 MULTI-GPU INFERENCE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total wall time: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
    logger.info(f"Successful jobs: {successful_jobs}/{len(results)}")
    logger.info(f"Structures generated: {total_structures_generated}/{args.total_structures}")
    logger.info(f"Throughput: {total_structures_generated/total_duration:.3f} structures/second")
    logger.info(f"Results saved to: {args.output_base_path}")
    
    if successful_jobs < len(results):
        logger.warning(f"⚠️  {len(results) - successful_jobs} jobs failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()