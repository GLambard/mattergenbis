#!/usr/bin/env python3
"""
Multi-GPU Inference Launcher for MatterGen
==========================================

This script implements true multi-GPU inference by launching multiple single-GPU
processes in parallel, each targeting a different GPU. This is the most effective
way to utilize multiple GPUs for inference workloads.

Usage:
    python multi_gpu_inference.py --help
    
Example:
    python multi_gpu_inference.py \
        --output_base_path "results/multi_gpu_test" \
        --pretrained_name "chemical_system" \
        --total_batches 8 \
        --batch_size 16 \
        --num_gpus 4 \
        --properties_to_condition_on "{'chemical_system':'Pd-Ni-H'}" \
        --guidance "{'volume': 30.935}"

Features:
- Automatic GPU detection and assignment
- Load balancing across available GPUs
- Progress monitoring and logging
- Robust error handling and cleanup
- Results consolidation
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
    format='%(asctime)s - %(levelname)s - %(message)s',
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


def run_single_gpu_job(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single GPU inference job."""
    gpu_id = args['gpu_id']
    job_id = args['job_id']
    output_path = args['output_path']
    batches_for_this_gpu = args['batches_for_this_gpu']
    mattergen_args = args['mattergen_args']
    
    start_time = time.time()
    
    # Set CUDA_VISIBLE_DEVICES to restrict to this GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Build the command
    cmd = [
        sys.executable, '-m', 'mattergen.scripts.generate',
        output_path,
        '--num_batches', str(batches_for_this_gpu),
        '--enable_multi_gpu', 'False',  # Disable multi-GPU for single-GPU jobs
    ]
    
    # Add all other mattergen arguments
    for key, value in mattergen_args.items():
        if key in ['output_path', 'num_batches', 'enable_multi_gpu']:
            continue  # Skip already handled args
        
        cmd.extend([f'--{key}', str(value)])
    
    logger.info(f"GPU {gpu_id} Job {job_id}: Starting {batches_for_this_gpu} batches")
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
        
        logger.info(f"GPU {gpu_id} Job {job_id}: Completed in {duration:.2f}s")
        
        return {
            'job_id': job_id,
            'gpu_id': gpu_id,
            'success': True,
            'duration': duration,
            'batches': batches_for_this_gpu,
            'output_path': output_path,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"GPU {gpu_id} Job {job_id}: Failed after {duration:.2f}s")
        logger.error(f"GPU {gpu_id} Job {job_id}: Error: {e}")
        logger.error(f"GPU {gpu_id} Job {job_id}: Stdout: {e.stdout}")
        logger.error(f"GPU {gpu_id} Job {job_id}: Stderr: {e.stderr}")
        
        return {
            'job_id': job_id,
            'gpu_id': gpu_id,
            'success': False,
            'duration': duration,
            'batches': batches_for_this_gpu,
            'output_path': output_path,
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }


def distribute_batches(total_batches: int, num_gpus: int) -> List[int]:
    """Distribute batches across GPUs as evenly as possible."""
    base_batches = total_batches // num_gpus
    extra_batches = total_batches % num_gpus
    
    distribution = []
    for i in range(num_gpus):
        batches_for_gpu = base_batches + (1 if i < extra_batches else 0)
        distribution.append(batches_for_gpu)
    
    return distribution


def consolidate_results(job_results: List[Dict], output_base_path: str) -> None:
    """Consolidate results from all GPU jobs."""
    logger.info("Consolidating results...")
    
    # Create summary
    summary = {
        'total_jobs': len(job_results),
        'successful_jobs': sum(1 for r in job_results if r['success']),
        'failed_jobs': sum(1 for r in job_results if not r['success']),
        'total_duration': max(r['duration'] for r in job_results),
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
                'batches': 0,
                'total_duration': 0,
                'successful_jobs': 0
            }
        
        summary['gpu_utilization'][gpu_id]['jobs'] += 1
        summary['gpu_utilization'][gpu_id]['batches'] += result['batches']
        summary['gpu_utilization'][gpu_id]['total_duration'] += result['duration']
        if result['success']:
            summary['gpu_utilization'][gpu_id]['successful_jobs'] += 1
    
    # Save summary
    summary_path = Path(output_base_path) / 'multi_gpu_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results summary saved to: {summary_path}")
    logger.info(f"Successful jobs: {summary['successful_jobs']}/{summary['total_jobs']}")
    logger.info(f"Total batches processed: {summary['total_batches']}")
    logger.info(f"Total wall time: {summary['total_duration']:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Inference Launcher for MatterGen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic multi-GPU run
  python multi_gpu_inference.py \\
    --output_base_path "results/multi_gpu_test" \\
    --pretrained_name "chemical_system" \\
    --total_batches 16 \\
    --batch_size 8 \\
    --num_gpus 4

  # With conditioning and guidance
  python multi_gpu_inference.py \\
    --output_base_path "results/conditioned_run" \\
    --pretrained_name "chemical_system" \\
    --total_batches 32 \\
    --batch_size 16 \\
    --properties_to_condition_on "{'chemical_system':'Pd-Ni-H'}" \\
    --guidance "{'volume': 30.935}" \\
    --diffusion_guidance_factor 2.0

  # Using custom model
  python multi_gpu_inference.py \\
    --output_base_path "results/custom_model" \\
    --model_path "/path/to/model" \\
    --total_batches 8 \\
    --batch_size 32
        """
    )
    
    # Required arguments
    parser.add_argument('--output_base_path', required=True,
                        help='Base output directory (subdirs will be created for each GPU)')
    parser.add_argument('--total_batches', type=int, required=True,
                        help='Total number of batches to process across all GPUs')
    
    # Model selection (one required)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--pretrained_name', 
                           help='Name of pretrained model')
    model_group.add_argument('--model_path',
                           help='Path to custom model checkpoint')
    
    # Multi-GPU settings
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: auto-detect)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Max concurrent processes (default: num_gpus)')
    
    # MatterGen arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--properties_to_condition_on', type=str, default=None,
                        help='Properties to condition on (JSON string)')
    parser.add_argument('--guidance', type=str, default=None,
                        help='Guidance settings (JSON string)')
    parser.add_argument('--diffusion_guidance_factor', type=float, default=None,
                        help='Diffusion guidance factor')
    parser.add_argument('--diffusion_loss_weight', type=float, default=1.0,
                        help='Diffusion loss weight')
    parser.add_argument('--sampling_config_name', default='optimized_compatible',
                        help='Sampling config name')
    parser.add_argument('--sampling_config_path', default=None,
                        help='Path to sampling config directory')
    parser.add_argument('--record_trajectories', type=bool, default=False,
                        help='Whether to record trajectories')
    parser.add_argument('--print_loss', type=bool, default=False,
                        help='Whether to print loss information')
    
    # Performance optimization flags
    parser.add_argument('--enable_optimizations', type=bool, default=True,
                        help='Enable performance optimizations')
    parser.add_argument('--enable_mixed_precision', type=bool, default=True,
                        help='Enable mixed precision')
    parser.add_argument('--enable_model_compilation', type=bool, default=True,
                        help='Enable model compilation')
    parser.add_argument('--enable_graph_caching', type=bool, default=True,
                        help='Enable graph caching')
    
    args = parser.parse_args()
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    num_gpus = args.num_gpus if args.num_gpus is not None else len(available_gpus)
    num_gpus = min(num_gpus, len(available_gpus))
    
    if num_gpus <= 0:
        logger.error("No GPUs available or specified")
        sys.exit(1)
    
    gpus_to_use = available_gpus[:num_gpus]
    logger.info(f"Using {num_gpus} GPUs: {gpus_to_use}")
    
    # Distribute batches
    batch_distribution = distribute_batches(args.total_batches, num_gpus)
    logger.info(f"Batch distribution: {batch_distribution}")
    
    # Create output directories and job configurations
    output_base = Path(args.output_base_path)
    output_base.mkdir(parents=True, exist_ok=True)
    
    jobs = []
    for i, (gpu_id, batches) in enumerate(zip(gpus_to_use, batch_distribution)):
        if batches == 0:
            continue
        
        job_output_path = str(output_base / f"gpu_{gpu_id}")
        
        # Prepare mattergen arguments
        mattergen_args = {
            'batch_size': args.batch_size,
            'sampling_config_name': args.sampling_config_name,
            'record_trajectories': args.record_trajectories,
            'print_loss': args.print_loss,
            'diffusion_loss_weight': args.diffusion_loss_weight,
            'enable_optimizations': args.enable_optimizations,
            'enable_mixed_precision': args.enable_mixed_precision,
            'enable_model_compilation': args.enable_model_compilation,
            'enable_graph_caching': args.enable_graph_caching,
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
        
        job_config = {
            'job_id': i,
            'gpu_id': gpu_id,
            'output_path': job_output_path,
            'batches_for_this_gpu': batches,
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
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(run_single_gpu_job, job): job for job in jobs}
        
        # Collect results as they complete
        results = []
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            
            if result['success']:
                logger.info(f"Job {result['job_id']} (GPU {result['gpu_id']}) completed successfully")
            else:
                logger.error(f"Job {result['job_id']} (GPU {result['gpu_id']}) failed")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Consolidate results
    consolidate_results(results, args.output_base_path)
    
    # Final summary
    successful_jobs = sum(1 for r in results if r['success'])
    total_batches_processed = sum(r['batches'] for r in results if r['success'])
    
    logger.info("\n" + "="*60)
    logger.info("MULTI-GPU INFERENCE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total wall time: {total_duration:.2f}s")
    logger.info(f"Successful jobs: {successful_jobs}/{len(results)}")
    logger.info(f"Total batches processed: {total_batches_processed}/{args.total_batches}")
    logger.info(f"Average throughput: {total_batches_processed/total_duration:.2f} batches/second")
    logger.info(f"Results saved to: {args.output_base_path}")
    
    if successful_jobs < len(results):
        logger.warning(f"{len(results) - successful_jobs} jobs failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
