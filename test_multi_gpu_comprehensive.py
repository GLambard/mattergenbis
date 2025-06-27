#!/usr/bin/env python3
"""
Comprehensive Multi-GPU Testing Script for MatterGen

This script tests various multi-GPU configurations and measures performance improvements.
"""

import os
import time
import subprocess
import sys
from pathlib import Path
import json

def run_command_with_timing(cmd, description, timeout=300):
    """Run a command and measure execution time."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Exit code: {result.returncode}")
        print(f"Execution time: {elapsed:.2f} seconds")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout[-1000:]}")  # Last 1000 chars
        if result.stderr:
            print(f"STDERR:\n{result.stderr[-1000:]}")  # Last 1000 chars
            
        return {
            'description': description,
            'command': cmd,
            'exit_code': result.returncode,
            'elapsed_time': elapsed,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return {
            'description': description,
            'command': cmd,
            'exit_code': -1,
            'elapsed_time': timeout,
            'success': False,
            'stdout': '',
            'stderr': 'Timeout expired'
        }
    except Exception as e:
        print(f"Error running command: {e}")
        return {
            'description': description,
            'command': cmd,
            'exit_code': -2,
            'elapsed_time': 0,
            'success': False,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    # Create test output directories
    test_base_dir = "test_multi_gpu_results"
    os.makedirs(test_base_dir, exist_ok=True)
    
    # Test configurations
    test_configs = [
        {
            'name': 'single_gpu_baseline',
            'description': 'Single GPU Baseline (GPU 0)',
            'cmd': f'''mattergen-generate "{test_base_dir}/single_gpu" \\
                --pretrained-name=chemical_system \\
                --batch_size=4 \\
                --num_batches=2 \\
                --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \\
                --record_trajectories=False \\
                --sampling_config_name=optimized_compatible \\
                --enable_multi_gpu=False \\
                --max_gpus=1 \\
                --print_optimization_info=True'''
        },
        {
            'name': 'multi_gpu_dp',
            'description': 'Multi-GPU DataParallel (2 GPUs)',
            'cmd': f'''mattergen-generate "{test_base_dir}/multi_gpu_dp" \\
                --pretrained-name=chemical_system \\
                --batch_size=4 \\
                --num_batches=2 \\
                --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \\
                --record_trajectories=False \\
                --sampling_config_name=optimized_compatible \\
                --enable_multi_gpu=True \\
                --multi_gpu_strategy=dp \\
                --max_gpus=2 \\
                --print_optimization_info=True'''
        },
        {
            'name': 'multi_gpu_ddp',
            'description': 'Multi-GPU DistributedDataParallel (2 GPUs)',
            'cmd': f'''mattergen-generate "{test_base_dir}/multi_gpu_ddp" \\
                --pretrained-name=chemical_system \\
                --batch_size=4 \\
                --num_batches=2 \\
                --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \\
                --record_trajectories=False \\
                --sampling_config_name=optimized_compatible \\
                --enable_multi_gpu=True \\
                --multi_gpu_strategy=ddp \\
                --max_gpus=2 \\
                --print_optimization_info=True'''
        },
        {
            'name': 'multi_process_launcher_2gpu',
            'description': 'Multi-Process Launcher (2 GPUs)',
            'cmd': f'''python multi_gpu_inference.py \\
                --output_dir "{test_base_dir}/multi_process_2gpu" \\
                --pretrained_name chemical_system \\
                --num_gpus 2 \\
                --samples_per_gpu 8 \\
                --properties_to_condition_on "{{'chemical_system':'Pd-Ni-H'}}" \\
                --sampling_config_name optimized_compatible'''
        },
        {
            'name': 'multi_process_launcher_4gpu',
            'description': 'Multi-Process Launcher (4 GPUs)',
            'cmd': f'''python multi_gpu_inference.py \\
                --output_dir "{test_base_dir}/multi_process_4gpu" \\
                --pretrained_name chemical_system \\
                --num_gpus 4 \\
                --samples_per_gpu 4 \\
                --properties_to_condition_on "{{'chemical_system':'Pd-Ni-H'}}" \\
                --sampling_config_name optimized_compatible'''
        }
    ]
    
    # Run tests
    results = []
    
    print("Starting Multi-GPU Performance Testing")
    print(f"Available GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")
    
    for config in test_configs:
        result = run_command_with_timing(
            config['cmd'], 
            config['description'],
            timeout=600  # 10 minutes timeout
        )
        result['test_name'] = config['name']
        results.append(result)
        
        # Short break between tests
        time.sleep(5)
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("MULTI-GPU TESTING SUMMARY")
    print('='*80)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        print("\nPerformance Comparison:")
        baseline_time = None
        for result in successful_tests:
            if 'baseline' in result['test_name']:
                baseline_time = result['elapsed_time']
                break
        
        for result in successful_tests:
            speedup = ""
            if baseline_time and result['elapsed_time'] > 0:
                speedup_factor = baseline_time / result['elapsed_time']
                speedup = f" (Speedup: {speedup_factor:.2f}x)"
            
            print(f"  {result['description']}: {result['elapsed_time']:.2f}s{speedup}")
    
    if failed_tests:
        print("\nFailed Tests:")
        for result in failed_tests:
            print(f"  {result['description']}: {result['stderr'][:100]}...")
    
    # Save detailed results
    results_file = f"{test_base_dir}/test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print('='*80)
    
    if any(r['success'] and 'multi_process' in r['test_name'] for r in results):
        print("✓ Multi-process launcher appears to work - this is the recommended approach")
        print("  for true multi-GPU inference with maximum throughput.")
    
    if any(r['success'] and ('dp' in r['test_name'] or 'ddp' in r['test_name']) for r in results):
        print("✓ DataParallel/DistributedDataParallel work but may not provide true")
        print("  multi-GPU inference for this type of generative model.")
    
    print("\nFor production workloads:")
    print("1. Use the multi-process launcher (multi_gpu_inference.py)")
    print("2. Adjust samples_per_gpu based on GPU memory")
    print("3. Use optimized_compatible sampling config")
    print("4. Monitor GPU utilization with nvidia-smi")

if __name__ == "__main__":
    main()
