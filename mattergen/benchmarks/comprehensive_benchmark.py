#!/usr/bin/env python3
"""
MatterGen Multi-GPU Performance Benchmark
=========================================

This script runs comprehensive benchmarks to measure the performance improvements
from our 3-phase optimization strategy:
- Phase 1: Sampling config optimizations
- Phase 2: Model inference optimizations (mixed precision, compilation, etc.)
- Phase 3: Multi-GPU support and advanced caching

Results are saved with detailed timing and throughput metrics.
"""

import os
import time
import subprocess
import json
from pathlib import Path
import argparse

def run_benchmark_test(name, description, cmd, timeout=900):
    """Run a single benchmark test and measure performance."""
    print(f"\n{'='*80}")
    print(f"🏃‍♂️ Running: {name}")
    print(f"📝 Description: {description}")
    print(f"🔧 Command: {cmd}")
    print('='*80)
    
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
        
        success = result.returncode == 0
        
        # Extract generated structures count from output
        structures_generated = 0
        if success and "Generated" in result.stdout:
            try:
                # Look for "Generated X structures"
                import re
                match = re.search(r'Generated (\d+) structures', result.stdout)
                if match:
                    structures_generated = int(match.group(1))
            except:
                pass
        
        # Calculate throughput
        throughput = structures_generated / elapsed if elapsed > 0 else 0
        
        result_data = {
            'name': name,
            'description': description,
            'command': cmd,
            'success': success,
            'elapsed_time': elapsed,
            'structures_generated': structures_generated,
            'throughput_structures_per_second': throughput,
            'stdout': result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
            'stderr': result.stderr[-1000:] if result.stderr else "",
            'return_code': result.returncode
        }
        
        if success:
            print(f"✅ SUCCESS in {elapsed:.2f}s")
            print(f"📊 Generated {structures_generated} structures")
            print(f"⚡ Throughput: {throughput:.3f} structures/second")
        else:
            print(f"❌ FAILED after {elapsed:.2f}s (return code: {result.returncode})")
            if result.stderr:
                print(f"💥 Error: {result.stderr[-200:]}")
        
        return result_data
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after {timeout} seconds")
        return {
            'name': name,
            'description': description,
            'command': cmd,
            'success': False,
            'elapsed_time': timeout,
            'structures_generated': 0,
            'throughput_structures_per_second': 0,
            'stdout': "",
            'stderr': "Timeout expired",
            'return_code': -1
        }
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return {
            'name': name,
            'description': description,
            'command': cmd,
            'success': False,
            'elapsed_time': 0,
            'structures_generated': 0,
            'throughput_structures_per_second': 0,
            'stdout': "",
            'stderr': str(e),
            'return_code': -2
        }

def main():
    parser = argparse.ArgumentParser(description="MatterGen Multi-GPU Performance Benchmark")
    parser.add_argument('--output_dir', default='benchmark_results', 
                       help='Directory to save benchmark results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for tests')
    parser.add_argument('--num_batches', type=int, default=2,
                       help='Number of batches for single-GPU tests')
    parser.add_argument('--total_batches', type=int, default=8,
                       help='Total batches for multi-GPU tests')
    parser.add_argument('--max_gpus', type=int, default=4,
                       help='Maximum number of GPUs to test')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per test in seconds')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 MatterGen Multi-GPU Performance Benchmark")
    print(f"📁 Results will be saved to: {output_dir}")
    print(f"⚙️  Configuration: batch_size={args.batch_size}, num_batches={args.num_batches}")
    
    # Define benchmark tests
    benchmark_tests = [
        {
            'name': 'baseline_default_config',
            'description': 'Baseline with default sampling config (no optimizations)',
            'cmd': f'''mattergen-generate "{output_dir}/baseline_default" \\
                --pretrained_name=chemical_system \\
                --batch_size={args.batch_size} \\
                --num_batches={args.num_batches} \\
                --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \\
                --record_trajectories=False \\
                --sampling_config_name=default \\
                --enable_optimizations=False \\
                --enable_multi_gpu=False'''
        },
        {
            'name': 'phase1_optimized_config',
            'description': 'Phase 1: Optimized sampling config only',
            'cmd': f'''mattergen-generate "{output_dir}/phase1_optimized" \\
                --pretrained_name=chemical_system \\
                --batch_size={args.batch_size} \\
                --num_batches={args.num_batches} \\
                --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \\
                --record_trajectories=False \\
                --sampling_config_name=optimized_compatible \\
                --enable_optimizations=False \\
                --enable_multi_gpu=False'''
        },
        {
            'name': 'phase2_inference_optimizations',
            'description': 'Phase 2: Sampling config + inference optimizations',
            'cmd': f'''mattergen-generate "{output_dir}/phase2_inference" \\
                --pretrained_name=chemical_system \\
                --batch_size={args.batch_size} \\
                --num_batches={args.num_batches} \\
                --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \\
                --record_trajectories=False \\
                --sampling_config_name=optimized_compatible \\
                --enable_optimizations=True \\
                --enable_mixed_precision=True \\
                --enable_model_compilation=True \\
                --enable_multi_gpu=False \\
                --print_optimization_info=True'''
        },
        {
            'name': 'phase3_single_gpu_full',
            'description': 'Phase 3: All optimizations on single GPU',
            'cmd': f'''mattergen-generate "{output_dir}/phase3_single" \\
                --pretrained_name=chemical_system \\
                --batch_size={args.batch_size} \\
                --num_batches={args.num_batches} \\
                --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \\
                --record_trajectories=False \\
                --sampling_config_name=optimized_compatible \\
                --enable_optimizations=True \\
                --enable_mixed_precision=True \\
                --enable_model_compilation=True \\
                --enable_graph_caching=True \\
                --enable_multi_gpu=False \\
                --max_gpus=1 \\
                --print_optimization_info=True'''
        },
        {
            'name': 'phase3_multi_gpu_2',
            'description': 'Phase 3: Multi-GPU (2 GPUs) with multi-process',
            'cmd': f'''python multi_gpu_inference.py \\
                --output_base_path "{output_dir}/phase3_multi_2gpu" \\
                --pretrained_name chemical_system \\
                --num_gpus 2 \\
                --total_batches {args.total_batches} \\
                --batch_size {args.batch_size} \\
                --properties_to_condition_on "{{'chemical_system':'Pd-Ni-H'}}" \\
                --sampling_config_name optimized_compatible \\
                --enable_optimizations True \\
                --enable_mixed_precision True \\
                --enable_model_compilation True \\
                --enable_graph_caching True'''
        },
        {
            'name': 'phase3_multi_gpu_4',
            'description': 'Phase 3: Multi-GPU (4 GPUs) with multi-process',
            'cmd': f'''python multi_gpu_inference.py \\
                --output_base_path "{output_dir}/phase3_multi_4gpu" \\
                --pretrained_name chemical_system \\
                --num_gpus {min(4, args.max_gpus)} \\
                --total_batches {args.total_batches * 2} \\
                --batch_size {args.batch_size} \\
                --properties_to_condition_on "{{'chemical_system':'Pd-Ni-H'}}" \\
                --sampling_config_name optimized_compatible \\
                --enable_optimizations True \\
                --enable_mixed_precision True \\
                --enable_model_compilation True \\
                --enable_graph_caching True'''
        }
    ]
    
    # Run benchmarks
    results = []
    
    for i, test in enumerate(benchmark_tests):
        print(f"\n🔄 Running test {i+1}/{len(benchmark_tests)}")
        result = run_benchmark_test(
            test['name'], 
            test['description'], 
            test['cmd'],
            timeout=args.timeout
        )
        results.append(result)
        
        # Brief pause between tests
        time.sleep(3)
    
    # Generate comprehensive report
    print(f"\n{'='*100}")
    print("📊 COMPREHENSIVE BENCHMARK RESULTS")
    print('='*100)
    
    # Calculate speedups relative to baseline
    baseline_time = None
    baseline_throughput = None
    
    for result in results:
        if 'baseline' in result['name'] and result['success']:
            baseline_time = result['elapsed_time']
            baseline_throughput = result['throughput_structures_per_second']
            break
    
    # Print summary table
    print(f"{'Test Name':<30} {'Status':<10} {'Time (s)':<10} {'Structures':<12} {'Throughput':<15} {'Speedup':<10}")
    print('-' * 100)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        time_str = f"{result['elapsed_time']:.1f}" if result['success'] else "N/A"
        structures = str(result['structures_generated']) if result['success'] else "0"
        throughput = f"{result['throughput_structures_per_second']:.3f}/s" if result['success'] else "0.000/s"
        
        speedup = ""
        if result['success'] and baseline_time and result['elapsed_time'] > 0:
            speedup_factor = baseline_time / result['elapsed_time']
            speedup = f"{speedup_factor:.2f}x"
        elif result['success'] and baseline_throughput and result['throughput_structures_per_second'] > 0:
            throughput_speedup = result['throughput_structures_per_second'] / baseline_throughput
            speedup = f"{throughput_speedup:.2f}x"
        
        print(f"{result['name']:<30} {status:<10} {time_str:<10} {structures:<12} {throughput:<15} {speedup:<10}")
    
    # Save detailed results
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'benchmark_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'configuration': {
                    'batch_size': args.batch_size,
                    'num_batches': args.num_batches,
                    'total_batches': args.total_batches,
                    'max_gpus': args.max_gpus,
                    'timeout': args.timeout
                }
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    # Performance analysis
    successful_tests = [r for r in results if r['success']]
    if len(successful_tests) >= 2:
        print(f"\n🎯 PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        if baseline_time:
            best_single_gpu = min([r for r in successful_tests if 'multi_gpu' not in r['name']], 
                                key=lambda x: x['elapsed_time'])
            best_multi_gpu = min([r for r in successful_tests if 'multi_gpu' in r['name']], 
                               key=lambda x: x['elapsed_time']) if any('multi_gpu' in r['name'] for r in successful_tests) else None
            
            print(f"Best single-GPU speedup: {baseline_time / best_single_gpu['elapsed_time']:.2f}x ({best_single_gpu['name']})")
            if best_multi_gpu:
                print(f"Best multi-GPU speedup: {baseline_time / best_multi_gpu['elapsed_time']:.2f}x ({best_multi_gpu['name']})")
                print(f"Multi-GPU vs best single-GPU: {best_single_gpu['elapsed_time'] / best_multi_gpu['elapsed_time']:.2f}x faster")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 50)
    print("✅ For production workloads:")
    print("   - Use 'optimized_compatible' sampling config")
    print("   - Enable all Phase 2 optimizations (mixed precision, compilation)")
    print("   - Use multi-GPU launcher for >4 batches")
    print("   - Monitor GPU memory usage and adjust batch sizes accordingly")
    
    print(f"\n🎉 Benchmark complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
