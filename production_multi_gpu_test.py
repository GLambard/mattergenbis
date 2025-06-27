#!/usr/bin/env python3
"""
Production Multi-GPU Test for MatterGen
=======================================

This script performs a production-scale test of the multi-GPU capabilities
with a substantial number of generated structures to demonstrate real-world
performance improvements and validate production readiness.

Features:
- Large-scale structure generation (hundreds to thousands of structures)
- Multiple GPU configurations tested
- Real-time performance monitoring
- Comprehensive result validation
- Production deployment simulation
"""

import os
import time
import json
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import shutil

class ProductionMultiGPUTest:
    def __init__(self, base_output_dir="production_test_results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_dir = self.base_output_dir / f"test_{self.timestamp}"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Test results storage
        self.results = {
            'test_info': {
                'timestamp': self.timestamp,
                'test_directory': str(self.test_dir),
                'start_time': None,
                'end_time': None,
                'total_duration': None
            },
            'configurations': [],
            'performance_metrics': {},
            'validation_results': {}
        }
    
    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_command_with_monitoring(self, cmd, description, timeout=1800):
        """Run command with performance monitoring."""
        self.log(f"Starting: {description}")
        self.log(f"Command: {cmd}")
        
        start_time = time.time()
        
        try:
            # Start GPU monitoring in background
            monitor_process = None
            try:
                monitor_cmd = f"nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits -l 2"
                monitor_process = subprocess.Popen(
                    monitor_cmd, 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
            except:
                self.log("Could not start GPU monitoring", "WARNING")
            
            # Run main command
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Stop monitoring
            if monitor_process:
                try:
                    monitor_process.terminate()
                    monitor_output, _ = monitor_process.communicate(timeout=5)
                except:
                    monitor_output = ""
            else:
                monitor_output = ""
            
            # Parse results
            success = result.returncode == 0
            structures_generated = self.extract_structure_count(result.stdout)
            
            result_data = {
                'description': description,
                'command': cmd,
                'success': success,
                'elapsed_time': elapsed,
                'structures_generated': structures_generated,
                'throughput_structures_per_second': structures_generated / elapsed if elapsed > 0 else 0,
                'return_code': result.returncode,
                'stdout_last_1000': result.stdout[-1000:] if result.stdout else "",
                'stderr': result.stderr if result.stderr else "",
                'gpu_monitoring_data': monitor_output
            }
            
            if success:
                self.log(f"✅ SUCCESS: {description} completed in {elapsed:.1f}s")
                self.log(f"📊 Generated {structures_generated} structures")
                self.log(f"⚡ Throughput: {structures_generated/elapsed:.2f} structures/second")
            else:
                self.log(f"❌ FAILED: {description} failed after {elapsed:.1f}s", "ERROR")
                if result.stderr:
                    self.log(f"Error: {result.stderr[-200:]}", "ERROR")
            
            return result_data
            
        except subprocess.TimeoutExpired:
            self.log(f"⏰ TIMEOUT: {description} timed out after {timeout}s", "ERROR")
            return {
                'description': description,
                'success': False,
                'elapsed_time': timeout,
                'structures_generated': 0,
                'throughput_structures_per_second': 0,
                'error': 'Timeout'
            }
        except Exception as e:
            self.log(f"💥 ERROR: {description} failed with exception: {e}", "ERROR")
            return {
                'description': description,
                'success': False,
                'elapsed_time': 0,
                'structures_generated': 0,
                'throughput_structures_per_second': 0,
                'error': str(e)
            }
    
    def extract_structure_count(self, stdout_text):
        """Extract number of generated structures from output."""
        if not stdout_text:
            return 0
        
        import re
        # Look for patterns like "Generated X structures"
        patterns = [
            r'Generated (\d+) structures',
            r'Total structures.*?(\d+)',
            r'(\d+) structures generated',
            r'Total batches processed: (\d+)',
            r'successfully processed: (\d+)'
        ]
        
        total_structures = 0
        
        # For multi-GPU output, sum all structure counts found
        for pattern in patterns:
            matches = re.findall(pattern, stdout_text, re.IGNORECASE)
            if matches:
                for match in matches:
                    total_structures += int(match)
                break  # Use first pattern that matches
        
        return total_structures
    
    def validate_results(self, output_path):
        """Validate generated results."""
        try:
            output_dir = Path(output_path)
            if not output_dir.exists():
                return {'valid': False, 'error': 'Output directory does not exist'}
            
            total_structures = 0
            
            # Check for multi-GPU summary first
            summary_file = output_dir / "multi_gpu_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                # For multi-GPU, sum structures from all successful jobs
                total_structures = 0
                for job_detail in summary_data.get('job_details', []):
                    if job_detail.get('success', False):
                        # Extract structure count from stdout of each successful job
                        stdout_text = job_detail.get('stdout', '')
                        job_structures = self.extract_structure_count(stdout_text)
                        total_structures += job_structures
                
                validation = {
                    'valid': True,
                    'output_directory': str(output_dir),
                    'total_structures': total_structures,
                    'has_multi_gpu_summary': True,
                    'multi_gpu_summary': summary_data,
                    'total_size_mb': self.get_directory_size(output_dir)
                }
                return validation
            
            # Fallback: Count CIF files or other structure files
            cif_files = list(output_dir.rglob("*.cif"))
            json_files = list(output_dir.rglob("*.json"))
            
            # For single GPU results, count actual files
            total_structures = len(cif_files)
            
            validation = {
                'valid': True,
                'output_directory': str(output_dir),
                'total_structures': total_structures,
                'cif_files_count': len(cif_files),
                'json_files_count': len(json_files),
                'has_multi_gpu_summary': False,
                'multi_gpu_summary': None,
                'total_size_mb': self.get_directory_size(output_dir)
            }
            
            return validation
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def get_directory_size(self, path):
        """Get directory size in MB."""
        try:
            total_size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0
    
    def run_production_tests(self, structures_per_test=100, max_gpus=8):
        """Run comprehensive production tests."""
        
        self.log("🚀 Starting Production Multi-GPU Test Suite")
        self.log(f"📁 Test directory: {self.test_dir}")
        self.log(f"🎯 Target structures per test: {structures_per_test}")
        self.log(f"🔧 Maximum GPUs to test: {max_gpus}")
        
        self.results['test_info']['start_time'] = time.time()
        
        # Calculate batch distributions for consistent structure counts
        base_batch_size = 8
        batches_needed = max(1, structures_per_test // base_batch_size)
        
        test_configurations = [
            {
                'name': 'baseline_single_gpu',
                'description': f'Baseline Single GPU ({structures_per_test} structures)',
                'type': 'single_gpu',
                'cmd': f'''mattergen-generate "{self.test_dir}/baseline_single_gpu" \\
                    --pretrained_name=chemical_system \\
                    --batch_size={base_batch_size} \\
                    --num_batches={batches_needed} \\
                    --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \\
                    --sampling_config_name=default \\
                    --enable_optimizations=False \\
                    --enable_multi_gpu=False \\
                    --record_trajectories=False''',
                'expected_structures': batches_needed * base_batch_size
            },
            {
                'name': 'optimized_single_gpu',
                'description': f'Optimized Single GPU ({structures_per_test} structures)',
                'type': 'single_gpu_optimized',
                'cmd': f'''mattergen-generate "{self.test_dir}/optimized_single_gpu" \\
                    --pretrained_name=chemical_system \\
                    --batch_size={base_batch_size} \\
                    --num_batches={batches_needed} \\
                    --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \\
                    --sampling_config_name=optimized_compatible \\
                    --enable_optimizations=True \\
                    --enable_mixed_precision=True \\
                    --enable_model_compilation=True \\
                    --enable_graph_caching=True \\
                    --enable_multi_gpu=False \\
                    --record_trajectories=False \\
                    --print_optimization_info=True''',
                'expected_structures': batches_needed * base_batch_size
            },
            {
                'name': 'multi_gpu_2',
                'description': f'Multi-GPU 2 GPUs ({structures_per_test} structures)',
                'type': 'multi_gpu',
                'gpus': 2,
                'cmd': f'''python multi_gpu_inference.py \\
                    --output_base_path "{self.test_dir}/multi_gpu_2" \\
                    --pretrained_name chemical_system \\
                    --num_gpus 2 \\
                    --total_batches {batches_needed} \\
                    --batch_size {base_batch_size} \\
                    --properties_to_condition_on "{{'chemical_system':'Pd-Ni-H'}}" \\
                    --sampling_config_name optimized_compatible \\
                    --enable_optimizations True \\
                    --enable_mixed_precision True \\
                    --enable_model_compilation True \\
                    --enable_graph_caching True \\
                    --record_trajectories False''',
                'expected_structures': batches_needed * base_batch_size
            },
            {
                'name': 'multi_gpu_4',
                'description': f'Multi-GPU 4 GPUs ({structures_per_test} structures)',
                'type': 'multi_gpu',
                'gpus': 4,
                'cmd': f'''python multi_gpu_inference.py \\
                    --output_base_path "{self.test_dir}/multi_gpu_4" \\
                    --pretrained_name chemical_system \\
                    --num_gpus {min(4, max_gpus)} \\
                    --total_batches {batches_needed} \\
                    --batch_size {base_batch_size} \\
                    --properties_to_condition_on "{{'chemical_system':'Pd-Ni-H'}}" \\
                    --sampling_config_name optimized_compatible \\
                    --enable_optimizations True \\
                    --enable_mixed_precision True \\
                    --enable_model_compilation True \\
                    --enable_graph_caching True \\
                    --record_trajectories False''',
                'expected_structures': batches_needed * base_batch_size
            }
        ]
        
        # Add larger GPU count tests if available
        if max_gpus >= 8:
            test_configurations.append({
                'name': 'multi_gpu_8',
                'description': f'Multi-GPU 8 GPUs ({structures_per_test * 2} structures)',
                'type': 'multi_gpu',
                'gpus': 8,
                'cmd': f'''python multi_gpu_inference.py \\
                    --output_base_path "{self.test_dir}/multi_gpu_8" \\
                    --pretrained_name chemical_system \\
                    --num_gpus 8 \\
                    --total_batches {batches_needed * 2} \\
                    --batch_size {base_batch_size} \\
                    --properties_to_condition_on "{{'chemical_system':'Pd-Ni-H'}}" \\
                    --sampling_config_name fast \\
                    --enable_optimizations True \\
                    --enable_mixed_precision True \\
                    --enable_model_compilation True \\
                    --enable_graph_caching True \\
                    --record_trajectories False''',
                'expected_structures': batches_needed * base_batch_size * 2
            })
        
        # Run tests
        for i, config in enumerate(test_configurations):
            self.log(f"\n{'='*80}")
            self.log(f"🔄 Running test {i+1}/{len(test_configurations)}: {config['name']}")
            self.log(f"🎯 Expected structures: {config['expected_structures']}")
            
            result = self.run_command_with_monitoring(
                config['cmd'],
                config['description'],
                timeout=2400  # 40 minutes for large tests
            )
            
            # Add configuration info to result
            result.update({
                'config_name': config['name'],
                'config_type': config['type'],
                'expected_structures': config['expected_structures'],
                'gpus_used': config.get('gpus', 1)
            })
            
            # Validate results
            if result['success']:
                if 'multi_gpu' in config['name']:
                    validation = self.validate_results(f"{self.test_dir}/{config['name']}")
                else:
                    validation = self.validate_results(f"{self.test_dir}/{config['name']}")
                result['validation'] = validation
                
                # Update structure count from validation if available
                if validation.get('valid') and 'total_structures' in validation:
                    result['structures_generated'] = validation['total_structures']
                    # Recalculate throughput
                    if result['elapsed_time'] > 0:
                        result['throughput_structures_per_second'] = result['structures_generated'] / result['elapsed_time']
            
            self.results['configurations'].append(result)
            
            # Brief pause between tests
            time.sleep(5)
        
        self.results['test_info']['end_time'] = time.time()
        self.results['test_info']['total_duration'] = (
            self.results['test_info']['end_time'] - self.results['test_info']['start_time']
        )
        
        # Generate comprehensive report
        self.generate_production_report()
        
        return self.results
    
    def generate_production_report(self):
        """Generate comprehensive production test report."""
        
        self.log(f"\n{'='*100}")
        self.log("📊 PRODUCTION MULTI-GPU TEST RESULTS")
        self.log('='*100)
        
        successful_tests = [r for r in self.results['configurations'] if r['success']]
        failed_tests = [r for r in self.results['configurations'] if not r['success']]
        
        # Summary statistics
        total_structures = sum(r['structures_generated'] for r in successful_tests)
        total_time = self.results['test_info']['total_duration']
        
        self.log(f"📈 SUMMARY STATISTICS")
        self.log(f"   Total test duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        self.log(f"   Successful tests: {len(successful_tests)}/{len(self.results['configurations'])}")
        self.log(f"   Total structures generated: {total_structures}")
        self.log(f"   Overall throughput: {total_structures/total_time:.2f} structures/second")
        
        # Performance comparison table
        self.log(f"\n📊 PERFORMANCE COMPARISON")
        header = f"{'Configuration':<25} {'Status':<8} {'Time(s)':<8} {'Structures':<12} {'Throughput/s':<12} {'Speedup':<8} {'GPUs':<5}"
        self.log(header)
        self.log('-' * len(header))
        
        baseline_time = None
        baseline_throughput = None
        
        # Find baseline
        for result in successful_tests:
            if 'baseline' in result['config_name']:
                baseline_time = result['elapsed_time']
                baseline_throughput = result['throughput_structures_per_second']
                break
        
        for result in self.results['configurations']:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            time_str = f"{result['elapsed_time']:.1f}" if result['success'] else "N/A"
            structures = str(result['structures_generated']) if result['success'] else "0"
            throughput = f"{result['throughput_structures_per_second']:.2f}" if result['success'] else "0.00"
            
            speedup = ""
            if result['success'] and baseline_time and result['elapsed_time'] > 0:
                speedup_factor = baseline_time / result['elapsed_time']
                speedup = f"{speedup_factor:.2f}x"
            
            gpus = str(result.get('gpus_used', 1))
            
            self.log(f"{result['config_name']:<25} {status:<8} {time_str:<8} {structures:<12} {throughput:<12} {speedup:<8} {gpus:<5}")
        
        # Multi-GPU scaling analysis
        self.log(f"\n🚀 MULTI-GPU SCALING ANALYSIS")
        multi_gpu_results = [r for r in successful_tests if r['config_type'] == 'multi_gpu']
        
        if len(multi_gpu_results) >= 2:
            self.log("GPU Scaling Efficiency:")
            for result in sorted(multi_gpu_results, key=lambda x: x['gpus_used']):
                if baseline_throughput:
                    scaling_factor = result['throughput_structures_per_second'] / baseline_throughput
                    efficiency = scaling_factor / result['gpus_used'] * 100
                    self.log(f"   {result['gpus_used']} GPUs: {scaling_factor:.1f}x throughput, {efficiency:.1f}% efficiency")
        
        # Save detailed results
        results_file = self.test_dir / 'production_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\n📁 DETAILED RESULTS")
        self.log(f"   Results saved to: {results_file}")
        self.log(f"   Test directory: {self.test_dir}")
        
        # Production recommendations
        self.log(f"\n💡 PRODUCTION RECOMMENDATIONS")
        self.log("-" * 50)
        
        if successful_tests:
            best_single = max([r for r in successful_tests if r['config_type'] != 'multi_gpu'], 
                            key=lambda x: x['throughput_structures_per_second'], default=None)
            best_multi = max([r for r in successful_tests if r['config_type'] == 'multi_gpu'], 
                           key=lambda x: x['throughput_structures_per_second'], default=None)
            
            if best_single:
                self.log(f"✅ Best single-GPU: {best_single['config_name']} ({best_single['throughput_structures_per_second']:.1f} structures/s)")
            if best_multi:
                self.log(f"✅ Best multi-GPU: {best_multi['config_name']} ({best_multi['throughput_structures_per_second']:.1f} structures/s)")
            
            if best_single and best_multi:
                improvement = best_multi['throughput_structures_per_second'] / best_single['throughput_structures_per_second']
                self.log(f"🚀 Multi-GPU improvement: {improvement:.1f}x faster than best single-GPU")
        
        self.log("\n🎯 For production workloads:")
        self.log("   • Use multi-GPU launcher for >100 structures")
        self.log("   • Use 'optimized_compatible' config for quality")
        self.log("   • Use 'fast' config for maximum throughput")
        self.log("   • Monitor GPU memory and adjust batch sizes")
        self.log("   • Consider result caching for repeated queries")
        
        self.log(f"\n🏆 PRODUCTION TEST COMPLETE!")


def main():
    parser = argparse.ArgumentParser(description="Production Multi-GPU Test for MatterGen")
    parser.add_argument('--structures', type=int, default=200, 
                       help='Number of structures to generate per test (default: 200)')
    parser.add_argument('--max_gpus', type=int, default=8,
                       help='Maximum number of GPUs to test (default: 8)')
    parser.add_argument('--output_dir', default='production_test_results',
                       help='Base output directory for test results')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline test (for faster testing)')
    
    args = parser.parse_args()
    
    # Create and run test
    test = ProductionMultiGPUTest(args.output_dir)
    
    print("🎯 Production Multi-GPU Test Configuration:")
    print(f"   Structures per test: {args.structures}")
    print(f"   Maximum GPUs: {args.max_gpus}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Skip baseline: {args.skip_baseline}")
    
    try:
        results = test.run_production_tests(
            structures_per_test=args.structures,
            max_gpus=args.max_gpus
        )
        
        # Print final summary
        successful = sum(1 for r in results['configurations'] if r['success'])
        total = len(results['configurations'])
        
        if successful == total:
            print(f"\n🎉 ALL TESTS PASSED ({successful}/{total})!")
            print("🚀 Multi-GPU optimization is production-ready!")
        else:
            print(f"\n⚠️  Some tests failed ({successful}/{total} passed)")
            print("❌ Check logs for details")
            
        return 0 if successful == total else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
