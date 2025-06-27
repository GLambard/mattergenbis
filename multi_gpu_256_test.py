#!/usr/bin/env python3
"""
Multi-GPU Only Production Test - 256 structures on 4 GPUs
========================================================

This script tests multi-GPU inference performance by generating 256 structures
across 4 GPUs using the optimized configuration.
"""

import os
import time
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

class MultiGPUOnlyTest:
    def __init__(self, output_dir="multi_gpu_256_test"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_dir = self.output_dir / f"test_{self.timestamp}"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def extract_structure_count(self, stdout_text):
        """Extract structure count from stdout text."""
        if not stdout_text:
            return 0
        
        import re
        patterns = [
            r'Generated (\d+) structures',
            r'Total structures.*?(\d+)',
            r'(\d+) structures generated',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stdout_text, re.IGNORECASE)
            if matches:
                return sum(int(match) for match in matches)
        
        return 0
    
    def run_multi_gpu_test(self, num_structures=256, num_gpus=4):
        """Run multi-GPU test with specified parameters."""
        
        self.log(f"🚀 Starting Multi-GPU Test: {num_structures} structures on {num_gpus} GPUs")
        self.log(f"📁 Output directory: {self.test_dir}")
        
        # Calculate batch distribution
        base_batch_size = 8
        total_batches = max(1, num_structures // base_batch_size)
        
        self.log(f"🔧 Configuration:")
        self.log(f"   Target structures: {num_structures}")
        self.log(f"   Batch size: {base_batch_size}")
        self.log(f"   Total batches: {total_batches}")
        self.log(f"   GPUs: {num_gpus}")
        
        # Multi-GPU test configuration
        output_path = self.test_dir / f"multi_gpu_{num_gpus}"
        
        cmd = f'''bash -c "source .venv/bin/activate && python multi_gpu_inference.py \\
            --output_base_path '{output_path}' \\
            --pretrained_name chemical_system \\
            --num_gpus {num_gpus} \\
            --total_batches {total_batches} \\
            --batch_size {base_batch_size} \\
            --properties_to_condition_on \\"{{'chemical_system':'Pd-Ni-H'}}\\" \\
            --sampling_config_name optimized_compatible \\
            --enable_optimizations True \\
            --enable_mixed_precision True \\
            --enable_model_compilation True \\
            --enable_graph_caching True \\
            --record_trajectories False"'''
        
        self.log(f"🔄 Running Multi-GPU Command:")
        self.log(f"Command: {cmd}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            if result.returncode == 0:
                self.log(f"✅ Multi-GPU test completed successfully in {elapsed:.1f}s")
                
                # Validate results
                validation = self.validate_multi_gpu_results(output_path, num_structures)
                
                if validation['success']:
                    structures_generated = validation['total_structures']
                    throughput = structures_generated / elapsed if elapsed > 0 else 0
                    
                    self.log(f"📊 RESULTS:")
                    self.log(f"   Structures generated: {structures_generated}")
                    self.log(f"   Target structures: {num_structures}")
                    self.log(f"   Success rate: {structures_generated/num_structures*100:.1f}%")
                    self.log(f"   Duration: {elapsed:.1f}s")
                    self.log(f"   Throughput: {throughput:.3f} structures/second")
                    self.log(f"   Per-GPU throughput: {throughput/num_gpus:.3f} structures/second/GPU")
                    
                    return {
                        'success': True,
                        'structures_generated': structures_generated,
                        'target_structures': num_structures,
                        'duration': elapsed,
                        'throughput': throughput,
                        'num_gpus': num_gpus,
                        'validation': validation
                    }
                else:
                    self.log(f"❌ Validation failed: {validation.get('error', 'Unknown error')}")
                    return {'success': False, 'error': 'Validation failed'}
            else:
                self.log(f"❌ Multi-GPU test failed with return code {result.returncode}")
                self.log(f"Error output: {result.stderr}")
                return {'success': False, 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            self.log(f"❌ Multi-GPU test timed out after 1 hour")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            self.log(f"❌ Multi-GPU test failed with exception: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_multi_gpu_results(self, output_path, expected_structures):
        """Validate multi-GPU results."""
        try:
            output_dir = Path(output_path)
            if not output_dir.exists():
                return {'success': False, 'error': 'Output directory does not exist'}
            
            # Check for multi-GPU summary
            summary_file = output_dir / "multi_gpu_summary.json"
            if not summary_file.exists():
                return {'success': False, 'error': 'Multi-GPU summary file not found'}
            
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            # Extract structure counts from job outputs (using our fixed logic)
            total_structures = 0
            successful_jobs = 0
            failed_jobs = 0
            
            for job_detail in summary_data.get('job_details', []):
                if job_detail.get('success', False):
                    successful_jobs += 1
                    stdout_text = job_detail.get('stdout', '')
                    job_structures = self.extract_structure_count(stdout_text)
                    total_structures += job_structures
                else:
                    failed_jobs += 1
            
            return {
                'success': True,
                'total_structures': total_structures,
                'successful_jobs': successful_jobs,
                'failed_jobs': failed_jobs,
                'summary_data': summary_data,
                'output_directory': str(output_dir)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def display_final_results(self, result):
        """Display comprehensive final results."""
        
        self.log("🎯 FINAL TEST RESULTS")
        self.log("=" * 50)
        
        if result['success']:
            structures = result['structures_generated']
            target = result['target_structures']
            duration = result['duration']
            throughput = result['throughput']
            num_gpus = result['num_gpus']
            
            self.log(f"✅ SUCCESS: Multi-GPU test completed")
            self.log(f"📊 Performance Metrics:")
            self.log(f"   Structures: {structures}/{target} ({structures/target*100:.1f}%)")
            self.log(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            self.log(f"   Total throughput: {throughput:.3f} structures/second")
            self.log(f"   Per-GPU throughput: {throughput/num_gpus:.3f} structures/second/GPU")
            
            # GPU utilization breakdown
            validation = result.get('validation', {})
            if 'summary_data' in validation:
                summary = validation['summary_data']
                self.log(f"🔧 GPU Utilization:")
                for gpu_id, stats in summary.get('gpu_utilization', {}).items():
                    gpu_structures = 0
                    # Find structures for this GPU
                    for job in summary.get('job_details', []):
                        if job.get('gpu_id') == int(gpu_id) and job.get('success'):
                            gpu_structures += self.extract_structure_count(job.get('stdout', ''))
                    
                    self.log(f"   GPU {gpu_id}: {gpu_structures} structures, {stats['total_duration']:.1f}s")
            
            # Efficiency calculation
            theoretical_max = throughput * num_gpus  # If perfect scaling
            single_gpu_baseline = 0.08  # From previous tests (structures/second)
            expected_multi_gpu = single_gpu_baseline * num_gpus
            efficiency = (throughput / expected_multi_gpu) * 100 if expected_multi_gpu > 0 else 0
            
            self.log(f"📈 Scaling Analysis:")
            self.log(f"   Single-GPU baseline: ~{single_gpu_baseline:.3f} structures/second")
            self.log(f"   Expected {num_gpus}-GPU: {expected_multi_gpu:.3f} structures/second")
            self.log(f"   Actual {num_gpus}-GPU: {throughput:.3f} structures/second")
            self.log(f"   Scaling efficiency: {efficiency:.1f}%")
            
        else:
            self.log(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        
        self.log("=" * 50)

def main():
    """Main function to run the multi-GPU only test."""
    
    print("🚀 Multi-GPU Only Production Test")
    print("🎯 Target: 256 structures on 4 GPUs")
    print("=" * 50)
    
    # Create test instance
    test = MultiGPUOnlyTest()
    
    # Run the test
    result = test.run_multi_gpu_test(num_structures=256, num_gpus=4)
    
    # Display results
    test.display_final_results(result)
    
    # Save results for future reference
    results_file = test.test_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    test.log(f"💾 Results saved to: {results_file}")
    
    return result['success'] if result else False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
