#!/usr/bin/env python3
"""
MatterGen Phase 3 Enhanced Multi-GPU Launcher
Includes advanced optimizations, intelligent resource management, and comprehensive monitoring
"""

import os
import sys
import subprocess
import time
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

# Add mattergen to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GPUResourceInfo:
    """GPU resource information"""
    gpu_id: int
    memory_total: float
    memory_free: float
    utilization: float
    temperature: Optional[float] = None
    power_usage: Optional[float] = None

@dataclass
class JobConfig:
    """Job configuration for Phase 3 launcher"""
    job_id: int
    gpu_id: int
    num_samples: int
    batch_size: int
    output_path: str
    config_path: str
    model_path: str
    
class Phase3GPUManager:
    """Advanced GPU resource management"""
    
    def __init__(self):
        self.gpu_count = self._get_gpu_count()
        self.resource_history = []
    
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            return len([line for line in result.stdout.split('\n') if 'GPU' in line])
        except:
            return 0
    
    def get_gpu_resources(self) -> List[GPUResourceInfo]:
        """Get current GPU resource information"""
        resources = []
        
        try:
            # Get GPU info using nvidia-ml-py if available, otherwise nvidia-smi
            for gpu_id in range(self.gpu_count):
                try:
                    # Try nvidia-ml-py first
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    resource = GPUResourceInfo(
                        gpu_id=gpu_id,
                        memory_total=memory_info.total / (1024**3),
                        memory_free=memory_info.free / (1024**3),
                        utilization=utilization.gpu
                    )
                    
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        resource.temperature = temp
                    except:
                        pass
                    
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                        resource.power_usage = power
                    except:
                        pass
                    
                    resources.append(resource)
                    
                except ImportError:
                    # Fallback to nvidia-smi parsing
                    resource = self._get_gpu_info_nvidia_smi(gpu_id)
                    if resource:
                        resources.append(resource)
                        
        except Exception as e:
            logger.warning(f"Failed to get GPU resources: {e}")
        
        return resources
    
    def _get_gpu_info_nvidia_smi(self, gpu_id: int) -> Optional[GPUResourceInfo]:
        """Get GPU info using nvidia-smi as fallback"""
        try:
            cmd = [
                'nvidia-smi', 
                '--query-gpu=memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits',
                f'--id={gpu_id}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return GPUResourceInfo(
                    gpu_id=gpu_id,
                    memory_total=float(values[0]) / 1024,  # Convert MB to GB
                    memory_free=float(values[1]) / 1024,
                    utilization=float(values[2]),
                    temperature=float(values[3]) if values[3] != '[Not Supported]' else None,
                    power_usage=float(values[4]) if values[4] != '[Not Supported]' else None
                )
        except Exception as e:
            logger.warning(f"Failed to get GPU {gpu_id} info: {e}")
        
        return None
    
    def select_optimal_gpus(self, num_gpus: int) -> List[int]:
        """Select optimal GPUs based on current resource usage"""
        resources = self.get_gpu_resources()
        
        if not resources:
            return list(range(min(num_gpus, 4)))  # Fallback
        
        # Score GPUs based on available memory and low utilization
        gpu_scores = []
        for resource in resources:
            score = (resource.memory_free / resource.memory_total) * (100 - resource.utilization)
            gpu_scores.append((resource.gpu_id, score))
        
        # Sort by score and select top N
        gpu_scores.sort(key=lambda x: x[1], reverse=True)
        selected_gpus = [gpu_id for gpu_id, _ in gpu_scores[:num_gpus]]
        
        logger.info(f"Selected optimal GPUs: {selected_gpus}")
        return selected_gpus
    
    def monitor_gpu_usage(self, duration: float = 60.0, interval: float = 5.0):
        """Monitor GPU usage over time"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            resources = self.get_gpu_resources()
            timestamp = datetime.now().isoformat()
            
            self.resource_history.append({
                'timestamp': timestamp,
                'resources': [
                    {
                        'gpu_id': r.gpu_id,
                        'memory_usage_percent': ((r.memory_total - r.memory_free) / r.memory_total) * 100,
                        'utilization': r.utilization,
                        'temperature': r.temperature,
                        'power_usage': r.power_usage
                    }
                    for r in resources
                ]
            })
            
            time.sleep(interval)

class Phase3JobLauncher:
    """Phase 3 enhanced job launcher with advanced features"""
    
    def __init__(self, base_output_dir: str = "phase3_multi_gpu_output"):
        self.base_output_dir = Path(base_output_dir)
        self.gpu_manager = Phase3GPUManager()
        self.jobs = []
        self.results = {}
        self.start_time = None
        
    def create_job_configs(self, 
                          total_samples: int, 
                          num_gpus: int,
                          config_path: str,
                          model_path: str = "checkpoints/mattergen_base") -> List[JobConfig]:
        """Create optimized job configurations"""
        
        # Select optimal GPUs
        selected_gpus = self.gpu_manager.select_optimal_gpus(num_gpus)
        
        # Calculate samples per GPU
        samples_per_gpu = total_samples // len(selected_gpus)
        remaining_samples = total_samples % len(selected_gpus)
        
        jobs = []
        for i, gpu_id in enumerate(selected_gpus):
            # Distribute remaining samples to first few GPUs
            num_samples = samples_per_gpu + (1 if i < remaining_samples else 0)
            
            # Intelligent batch sizing based on GPU memory
            resources = self.gpu_manager.get_gpu_resources()
            gpu_resource = next((r for r in resources if r.gpu_id == gpu_id), None)
            
            if gpu_resource and gpu_resource.memory_total >= 24:  # High-memory GPU
                batch_size = 16
            elif gpu_resource and gpu_resource.memory_total >= 12:  # Medium-memory GPU  
                batch_size = 8
            else:  # Lower-memory GPU
                batch_size = 4
            
            output_path = self.base_output_dir / f"gpu_{gpu_id}"
            
            job = JobConfig(
                job_id=i,
                gpu_id=gpu_id,
                num_samples=num_samples,
                batch_size=batch_size,
                output_path=str(output_path),
                config_path=config_path,
                model_path=model_path
            )
            jobs.append(job)
        
        self.jobs = jobs
        logger.info(f"Created {len(jobs)} job configurations for {total_samples} total samples")
        return jobs
    
    def run_job(self, job: JobConfig) -> Dict[str, Any]:
        """Run a single job with Phase 3 optimizations"""
        start_time = time.time()
        
        # Create output directory
        os.makedirs(job.output_path, exist_ok=True)
        
        # Prepare environment with virtual environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(job.gpu_id)
        
        # Construct command with Phase 3 optimizations
        cmd = [
            sys.executable, '-m', 'mattergen.scripts.generate',
            job.output_path,  # First positional argument: output_path
            '--model_path', job.model_path,
            '--sampling_config_path', job.config_path,
            '--num_batches', str((job.num_samples + job.batch_size - 1) // job.batch_size),  # Calculate num_batches
            '--batch_size', str(job.batch_size),
            '--enable_optimizations',  # Enable optimizations
            '--enable_mixed_precision',
            '--enable_model_compilation', 
            '--record_trajectories', 'false'  # Disable trajectories for speed
        ]
        
        logger.info(f"Starting job {job.job_id} on GPU {job.gpu_id}: {' '.join(cmd)}")
        
        try:
            # Run job
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(__file__),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Parse generated structure count from output
            generated_count = self._parse_structure_count(result.stdout)
            
            job_result = {
                'job_id': job.job_id,
                'gpu_id': job.gpu_id,
                'success': success,
                'duration': duration,
                'generated_structures': generated_count,
                'batch_size': job.batch_size,
                'output_path': job.output_path,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            if success:
                logger.info(f"Job {job.job_id} completed successfully in {duration:.2f}s, generated {generated_count} structures")
            else:
                logger.error(f"Job {job.job_id} failed: {result.stderr}")
            
            return job_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Job {job.job_id} timed out")
            return {
                'job_id': job.job_id,
                'gpu_id': job.gpu_id,
                'success': False,
                'duration': time.time() - start_time,
                'error': 'Timeout',
                'generated_structures': 0
            }
        except Exception as e:
            logger.error(f"Job {job.job_id} failed with exception: {e}")
            return {
                'job_id': job.job_id,
                'gpu_id': job.gpu_id,
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'generated_structures': 0
            }
    
    def _parse_structure_count(self, stdout: str) -> int:
        """Parse generated structure count from job output"""
        for line in stdout.split('\n'):
            if 'Generated' in line and 'structures' in line:
                try:
                    # Extract number from "Generated X structures"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Generated' and i + 1 < len(parts):
                            return int(parts[i + 1])
                except:
                    continue
        return 0
    
    def run_all_jobs(self, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Run all jobs with Phase 3 optimizations"""
        if not self.jobs:
            raise ValueError("No jobs configured. Call create_job_configs first.")
        
        self.start_time = time.time()
        max_workers = max_workers or len(self.jobs)
        
        # Start GPU monitoring in background
        monitor_thread = threading.Thread(
            target=self.gpu_manager.monitor_gpu_usage,
            args=(600,),  # Monitor for 10 minutes
            daemon=True
        )
        monitor_thread.start()
        
        # Run jobs in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(self.run_job, job): job for job in self.jobs}
            
            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)
        
        # Calculate summary statistics
        total_duration = time.time() - self.start_time
        successful_jobs = [r for r in results if r['success']]
        failed_jobs = [r for r in results if not r['success']]
        total_structures = sum(r.get('generated_structures', 0) for r in results)
        
        # GPU utilization summary
        gpu_utilization = {}
        for result in results:
            gpu_id = result['gpu_id']
            if gpu_id not in gpu_utilization:
                gpu_utilization[gpu_id] = {
                    'jobs': 0,
                    'total_duration': 0,
                    'successful_jobs': 0,
                    'generated_structures': 0
                }
            
            gpu_utilization[gpu_id]['jobs'] += 1
            gpu_utilization[gpu_id]['total_duration'] += result.get('duration', 0)
            gpu_utilization[gpu_id]['generated_structures'] += result.get('generated_structures', 0)
            if result['success']:
                gpu_utilization[gpu_id]['successful_jobs'] += 1
        
        summary = {
            'phase': 'Phase 3 - Enhanced Multi-GPU',
            'timestamp': datetime.now().isoformat(),
            'total_jobs': len(self.jobs),
            'successful_jobs': len(successful_jobs),
            'failed_jobs': len(failed_jobs),
            'total_duration': total_duration,
            'total_structures_generated': total_structures,
            'structures_per_minute': total_structures / (total_duration / 60) if total_duration > 0 else 0,
            'gpu_utilization': gpu_utilization,
            'job_details': results,
            'resource_monitoring': self.gpu_manager.resource_history[-10:] if self.gpu_manager.resource_history else []
        }
        
        # Save summary
        summary_path = self.base_output_dir / "phase3_summary.json"
        os.makedirs(self.base_output_dir, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Phase 3 multi-GPU job completed: {len(successful_jobs)}/{len(self.jobs)} jobs successful")
        logger.info(f"Total structures generated: {total_structures}")
        logger.info(f"Throughput: {summary['structures_per_minute']:.2f} structures/minute")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Enhanced Multi-GPU MatterGen Launcher")
    parser.add_argument('--total_samples', type=int, default=256, help='Total number of samples to generate')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--config_path', type=str, default='sampling_conf/phase3_optimized.yaml', help='Sampling configuration path')
    parser.add_argument('--model_path', type=str, default='checkpoints/mattergen_base', help='Model path')
    parser.add_argument('--output_dir', type=str, default='phase3_multi_gpu_output', help='Output directory')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum worker threads')
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = Phase3JobLauncher(args.output_dir)
    
    # Create job configurations
    jobs = launcher.create_job_configs(
        total_samples=args.total_samples,
        num_gpus=args.num_gpus,
        config_path=args.config_path,
        model_path=args.model_path
    )
    
    # Run all jobs
    summary = launcher.run_all_jobs(max_workers=args.max_workers)
    
    # Print summary
    print(f"\n🚀 Phase 3 Multi-GPU Generation Complete!")
    print(f"✅ Successful jobs: {summary['successful_jobs']}/{summary['total_jobs']}")
    print(f"📊 Total structures: {summary['total_structures_generated']}")
    print(f"⚡ Throughput: {summary['structures_per_minute']:.2f} structures/minute")
    print(f"⏱️ Total time: {summary['total_duration']:.2f} seconds")

if __name__ == "__main__":
    main()
