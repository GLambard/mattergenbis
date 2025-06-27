#!/usr/bin/env python3
"""
Monitor the 4-GPU, 256-structure test progress
"""

import time
import subprocess
import json
from pathlib import Path

def monitor_test():
    """Monitor the running test."""
    
    print("📊 Monitoring Multi-GPU Test Progress")
    print("=" * 50)
    
    start_time = time.time()
    last_gpu_check = 0
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Check if test process is still running
        try:
            result = subprocess.run(
                ["pgrep", "-f", "multi_gpu_256_test.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"\n✅ Test process completed after {elapsed/60:.1f} minutes")
                break
                
        except:
            pass
        
        # Check GPU utilization every 30 seconds
        if current_time - last_gpu_check > 30:
            try:
                gpu_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True
                )
                
                if gpu_result.returncode == 0:
                    print(f"\n🔧 GPU Status at {elapsed/60:.1f}min:")
                    lines = gpu_result.stdout.strip().split('\n')
                    for line in lines[:4]:  # Show first 4 GPUs
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_id, util, mem_used, mem_total = parts
                            print(f"   GPU {gpu_id}: {util}% util, {mem_used}/{mem_total}MB memory")
            except:
                pass
            
            last_gpu_check = current_time
        
        # Check for partial results
        test_dirs = list(Path("multi_gpu_256_test").glob("test_*"))
        if test_dirs:
            latest_dir = max(test_dirs, key=lambda x: x.stat().st_mtime)
            multi_gpu_dir = latest_dir / "multi_gpu_4"
            
            if multi_gpu_dir.exists():
                summary_file = multi_gpu_dir / "multi_gpu_summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                        
                        completed_jobs = summary.get('successful_jobs', 0)
                        total_jobs = summary.get('total_jobs', 4)
                        
                        print(f"📈 Progress: {completed_jobs}/{total_jobs} GPU jobs completed")
                        
                    except:
                        pass
        
        time.sleep(10)  # Check every 10 seconds
    
    # Final check for results
    print("\n🔍 Checking final results...")
    test_dirs = list(Path("multi_gpu_256_test").glob("test_*"))
    if test_dirs:
        latest_dir = max(test_dirs, key=lambda x: x.stat().st_mtime)
        results_file = latest_dir / "test_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            if results.get('success'):
                print(f"🎉 SUCCESS!")
                print(f"   Structures: {results.get('structures_generated', 0)}")
                print(f"   Duration: {results.get('duration', 0):.1f}s")
                print(f"   Throughput: {results.get('throughput', 0):.3f} structures/second")
            else:
                print(f"❌ Test failed: {results.get('error', 'Unknown error')}")
        else:
            print("📄 No results file found")

if __name__ == "__main__":
    try:
        monitor_test()
    except KeyboardInterrupt:
        print("\n👋 Monitoring interrupted by user")
