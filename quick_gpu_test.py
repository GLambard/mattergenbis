#!/usr/bin/env python3
"""
Quick Multi-GPU Test - validates basic functionality
"""

import subprocess
import time
import os

def run_quick_test(description, cmd, timeout=180):
    """Run a quick test with timeout."""
    print(f"\n🧪 Testing: {description}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, timeout=timeout, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        print(f"✅ Exit code: {result.returncode}")
        print(f"⏱️  Time: {elapsed:.1f}s")
        
        if result.returncode == 0:
            print("✅ SUCCESS")
        else:
            print("❌ FAILED")
            if result.stderr:
                print(f"Error: {result.stderr[-500:]}")
        
        return result.returncode == 0, elapsed
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after {timeout}s")
        return False, timeout
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, 0

def main():
    print("🚀 MatterGen Multi-GPU Quick Test")
    print("=" * 60)
    
    # Create test output dir
    test_dir = "quick_gpu_test"
    os.makedirs(test_dir, exist_ok=True)
    
    tests = [
        {
            'name': 'Single GPU Test',
            'cmd': f'''mattergen-generate "{test_dir}/single" \
                --pretrained-name=chemical_system \
                --batch_size=2 \
                --num_batches=1 \
                --properties_to_condition_on="{{'chemical_system':'Pd-Ni-H'}}" \
                --record_trajectories=False \
                --sampling_config_name=optimized_compatible \
                --enable_multi_gpu=False \
                --max_gpus=1'''
        },
        {
            'name': 'Multi-Process 2 GPU Test',
            'cmd': f'''python multi_gpu_inference.py \
                --output_base_path "{test_dir}/multi_2gpu" \
                --pretrained_name chemical_system \
                --num_gpus 2 \
                --total_batches 2 \
                --batch_size 2 \
                --properties_to_condition_on "{{'chemical_system':'Pd-Ni-H'}}" \
                --sampling_config_name optimized_compatible'''
        }
    ]
    
    results = []
    for test in tests:
        success, elapsed = run_quick_test(test['name'], test['cmd'])
        results.append((test['name'], success, elapsed))
    
    print(f"\n📊 QUICK TEST SUMMARY")
    print("=" * 60)
    
    for name, success, elapsed in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:25} {status:10} {elapsed:6.1f}s")
    
    # Calculate speedup if both tests passed
    if len(results) == 2 and all(r[1] for r in results):
        single_time = results[0][2]
        multi_time = results[1][2]
        speedup = single_time / multi_time if multi_time > 0 else 0
        print(f"\n🏃‍♂️ Multi-GPU Speedup: {speedup:.2f}x")
    
    print(f"\n🔍 Check results in: {test_dir}/")

if __name__ == "__main__":
    main()
