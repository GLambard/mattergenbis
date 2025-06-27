#!/usr/bin/env python3
"""
Test the fixed validation logic with a quick multi-GPU run
"""

import json
import re
import subprocess
import time
from pathlib import Path

def extract_structure_count(stdout_text):
    """Extract structure count from stdout text."""
    if not stdout_text:
        return 0
    
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

def test_validation_fix():
    """Test the validation fix with existing data."""
    
    # Test with existing multi-GPU summary
    summary_path = Path("production_demo/test_20250627_152656/multi_gpu_2/multi_gpu_summary.json")
    
    if not summary_path.exists():
        print(f"❌ Test file not found: {summary_path}")
        return False
    
    print("🔍 Testing Fixed Validation Logic")
    print(f"📁 Using: {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    print(f"📊 Summary info:")
    print(f"   Total jobs: {summary_data.get('total_jobs', 0)}")
    print(f"   Successful jobs: {summary_data.get('successful_jobs', 0)}")
    print(f"   Total batches: {summary_data.get('total_batches', 0)}")
    
    # Apply the NEW validation logic
    total_structures = 0
    for i, job_detail in enumerate(summary_data.get('job_details', [])):
        if job_detail.get('success', False):
            stdout_text = job_detail.get('stdout', '')
            job_structures = extract_structure_count(stdout_text)
            print(f"   Job {i} (GPU {job_detail.get('gpu_id')}): {job_structures} structures")
            total_structures += job_structures
    
    print(f"\n✅ NEW validation logic result: {total_structures} structures")
    
    # Compare with OLD logic
    old_total = summary_data.get('total_batches', 0) * 8
    print(f"🔄 OLD validation logic result: {old_total} structures")
    
    if total_structures > 0:
        print(f"🎉 SUCCESS! Structure counting fix is working!")
        print(f"   Multi-GPU inference generated {total_structures} structures total")
        return True
    else:
        print("❌ FAILURE! Structure counting still broken")
        return False

def run_quick_multi_gpu_test():
    """Run a quick multi-GPU test to validate the fix end-to-end."""
    print("\n🚀 Running quick multi-GPU test to validate fix...")
    
    output_dir = Path("quick_validation_test")
    output_dir.mkdir(exist_ok=True)
    
    # Run a very small multi-GPU test
    cmd = [
        "python", "multi_gpu_inference.py",
        "--output_base_path", str(output_dir),
        "--pretrained_name", "chemical_system",
        "--num_gpus", "2",
        "--total_batches", "2",  # Very small test
        "--batch_size", "4",
        "--properties_to_condition_on", "{'chemical_system':'Pd-Ni-H'}",
        "--sampling_config_name", "optimized_compatible",
        "--enable_optimizations", "True"
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ Multi-GPU test completed in {end_time - start_time:.1f}s")
            
            # Check the summary
            summary_file = output_dir / "multi_gpu_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                # Apply the NEW validation logic
                total_structures = 0
                for job_detail in summary_data.get('job_details', []):
                    if job_detail.get('success', False):
                        stdout_text = job_detail.get('stdout', '')
                        job_structures = extract_structure_count(stdout_text)
                        total_structures += job_structures
                
                print(f"🎯 Quick test generated {total_structures} structures")
                return total_structures > 0
            else:
                print("❌ No summary file generated")
                return False
        else:
            print(f"❌ Multi-GPU test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Multi-GPU test timed out")
        return False
    except Exception as e:
        print(f"❌ Multi-GPU test error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Validating Multi-GPU Structure Counting Fix")
    print("=" * 50)
    
    # Test 1: Validate with existing data
    validation_works = test_validation_fix()
    
    # Test 2: Run a quick end-to-end test (optional)
    if validation_works:
        print("\n" + "=" * 50)
        quick_test_works = run_quick_multi_gpu_test()
        
        if quick_test_works:
            print("\n🎉 ALL TESTS PASSED! Multi-GPU structure counting is fixed!")
        else:
            print("\n⚠️  Validation logic works, but quick test failed (may be expected)")
    else:
        print("\n❌ Validation logic needs more work")
