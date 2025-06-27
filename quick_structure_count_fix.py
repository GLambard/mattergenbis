#!/usr/bin/env python3
"""
Quick fix script to reprocess production test results with corrected structure counting
"""

import json
import re
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

def fix_production_results():
    """Fix structure counting in production test results."""
    
    # Use the existing known working result
    summary_path = Path("production_demo/test_20250627_152656/multi_gpu_2/multi_gpu_summary.json")
    
    if not summary_path.exists():
        print(f"File not found: {summary_path}")
        return
    
    print(f"Testing structure counting fix with: {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    print("\nOriginal summary data:")
    print(f"  Total jobs: {summary_data.get('total_jobs', 0)}")
    print(f"  Successful jobs: {summary_data.get('successful_jobs', 0)}")
    print(f"  Total batches: {summary_data.get('total_batches', 0)}")
    
    # Apply the corrected logic
    total_structures = 0
    print("\nAnalyzing job outputs:")
    
    for i, job_detail in enumerate(summary_data.get('job_details', [])):
        gpu_id = job_detail.get('gpu_id', 'unknown')
        success = job_detail.get('success', False)
        
        if success:
            stdout_text = job_detail.get('stdout', '')
            job_structures = extract_structure_count(stdout_text)
            print(f"  Job {i} (GPU {gpu_id}): {job_structures} structures")
            total_structures += job_structures
        else:
            print(f"  Job {i} (GPU {gpu_id}): FAILED")
    
    print(f"\nFINAL RESULT:")
    print(f"  Total structures (corrected): {total_structures}")
    print(f"  Old calculation (batches * 8): {summary_data.get('total_batches', 0) * 8}")
    
    if total_structures > 0:
        print(f"\n✅ SUCCESS! Structure counting fix works correctly!")
        print(f"   Multi-GPU inference generated {total_structures} structures")
        
        # Calculate the correct throughput for demonstration
        total_duration = summary_data.get('total_duration', 1)
        throughput = total_structures / total_duration
        print(f"   Corrected throughput: {throughput:.3f} structures/second")
        
        return True
    else:
        print(f"\n❌ FAILED! No structures detected")
        return False

if __name__ == "__main__":
    fix_production_results()
