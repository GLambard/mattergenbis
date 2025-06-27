#!/usr/bin/env python3
"""
Quick test to validate the structure counting fix
"""

import json
import re
from pathlib import Path

def extract_structure_count(stdout_text):
    """Extract structure count from stdout text."""
    if not stdout_text:
        return 0
    
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

def test_multi_gpu_validation():
    """Test the multi-GPU validation logic."""
    
    # Test with existing multi-GPU summary
    summary_path = Path("production_demo/test_20250627_152656/multi_gpu_2/multi_gpu_summary.json")
    
    if not summary_path.exists():
        print(f"❌ Test file not found: {summary_path}")
        return
    
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    print("📊 Testing Multi-GPU Structure Counting")
    print(f"Summary file: {summary_path}")
    print(f"Total jobs: {summary_data.get('total_jobs', 0)}")
    print(f"Successful jobs: {summary_data.get('successful_jobs', 0)}")
    print(f"Total batches: {summary_data.get('total_batches', 0)}")
    
    # Test the new validation logic
    total_structures = 0
    for i, job_detail in enumerate(summary_data.get('job_details', [])):
        if job_detail.get('success', False):
            stdout_text = job_detail.get('stdout', '')
            job_structures = extract_structure_count(stdout_text)
            print(f"Job {i} (GPU {job_detail.get('gpu_id')}): {job_structures} structures")
            total_structures += job_structures
    
    print(f"\n✅ Total structures (new logic): {total_structures}")
    
    # Compare with old logic
    old_total = summary_data.get('total_batches', 0) * 8
    print(f"🔍 Total structures (old logic): {old_total}")
    
    if total_structures > 0:
        print(f"✅ Structure counting fix working! Found {total_structures} structures")
    else:
        print("❌ Structure counting still broken")

if __name__ == "__main__":
    test_multi_gpu_validation()
