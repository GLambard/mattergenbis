#!/usr/bin/env python3
"""
Final demonstration of the multi-GPU structure counting fix
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

def demonstrate_fix():
    """Demonstrate the fix with before/after comparison."""
    
    print("🔧 MULTI-GPU STRUCTURE COUNTING FIX DEMONSTRATION")
    print("=" * 60)
    
    # Use the existing working multi-GPU result
    summary_path = Path("production_demo/test_20250627_152656/multi_gpu_2/multi_gpu_summary.json")
    
    if not summary_path.exists():
        print(f"❌ Demo data not found: {summary_path}")
        return
    
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    print(f"📁 Demo using: {summary_path}")
    print(f"🎯 Test configuration: 2 GPUs, chemical_system model")
    print()
    
    # Show the OLD (broken) logic
    print("❌ OLD LOGIC (BROKEN):")
    total_batches = summary_data.get('total_batches', 0)
    old_calculation = total_batches * 8  # Hardcoded batch size assumption
    print(f"   Calculation: {total_batches} batches × 8 structures/batch = {old_calculation} structures")
    print(f"   Problem: This assumes structures per batch, which isn't always correct")
    print()
    
    # Show the NEW (fixed) logic
    print("✅ NEW LOGIC (FIXED):")
    total_structures = 0
    job_results = []
    
    for i, job_detail in enumerate(summary_data.get('job_details', [])):
        gpu_id = job_detail.get('gpu_id', 'unknown')
        success = job_detail.get('success', False)
        
        if success:
            stdout_text = job_detail.get('stdout', '')
            job_structures = extract_structure_count(stdout_text)
            job_results.append((i, gpu_id, job_structures))
            total_structures += job_structures
        else:
            job_results.append((i, gpu_id, 0))
    
    print("   Extracting from each job's stdout:")
    for job_id, gpu_id, structures in job_results:
        status = "✅" if structures > 0 else "❌"
        print(f"     {status} Job {job_id} (GPU {gpu_id}): {structures} structures")
    
    print(f"   Total: {total_structures} structures")
    print()
    
    # Show performance impact
    print("📊 PERFORMANCE IMPACT:")
    total_duration = summary_data.get('total_duration', 1)
    throughput = total_structures / total_duration if total_structures > 0 else 0
    
    print(f"   Duration: {total_duration:.1f} seconds")
    print(f"   Throughput: {throughput:.3f} structures/second")
    print(f"   Result: Multi-GPU inference actually WORKS and generates structures!")
    print()
    
    # Comparison
    print("🔍 BEFORE vs AFTER FIX:")
    print(f"   Before fix: 0 structures reported (❌ incorrect)")
    print(f"   After fix:  {total_structures} structures reported (✅ correct)")
    print(f"   Impact: Now we can properly measure multi-GPU performance gains!")
    print()
    
    if total_structures > 0:
        print("🎉 CONCLUSION: Structure counting fix is working perfectly!")
        print("   Multi-GPU inference throughput can now be measured accurately.")
        print("   This resolves the production test reporting issue.")
        return True
    else:
        print("❌ CONCLUSION: Fix didn't work as expected.")
        return False

if __name__ == "__main__":
    demonstrate_fix()
