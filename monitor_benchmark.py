#!/usr/bin/env python3
"""
Monitor High-Volume Benchmark Progress
=====================================

Monitors the progress of the high-volume benchmark execution.
"""

import json
import time
from pathlib import Path
from datetime import datetime

def check_benchmark_progress():
    """Check current benchmark progress."""
    
    results_dir = Path("benchmark_results_high_volume")
    if not results_dir.exists():
        print("❌ No benchmark results directory found")
        return
    
    print("🔍 BENCHMARK PROGRESS REPORT")
    print("=" * 50)
    print(f"📅 Report time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    phases = ['baseline', 'phase1', 'phase2', 'phase3', 'phase4_3']
    phase_names = {
        'baseline': 'Baseline (No Optimizations)',
        'phase1': 'Phase 1: Core Optimizations', 
        'phase2': 'Phase 2: Performance Optimization',
        'phase3': 'Phase 3: Multi-GPU Scaling',
        'phase4_3': 'Phase 4.3: Enterprise Features'
    }
    
    completed_phases = 0
    total_structures = 0
    
    for phase in phases:
        phase_dir = results_dir / f"phase_{phase}"
        summary_file = phase_dir / "multi_gpu_summary.json"
        
        print(f"\n📋 {phase_names[phase]}:")
        
        if phase_dir.exists():
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                    
                    structures = summary.get('total_structures_generated', 0)
                    total_structures += structures
                    completed_phases += 1
                    
                    print(f"   ✅ COMPLETED")
                    print(f"   • Structures Generated: {structures}")
                    print(f"   • Success Rate: {summary.get('successful_jobs', 0) / max(summary.get('total_jobs', 1), 1):.2%}")
                    print(f"   • Duration: {summary.get('total_time_seconds', 0)/60:.1f} minutes")
                    print(f"   • Throughput: {summary.get('throughput_structures_per_second', 0):.2f} structures/sec")
                    
                except Exception as e:
                    print(f"   ⚠️ Summary file exists but cannot be read: {e}")
            else:
                # Check if phase is in progress
                structure_files = list(phase_dir.glob("*.xyz"))
                if structure_files:
                    print(f"   🔄 IN PROGRESS")
                    print(f"   • Structure files found: {len(structure_files)}")
                else:
                    print(f"   📁 Directory exists but no files yet")
        else:
            print(f"   ⏳ NOT STARTED")
    
    print(f"\n📊 OVERALL PROGRESS:")
    print(f"   • Completed Phases: {completed_phases}/5 ({completed_phases/5:.1%})")
    print(f"   • Total Structures Generated: {total_structures}")
    print(f"   • Target Structures: 1280")
    
    if completed_phases > 0:
        avg_time_per_phase = 0  # We'd need to track this from logs
        estimated_remaining = (5 - completed_phases) * 60  # Assume 60 min per phase
        print(f"   • Estimated Remaining Time: ~{estimated_remaining} minutes")
    
    # Check for comprehensive summary
    comp_summary = results_dir / "comprehensive_summary.json"
    if comp_summary.exists():
        print(f"\n🎉 BENCHMARK COMPLETE!")
        print(f"   📋 Final summary available: {comp_summary}")
    
    print("\n" + "=" * 50)

def main():
    check_benchmark_progress()

if __name__ == "__main__":
    main()
