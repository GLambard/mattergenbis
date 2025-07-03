#!/usr/bin/env python3
"""
Benchmark Analysis and Query Tool
=================================

Simple tool to query and analyze MatterGen benchmark results.
"""

import sys
from pathlib import Path
from benchmark_tracker import BenchmarkTracker
import json
from datetime import datetime

def print_benchmark_summary(tracker):
    """Print a summary of all benchmarks."""
    print("\n🔍 BENCHMARK DATABASE SUMMARY")
    print("=" * 50)
    
    # Get all benchmarks
    all_benchmarks = tracker.get_benchmarks()
    
    if not all_benchmarks:
        print("No benchmarks found in database.")
        return
    
    # Group by category
    categories = {}
    for benchmark in all_benchmarks:
        cat = benchmark.test_category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(benchmark)
    
    # Print summary by category
    total_benchmarks = len(all_benchmarks)
    successful_benchmarks = sum(1 for b in all_benchmarks if b.success)
    
    print(f"Total Benchmarks: {total_benchmarks}")
    print(f"Successful: {successful_benchmarks}")
    print(f"Failed: {total_benchmarks - successful_benchmarks}")
    print(f"Success Rate: {successful_benchmarks/total_benchmarks:.1%}")
    
    print(f"\nBy Category:")
    for category, benchmarks in categories.items():
        successful = sum(1 for b in benchmarks if b.success)
        print(f"  {category}: {successful}/{len(benchmarks)} passed ({successful/len(benchmarks):.1%})")
    
    # Print recent benchmarks
    print(f"\nRecent Benchmarks:")
    recent = sorted(all_benchmarks, key=lambda x: x.timestamp, reverse=True)[:5]
    for benchmark in recent:
        status = "✅" if benchmark.success else "❌"
        dt = datetime.fromtimestamp(benchmark.timestamp).strftime("%Y-%m-%d %H:%M")
        print(f"  {status} {dt} - {benchmark.test_name} ({benchmark.test_category})")

def compare_precision_benchmarks(tracker):
    """Compare precision benchmark results."""
    print("\n🔬 PRECISION BENCHMARK ANALYSIS")
    print("=" * 50)
    
    precision_benchmarks = tracker.get_benchmarks(category="precision")
    
    if len(precision_benchmarks) < 2:
        print("Need at least 2 precision benchmarks for comparison.")
        return
    
    # Find FP32 and FP16 results
    fp32_result = None
    fp16_result = None
    
    for benchmark in precision_benchmarks:
        if not benchmark.configuration.get("mixed_precision", True):
            fp32_result = benchmark
        else:
            fp16_result = benchmark
    
    if not fp32_result or not fp16_result:
        print("Could not find both FP32 and FP16 benchmark results.")
        return
    
    print("Performance Comparison:")
    print(f"  FP32 Duration: {fp32_result.metrics['duration']:.1f}s")
    print(f"  FP16 Duration: {fp16_result.metrics['duration']:.1f}s")
    
    speedup = fp32_result.metrics['duration'] / fp16_result.metrics['duration']
    print(f"  Speedup Factor: {speedup:.3f}x")
    
    if speedup > 1.02:
        print(f"  Result: ⚡ FP16 is {(speedup-1)*100:.1f}% faster")
    elif speedup < 0.98:
        print(f"  Result: 🐌 FP16 is {(1-speedup)*100:.1f}% slower")
    else:
        print(f"  Result: ⚖️ Performance is equivalent")
    
    print(f"\nQuality Comparison:")
    print(f"  FP32 Structures: {fp32_result.metrics['structures_generated']}")
    print(f"  FP16 Structures: {fp16_result.metrics['structures_generated']}")
    print(f"  FP32 Validity: {fp32_result.metrics['validity_rate']:.1%}")
    print(f"  FP16 Validity: {fp16_result.metrics['validity_rate']:.1%}")
    
    quality_diff = fp16_result.metrics['validity_rate'] - fp32_result.metrics['validity_rate']
    if abs(quality_diff) < 0.01:
        print(f"  Result: ✅ Quality is equivalent")
    else:
        print(f"  Result: ⚠️ Quality difference: {quality_diff:.1%}")

def analyze_trends(tracker):
    """Analyze performance trends."""
    print("\n📈 TREND ANALYSIS")
    print("=" * 50)
    
    categories = ["precision", "multi_gpu", "optimization"]
    metrics = ["duration", "structures_generated", "validity_rate"]
    
    for category in categories:
        benchmarks = tracker.get_benchmarks(category=category)
        if not benchmarks:
            continue
            
        print(f"\n{category.upper()} Category:")
        for metric in metrics:
            trend = tracker.analyze_performance_trends(category, metric)
            if "error" in trend:
                continue
                
            direction = trend["trend"]["direction"]
            if direction == "increasing":
                emoji = "📈"
            elif direction == "decreasing":
                emoji = "📉"
            else:
                emoji = "➡️"
                
            print(f"  {emoji} {metric}: {direction} ({trend['data_points']} points)")

def check_regressions(tracker):
    """Check for performance regressions."""
    print("\n🚨 REGRESSION CHECK")
    print("=" * 50)
    
    categories = ["precision", "multi_gpu", "optimization"]
    metrics = ["duration", "validity_rate"]
    
    total_regressions = 0
    
    for category in categories:
        for metric in metrics:
            regressions = tracker.detect_regressions(category, metric, threshold_percent=5.0)
            if regressions:
                total_regressions += len(regressions)
                print(f"\n⚠️ {category}.{metric}:")
                for regression in regressions:
                    print(f"  Change: {regression['change_percent']:.1f}%")
                    print(f"  From: {regression['previous_value']:.2f} to {regression['current_value']:.2f}")
    
    if total_regressions == 0:
        print("✅ No performance regressions detected!")
    else:
        print(f"\n🚨 Found {total_regressions} performance regressions!")

def export_summary(tracker):
    """Export a summary report."""
    print("\n📁 EXPORTING SUMMARY")
    print("=" * 50)
    
    # Generate report
    report = tracker.generate_report("results", since_days=30)
    
    # Save to file
    summary_file = Path("results/benchmark_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Summary report saved to: {summary_file}")
    
    # Export CSV
    csv_file = tracker.export_to_csv("results")
    print(f"CSV data exported to: {csv_file}")

def main():
    """Main analysis function."""
    print("🧬 MatterGen Benchmark Analysis Tool")
    print("=" * 60)
    
    # Initialize tracker
    tracker = BenchmarkTracker("mattergen_benchmarks.db")
    
    # Run all analyses
    print_benchmark_summary(tracker)
    compare_precision_benchmarks(tracker)
    analyze_trends(tracker)
    check_regressions(tracker)
    export_summary(tracker)
    
    print(f"\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
