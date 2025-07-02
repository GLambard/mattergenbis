#!/usr/bin/env python3
"""
High-Volume MatterGen Benchmark Execution
========================================

Runs the full benchmark suite with 256 structures per phase for robust statistical validation.
Includes progress monitoring, detailed logging, and result analysis.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from comprehensive_benchmark_suite import ComprehensiveBenchmarkRunner

def _json_serializer(obj):
    """JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return str(obj)

# Configure detailed logging
log_file = f"high_volume_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    print("🚀 Starting High-Volume MatterGen Benchmark Suite")
    print("=" * 60)
    print(f"📝 Detailed logging: {log_file}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Configuration: 256 structures per phase, 1280 total structures")
    print(f"📊 Expected duration: Up to 5 hours")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialize benchmark runner for full mode (not test mode)
        runner = ComprehensiveBenchmarkRunner(
            output_dir="benchmark_results_high_volume",
            test_mode=False,  # Full benchmark mode
            timeout_minutes=60,  # 1 hour timeout per phase
            skip_existing=True  # Skip existing results to resume if needed
        )
        
        logger.info("🔬 Starting comprehensive benchmark execution...")
        
        # Run the full benchmark suite
        results = runner.run_full_benchmark_suite()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("✅ HIGH-VOLUME BENCHMARK COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Total Duration: {total_duration/3600:.2f} hours ({total_duration/60:.1f} minutes)")
        print(f"📊 Total Structures Generated: {sum(r.get('performance', {}).get('structures_generated', 0) for r in results.values())}")
        print(f"📈 Average Success Rate: {sum(r.get('performance', {}).get('success_rate', 0) for r in results.values()) / len(results):.2%}")
        print(f"🎯 Average Accuracy Score: {sum(r.get('accuracy_score', 0) for r in results.values()) / len(results):.3f}")
        
        # Save final summary
        summary = {
            'execution_info': {
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'total_duration_hours': total_duration / 3600,
                'total_structures': sum(r.get('performance', {}).get('structures_generated', 0) for r in results.values()),
                'phases_completed': len(results),
                'log_file': log_file
            },
            'phase_results': results,
            'overall_metrics': {
                'average_success_rate': sum(r.get('performance', {}).get('success_rate', 0) for r in results.values()) / len(results),
                'average_accuracy_score': sum(r.get('accuracy_score', 0) for r in results.values()) / len(results),
                'total_throughput': sum(r.get('performance', {}).get('throughput_structures_per_second', 0) for r in results.values()),
                'statistical_robustness': 'HIGH - 256 structures per phase'
            }
        }
        
        # Convert numpy types to Python native types for JSON serialization
        summary = json.loads(json.dumps(summary, default=_json_serializer))
        
        summary_path = Path("benchmark_results_high_volume") / "comprehensive_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_json_serializer)
        
        print(f"📋 Comprehensive summary saved: {summary_path}")
        print(f"📝 Detailed logs: {log_file}")
        
        # Display key findings
        print("\n🔍 KEY FINDINGS:")
        for phase_name, phase_results in results.items():
            if phase_results.get('success'):
                perf = phase_results.get('performance', {})
                acc_score = phase_results.get('accuracy_score', 0)
                print(f"  {phase_results['name']}:")
                print(f"    • Structures Generated: {perf.get('structures_generated', 0)}")
                print(f"    • Success Rate: {perf.get('success_rate', 0):.2%}")
                print(f"    • Throughput: {perf.get('throughput_structures_per_second', 0):.2f} structures/sec")
                print(f"    • Accuracy Score: {acc_score:.3f}")
            else:
                print(f"  {phase_results['name']}: ❌ FAILED")
        
        print("\n🎉 High-volume validation complete - results ready for analysis!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        logger.warning("Benchmark execution interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        logger.error(f"Benchmark execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
