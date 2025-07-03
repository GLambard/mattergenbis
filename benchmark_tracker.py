#!/usr/bin/env python3
"""
MatterGen Benchmark Results Tracker
==================================

This module provides comprehensive tracking and analysis of MatterGen benchmarks,
including performance comparisons, optimization studies, and regression testing.

Features:
- Centralized benchmark result storage
- Performance trend analysis
- Regression detection
- Comparison across configurations
- Export capabilities for reporting
"""

import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import hashlib
from dataclasses import dataclass, asdict

# Optional dependencies for advanced features
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Structure for storing individual benchmark results."""
    benchmark_id: str
    timestamp: float
    test_name: str
    test_category: str  # "precision", "multi_gpu", "optimization", "regression"
    configuration: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary."""
        return cls(**data)


class BenchmarkTracker:
    """Centralized benchmark tracking and analysis system."""
    
    def __init__(self, database_path: str = "mattergen_benchmarks.db"):
        self.database_path = Path(database_path)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for benchmark storage."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    benchmark_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    test_name TEXT,
                    test_category TEXT,
                    configuration TEXT,
                    metrics TEXT,
                    metadata TEXT,
                    success BOOLEAN,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    comparison_name TEXT,
                    benchmark_ids TEXT,
                    comparison_results TEXT,
                    conclusions TEXT
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmarks(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON benchmarks(test_category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_test_name ON benchmarks(test_name)")
    
    def add_benchmark(self, result: BenchmarkResult) -> str:
        """Add a benchmark result to the database."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO benchmarks 
                (benchmark_id, timestamp, test_name, test_category, configuration, 
                 metrics, metadata, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.benchmark_id,
                result.timestamp,
                result.test_name,
                result.test_category,
                json.dumps(result.configuration),
                json.dumps(result.metrics),
                json.dumps(result.metadata),
                result.success,
                result.error_message
            ))
        
        logger.info(f"Added benchmark result: {result.benchmark_id}")
        return result.benchmark_id
    
    def get_benchmarks(self, 
                      category: Optional[str] = None,
                      test_name: Optional[str] = None,
                      since_timestamp: Optional[float] = None) -> List[BenchmarkResult]:
        """Retrieve benchmark results with filtering."""
        
        query = "SELECT * FROM benchmarks WHERE 1=1"
        params = []
        
        if category:
            query += " AND test_category = ?"
            params.append(category)
        
        if test_name:
            query += " AND test_name = ?"
            params.append(test_name)
        
        if since_timestamp:
            query += " AND timestamp >= ?"
            params.append(since_timestamp)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor.fetchall():
                results.append(BenchmarkResult(
                    benchmark_id=row[0],
                    timestamp=row[1],
                    test_name=row[2],
                    test_category=row[3],
                    configuration=json.loads(row[4]),
                    metrics=json.loads(row[5]),
                    metadata=json.loads(row[6]),
                    success=bool(row[7]),
                    error_message=row[8]
                ))
        
        return results
    
    def add_comparison(self, 
                      comparison_name: str,
                      benchmark_ids: List[str],
                      comparison_results: Dict[str, Any],
                      conclusions: str) -> str:
        """Add a benchmark comparison result."""
        
        comparison_id = f"comp_{int(time.time())}_{hashlib.md5(comparison_name.encode()).hexdigest()[:8]}"
        
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO benchmark_comparisons
                (comparison_id, timestamp, comparison_name, benchmark_ids, 
                 comparison_results, conclusions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                comparison_id,
                time.time(),
                comparison_name,
                json.dumps(benchmark_ids),
                json.dumps(comparison_results),
                conclusions
            ))
        
        logger.info(f"Added comparison: {comparison_id}")
        return comparison_id
    
    def analyze_performance_trends(self, test_category: str, metric_name: str) -> Dict[str, Any]:
        """Analyze performance trends for a specific metric over time."""
        
        benchmarks = self.get_benchmarks(category=test_category)
        
        if not benchmarks:
            return {"error": "No benchmarks found for category"}
        
        # Extract time series data
        timestamps = []
        values = []
        configurations = []
        
        for benchmark in benchmarks:
            if metric_name in benchmark.metrics:
                timestamps.append(benchmark.timestamp)
                values.append(benchmark.metrics[metric_name])
                configurations.append(benchmark.configuration)
        
        if not values:
            return {"error": f"No data found for metric {metric_name}"}
        
        # Calculate trends
        if NUMPY_AVAILABLE:
            values_array = np.array(values)
            analysis = {
                "metric_name": metric_name,
                "test_category": test_category,
                "data_points": len(values),
                "time_range": {
                    "start": min(timestamps),
                    "end": max(timestamps),
                    "duration_days": (max(timestamps) - min(timestamps)) / 86400
                },
                "statistics": {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "latest": values[0] if values else None  # Most recent first
                },
                "trend": self._calculate_trend(timestamps, values),
                "configurations_tested": len(set(str(cfg) for cfg in configurations))
            }
        else:
            # Simple statistics without numpy
            analysis = {
                "metric_name": metric_name,
                "test_category": test_category,
                "data_points": len(values),
                "time_range": {
                    "start": min(timestamps),
                    "end": max(timestamps),
                    "duration_days": (max(timestamps) - min(timestamps)) / 86400
                },
                "statistics": {
                    "mean": sum(values) / len(values),
                    "std": 0,  # Not calculated without numpy
                    "min": min(values),
                    "max": max(values),
                    "latest": values[0] if values else None  # Most recent first
                },
                "trend": self._calculate_trend(timestamps, values),
                "configurations_tested": len(set(str(cfg) for cfg in configurations))
            }
        
        return analysis
    
    def _calculate_trend(self, timestamps: List[float], values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics."""
        if len(values) < 2:
            return {"status": "insufficient_data"}
        
        if not NUMPY_AVAILABLE:
            # Simple trend calculation without numpy
            sorted_data = sorted(zip(timestamps, values))
            sorted_values = [v for _, v in sorted_data]
            
            if len(sorted_values) < 2:
                return {"status": "insufficient_data"}
            
            # Simple slope calculation
            first_val = sorted_values[0]
            last_val = sorted_values[-1]
            change_percent = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
            
            if abs(change_percent) < 1:
                direction = "stable"
            elif change_percent > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
            
            return {
                "direction": direction,
                "slope": (last_val - first_val) / len(sorted_values),
                "change_percent": change_percent,
                "r_squared": 0  # Not calculated without numpy
            }
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values))
        sorted_values = [v for _, v in sorted_data]
        
        # Linear regression for trend
        x = np.arange(len(sorted_values))
        coeffs = np.polyfit(x, sorted_values, 1)
        slope = coeffs[0]
        
        # Determine trend direction
        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {
            "direction": direction,
            "slope": float(slope),
            "change_percent": float((sorted_values[-1] - sorted_values[0]) / sorted_values[0] * 100) if sorted_values[0] != 0 else 0,
            "r_squared": float(np.corrcoef(x, sorted_values)[0, 1]**2) if len(sorted_values) > 1 else 0
        }
    
    def detect_regressions(self, 
                          test_category: str,
                          metric_name: str,
                          threshold_percent: float = 5.0) -> List[Dict[str, Any]]:
        """Detect performance regressions in benchmark results."""
        
        benchmarks = self.get_benchmarks(category=test_category)
        regressions = []
        
        if len(benchmarks) < 2:
            return regressions
        
        # Sort by timestamp (newest first)
        benchmarks.sort(key=lambda x: x.timestamp, reverse=True)
        
        for i in range(len(benchmarks) - 1):
            current = benchmarks[i]
            previous = benchmarks[i + 1]
            
            if metric_name not in current.metrics or metric_name not in previous.metrics:
                continue
            
            current_value = current.metrics[metric_name]
            previous_value = previous.metrics[metric_name]
            
            if previous_value == 0:
                continue
            
            change_percent = ((current_value - previous_value) / previous_value) * 100
            
            # For metrics like duration, higher is worse
            # For metrics like accuracy, lower is worse
            # This is configurable based on metric name
            is_regression = False
            if "duration" in metric_name.lower() or "time" in metric_name.lower():
                is_regression = change_percent > threshold_percent
            else:
                is_regression = change_percent < -threshold_percent
            
            if is_regression:
                regressions.append({
                    "current_benchmark": current.benchmark_id,
                    "previous_benchmark": previous.benchmark_id,
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "previous_value": previous_value,
                    "change_percent": change_percent,
                    "regression_type": "performance_degradation" if "duration" in metric_name.lower() else "quality_degradation",
                    "detected_at": current.timestamp
                })
        
        return regressions
    
    def generate_report(self, 
                       output_path: str,
                       categories: Optional[List[str]] = None,
                       since_days: int = 30) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report."""
        
        since_timestamp = time.time() - (since_days * 86400)
        
        if categories is None:
            categories = ["precision", "multi_gpu", "optimization", "regression"]
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "time_range_days": since_days,
            "categories_analyzed": categories,
            "summary": {},
            "trends": {},
            "regressions": {},
            "recommendations": []
        }
        
        # Summary statistics
        total_benchmarks = 0
        successful_benchmarks = 0
        
        for category in categories:
            benchmarks = self.get_benchmarks(category=category, since_timestamp=since_timestamp)
            category_successful = sum(1 for b in benchmarks if b.success)
            
            total_benchmarks += len(benchmarks)
            successful_benchmarks += category_successful
            
            report["summary"][category] = {
                "total_tests": len(benchmarks),
                "successful_tests": category_successful,
                "success_rate": category_successful / len(benchmarks) if benchmarks else 0,
                "latest_test": benchmarks[0].timestamp if benchmarks else None
            }
        
        report["summary"]["overall"] = {
            "total_benchmarks": total_benchmarks,
            "successful_benchmarks": successful_benchmarks,
            "overall_success_rate": successful_benchmarks / total_benchmarks if total_benchmarks else 0
        }
        
        # Trend analysis for key metrics
        key_metrics = ["duration", "structures_generated", "validity_rate", "speedup_factor"]
        
        for category in categories:
            report["trends"][category] = {}
            for metric in key_metrics:
                trend = self.analyze_performance_trends(category, metric)
                if "error" not in trend:
                    report["trends"][category][metric] = trend
        
        # Regression detection
        for category in categories:
            for metric in key_metrics:
                regressions = self.detect_regressions(category, metric)
                if regressions:
                    if category not in report["regressions"]:
                        report["regressions"][category] = {}
                    report["regressions"][category][metric] = regressions
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        # Save report
        output_file = Path(output_path) / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Benchmark report saved to: {output_file}")
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark analysis."""
        recommendations = []
        
        # Check overall success rate
        overall_success = report["summary"]["overall"]["overall_success_rate"]
        if overall_success < 0.95:
            recommendations.append(f"⚠️ Overall success rate is {overall_success:.1%} - investigate failing tests")
        
        # Check for regressions
        total_regressions = sum(
            len(metrics) for category_regressions in report["regressions"].values()
            for metrics in category_regressions.values()
        )
        
        if total_regressions > 0:
            recommendations.append(f"🚨 {total_regressions} performance regressions detected - prioritize fixes")
        
        # Check trends
        for category, trends in report["trends"].items():
            for metric, trend_data in trends.items():
                if "trend" in trend_data and trend_data["trend"]["direction"] == "decreasing":
                    if "duration" not in metric.lower():  # Lower is bad for non-duration metrics
                        recommendations.append(f"📉 {category}.{metric} shows declining trend")
        
        if not recommendations:
            recommendations.append("✅ All benchmarks are performing well - no immediate issues detected")
        
        return recommendations
    
    def export_to_csv(self, output_path: str, category: Optional[str] = None):
        """Export benchmark data to CSV for external analysis."""
        benchmarks = self.get_benchmarks(category=category)
        
        if not PANDAS_AVAILABLE:
            # Simple CSV export without pandas
            output_file = Path(output_path) / f"benchmarks_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            with open(output_file, 'w') as f:
                # Write header
                headers = ['benchmark_id', 'timestamp', 'datetime', 'test_name', 'test_category', 'success', 'error_message']
                
                # Get all unique config, metric, and metadata keys
                all_config_keys = set()
                all_metric_keys = set()
                all_meta_keys = set()
                
                for benchmark in benchmarks:
                    all_config_keys.update(benchmark.configuration.keys())
                    all_metric_keys.update(benchmark.metrics.keys())
                    all_meta_keys.update(benchmark.metadata.keys())
                
                config_headers = [f'config_{key}' for key in sorted(all_config_keys)]
                metric_headers = [f'metric_{key}' for key in sorted(all_metric_keys)]
                meta_headers = [f'meta_{key}' for key in sorted(all_meta_keys)]
                
                all_headers = headers + config_headers + metric_headers + meta_headers
                f.write(','.join(all_headers) + '\n')
                
                # Write data
                for benchmark in benchmarks:
                    row = [
                        benchmark.benchmark_id,
                        str(benchmark.timestamp),
                        datetime.fromtimestamp(benchmark.timestamp).isoformat(),
                        benchmark.test_name,
                        benchmark.test_category,
                        str(benchmark.success),
                        benchmark.error_message or ''
                    ]
                    
                    # Add config values
                    for key in sorted(all_config_keys):
                        row.append(str(benchmark.configuration.get(key, '')))
                    
                    # Add metric values
                    for key in sorted(all_metric_keys):
                        row.append(str(benchmark.metrics.get(key, '')))
                    
                    # Add metadata values
                    for key in sorted(all_meta_keys):
                        row.append(str(benchmark.metadata.get(key, '')))
                    
                    f.write(','.join(row) + '\n')
            
            logger.info(f"Exported {len(benchmarks)} benchmark records to: {output_file}")
            return output_file
        
        # Pandas-based export (more robust)
        # Flatten data for CSV export
        rows = []
        for benchmark in benchmarks:
            base_row = {
                'benchmark_id': benchmark.benchmark_id,
                'timestamp': benchmark.timestamp,
                'datetime': datetime.fromtimestamp(benchmark.timestamp).isoformat(),
                'test_name': benchmark.test_name,
                'test_category': benchmark.test_category,
                'success': benchmark.success,
                'error_message': benchmark.error_message or ''
            }
            
            # Add configuration fields
            for key, value in benchmark.configuration.items():
                base_row[f'config_{key}'] = value
            
            # Add metric fields
            for key, value in benchmark.metrics.items():
                base_row[f'metric_{key}'] = value
            
            # Add metadata fields
            for key, value in benchmark.metadata.items():
                base_row[f'meta_{key}'] = value
            
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        output_file = Path(output_path) / f"benchmarks_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(rows)} benchmark records to: {output_file}")
        return output_file


def import_mixed_precision_results(tracker: BenchmarkTracker, results_file: str) -> List[str]:
    """Import the mixed precision comparison results into the tracker."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    benchmark_ids = []
    
    # Import FP32 benchmark
    fp32_result = BenchmarkResult(
        benchmark_id=f"mixed_precision_fp32_{int(data['timestamp'])}",
        timestamp=data['timestamp'],
        test_name="mixed_precision_comparison_fp32",
        test_category="precision",
        configuration={
            "mixed_precision": False,
            "batch_size": 32,
            "num_batches": 8,
            "enable_optimizations": True,
            "enable_model_compilation": True,
            "enable_graph_caching": True,
            "structures_requested": 256
        },
        metrics={
            "duration": data['experiments']['full_precision_fp32']['duration'],
            "structures_generated": data['experiments']['full_precision_fp32']['structure_count'],
            "validity_rate": data['experiments']['full_precision_fp32']['structural_metrics']['validity_rate'],
            "completion_rate": data['experiments']['full_precision_fp32']['structural_metrics']['completion_rate'],
            "error_rate": data['experiments']['full_precision_fp32']['structural_metrics']['error_rate']
        },
        metadata={
            "experiment_type": "precision_comparison",
            "output_dir": data['experiments']['full_precision_fp32']['output_dir'],
            "files_found": data['experiments']['full_precision_fp32']['files_found']
        },
        success=data['experiments']['full_precision_fp32']['success']
    )
    
    # Import FP16 benchmark
    fp16_result = BenchmarkResult(
        benchmark_id=f"mixed_precision_fp16_{int(data['timestamp'])}",
        timestamp=data['timestamp'],
        test_name="mixed_precision_comparison_fp16",
        test_category="precision",
        configuration={
            "mixed_precision": True,
            "batch_size": 32,
            "num_batches": 8,
            "enable_optimizations": True,
            "enable_model_compilation": True,
            "enable_graph_caching": True,
            "structures_requested": 256
        },
        metrics={
            "duration": data['experiments']['mixed_precision_fp16']['duration'],
            "structures_generated": data['experiments']['mixed_precision_fp16']['structure_count'],
            "validity_rate": data['experiments']['mixed_precision_fp16']['structural_metrics']['validity_rate'],
            "completion_rate": data['experiments']['mixed_precision_fp16']['structural_metrics']['completion_rate'],
            "error_rate": data['experiments']['mixed_precision_fp16']['structural_metrics']['error_rate']
        },
        metadata={
            "experiment_type": "precision_comparison",
            "output_dir": data['experiments']['mixed_precision_fp16']['output_dir'],
            "files_found": data['experiments']['mixed_precision_fp16']['files_found']
        },
        success=data['experiments']['mixed_precision_fp16']['success']
    )
    
    # Add benchmarks to tracker
    benchmark_ids.append(tracker.add_benchmark(fp32_result))
    benchmark_ids.append(tracker.add_benchmark(fp16_result))
    
    # Add comparison
    comparison_id = tracker.add_comparison(
        comparison_name="Mixed Precision vs Full Precision Accuracy Comparison",
        benchmark_ids=benchmark_ids,
        comparison_results=data['comparison'],
        conclusions="Mixed precision shows no performance benefit and slight quality degradation for MatterGen on this hardware. Full precision (FP32) is recommended for production use."
    )
    
    logger.info(f"Imported mixed precision comparison results - Comparison ID: {comparison_id}")
    return benchmark_ids


def main():
    """Main function to demonstrate benchmark tracking usage."""
    
    # Initialize tracker
    tracker = BenchmarkTracker("mattergen_benchmarks.db")
    
    # Import existing mixed precision results
    results_file = "results/mixed_precision_comparison/comparison_results.json"
    if Path(results_file).exists():
        benchmark_ids = import_mixed_precision_results(tracker, results_file)
        print(f"✅ Imported mixed precision benchmark results: {benchmark_ids}")
    
    # Generate initial report
    report = tracker.generate_report("results", since_days=30)
    print(f"📊 Generated benchmark report with {report['summary']['overall']['total_benchmarks']} benchmarks")
    
    # Export data for analysis
    csv_file = tracker.export_to_csv("results")
    print(f"📁 Exported benchmark data to: {csv_file}")
    
    print("\n🎯 Current benchmark status:")
    for category, summary in report['summary'].items():
        if category != 'overall':
            print(f"  {category}: {summary['successful_tests']}/{summary['total_tests']} tests passed")
    
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  {rec}")


if __name__ == "__main__":
    main()
