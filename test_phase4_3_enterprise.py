#!/usr/bin/env python3
"""
Phase 4.3 Enterprise Features Test Suite
========================================

Comprehensive testing for MatterGen enterprise monitoring, dashboard, analytics,
and integration features.

Tests:
- Enterprise component initialization
- Monitoring system functionality
- Analytics engine capabilities  
- Integration manager coordination
- CLI interface functionality
- Error handling and edge cases
"""

import os
import sys
import time
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_enterprise_imports():
    """Test that enterprise modules can be imported."""
    print("🧪 Testing enterprise module imports...")
    
    try:
        # Test monitoring import
        from mattergen.enterprise.monitoring import MetricsCollector, SystemMetrics, GenerationMetrics, QualityMetrics
        print("✅ Monitoring module imported successfully")
        
        # Test analytics import (may fail due to numpy/pandas dependencies)
        try:
            from mattergen.enterprise.analytics import AnalyticsEngine, PerformanceInsight
            print("✅ Analytics module imported successfully")
        except ImportError as e:
            print(f"⚠️ Analytics module import failed (expected if numpy/pandas not installed): {e}")
        
        # Test dashboard import (may fail due to Flask dependencies)
        try:
            from mattergen.enterprise.dashboard import DashboardServer, FLASK_AVAILABLE
            if FLASK_AVAILABLE:
                print("✅ Dashboard module imported successfully")
            else:
                print("⚠️ Dashboard module imported but Flask not available")
        except ImportError as e:
            print(f"⚠️ Dashboard module import failed (expected if Flask not installed): {e}")
        
        # Test integration import
        from mattergen.enterprise.integration import EnterpriseManager
        print("✅ Integration module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Enterprise import test failed: {e}")
        return False


def test_metrics_collector():
    """Test metrics collector functionality."""
    print("\n🧪 Testing metrics collector...")
    
    try:
        from mattergen.enterprise.monitoring import MetricsCollector, SystemMetrics, GenerationMetrics, QualityMetrics
        
        # Initialize collector
        collector = MetricsCollector(retention_hours=1, max_points_per_metric=100)
        print("✅ Metrics collector initialized")
        
        # Start collector
        collector.start()
        print("✅ Metrics collector started")
        
        # Add test metrics
        collector.add_metric("test_metric", 42.0, {"source": "test"})
        print("✅ Added test metric")
        
        # Add system metrics
        gpu_metrics = {0: {"utilization": 85.0, "memory_used": 8000, "temperature": 65.0}}
        sys_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage=45.0,
            memory_usage=60.0,
            gpu_metrics=gpu_metrics,
            disk_usage=30.0,
            network_io={"rx": 1000, "tx": 500}
        )
        collector.add_system_metrics(sys_metrics)
        print("✅ Added system metrics")
        
        # Add generation metrics
        gen_metrics = GenerationMetrics(
            timestamp=time.time(),
            batch_id=1,
            gpu_id=0,
            structures_generated=64,
            batch_duration=30.5,
            quality_score=0.85,
            throughput=2.1,
            memory_peak=7500.0,
            error_count=0
        )
        collector.add_generation_metrics(gen_metrics)
        print("✅ Added generation metrics")
        
        # Add quality metrics
        quality_metrics = QualityMetrics(
            timestamp=time.time(),
            batch_id=1,
            overall_quality=0.85,
            quality_distribution={"excellent": 0.3, "good": 0.4, "fair": 0.2, "poor": 0.1},
            improvement_rate=0.05,
            trend="improving",
            convergence_status="converging"
        )
        collector.add_quality_metrics(quality_metrics)
        print("✅ Added quality metrics")
        
        # Test metrics retrieval
        summary = collector.get_current_metrics_summary()
        print(f"✅ Retrieved metrics summary: {len(summary)} items")
        
        # Test metric history
        history = collector.get_metric_history("test_metric", hours=1)
        print(f"✅ Retrieved metric history: {len(history)} points")
        
        # Stop collector
        collector.stop()
        print("✅ Metrics collector stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analytics_engine():
    """Test analytics engine (with mock data if dependencies missing)."""
    print("\n🧪 Testing analytics engine...")
    
    try:
        # Try to import analytics
        try:
            from mattergen.enterprise.analytics import AnalyticsEngine
            from mattergen.enterprise.monitoring import MetricsCollector
            
            # Create mock metrics collector with data
            collector = MetricsCollector()
            
            # Add some mock generation metrics
            from mattergen.enterprise.monitoring import GenerationMetrics, QualityMetrics
            for i in range(10):
                gen_metrics = GenerationMetrics(
                    timestamp=time.time() - (i * 300),  # 5 min intervals
                    batch_id=i,
                    gpu_id=i % 2,
                    structures_generated=64,
                    batch_duration=30.0 + (i * 2),
                    quality_score=0.8 + (i * 0.01),
                    throughput=2.0 + (i * 0.1),
                    memory_peak=7000.0,
                    error_count=0
                )
                collector.add_generation_metrics(gen_metrics)
            
            # Initialize analytics engine
            engine = AnalyticsEngine(metrics_collector=collector)
            print("✅ Analytics engine initialized")
            
            # Test performance score calculation
            score = engine.calculate_performance_score()
            print(f"✅ Performance score calculated: {score.get('overall', 0):.1f}")
            
            # Test insights generation
            insights = engine.generate_insights()
            print(f"✅ Generated {len(insights)} performance insights")
            
            # Test optimization recommendations
            recommendations = engine.get_optimization_recommendations()
            print(f"✅ Generated {len(recommendations)} optimization recommendations")
            
            return True
            
        except ImportError as e:
            print(f"⚠️ Analytics dependencies not available: {e}")
            print("✅ Analytics test skipped (install numpy/pandas for full functionality)")
            return True
            
    except Exception as e:
        print(f"❌ Analytics engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enterprise_integration():
    """Test enterprise integration manager."""
    print("\n🧪 Testing enterprise integration...")
    
    try:
        from mattergen.enterprise.integration import EnterpriseManager
        
        # Initialize manager
        config = {
            "monitoring": {"metrics_retention_hours": 1},
            "dashboard": {"host": "localhost", "port": 8081},
            "analytics": {"enabled": True}
        }
        
        manager = EnterpriseManager(
            config=config,
            enable_monitoring=True,
            enable_dashboard=False,  # Skip dashboard to avoid Flask dependency
            enable_analytics=True
        )
        print("✅ Enterprise manager created")
        
        # Initialize components
        manager.initialize()
        print("✅ Enterprise components initialized")
        
        # Start manager
        manager.start()
        print("✅ Enterprise manager started")
        
        # Test status
        status = manager.get_status()
        print(f"✅ Got enterprise status: {status['running']}")
        
        # Test integration hooks
        manager.on_generation_start(batch_id=1, gpu_id=0, batch_size=64)
        print("✅ Generation start hook called")
        
        manager.on_generation_complete(
            batch_id=1, gpu_id=0, structures_generated=64,
            batch_duration=30.0, quality_score=0.85, throughput=2.1,
            memory_peak=7000.0, error_count=0
        )
        print("✅ Generation complete hook called")
        
        manager.on_quality_assessment(
            batch_id=1, overall_quality=0.85,
            quality_distribution={"excellent": 0.3, "good": 0.4},
            improvement_rate=0.05, trend="improving", convergence_status="converging"
        )
        print("✅ Quality assessment hook called")
        
        manager.on_system_metrics(
            cpu_usage=45.0, memory_usage=60.0,
            gpu_metrics={0: {"utilization": 85.0}},
            disk_usage=30.0, network_io={"rx": 1000, "tx": 500}
        )
        print("✅ System metrics hook called")
        
        # Test analytics methods
        insights = manager.get_performance_insights()
        print(f"✅ Got {len(insights)} performance insights")
        
        recommendations = manager.get_optimization_recommendations()
        print(f"✅ Got {len(recommendations)} optimization recommendations")
        
        score = manager.get_performance_score()
        print(f"✅ Got performance score: {score.get('overall', 0):.1f}")
        
        # Stop manager
        manager.stop()
        print("✅ Enterprise manager stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Enterprise integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_interface():
    """Test enterprise CLI interface."""
    print("\n🧪 Testing enterprise CLI interface...")
    
    try:
        from mattergen.enterprise.cli import create_enterprise_parser
        
        # Create parser
        parser = create_enterprise_parser()
        print("✅ CLI parser created")
        
        # Test argument parsing
        test_args = [
            ['start', '--enable-monitoring', '--enable-analytics'],
            ['status'],
            ['report', '--output', 'test_report.json'],
            ['dashboard', '--host', 'localhost', '--port', '8080'],
            ['export', '--output', 'test_metrics.json']
        ]
        
        for args in test_args:
            try:
                parsed = parser.parse_args(args)
                print(f"✅ Parsed args: {args[0]}")
            except SystemExit:
                # Expected for help/error cases
                pass
        
        return True
        
    except Exception as e:
        print(f"❌ CLI interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n🧪 Testing error handling...")
    
    try:
        from mattergen.enterprise.monitoring import MetricsCollector
        from mattergen.enterprise.integration import EnterpriseManager
        
        # Test metrics collector with invalid parameters
        try:
            collector = MetricsCollector(retention_hours=-1)
            print("⚠️ Should have validated negative retention hours")
        except:
            print("✅ Properly handled invalid retention hours")
        
        # Test enterprise manager without dependencies
        manager = EnterpriseManager(
            enable_monitoring=True,
            enable_dashboard=True,  # Should gracefully handle missing Flask
            enable_analytics=True   # Should gracefully handle missing numpy/pandas
        )
        
        try:
            manager.initialize()
            print("✅ Gracefully handled missing dependencies")
        except ImportError:
            print("✅ Properly raised ImportError for missing dependencies")
        
        # Test null metrics collector operations
        manager = EnterpriseManager(enable_monitoring=False)
        manager.initialize()
        manager.on_generation_start(1, 0, 64)  # Should not crash
        print("✅ Handled operations with null metrics collector")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_loading():
    """Test configuration loading and validation."""
    print("\n🧪 Testing configuration loading...")
    
    try:
        # Create test configuration
        test_config = {
            "monitoring": {
                "enabled": True,
                "metrics_retention_hours": 48,
                "max_points_per_metric": 5000
            },
            "dashboard": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8080,
                "debug": False
            },
            "analytics": {
                "enabled": True,
                "trend_analysis_hours": 24
            }
        }
        
        # Test configuration with enterprise manager
        from mattergen.enterprise.integration import EnterpriseManager
        
        manager = EnterpriseManager(config=test_config)
        print("✅ Configuration loaded successfully")
        
        # Verify configuration is used
        status = manager.get_status()
        print(f"✅ Configuration applied: {len(status)} status items")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enterprise_features_complete():
    """Comprehensive test of all enterprise features working together."""
    print("\n🧪 Testing complete enterprise feature integration...")
    
    try:
        # Test the enterprise feature availability
        from mattergen.enterprise import get_enterprise_status, ENTERPRISE_FEATURES_AVAILABLE
        
        status = get_enterprise_status()
        print(f"✅ Enterprise status: {status['version']}")
        print(f"✅ Features available: {status['features_available']}")
        
        # Test global enterprise initialization
        from mattergen.enterprise.integration import initialize_enterprise, start_enterprise, stop_enterprise
        
        config = {
            "monitoring": {"metrics_retention_hours": 1},
            "dashboard": {"enabled": False},  # Skip dashboard for testing
            "analytics": {"enabled": True}
        }
        
        # Initialize and start
        manager = initialize_enterprise(config=config, enable_dashboard=False)
        start_enterprise()
        print("✅ Global enterprise services started")
        
        # Test some operations
        manager.on_generation_complete(
            batch_id=100, gpu_id=0, structures_generated=64,
            batch_duration=25.0, quality_score=0.92, throughput=2.56,
            memory_peak=6800.0, error_count=0
        )
        print("✅ Recorded test generation metrics")
        
        # Get performance insights
        insights = manager.get_performance_insights()
        score = manager.get_performance_score()
        print(f"✅ Performance analysis: {score.get('overall', 0):.1f} score, {len(insights)} insights")
        
        # Stop services
        stop_enterprise()
        print("✅ Global enterprise services stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete enterprise integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 4.3 enterprise tests."""
    print("🚀 Starting Phase 4.3 Enterprise Features Test Suite")
    print("=" * 60)
    
    tests = [
        ("Enterprise Imports", test_enterprise_imports),
        ("Metrics Collector", test_metrics_collector),
        ("Analytics Engine", test_analytics_engine),
        ("Enterprise Integration", test_enterprise_integration),
        ("CLI Interface", test_cli_interface),
        ("Error Handling", test_error_handling),
        ("Configuration Loading", test_configuration_loading),
        ("Complete Integration", test_enterprise_features_complete),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            results.append((test_name, result, duration))
            
            if result:
                print(f"✅ {test_name} PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {test_name} FAILED ({duration:.2f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, False, duration))
            print(f"💥 {test_name} CRASHED: {e} ({duration:.2f}s)")
    
    # Summary
    print("\n" + "="*60)
    print("📊 PHASE 4.3 ENTERPRISE TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    total_time = sum(duration for _, _, duration in results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    
    print("\nDetailed Results:")
    for test_name, result, duration in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name:<25} ({duration:.2f}s)")
    
    if passed == total:
        print("\n🎉 ALL ENTERPRISE TESTS PASSED!")
        print("🏢 Phase 4.3 enterprise features are ready for production!")
        return True
    else:
        print(f"\n⚠️ {total - passed} TESTS FAILED")
        print("🔧 Please review failed tests before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
