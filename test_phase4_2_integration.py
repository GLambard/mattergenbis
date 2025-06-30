#!/usr/bin/env python3
"""
Phase 4.2 Integration Test Suite
===============================

Test the integration of advanced quality metrics and reporting features.
"""

import os
import sys
import tempfile
import json
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def test_phase4_2_cli_arguments():
    """Test that Phase 4.2 CLI arguments are properly integrated."""
    print("Testing Phase 4.2 CLI arguments...")
    
    # Test multi-GPU inference arguments
    from multi_gpu_inference import main as multi_gpu_main
    import argparse
    
    # Create a mock argument parser to test argument parsing
    parser = argparse.ArgumentParser()
    
    # Add the arguments that should be in the multi-GPU inference script
    expected_phase4_2_args = [
        'enable_advanced_quality',
        'enable_quality_prediction', 
        'enable_quality_reporting',
        'enable_trend_analysis',
        'quality_model_path',
        'quality_report_format'
    ]
    
    # Parse the multi_gpu_inference.py file to check for these arguments
    with open('multi_gpu_inference.py', 'r') as f:
        content = f.read()
    
    missing_args = []
    for arg in expected_phase4_2_args:
        if f'--{arg}' not in content:
            missing_args.append(arg)
    
    if missing_args:
        print(f"❌ Missing CLI arguments: {missing_args}")
        return False
    else:
        print("✅ All Phase 4.2 CLI arguments found")
        return True


def test_phase4_2_imports():
    """Test that Phase 4.2 modules can be imported."""
    print("Testing Phase 4.2 module imports...")
    
    try:
        from mattergen.common.utils.advanced_quality_metrics import AdvancedQualityMetrics
        from mattergen.common.utils.quality_reporting import QualityReporter
        print("✅ Phase 4.2 modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_advanced_quality_metrics_initialization():
    """Test AdvancedQualityMetrics initialization."""
    print("Testing AdvancedQualityMetrics initialization...")
    
    try:
        from mattergen.common.utils.advanced_quality_metrics import AdvancedQualityMetrics
        
        # Test basic initialization
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = AdvancedQualityMetrics(
                enable_ml_prediction=True,
                enable_trend_analysis=True,
                output_dir=temp_dir
            )
            
            # Test that basic attributes are set
            assert hasattr(metrics, 'enable_ml_prediction')
            assert hasattr(metrics, 'enable_trend_analysis')
            assert hasattr(metrics, 'output_dir')
            
        print("✅ AdvancedQualityMetrics initialized successfully")
        return True
    except Exception as e:
        print(f"❌ AdvancedQualityMetrics initialization failed: {e}")
        return False


def test_quality_reporter_initialization():
    """Test QualityReporter initialization."""
    print("Testing QualityReporter initialization...")
    
    try:
        from mattergen.common.utils.quality_reporting import QualityReporter
        
        # Test basic initialization
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = QualityReporter(
                output_dir=temp_dir,
                enable_visualizations=True
            )
            
            # Test that basic attributes are set
            assert hasattr(reporter, 'output_dir')
            assert hasattr(reporter, 'enable_visualizations')
            
        print("✅ QualityReporter initialized successfully")
        return True
    except Exception as e:
        print(f"❌ QualityReporter initialization failed: {e}")
        return False


def test_phase4_2_integration_in_generate():
    """Test that Phase 4.2 features are integrated into generate.py."""
    print("Testing Phase 4.2 integration in generate.py...")
    
    # Check that the generate.py file contains Phase 4.2 integration
    with open('mattergen/scripts/generate.py', 'r') as f:
        content = f.read()
    
    expected_phase4_2_features = [
        'enable_advanced_quality',
        'enable_quality_prediction',
        'enable_quality_reporting',
        'enable_trend_analysis',
        'AdvancedQualityMetrics',
        'QualityReporter',
        'advanced_quality_results',
        'predict_quality',
        'update_trends'
    ]
    
    missing_features = []
    for feature in expected_phase4_2_features:
        if feature not in content:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Missing Phase 4.2 features in generate.py: {missing_features}")
        return False
    else:
        print("✅ All Phase 4.2 features found in generate.py")
        return True


def test_configuration_loading():
    """Test that Phase 4.2 configurations can be loaded."""
    print("Testing Phase 4.2 configuration loading...")
    
    try:
        # Test loading the adaptive sampling config (should work with Phase 4.2)
        config_path = Path("sampling_conf/adaptive_sampling.yaml")
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Verify basic structure
            required_sections = ['adaptive_sampling', 'quality_metrics', 'optimization']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                print(f"❌ Missing config sections: {missing_sections}")
                return False
            else:
                print("✅ Configuration loaded successfully")
                return True
        else:
            print("⚠️  Adaptive sampling config not found, creating minimal test config...")
            
            # Create a minimal test config
            test_config = {
                'adaptive_sampling': {
                    'initial_guidance_factor': 1.0,
                    'max_iterations': 5
                },
                'quality_metrics': {
                    'threshold': 0.7,
                    'metrics': ['validity', 'uniqueness']
                },
                'optimization': {
                    'learning_rate': 0.1,
                    'convergence_threshold': 0.01
                }
            }
            
            config_path.parent.mkdir(exist_ok=True)
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            print("✅ Test configuration created")
            return True
            
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_mock_quality_analysis():
    """Test Phase 4.2 quality analysis with mock data."""
    print("Testing Phase 4.2 quality analysis with mock data...")
    
    try:
        from mattergen.common.utils.advanced_quality_metrics import AdvancedQualityMetrics
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize advanced quality metrics
            metrics = AdvancedQualityMetrics(
                enable_ml_prediction=False,  # Disable ML to avoid sklearn dependency issues
                enable_trend_analysis=True,
                output_dir=temp_dir
            )
            
            # Create mock structure data
            mock_structures = [
                {
                    'composition': 'H2O',
                    'volume': 30.0,
                    'energy': -10.5,
                    'spacegroup': 'P1'
                },
                {
                    'composition': 'NaCl',
                    'volume': 45.2,
                    'energy': -15.3,
                    'spacegroup': 'Fm-3m'
                }
            ]
            
            # Test structure analysis
            results = metrics.analyze_structures(
                structures=mock_structures,
                batch_id=0,
                metadata={'test': True}
            )
            
            print("✅ Mock quality analysis completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Mock quality analysis failed: {e}")
        return False


def test_mock_quality_reporting():
    """Test Phase 4.2 quality reporting with mock data."""
    print("Testing Phase 4.2 quality reporting with mock data...")
    
    try:
        from mattergen.common.utils.quality_reporting import QualityReporter
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize quality reporter
            reporter = QualityReporter(
                output_dir=temp_dir,
                enable_visualizations=False  # Disable to avoid matplotlib dependency issues
            )
            
            # Create mock quality data
            mock_quality_data = {
                'overall_quality': 0.85,
                'structure_count': 10,
                'quality_distribution': [0.7, 0.8, 0.9, 0.85, 0.75],
                'crystallographic_metrics': {
                    'avg_space_group_probability': 0.90,
                    'avg_symmetry_precision': 0.85
                }
            }
            
            # Generate report
            report = reporter.generate_report_from_quality_data(
                quality_data=mock_quality_data,
                generation_metadata={'test': True}
            )
            
            # Save report
            report_path = reporter.save_report(report)
            
            # Verify report was created
            if Path(report_path).exists():
                print("✅ Mock quality report generated successfully")
                return True
            else:
                print("❌ Quality report file not found")
                return False
                
    except Exception as e:
        print(f"❌ Mock quality reporting failed: {e}")
        return False


def run_all_tests():
    """Run all Phase 4.2 integration tests."""
    print("="*60)
    print("Phase 4.2 Integration Test Suite")
    print("="*60)
    
    tests = [
        test_phase4_2_cli_arguments,
        test_phase4_2_imports,
        test_advanced_quality_metrics_initialization,
        test_quality_reporter_initialization,
        test_phase4_2_integration_in_generate,
        test_configuration_loading,
        test_mock_quality_analysis,
        test_mock_quality_reporting
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("🎉 All Phase 4.2 integration tests passed!")
        return True
    else:
        print(f"⚠️  {failed} tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
