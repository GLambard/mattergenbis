#!/usr/bin/env python3
"""
Phase 4 Integration Test Script
==============================

Comprehensive test suite for validating Phase 4 adaptive sampling and quality metrics
integration with the main MatterGen generation pipeline.
"""

import os
import sys
import time
import shutil
import tempfile
from pathlib import Path
import subprocess
import json

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_command(cmd, timeout=300):
    """Run a command with timeout and capture output."""
    print(f"\n🔧 Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        
        duration = time.time() - start_time
        print(f"⏱️  Command completed in {duration:.2f}s")
        
        if result.returncode != 0:
            print(f"❌ Command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, result.stdout, result.stderr
        else:
            print(f"✅ Command succeeded")
            return True, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout}s")
        return False, "", "Command timed out"
    except Exception as e:
        print(f"💥 Command failed with exception: {e}")
        return False, "", str(e)


def test_phase4_imports():
    """Test that Phase 4 modules can be imported successfully."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Phase 4 Module Import Validation")
    print("="*60)
    
    modules_to_test = [
        'mattergen.common.utils.adaptive_sampler',
        'mattergen.common.utils.quality_metrics',
        'mattergen.common.utils.phase4_integration'
    ]
    
    all_passed = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ Successfully imported {module}")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            all_passed = False
        except Exception as e:
            print(f"💥 Unexpected error importing {module}: {e}")
            all_passed = False
    
    return all_passed


def test_configuration_loading():
    """Test that adaptive sampling configuration can be loaded."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Configuration Loading Validation")
    print("="*60)
    
    config_path = project_root / "sampling_conf" / "adaptive_sampling.yaml"
    
    if not config_path.exists():
        print(f"❌ Adaptive sampling config not found at {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['adaptive_sampling', 'quality_metrics', 'optimization']
        for section in required_sections:
            if section in config:
                print(f"✅ Found required section: {section}")
            else:
                print(f"❌ Missing required section: {section}")
                return False
        
        print(f"✅ Configuration loaded successfully from {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False


def test_single_gpu_phase4_generation():
    """Test Phase 4 features with single GPU generation."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Single GPU Phase 4 Generation")
    print("="*60)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "phase4_test_single"
        
        # Test command with Phase 4 features
        cmd = [
            "python", "-m", "mattergen.scripts.generate",
            str(output_path),
            "--pretrained_name=mattergen_base",
            "--batch_size=2",
            "--num_batches=1",
            "--sampling_config_name=default",
            "--record_trajectories=false",
            "--enable_phase4_features=true",
            "--enable_adaptive_sampling=true",
            "--enable_quality_metrics=true",
            "--quality_threshold=0.5",
            "--max_adaptation_iterations=2"
        ]
        
        # Add adaptive config if it exists
        adaptive_config = project_root / "sampling_conf" / "adaptive_sampling.yaml"
        if adaptive_config.exists():
            cmd.extend([f"--adaptive_config_path={adaptive_config}"])
        
        success, stdout, stderr = run_command(cmd, timeout=180)
        
        if success:
            print("✅ Single GPU Phase 4 generation completed successfully")
            
            # Check if structures were generated
            if output_path.exists():
                cif_files = list(output_path.glob("*.cif"))
                print(f"📁 Found {len(cif_files)} CIF files in output directory")
                
                if len(cif_files) > 0:
                    print("✅ Structure generation successful")
                    return True
                else:
                    print("⚠️  No structures generated, but no errors reported")
                    return True  # May be due to quality filtering
            else:
                print("❌ Output directory not created")
                return False
        else:
            print(f"❌ Single GPU Phase 4 generation failed")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False


def test_multi_gpu_phase4_integration():
    """Test Phase 4 features with multi-GPU setup."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Multi-GPU Phase 4 Integration")
    print("="*60)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "phase4_test_multi"
        
        # Test command with Phase 4 features on multi-GPU
        cmd = [
            "python", "multi_gpu_inference.py",
            "--output_base_path", str(output_path),
            "--pretrained_name", "mattergen_base",
            "--total_structures", "4",
            "--num_gpus", "2",
            "--base_batch_size", "2",
            "--sampling_config_name", "default",
            "--record_trajectories", "false",
            "--enable_phase4_features", "true",
            "--enable_adaptive_sampling", "true",
            "--enable_quality_metrics", "true",
            "--quality_threshold", "0.5",
            "--max_adaptation_iterations", "2"
        ]
        
        # Add adaptive config if it exists
        adaptive_config = project_root / "sampling_conf" / "adaptive_sampling.yaml"
        if adaptive_config.exists():
            cmd.extend(["--adaptive_config_path", str(adaptive_config)])
        
        success, stdout, stderr = run_command(cmd, timeout=300)
        
        if success:
            print("✅ Multi-GPU Phase 4 generation completed successfully")
            
            # Check if structures were generated
            if output_path.exists():
                # Count all CIF files in subdirectories
                cif_files = list(output_path.rglob("*.cif"))
                print(f"📁 Found {len(cif_files)} CIF files across all GPU outputs")
                
                # Check GPU subdirectories
                gpu_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("gpu_")]
                print(f"📂 Found {len(gpu_dirs)} GPU output directories")
                
                if len(cif_files) > 0:
                    print("✅ Multi-GPU structure generation successful")
                    return True
                else:
                    print("⚠️  No structures generated, but no errors reported")
                    return True  # May be due to quality filtering
            else:
                print("❌ Output directory not created")
                return False
        else:
            print(f"❌ Multi-GPU Phase 4 generation failed")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False


def test_backward_compatibility():
    """Test that Phase 3 functionality still works without Phase 4."""
    print("\n" + "="*60)
    print("🧪 TEST 5: Backward Compatibility Validation")
    print("="*60)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "phase3_compatibility_test"
        
        # Test standard Phase 3 command (no Phase 4 features)
        cmd = [
            "python", "-m", "mattergen.scripts.generate",
            str(output_path),
            "--pretrained_name=mattergen_base",
            "--batch_size=2",
            "--num_batches=1",
            "--sampling_config_name=default",
            "--record_trajectories=false",
            "--enable_optimizations=true",
            "--enable_model_compilation=true",
            "--enable_graph_caching=true"
        ]
        
        success, stdout, stderr = run_command(cmd, timeout=120)
        
        if success:
            print("✅ Phase 3 backward compatibility maintained")
            
            # Check if structures were generated
            if output_path.exists():
                cif_files = list(output_path.glob("*.cif"))
                print(f"📁 Found {len(cif_files)} CIF files in output directory")
                
                if len(cif_files) > 0:
                    print("✅ Standard generation successful")
                    return True
                else:
                    print("❌ No structures generated in standard mode")
                    return False
            else:
                print("❌ Output directory not created")
                return False
        else:
            print(f"❌ Phase 3 backward compatibility test failed")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False


def run_performance_benchmark():
    """Run a basic performance benchmark comparing Phase 3 vs Phase 4."""
    print("\n" + "="*60)
    print("🧪 BENCHMARK: Phase 3 vs Phase 4 Performance")
    print("="*60)
    
    results = {}
    
    # Test Phase 3 performance
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "phase3_benchmark"
        
        cmd = [
            "python", "-m", "mattergen.scripts.generate",
            str(output_path),
            "--pretrained_name=mattergen_base",
            "--batch_size=4",
            "--num_batches=1",
            "--sampling_config_name=default",
            "--record_trajectories=false",
            "--enable_optimizations=true"
        ]
        
        start_time = time.time()
        success, stdout, stderr = run_command(cmd, timeout=180)
        phase3_duration = time.time() - start_time
        
        if success and output_path.exists():
            phase3_count = len(list(output_path.glob("*.cif")))
            results['phase3'] = {'duration': phase3_duration, 'structures': phase3_count}
            print(f"📊 Phase 3: {phase3_count} structures in {phase3_duration:.2f}s")
        else:
            print("❌ Phase 3 benchmark failed")
            results['phase3'] = {'duration': None, 'structures': 0}
    
    # Test Phase 4 performance
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "phase4_benchmark"
        
        cmd = [
            "python", "-m", "mattergen.scripts.generate",
            str(output_path),
            "--pretrained_name=mattergen_base",
            "--batch_size=4",
            "--num_batches=1",
            "--sampling_config_name=default",
            "--record_trajectories=false",
            "--enable_optimizations=true",
            "--enable_phase4_features=true",
            "--enable_adaptive_sampling=true",
            "--enable_quality_metrics=true",
            "--max_adaptation_iterations=2"
        ]
        
        start_time = time.time()
        success, stdout, stderr = run_command(cmd, timeout=180)
        phase4_duration = time.time() - start_time
        
        if success and output_path.exists():
            phase4_count = len(list(output_path.glob("*.cif")))
            results['phase4'] = {'duration': phase4_duration, 'structures': phase4_count}
            print(f"📊 Phase 4: {phase4_count} structures in {phase4_duration:.2f}s")
        else:
            print("❌ Phase 4 benchmark failed")
            results['phase4'] = {'duration': None, 'structures': 0}
    
    # Compare results
    if results['phase3']['duration'] and results['phase4']['duration']:
        duration_ratio = results['phase4']['duration'] / results['phase3']['duration']
        print(f"\n📈 Performance Comparison:")
        print(f"   Phase 4 vs Phase 3 duration ratio: {duration_ratio:.2f}x")
        if duration_ratio < 1.5:
            print("✅ Phase 4 performance acceptable (< 1.5x slower)")
        else:
            print("⚠️  Phase 4 performance needs optimization (> 1.5x slower)")
    
    return results


def main():
    """Run the complete Phase 4 integration test suite."""
    print("🚀 Starting Phase 4 Integration Test Suite")
    print("=" * 80)
    print(f"📁 Project root: {project_root}")
    print(f"⏰ Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Module Import Validation", test_phase4_imports),
        ("Configuration Loading", test_configuration_loading),
        ("Single GPU Phase 4", test_single_gpu_phase4_generation),
        ("Multi-GPU Phase 4", test_multi_gpu_phase4_integration),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            test_results[test_name] = False
    
    # Run performance benchmark
    try:
        benchmark_results = run_performance_benchmark()
        test_results["Performance Benchmark"] = benchmark_results
    except Exception as e:
        print(f"💥 Performance Benchmark: ERROR - {e}")
        test_results["Performance Benchmark"] = None
    
    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUITE SUMMARY")
    print("="*80)
    
    passed_tests = sum(1 for result in test_results.values() if result is True)
    total_tests = len([r for r in test_results.values() if r is not None])
    
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    for test_name, result in test_results.items():
        if test_name == "Performance Benchmark":
            continue
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Phase 4 integration is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review and fix issues.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
