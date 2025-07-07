#!/usr/bin/env python3
"""
Validation Script for MatterGen Code Reorganization
==================================================

Validates that all benchmark and debug scripts have been successfully moved
to mattergen/benchmarks/ and that all imports work correctly.
"""

import sys
import traceback
from pathlib import Path

def test_import(module_name, description):
    """Test importing a module and return success status."""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except Exception as e:
        print(f"❌ {description}: {module_name} - Error: {e}")
        return False

def test_class_instantiation(module_name, class_name, description):
    """Test instantiating a class from a module."""
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        instance = cls()
        print(f"✅ {description}: {class_name} from {module_name}")
        return True
    except Exception as e:
        print(f"❌ {description}: {class_name} from {module_name} - Error: {e}")
        return False

def test_function_call(module_name, function_name, description):
    """Test calling a function from a module."""
    try:
        module = __import__(module_name, fromlist=[function_name])
        func = getattr(module, function_name)
        # Just verify the function exists and is callable
        if callable(func):
            print(f"✅ {description}: {function_name} from {module_name}")
            return True
        else:
            print(f"❌ {description}: {function_name} from {module_name} - Not callable")
            return False
    except Exception as e:
        print(f"❌ {description}: {function_name} from {module_name} - Error: {e}")
        return False

def validate_file_structure():
    """Validate that files have been moved to the correct locations."""
    print("\n📁 Validating File Structure")
    print("=" * 50)
    
    # Expected files in mattergen/benchmarks/
    expected_files = [
        "mattergen/benchmarks/__init__.py",
        "mattergen/benchmarks/comprehensive_benchmark_suite.py",
        "mattergen/benchmarks/comprehensive_benchmark.py", 
        "mattergen/benchmarks/crystalline_accuracy_validator.py",
        "mattergen/benchmarks/real_crystalline_baseline.py",
        "mattergen/benchmarks/debug_imports.py",
        "mattergen/benchmarks/debug_multi_gpu.py"
    ]
    
    all_exist = True
    for file_path in expected_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            all_exist = False
    
    return all_exist

def validate_imports():
    """Validate that all moved modules can be imported."""
    print("\n📦 Validating Module Imports")
    print("=" * 50)
    
    import_tests = [
        ("mattergen.benchmarks", "Main benchmarks package"),
        ("mattergen.benchmarks.comprehensive_benchmark_suite", "Comprehensive benchmark suite"),
        ("mattergen.benchmarks.comprehensive_benchmark", "Comprehensive benchmark"),
        ("mattergen.benchmarks.crystalline_accuracy_validator", "Crystalline accuracy validator"),
        ("mattergen.benchmarks.real_crystalline_baseline", "Real crystalline baseline"),
        ("mattergen.benchmarks.debug_imports", "Debug imports"),
        ("mattergen.benchmarks.debug_multi_gpu", "Debug multi-GPU")
    ]
    
    success_count = 0
    for module_name, description in import_tests:
        if test_import(module_name, description):
            success_count += 1
    
    return success_count, len(import_tests)

def validate_benchmark_functionality():
    """Validate that benchmark classes and functions are accessible."""
    print("\n🔧 Validating Benchmark Functionality")
    print("=" * 50)
    
    functionality_tests = [
        # Test class instantiation
        ("mattergen.benchmarks.comprehensive_benchmark_suite", "ComprehensiveBenchmarkRunner", "Benchmark runner class"),
        ("mattergen.benchmarks.crystalline_accuracy_validator", "CrystallineAccuracyValidator", "Accuracy validator class"),
        ("mattergen.benchmarks.real_crystalline_baseline", "RealCrystallineBaseline", "Real baseline class"),
    ]
    
    success_count = 0
    for module_name, class_name, description in functionality_tests:
        if test_class_instantiation(module_name, class_name, description):
            success_count += 1
    
    return success_count, len(functionality_tests)

def validate_api_exposure():
    """Validate that the benchmarks API is properly exposed via __init__.py."""
    print("\n🚀 Validating API Exposure")
    print("=" * 50)
    
    try:
        from mattergen.benchmarks import (
            ComprehensiveBenchmarkRunner,
            CrystallineAccuracyValidator, 
            RealCrystallineBaseline
        )
        print("✅ All main benchmark classes can be imported from mattergen.benchmarks")
        return True
    except ImportError as e:
        print(f"❌ API exposure issue: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 MatterGen Code Reorganization Validation")
    print("=" * 60)
    print("Validating that benchmark and debug scripts have been successfully")
    print("moved to mattergen/benchmarks/ and all imports work correctly.")
    print("=" * 60)
    
    # Track overall success
    all_passed = True
    
    # Test file structure
    if not validate_file_structure():
        all_passed = False
    
    # Test imports
    import_success, import_total = validate_imports()
    if import_success < import_total:
        all_passed = False
    
    # Test functionality
    func_success, func_total = validate_benchmark_functionality()
    if func_success < func_total:
        all_passed = False
    
    # Test API exposure
    if not validate_api_exposure():
        all_passed = False
    
    # Final summary
    print("\n📊 Validation Summary")
    print("=" * 50)
    print(f"File Structure: {'✅ PASS' if validate_file_structure() else '❌ FAIL'}")
    print(f"Module Imports: {import_success}/{import_total} ({'✅ PASS' if import_success == import_total else '❌ FAIL'})")
    print(f"Functionality: {func_success}/{func_total} ({'✅ PASS' if func_success == func_total else '❌ FAIL'})")
    print(f"API Exposure: {'✅ PASS' if validate_api_exposure() else '❌ FAIL'}")
    
    if all_passed:
        print("\n🎉 CODE REORGANIZATION VALIDATION: SUCCESS")
        print("✅ All benchmark and debug scripts successfully moved to mattergen/benchmarks/")
        print("✅ All imports working correctly")
        print("✅ API properly exposed")
        print("✅ MatterGen structure generation pipeline ready")
        return 0
    else:
        print("\n❌ CODE REORGANIZATION VALIDATION: ISSUES DETECTED")
        print("Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
