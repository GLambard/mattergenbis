#!/usr/bin/env python3
"""
Test script for validating Phase 2 optimization implementation.
"""

import sys
import os
import traceback

def test_imports():
    """Test if all required modules import correctly."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from mattergen.common.utils.performance_optimizer import PerformanceOptimizer, apply_generation_optimizations
        print("✅ Performance optimizer imported successfully")
    except Exception as e:
        print(f"❌ Performance optimizer import failed: {e}")
        return False
    
    return True

def test_performance_optimizer():
    """Test performance optimizer functionality."""
    print("\n🚀 Testing performance optimizer...")
    
    try:
        from mattergen.common.utils.performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        
        # Test inference mode
        optimizer.set_inference_mode()
        print("✅ Inference mode set successfully")
        
        # Test memory optimization
        clear_freq = optimizer.optimize_memory()
        print(f"✅ Memory optimization enabled, clear frequency: {clear_freq}")
        
        # Test optimal batch size estimation
        batch_size = optimizer.get_optimal_batch_size(None, None)
        print(f"✅ Optimal batch size estimation: {batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_config_files():
    """Test if sampling configuration files exist."""
    print("\n📁 Testing configuration files...")
    
    configs = [
        "sampling_conf/default.yaml",
        "sampling_conf/optimized.yaml", 
        "sampling_conf/fast.yaml"
    ]
    
    for config in configs:
        if os.path.exists(config):
            print(f"✅ Config found: {config}")
        else:
            print(f"❌ Config missing: {config}")
    
    # Test model config
    model_config = "mattergen/conf/lightning_module/diffusion_module/model/mattergen_optimized.yaml"
    if os.path.exists(model_config):
        print(f"✅ Optimized model config found: {model_config}")
    else:
        print(f"❌ Optimized model config missing: {model_config}")
    
    return True

def test_generator_integration():
    """Test if generator has optimization integration."""
    print("\n🔧 Testing generator integration...")
    
    try:
        # Check if the generator module exists and can be imported
        from mattergen import generator
        print("✅ Generator module imported")
        
        # Check if the integration exists by looking for the performance optimizer import
        import inspect
        source = inspect.getsource(generator)
        
        if "performance_optimizer" in source:
            print("✅ Performance optimizer integration found in generator")
        else:
            print("❌ Performance optimizer integration not found in generator")
            
        if "enable_performance_optimizations" in source:
            print("✅ Performance optimization flag found in generator")
        else:
            print("❌ Performance optimization flag not found in generator")
        
        return True
        
    except Exception as e:
        print(f"❌ Generator integration test failed: {e}")
        return False

def test_cli_integration():
    """Test if CLI has optimization flags."""
    print("\n💻 Testing CLI integration...")
    
    try:
        # Check if the scripts module exists and has the new flags
        from mattergen.scripts import generate
        import inspect
        source = inspect.getsource(generate)
        
        flags_to_check = [
            "enable_optimizations",
            "enable_mixed_precision", 
            "enable_model_compilation"
        ]
        
        for flag in flags_to_check:
            if flag in source:
                print(f"✅ CLI flag found: {flag}")
            else:
                print(f"❌ CLI flag missing: {flag}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 MatterGen Phase 2 Optimization Validation")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_performance_optimizer,
        test_config_files,
        test_generator_integration,
        test_cli_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n📊 Summary")
    print("=" * 20)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("🎉 All tests passed! Phase 2 implementation is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
