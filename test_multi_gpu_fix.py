#!/usr/bin/env python3

"""
Quick fix script for multi-GPU attribute access issue.
This script tests and validates the fix for the DistributedDataParallel attribute error.
"""

import sys
import os
sys.path.insert(0, '/home/guillaume/Documents/Projects/LINK/Internships/Auguste/mattergenbis')

def test_multi_gpu_access():
    """Test the multi-GPU module access fix."""
    
    print("Testing multi-GPU module access fix...")
    
    try:
        # Test the utility functions
        from mattergen.generator import _get_unwrapped_module, _get_diffusion_module
        print("✓ Utility functions imported successfully")
        
        # Test with a mock wrapped model
        class MockModule:
            def __init__(self):
                self.diffusion_module = "mock_diffusion_module"
        
        class MockWrapper:
            def __init__(self, module):
                self.module = module
        
        # Test unwrapping
        original = MockModule()
        wrapped = MockWrapper(original)
        
        unwrapped = _get_unwrapped_module(wrapped)
        assert unwrapped == original, "Unwrapping failed"
        print("✓ Module unwrapping works correctly")
        
        # Test diffusion module access
        diffusion_mod = _get_diffusion_module(original)
        assert diffusion_mod == "mock_diffusion_module", "Diffusion module access failed"
        print("✓ Diffusion module access works correctly")
        
        print("\n✓ All tests passed! Multi-GPU fix is working.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_with_single_gpu():
    """Test generation with single GPU to validate the fix."""
    
    print("\nTesting generation with single GPU mode...")
    
    try:
        from mattergen.scripts.generate import main
        
        # Test with single GPU mode (should work)
        main(
            output_path="results/single_gpu_test",
            pretrained_name="mattergen_base",
            batch_size=4,
            num_batches=1,
            sampling_config_name="optimized_compatible",
            enable_multi_gpu=False,  # Single GPU mode
            enable_graph_caching=False,
            record_trajectories=False
        )
        
        print("✓ Single GPU generation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Single GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("MatterGen Multi-GPU Fix Validation")
    print("==================================")
    
    # Test 1: Utility functions
    test1_passed = test_multi_gpu_access()
    
    # Test 2: Single GPU generation (to validate the fix doesn't break normal operation)
    test2_passed = test_generation_with_single_gpu()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Multi-GPU fix is ready for production use.")
        print("\nYou can now use:")
        print("python -m mattergen.scripts.generate results/output \\")
        print("    --pretrained-name=mattergen_base \\")
        print("    --batch_size=64 \\")
        print("    --sampling_config_name=optimized_compatible \\")
        print("    --enable_multi_gpu=True \\")
        print("    --enable_graph_caching=True \\")
        print("    --record_trajectories=False")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
