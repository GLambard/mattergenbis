#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

def test_basic_multi_gpu():
    """Test basic multi-GPU setup with minimal parameters"""
    
    print("🔍 Debugging Multi-GPU Setup")
    print("=" * 50)
    
    # Test 1: Check if mattergen-generate command exists
    print("1. Testing mattergen-generate command availability...")
    try:
        result = subprocess.run(
            ["mattergen-generate", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("✅ mattergen-generate command is available")
        else:
            print(f"❌ mattergen-generate command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running mattergen-generate: {e}")
        return False
    
    # Test 2: Test basic single GPU command construction
    print("\n2. Testing basic command construction...")
    output_path = "debug_test_output"
    
    cmd = [
        "mattergen-generate",
        output_path,
        "--pretrained_name=chemical_system",
        "--batch_size=1",
        "--num_batches=1",
        "--record_trajectories=False",
        "--enable_multi_gpu=False",  # Single GPU for debug
        "--sampling_config_name=optimized_compatible"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Test 3: Run with explicit GPU setting
    print("\n3. Testing with explicit GPU environment...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("✅ Basic command succeeded")
            print(f"Stdout: {result.stdout[-500:]}")  # Last 500 chars
            return True
        else:
            print(f"❌ Basic command failed with return code {result.returncode}")
            print(f"Stderr: {result.stderr}")
            print(f"Stdout: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

if __name__ == "__main__":
    test_basic_multi_gpu()
