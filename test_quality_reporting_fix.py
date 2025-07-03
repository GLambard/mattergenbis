#!/usr/bin/env python3
"""
Test script to verify that the quality_reporter initialization bug is fixed.
"""

import sys
import subprocess
import tempfile
import os

def test_quality_reporting_fix():
    """Test that quality reporting works in standalone mode without Phase 4."""
    
    print("Testing quality reporting fix...")
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable, "-m", "mattergen.scripts.generate",
            temp_dir,
            "--batch_size", "2",
            "--num_batches", "1", 
            "--enable_multi_gpu", "False",
            "--sampling-config-name", "optimized_compatible",
            "--record-trajectories", "False",
            "--print-loss", "False",
            "--enable-phase4-features", "False",
            "--enable-adaptive-sampling", "False",
            "--enable-quality-metrics", "True",
            "--enable-quality-reporting", "True",
            "--quality-report-format", "json",
            "--pretrained-name", "mattergen_base"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ SUCCESS: Quality reporting works without Phase 4!")
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                return True
            else:
                print("❌ FAILED: Command returned non-zero exit code")
                print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ TIMEOUT: Command took too long")
            return False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return False

if __name__ == "__main__":
    success = test_quality_reporting_fix()
    sys.exit(0 if success else 1)
