#!/usr/bin/env python3

"""
Production fix for multi-GPU AttributeError.
This script provides the corrected command that will work.
"""

import subprocess
import sys
import os

def run_corrected_command():
    """Run the corrected multi-GPU command."""
    
    print("Running corrected MatterGen command with multi-GPU fix...")
    print("=" * 60)
    
    # The key fix is to disable multi-GPU for now and use single GPU with optimizations
    cmd = [
        "python", "-m", "mattergen.scripts.generate",
        "results/corrected_output",
        "--pretrained-name=mattergen_base",
        "--batch_size=64",
        "--num_batches=1", 
        "--sampling_config_name=optimized_compatible",
        "--enable_multi_gpu=False",  # Disable multi-GPU to avoid wrapper issue
        "--enable_graph_caching=True",  # Keep caching enabled
        "--enable_mixed_precision=True",  # Keep other optimizations
        "--enable_model_compilation=True",
        "--record_trajectories=False"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, cwd="/home/guillaume/Documents/Projects/LINK/Internships/Auguste/mattergenbis", 
                              capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        if result.returncode == 0:
            print("\n✅ SUCCESS! Generation completed without errors.")
        else:
            print("\n❌ FAILED! See error output above.")
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 5 minutes (this may be normal for generation)")
    except Exception as e:
        print(f"❌ Error running command: {e}")

if __name__ == "__main__":
    print("MatterGen Multi-GPU Fix - Production Workaround")
    print("=" * 50)
    print("This uses single GPU mode with all other optimizations enabled")
    print("until the multi-GPU wrapper issue is fully resolved.")
    print()
    
    run_corrected_command()
