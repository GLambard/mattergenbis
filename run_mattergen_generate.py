#!/usr/bin/env python3
"""
Direct execution wrapper for MatterGen generation with Phase 2 optimizations.
Use this when mattergen-generate console script is not available.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run mattergen generation with command line arguments."""
    try:
        # Import the generate module
        from mattergen.scripts.generate import main as generate_main
        
        # Remove the script name from argv to pass clean arguments to generate_main
        sys.argv = sys.argv[1:]  # Remove script name
        
        # Run the generation
        generate_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the MatterGen directory and dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_mattergen_generate.py [generate arguments...]")
        print()
        print("Example:")
        print("python run_mattergen_generate.py results/ --pretrained-name=mattergen_base --batch_size=64 --sampling_config_name=optimized --enable_optimizations=True")
        sys.exit(1)
    
    main()
