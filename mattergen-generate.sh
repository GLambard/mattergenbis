#!/bin/bash

# MatterGen Direct Execution Script
# Use this when mattergen-generate command is not available

cd "$(dirname "$0")"

echo "🚀 Running MatterGen with Phase 2 Optimizations"
echo "=============================================="

# Run the generation script directly with Python
python -c "
import sys
sys.path.insert(0, '.')
from mattergen.scripts.generate import main
import fire
fire.Fire(main)
" "$@"
