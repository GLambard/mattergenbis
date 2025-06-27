#!/bin/bash

# Production-ready Phase 3 usage script
# This script provides working commands that avoid configuration conflicts

echo "MatterGen Phase 3 Production Usage"
echo "===================================="

# Set working directory
cd /home/guillaume/Documents/Projects/LINK/Internships/Auguste/mattergenbis

echo ""
echo "Option 1: Phase 3 with proven optimized_compatible config"
echo "Command:"
echo "python -m mattergen.scripts.generate results/my_output \\"
echo "    --pretrained-name=mattergen_base \\"
echo "    --batch_size=64 \\"
echo "    --num_batches=1 \\"
echo "    --sampling_config_name=optimized_compatible \\"
echo "    --enable_multi_gpu=True \\"
echo "    --enable_graph_caching=True \\"
echo "    --record_trajectories=False"

echo ""
echo "Option 2: Conservative Phase 3 (single GPU mode)"
echo "Command:"
echo "python -m mattergen.scripts.generate results/my_output \\"
echo "    --pretrained-name=mattergen_base \\"
echo "    --batch_size=32 \\"
echo "    --num_batches=2 \\"
echo "    --sampling_config_name=optimized_compatible \\"
echo "    --enable_multi_gpu=False \\"
echo "    --enable_graph_caching=True \\"
echo "    --record_trajectories=False"

echo ""
echo "Option 3: Phase 2 optimizations only (guaranteed to work)"
echo "Command:"
echo "python -m mattergen.scripts.generate results/my_output \\"
echo "    --pretrained-name=mattergen_base \\"
echo "    --batch_size=64 \\"
echo "    --num_batches=1 \\"
echo "    --sampling_config_name=optimized_compatible \\"
echo "    --enable_multi_gpu=False \\"
echo "    --enable_graph_caching=False \\"
echo "    --record_trajectories=False"

echo ""
echo "Recommended: Start with Option 3, then try Option 2, then Option 1"

echo ""
echo "To run Option 1 now:"
read -p "Press Enter to continue or Ctrl+C to cancel..."

echo "Running Option 1..."
python -m mattergen.scripts.generate results/my_output \
    --pretrained-name=mattergen_base \
    --batch_size=64 \
    --num_batches=1 \
    --sampling_config_name=optimized_compatible \
    --enable_multi_gpu=True \
    --enable_graph_caching=True \
    --record_trajectories=False

echo "Generation complete! Check results/my_output/ for generated structures."
