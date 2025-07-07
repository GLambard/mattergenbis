#!/usr/bin/env python3
"""
Discover Supported Conditioning Fields for MatterGen Models
==========================================================
"""

import json
from pathlib import Path
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator

def discover_model_conditioning_fields(model_name="mattergen_base"):
    """Discover what conditioning fields a model supports."""
    print(f"🔍 Discovering conditioning fields for {model_name}...")
    
    try:
        # Create checkpoint info
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(model_name)
        print(f"✅ Created checkpoint info for {model_name}")
        
        # Create generator
        generator = CrystalGenerator(checkpoint_info=checkpoint_info)
        print(f"✅ Created generator")
        
        # Prepare the generator (this loads the model)
        generator.prepare()
        print(f"✅ Generator prepared")
        
        # Get conditioning fields - handle DistributedDataParallel wrapper
        model = generator._model
        
        # If wrapped in DistributedDataParallel, get the underlying module
        if hasattr(model, 'module'):
            model = model.module
        
        diffusion_module = model.diffusion_module
        supported_fields = diffusion_module.model.cond_fields_model_was_trained_on
        
        print(f"\n📋 Supported conditioning fields for {model_name}:")
        for i, field in enumerate(supported_fields, 1):
            print(f"  {i}. {field}")
        
        return supported_fields
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_supported_conditioning():
    """Test generation with supported conditioning fields."""
    
    # First discover what's supported
    supported_fields = discover_model_conditioning_fields("mattergen_base")
    
    if not supported_fields:
        print("❌ Could not determine supported fields")
        return
    
    print(f"\n🧪 Testing with supported conditioning fields...")
    
    # Test examples for common fields
    test_examples = {
        "space_group": "225",  # Face-centered cubic
        "formation_energy_per_atom": "-2.5",
        "energy_above_hull": "0.1",
        "dft_band_gap": "1.5",
        "ml_bulk_modulus": "150"
    }
    
    print(f"\n📝 Suggested test commands:")
    print("=" * 50)
    
    for field in supported_fields:
        if field in test_examples:
            value = test_examples[field]
            print(f"\n# Test with {field} conditioning:")
            print(f"python multi_gpu_inference.py \\")
            print(f"  --output_base_path \"results/{field}_conditioned_64_2gpu\" \\")
            print(f"  --pretrained_name \"mattergen_base\" \\")
            print(f"  --total_structures 64 \\")
            print(f"  --num_gpus 2 \\")
            print(f"  --base_batch_size 16 \\")
            print(f"  --sampling_config_name \"default\" \\")
            print(f"  --properties_to_condition_on '{{\"{{field}}\": \"{value}\"}}' \\")
            print(f"  --enable_optimizations true \\")
            print(f"  --enable_model_compilation true \\")
            print(f"  --enable_graph_caching true \\")
            print(f"  --record_trajectories false")

if __name__ == "__main__":
    test_supported_conditioning()
