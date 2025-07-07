#!/usr/bin/env python3
"""
Simple script to discover conditioning fields without full model loading
"""

# Based on what we know, let's create example commands for likely supported properties
# From the PROPERTY_SOURCE_IDS we saw earlier

LIKELY_SUPPORTED_FIELDS = [
    "space_group",
    "formation_energy_per_atom", 
    "energy_above_hull",
    "dft_band_gap",
    "ml_bulk_modulus",
    "dft_bulk_modulus",
    "dft_shear_modulus",
    "dft_mag_density",
    "hhi_score"
]

def create_test_commands():
    print("🧪 Test Commands for Likely Supported Conditioning Fields")
    print("=" * 60)
    
    test_values = {
        "space_group": "225",  # Face-centered cubic
        "formation_energy_per_atom": "-2.5",
        "energy_above_hull": "0.1", 
        "dft_band_gap": "1.5",
        "ml_bulk_modulus": "150",
        "dft_bulk_modulus": "200",
        "dft_shear_modulus": "80",
        "dft_mag_density": "1.2",
        "hhi_score": "0.3"
    }
    
    for field in LIKELY_SUPPORTED_FIELDS:
        if field in test_values:
            value = test_values[field]
            print(f"\n# Test {field} conditioning:")
            print(f"python multi_gpu_inference.py \\")
            print(f"  --output_base_path \"results/{field}_test_32_2gpu\" \\")
            print(f"  --pretrained_name \"mattergen_base\" \\")
            print(f"  --total_structures 32 \\")
            print(f"  --num_gpus 2 \\")
            print(f"  --base_batch_size 8 \\")
            print(f"  --sampling_config_name \"default\" \\")
            print(f"  --properties_to_condition_on '{{\"{{field}}\": \"{value}\"}}' \\")
            print(f"  --enable_optimizations true \\")
            print(f"  --record_trajectories false")

if __name__ == "__main__":
    create_test_commands()
    
    print(f"\n\n🎯 RECOMMENDATION:")
    print("=" * 30)
    print("Since 'chemical_system' is NOT supported by mattergen_base,")
    print("let's test with 'space_group' which is very likely to be supported:")
    print()
    print("python multi_gpu_inference.py \\")
    print("  --output_base_path \"results/space_group_test_32_2gpu\" \\") 
    print("  --pretrained_name \"mattergen_base\" \\")
    print("  --total_structures 32 \\")
    print("  --num_gpus 2 \\")
    print("  --base_batch_size 8 \\")
    print("  --sampling_config_name \"default\" \\")
    print("  --properties_to_condition_on '{\"space_group\": \"225\"}' \\")
    print("  --enable_optimizations true \\")
    print("  --record_trajectories false")
