#!/usr/bin/env python3
"""
Display the current high-volume benchmark configuration
"""

from crystalline_accuracy_validator import create_comprehensive_benchmark_suite
import json

def main():
    print("🚀 MatterGen High-Volume Benchmark Configuration")
    print("=" * 50)
    
    config = create_comprehensive_benchmark_suite()
    
    print("\n📊 PHASE CONFIGURATIONS:")
    total_structures = 0
    
    for phase_name, phase_config in config['phases'].items():
        structures = phase_config['structures']
        total_structures += structures
        
        print(f"\n{phase_config['name']}:")
        print(f"  • Structures: {structures}")
        print(f"  • GPUs: {phase_config['gpus']}")
        print(f"  • Batch Size: {phase_config['batch_size']}")
        print(f"  • Features: {', '.join(phase_config['features']) if phase_config['features'] else 'None'}")
    
    print(f"\n📈 TOTAL STRUCTURES: {total_structures}")
    print(f"💡 Statistical Robustness: {total_structures} structures across {len(config['phases'])} phases")
    
    print(f"\n🎯 VALIDATION METRICS:")
    for metric in config['validation_metrics']:
        print(f"  • {metric}")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    for metric in config['performance_metrics']:
        print(f"  • {metric}")
    
    print(f"\n🕒 ESTIMATED RUNTIME:")
    print(f"  • With timeout: 60 minutes per phase")
    print(f"  • Total phases: {len(config['phases'])}")
    print(f"  • Maximum total time: {60 * len(config['phases'])} minutes ({60 * len(config['phases']) / 60:.1f} hours)")
    
    print("\n✅ Configuration ready for robust statistical validation!")

if __name__ == "__main__":
    main()
