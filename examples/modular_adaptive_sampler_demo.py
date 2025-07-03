#!/usr/bin/env python3
"""
Modular Adaptive Sampler Usage Examples
=======================================

This script demonstrates how to use the new modular adaptive sampler
architecture with different configurations and components.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mattergen.common.utils.adaptive_sampler import (
    ModularAdaptiveSampler,
    AdaptiveSampler,
    create_adaptive_sampler,
    create_full_adaptive_pipeline,
    SamplingPresets,
    AdaptiveSamplingConfig,
    ConvergenceMetrics,
    QualityAssessmentAdapter,
    EnterpriseCallbackAdapter,
    create_sampling_strategy,
    create_convergence_detector
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Create a simple adaptive sampler (legacy compatible)
    sampler = create_adaptive_sampler(
        quality_threshold=0.8,
        sampling_strategy="adaptive"
    )
    
    print(f"Created sampler: {type(sampler).__name__}")
    print(f"Configuration: {sampler.config}")
    
    # Simulate some adaptation iterations
    for i in range(3):
        # Simulate structure data
        structures = [f"structure_{i}_{j}" for j in range(10)]
        quality_score = 0.6 + (i * 0.1)  # Improving quality
        guidance_factor = 1.0
        
        result = sampler.adapt_parameters(
            structures=structures,
            quality_score=quality_score,
            current_guidance_factor=guidance_factor,
            iteration=i
        )
        
        print(f"Iteration {i}: Quality={quality_score:.2f}, "
              f"New guidance={result['new_guidance_factor']:.2f}, "
              f"Continue={result['should_continue']}")
    
    print(f"Final summary: {sampler.get_optimization_summary()}")


def example_modular_configuration():
    """Example 2: Modular configuration with custom components."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Modular Configuration")
    print("="*60)
    
    # Create a modular sampler with custom components
    config = SamplingPresets.aggressive()
    
    sampler = ModularAdaptiveSampler(
        config=config,
        sampling_strategy=create_sampling_strategy("aggressive", config),
        convergence_detector=create_convergence_detector("lenient", config)
    )
    
    print(f"Created modular sampler with components:")
    print(f"  - Sampling strategy: {type(sampler.sampling_strategy).__name__}")
    print(f"  - Convergence detector: {type(sampler.convergence_detector).__name__}")
    print(f"  - Component usage: {sampler.sampling_stats['component_usage']}")
    
    # Simulate adaptation with more aggressive settings
    for i in range(2):
        structures = [f"structure_{i}_{j}" for j in range(5)]
        quality_score = 0.85 + (i * 0.05)  # High quality
        
        result = sampler.adapt_parameters(
            structures=structures,
            quality_score=quality_score,
            current_guidance_factor=1.0,
            iteration=i
        )
        
        print(f"Iteration {i}: Quality={quality_score:.2f}, "
              f"Recommendations={result.recommendations}")


def example_quality_integration():
    """Example 3: Quality assessment integration."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Quality Assessment Integration")
    print("="*60)
    
    # Mock quality assessor
    class MockQualityAssessor:
        def assess_quality(self, structures):
            # Mock assessment
            quality = min(1.0, len(structures) * 0.1)
            report = {
                'method': 'mock_assessment',
                'structure_count': len(structures),
                'quality_distribution': {'high': 0.3, 'medium': 0.5, 'low': 0.2}
            }
            return quality, report
        
        def filter_structures(self, structures, threshold):
            # Mock filtering - keep 80% of structures
            return structures[:int(len(structures) * 0.8)]
    
    # Create sampler with quality integration
    quality_assessor = MockQualityAssessor()
    
    sampler = create_full_adaptive_pipeline(
        sampling_preset="production",
        quality_assessor=quality_assessor,
        enterprise_monitoring=False
    )
    
    print(f"Created pipeline with quality integration enabled")
    print(f"Quality adapter enabled: {sampler.quality_adapter.enabled}")
    
    # Test quality assessment
    structures = [f"structure_{i}" for i in range(20)]
    quality_score, quality_report = sampler.quality_adapter.assess_quality(structures)
    filtered_structures = sampler.quality_adapter.filter_structures(structures, 0.7)
    
    print(f"Quality assessment: {quality_score:.2f}")
    print(f"Quality report: {quality_report}")
    print(f"Filtered structures: {len(filtered_structures)}/{len(structures)}")


def example_enterprise_callbacks():
    """Example 4: Enterprise monitoring callbacks."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Enterprise Monitoring Integration")
    print("="*60)
    
    # Create enterprise callback adapter
    enterprise_callbacks = EnterpriseCallbackAdapter()
    
    # Register custom callbacks
    def monitoring_callback(event_type, **kwargs):
        print(f"📊 Enterprise Event: {event_type}")
        if event_type == 'adaptation_step':
            metrics = kwargs.get('metrics')
            result = kwargs.get('result')
            if metrics and result:
                print(f"   Quality: {metrics.quality_score:.3f}")
                print(f"   Action: {result.recommendations['action']}")
        elif event_type == 'adaptation_complete':
            stats = kwargs.get('stats', {})
            print(f"   Final stats: {stats}")
    
    def analytics_callback(event_type, **kwargs):
        print(f"📈 Analytics: Logged {event_type} with {len(kwargs)} parameters")
    
    enterprise_callbacks.register_callback(monitoring_callback)
    enterprise_callbacks.register_callback(analytics_callback)
    
    # Create sampler with enterprise integration
    config = SamplingPresets.production()
    
    sampler = ModularAdaptiveSampler(
        config=config,
        enterprise_callbacks=enterprise_callbacks
    )
    
    print(f"Enterprise callbacks enabled: {sampler.enterprise_callbacks.enabled}")
    print(f"Registered callbacks: {len(sampler.enterprise_callbacks.callbacks)}")
    
    # Trigger enterprise events
    sampler.enterprise_callbacks.on_adaptation_start(config, {'initial_guidance': 1.0})
    
    # Simulate adaptation with enterprise monitoring
    result = sampler.adapt_parameters(
        structures=['structure_1', 'structure_2'],
        quality_score=0.85,
        current_guidance_factor=1.0,
        iteration=0
    )
    
    sampler.enterprise_callbacks.on_adaptation_complete({'total_iterations': 1})


def example_preset_comparisons():
    """Example 5: Compare different presets."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Preset Comparisons")
    print("="*60)
    
    presets = ["conservative", "balanced", "aggressive", "production"]
    
    for preset_name in presets:
        sampler = create_full_adaptive_pipeline(sampling_preset=preset_name)
        config = sampler.config
        
        print(f"\n{preset_name.upper()} preset:")
        print(f"  Steps range: {config.min_steps}-{config.max_steps} (initial: {config.initial_steps})")
        print(f"  Quality threshold: {config.quality_threshold}")
        print(f"  Convergence threshold: {config.convergence_threshold}")
        print(f"  Strategy: {config.sampling_strategy}/{config.convergence_strategy}")
        print(f"  Adjustment factor: {config.step_adjustment_factor}")


def example_standalone_components():
    """Example 6: Using components standalone (without adaptive sampling)."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Standalone Component Usage")
    print("="*60)
    
    config = AdaptiveSamplingConfig()
    
    # Use convergence detector standalone
    detector = create_convergence_detector("standard", config)
    print(f"Standalone convergence detector: {type(detector).__name__}")
    
    # Create some test metrics
    for i in range(5):
        metrics = ConvergenceMetrics(
            step=i,
            loss_value=1.0 - (i * 0.1),
            loss_change=0.05,
            gradient_norm=0.1 - (i * 0.01),
            structure_stability=0.7 + (i * 0.05),
            quality_score=0.6 + (i * 0.08),
            convergence_rate=i * 0.2,
            early_stop_score=0.7 + (i * 0.03)
        )
        
        converged = detector.update_metrics(metrics)
        print(f"Step {i}: Quality={metrics.quality_score:.2f}, Converged={converged}")
        
        if converged:
            print("Convergence detected!")
            break
    
    # Use sampling strategy standalone
    strategy = create_sampling_strategy("adaptive", config)
    print(f"\nStandalone sampling strategy: {type(strategy).__name__}")
    
    current_steps = 200
    for metrics in getattr(detector, 'metrics_history', []):
        new_steps = strategy.adjust_steps(metrics, current_steps)
        should_continue = strategy.should_continue(metrics, metrics.step, 5)
        
        if new_steps != current_steps:
            print(f"Step adjustment: {current_steps} → {new_steps}")
            current_steps = new_steps
        
        if not should_continue:
            print("Strategy recommends stopping")
            break


def main():
    """Run all examples."""
    print("MODULAR ADAPTIVE SAMPLER EXAMPLES")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_modular_configuration()
        example_quality_integration()
        example_enterprise_callbacks()
        example_preset_comparisons()
        example_standalone_components()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
