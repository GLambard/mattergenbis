# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
from pathlib import Path
from typing import Literal
from mattergen.diffusion.diffusion_loss import make_combined_loss


import fire

from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_classes import PRETRAINED_MODEL_NAME, MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator


def main(
    output_path: str,
    pretrained_name: PRETRAINED_MODEL_NAME | None = None,
    model_path: str | None = None,
    batch_size: int = 64,
    num_batches: int = 1,
    config_overrides: list[str] | None = None,
    checkpoint_epoch: Literal["best", "last"] | int = "last",
    properties_to_condition_on: TargetProperty | None = None,
    sampling_config_path: str | None = None,
    sampling_config_name: str = "default",
    sampling_config_overrides: list[str] | None = None,
    record_trajectories: bool = True,
    diffusion_guidance_factor: float | None = None,
    strict_checkpoint_loading: bool = True,
    target_compositions: list[dict[str, int]] | None = None,
    guidance: dict | None = None,
    diffusion_loss_weight: float = 1.0,
    print_loss: bool = False,
    # NEW: Performance optimization options (Phase 2 & 3)
    enable_optimizations: bool = True,
    enable_mixed_precision: bool = True,
    enable_model_compilation: bool = True,
    # Phase 3 options
    enable_multi_gpu: bool = True,
    enable_graph_caching: bool = True,
    enable_gradient_checkpointing: bool = False,
    multi_gpu_strategy: str = "auto",  # "auto", "dp", "ddp", "single"
    max_gpus: int | None = None,
    max_memory_usage_gb: float | None = None,
    print_optimization_info: bool = False,
    # NEW: Phase 3 advanced optimization options
    enable_phase3_optimizations: bool = False,
    optimize_memory: bool = False,
    enable_advanced_caching: bool = False,
    use_intelligent_batching: bool = False,
    hardware_optimization_level: str = "auto",  # "auto", "aggressive", "conservative"
    # NEW: Phase 4 adaptive sampling options
    enable_phase4_features: bool = False,
    enable_adaptive_sampling: bool = False,
    enable_quality_metrics: bool = False,
    adaptive_config_path: str | None = None,
    quality_threshold: float = 0.7,
    max_adaptation_iterations: int = 5,
    # NEW: Phase 4.2 advanced quality enhancement options
    enable_advanced_quality: bool = False,
    enable_quality_prediction: bool = False,
    enable_quality_reporting: bool = False,
    enable_trend_analysis: bool = False,
    quality_model_path: str | None = None,
    quality_report_format: str = "json",
    # NEW: Phase 4.3 enterprise monitoring options
    enable_enterprise_monitoring: bool = False,
    enable_enterprise_dashboard: bool = False,
    enable_enterprise_analytics: bool = False,
    enterprise_config_path: str | None = None
):
    """
    Evaluate diffusion model against molecular metrics.

    Args:
        model_path: Path to DiffusionLightningModule checkpoint directory.
        output_path: Path to output directory.
        config_overrides: Overrides for the model config, e.g., `model.num_layers=3 model.hidden_dim=128`.
        properties_to_condition_on: Property value to draw conditional sampling with respect to. When this value is an empty dictionary (default), unconditional samples are drawn.
        sampling_config_path: Path to the sampling config file. (default: None, in which case we use `DEFAULT_SAMPLING_CONFIG_PATH` from explorers.common.utils.utils.py)
        sampling_config_name: Name of the sampling config (corresponds to `{sampling_config_path}/{sampling_config_name}.yaml` on disk). (default: default)
        sampling_config_overrides: Overrides for the sampling config, e.g., `condition_loader_partial.batch_size=32`.
        load_epoch: Epoch to load from the checkpoint. If None, the best epoch is loaded. (default: None)
        record: Whether to record the trajectories of the generated structures. (default: True)
        strict_checkpoint_loading: Whether to raise an exception when not all parameters from the checkpoint can be matched to the model.
        target_compositions: List of dictionaries with target compositions to condition on. Each dictionary should have the form `{element: number_of_atoms}`. If None, the target compositions are not conditioned on.
           Only supported for models trained for crystal structure prediction (CSP) (default: None)
        diffusion_loss

    NOTE: When specifying dictionary values via the CLI, make sure there is no whitespace between the key and value, e.g., `--properties_to_condition_on={key1:value1}`.
    """
    assert (
        pretrained_name is not None or model_path is not None
    ), "Either pretrained_name or model_path must be provided."
    assert (
        pretrained_name is None or model_path is None
    ), "Only one of pretrained_name or model_path can be provided."

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sampling_config_overrides = sampling_config_overrides or []
    config_overrides = config_overrides or []
    # Disable generating element types which are not supported or not in the desired chemical
    # system (if provided).
    config_overrides += [
        "++lightning_module.diffusion_module.model.element_mask_func={_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
    ]
    properties_to_condition_on = properties_to_condition_on or {}
    target_compositions = target_compositions or []

    if pretrained_name is not None:
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
            pretrained_name, config_overrides=config_overrides
        )
    else:
        checkpoint_info = MatterGenCheckpointInfo(
            model_path=Path(model_path).resolve(),
            load_epoch=checkpoint_epoch,
            config_overrides=config_overrides,
            strict_checkpoint_loading=strict_checkpoint_loading,
        )
    _sampling_config_path = Path(sampling_config_path) if sampling_config_path is not None else None

    loss_fn = None
    if guidance is not None:
        # Ensure guidance is a dictionary with string keys and numeric values
        if not isinstance(guidance, dict) or not all(
            isinstance(k, str) and (isinstance(v, (int, float)) or isinstance(v, list))
            for k, v in guidance.items()
        ):
            raise ValueError(
                "Guidance must be a dictionary with string keys and numeric values or lists."
            )
        # Create the combined loss function based on the provided guidance
        loss_fn = make_combined_loss(guidance)

    # NEW: Print optimization info if requested
    if print_optimization_info:
        from mattergen.common.utils.performance_optimizer import print_optimization_info
        print_optimization_info()

    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        properties_to_condition_on=properties_to_condition_on,
        batch_size=batch_size,
        num_batches=num_batches,
        sampling_config_name=sampling_config_name,
        sampling_config_path=_sampling_config_path,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
        diffusion_guidance_factor=(
            diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
        ),
        target_compositions_dict=target_compositions,
        diffusion_loss_fn=loss_fn,           # NEW
        diffusion_loss_weight=diffusion_loss_weight,   # NEW
        print_loss=print_loss,  # NEW
        # Performance optimization settings (Phase 2 & 3)
        enable_performance_optimizations=enable_optimizations,
        enable_mixed_precision=enable_mixed_precision,
        enable_model_compilation=enable_model_compilation,
        enable_multi_gpu=enable_multi_gpu,
        enable_graph_caching=enable_graph_caching,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        multi_gpu_strategy=multi_gpu_strategy,
        max_gpus=max_gpus,
        max_memory_usage_gb=max_memory_usage_gb,
    )
    
    # Phase 3: Apply advanced optimizations if enabled
    phase3_optimizer = None
    if enable_phase3_optimizations:
        try:
            from mattergen.common.utils.performance_optimizer import Phase3PerformanceOptimizer
            phase3_optimizer = Phase3PerformanceOptimizer()
            print("INFO: Phase 3 advanced optimizations enabled")
        except ImportError as e:
            print(f"WARNING: Phase 3 optimizations not available: {e}")
            enable_phase3_optimizations = False

    # Phase 4: Initialize adaptive sampling and quality metrics if enabled
    adaptive_sampler = None
    quality_metrics = None
    phase4_integration = None
    
    if enable_phase4_features:
        try:
            from mattergen.common.utils.phase4_integration import Phase4IntegrationManager
            from mattergen.common.utils.adaptive_sampler import AdaptiveSampler, AdaptiveSamplingConfig
            from mattergen.common.utils.quality_metrics import StructureQualityAssessor, QualityThresholds
            
            # Initialize Phase 4 integration manager
            phase4_integration = Phase4IntegrationManager(
                enable_adaptive_sampling=enable_adaptive_sampling,
                enable_quality_assessment=enable_quality_metrics,
                config_path=adaptive_config_path
            )
            
            if enable_adaptive_sampling:
                # Create adaptive sampling config
                adaptive_config = AdaptiveSamplingConfig()
                adaptive_config.max_iterations = max_adaptation_iterations
                adaptive_sampler = AdaptiveSampler(config=adaptive_config)
                print("INFO: Phase 4 adaptive sampling enabled")
            
            if enable_quality_metrics:
                # Create quality thresholds with custom threshold
                quality_thresholds = QualityThresholds()
                quality_thresholds.overall_quality_threshold = quality_threshold
                quality_metrics = StructureQualityAssessor(thresholds=quality_thresholds)
                print("INFO: Phase 4 quality metrics enabled")
                
            # Phase 4.2: Initialize advanced quality features if enabled
            advanced_quality_metrics = None
            quality_reporter = None
            
            if enable_advanced_quality or enable_quality_prediction or enable_quality_reporting:
                try:
                    from mattergen.common.utils.advanced_quality_metrics import AdvancedQualityMetrics
                    from mattergen.common.utils.quality_reporting import QualityReporter
                    
                    if enable_advanced_quality or enable_quality_prediction:
                        advanced_quality_metrics = AdvancedQualityMetrics(
                            enable_ml_prediction=enable_quality_prediction,
                            enable_trend_analysis=enable_trend_analysis,
                            model_path=quality_model_path,
                            output_dir=output_path
                        )
                        print("INFO: Phase 4.2 advanced quality metrics enabled")
                    
                    if enable_quality_reporting:
                        quality_reporter = QualityReporter(
                            output_dir=output_path,
                            enable_visualizations=True
                        )
                        print("INFO: Phase 4.2 quality reporting enabled")
                        
                except ImportError as e:
                    print(f"WARNING: Phase 4.2 advanced quality features not available: {e}")
                    enable_advanced_quality = False
                    enable_quality_prediction = False
                    enable_quality_reporting = False
                
        except ImportError as e:
            print(f"WARNING: Phase 4 features not available: {e}")
            enable_phase4_features = False

    # Phase 4.3: Initialize enterprise monitoring if enabled
    enterprise_manager = None
    if enable_enterprise_monitoring or enable_enterprise_dashboard or enable_enterprise_analytics:
        try:
            from mattergen.enterprise.integration import initialize_enterprise, start_enterprise
            
            # Load enterprise configuration
            enterprise_config = {}
            if enterprise_config_path:
                import json
                with open(enterprise_config_path, 'r') as f:
                    enterprise_config = json.load(f)
            
            # Initialize enterprise manager
            enterprise_manager = initialize_enterprise(
                config=enterprise_config,
                enable_monitoring=enable_enterprise_monitoring,
                enable_dashboard=enable_enterprise_dashboard,
                enable_analytics=enable_enterprise_analytics
            )
            
            # Start enterprise services
            start_enterprise()
            print("INFO: Phase 4.3 enterprise monitoring enabled")
            
        except ImportError as e:
            print(f"WARNING: Phase 4.3 enterprise features not available: {e}")
            print("INFO: Install enterprise dependencies with: pip install -r requirements-enterprise.txt")
            enable_enterprise_monitoring = False
            enable_enterprise_dashboard = False
            enable_enterprise_analytics = False
        except Exception as e:
            print(f"WARNING: Failed to initialize enterprise features: {e}")
            enterprise_manager = None

    try:
        # Phase 4: Adaptive generation with quality assessment
        if enable_phase4_features and (adaptive_sampler or quality_metrics):
            print("INFO: Starting Phase 4 adaptive generation process...")
            
            # Initialize generation parameters
            current_guidance_factor = diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
            total_generated = 0
            adaptation_iteration = 0
            
            while total_generated < (batch_size * num_batches) and adaptation_iteration < max_adaptation_iterations:
                print(f"\nAdaptation iteration {adaptation_iteration + 1}/{max_adaptation_iterations}")
                print(f"Current guidance factor: {current_guidance_factor}")
                
                # Generate batch with current parameters
                batch_start_time = time.time()
                
                # Enterprise monitoring: Record batch start
                if enterprise_manager:
                    enterprise_manager.on_generation_start(
                        batch_id=adaptation_iteration, 
                        gpu_id=0,  # Default GPU ID for single GPU generation
                        batch_size=batch_size
                    )
                
                temp_generator = CrystalGenerator(
                    checkpoint_info=checkpoint_info,
                    properties_to_condition_on=properties_to_condition_on,
                    batch_size=batch_size,
                    num_batches=1,  # Generate one batch at a time for adaptation
                    sampling_config_name=sampling_config_name,
                    sampling_config_path=_sampling_config_path,
                    sampling_config_overrides=sampling_config_overrides,
                    record_trajectories=record_trajectories,
                    diffusion_guidance_factor=current_guidance_factor,
                    target_compositions_dict=target_compositions,
                    diffusion_loss_fn=loss_fn,
                    diffusion_loss_weight=diffusion_loss_weight,
                    print_loss=print_loss,
                    enable_performance_optimizations=enable_optimizations,
                    enable_mixed_precision=enable_mixed_precision,
                    enable_model_compilation=enable_model_compilation,
                    enable_multi_gpu=enable_multi_gpu,
                    enable_graph_caching=enable_graph_caching,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    multi_gpu_strategy=multi_gpu_strategy,
                    max_gpus=max_gpus,
                    max_memory_usage_gb=max_memory_usage_gb,
                )
                
                batch_structures = temp_generator.generate(output_dir=Path(output_path))
                temp_generator.cleanup()
                
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                structures_generated = len(batch_structures)
                throughput = structures_generated / batch_duration if batch_duration > 0 else 0
                
                # Assess quality if enabled
                if quality_metrics:
                    quality_score, quality_report = quality_metrics.assess_structures(batch_structures)
                    print(f"Batch quality score: {quality_score:.3f}")
                    
                    # Filter structures based on quality
                    filtered_structures = quality_metrics.filter_structures(batch_structures)
                    print(f"Filtered structures: {len(filtered_structures)}/{len(batch_structures)} passed quality threshold")
                else:
                    quality_score = 1.0
                    quality_report = {}
                    filtered_structures = batch_structures
                
                # Enterprise monitoring: Record batch completion
                if enterprise_manager:
                    enterprise_manager.on_generation_complete(
                        batch_id=adaptation_iteration,
                        gpu_id=0,  # Default GPU ID
                        structures_generated=structures_generated,
                        batch_duration=batch_duration,
                        quality_score=quality_score,
                        throughput=throughput,
                        memory_peak=0.0,  # TODO: Add memory tracking
                        error_count=0  # TODO: Add error tracking
                    )
                    
                    # Record quality assessment
                    if quality_metrics and hasattr(quality_report, 'get'):
                        enterprise_manager.on_quality_assessment(
                            batch_id=adaptation_iteration,
                            overall_quality=quality_score,
                            quality_distribution=quality_report.get('distribution', {}),
                            improvement_rate=quality_report.get('improvement_rate', 0.0),
                            trend=quality_report.get('trend', 'stable'),
                            convergence_status=quality_report.get('convergence', 'unknown')
                        )
                
                # Phase 4.2: Advanced quality analysis if enabled
                advanced_quality_results = None
                if advanced_quality_metrics and len(filtered_structures) > 0:
                    try:
                        # Perform advanced quality analysis
                        advanced_quality_results = advanced_quality_metrics.analyze_structures(
                            structures=filtered_structures,
                            batch_id=adaptation_iteration,
                            metadata={'guidance_factor': current_guidance_factor}
                        )
                        
                        # ML-based quality prediction if enabled
                        if enable_quality_prediction:
                            prediction_results = advanced_quality_metrics.predict_quality(filtered_structures)
                            print(f"ML quality predictions: mean={prediction_results['mean_predicted_quality']:.3f}")
                        
                        # Update trend analysis
                        if enable_trend_analysis:
                            trend_data = advanced_quality_metrics.update_trends(
                                quality_score=quality_score,
                                structures=filtered_structures,
                                iteration=adaptation_iteration
                            )
                            
                        print(f"Advanced quality analysis complete for {len(filtered_structures)} structures")
                        
                    except Exception as e:
                        print(f"WARNING: Advanced quality analysis failed: {e}")
                        advanced_quality_results = None
                
                # Adapt sampling parameters if enabled
                if adaptive_sampler:
                    adaptation_result = adaptive_sampler.adapt_parameters(
                        structures=filtered_structures,
                        quality_score=quality_score,
                        current_guidance_factor=current_guidance_factor,
                        iteration=adaptation_iteration
                    )
                    
                    current_guidance_factor = adaptation_result['new_guidance_factor']
                    should_continue = adaptation_result['should_continue']
                    
                    print(f"Adaptation result: {adaptation_result}")
                    
                    if not should_continue:
                        print("INFO: Adaptive sampling converged, stopping early")
                        break
                
                total_generated += len(filtered_structures)
                adaptation_iteration += 1
                
                # Check if we have enough structures
                if total_generated >= (batch_size * num_batches):
                    break
            
            # Final generation to reach target count if needed
            remaining = (batch_size * num_batches) - total_generated
            if remaining > 0:
                print(f"\nGenerating final {remaining} structures...")
                final_generator = CrystalGenerator(
                    checkpoint_info=checkpoint_info,
                    properties_to_condition_on=properties_to_condition_on,
                    batch_size=remaining,
                    num_batches=1,
                    sampling_config_name=sampling_config_name,
                    sampling_config_path=_sampling_config_path,
                    sampling_config_overrides=sampling_config_overrides,
                    record_trajectories=record_trajectories,
                    diffusion_guidance_factor=current_guidance_factor,
                    target_compositions_dict=target_compositions,
                    diffusion_loss_fn=loss_fn,
                    diffusion_loss_weight=diffusion_loss_weight,
                    print_loss=print_loss,
                    enable_performance_optimizations=enable_optimizations,
                    enable_mixed_precision=enable_mixed_precision,
                    enable_model_compilation=enable_model_compilation,
                    enable_multi_gpu=enable_multi_gpu,
                    enable_graph_caching=enable_graph_caching,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    multi_gpu_strategy=multi_gpu_strategy,
                    max_gpus=max_gpus,
                    max_memory_usage_gb=max_memory_usage_gb,
                )
                final_structures = final_generator.generate(output_dir=Path(output_path))
                final_generator.cleanup()
                total_generated += len(final_structures)
            
            print(f"\nPhase 4 adaptive generation complete! Generated {total_generated} structures in {adaptation_iteration} iterations.")
            generated_structures = []  # Structures already saved in adaptive process
            
        else:
            # Standard generation (Phase 1-3)
            generated_structures = generator.generate(output_dir=Path(output_path))
        
        # Print final optimization stats
        if print_optimization_info:
            print("\nFinal Optimization Statistics:")
            opt_info = generator.get_optimization_info()
            for key, value in opt_info.items():
                print(f"  {key}: {value}")
        
        if not enable_phase4_features or not (adaptive_sampler or quality_metrics):
            print(f"\nGeneration complete! Generated {len(generated_structures)} structures.")
        
        # Phase 4.2: Generate final quality report if enabled
        if enable_quality_reporting and quality_reporter and enable_phase4_features:
            try:
                print("\nGenerating comprehensive quality report...")
                
                # Collect all quality data from advanced metrics
                quality_data = {}
                if advanced_quality_metrics:
                    quality_data = advanced_quality_metrics.get_comprehensive_results()
                
                # Generate comprehensive report
                report = quality_reporter.generate_report_from_quality_data(
                    quality_data=quality_data,
                    generation_metadata={
                        'total_structures': total_generated if enable_phase4_features else len(generated_structures),
                        'enable_adaptive_sampling': enable_adaptive_sampling,
                        'enable_quality_metrics': enable_quality_metrics,
                        'enable_advanced_quality': enable_advanced_quality,
                        'quality_threshold': quality_threshold,
                        'adaptation_iterations': adaptation_iteration if enable_phase4_features else 0
                    }
                )
                
                # Save report
                report_path = quality_reporter.save_report(report)
                print(f"Quality report saved to: {report_path}")
                
                # Generate visualizations if requested
                if hasattr(quality_reporter, 'generate_visualizations'):
                    viz_paths = quality_reporter.generate_visualizations(quality_data)
                    print(f"Visualizations saved to: {viz_paths}")
                    
            except Exception as e:
                print(f"WARNING: Quality report generation failed: {e}")
        
    finally:
        # Clean up resources
        generator.cleanup()


def _main():
    # Use fire to allow for the specification of dictionary values via the CLI
    fire.Fire(main)


if __name__ == "__main__":
    _main()
#mattergen-generate "results/chemical_system/Pd-Ni-H_test"   --pretrained-name=chemical_system   --batch_size=1   --properties_to_condition_on="{'chemical_system':'Pd-Ni-H'}"   --record_trajectories=False   --diffusion_guidance_factor=2.0   --guidance="{'volume': 30.935}"   --diffusion_loss_weight=1   --print_loss=True