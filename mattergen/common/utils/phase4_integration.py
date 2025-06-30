"""
Phase 4 Adaptive Sampling Integration
====================================

This module integrates adaptive sampling intelligence with the MatterGen generation pipeline,
providing seamless quality-driven optimization and performance enhancement.

Key Features:
- Integration with existing generate.py script
- Adaptive sampling parameter adjustment
- Quality-driven early stopping
- Real-time performance optimization
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import yaml

# Import Phase 4 modules
from .adaptive_sampler import AdaptiveSampler, AdaptiveSamplingConfig, ConvergenceMetrics, create_quality_metrics
from .quality_metrics import StructureQualityAssessor, QualityThresholds, QualityMetrics, create_quality_assessor

logger = logging.getLogger(__name__)


class Phase4IntegrationManager:
    """Main manager for Phase 4 adaptive optimizations."""
    
    def __init__(self, 
                 config_path: Optional[Path] = None,
                 enable_adaptive_sampling: bool = True,
                 enable_quality_assessment: bool = True):
        self.config_path = config_path
        self.enable_adaptive_sampling = enable_adaptive_sampling
        self.enable_quality_assessment = enable_quality_assessment
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.adaptive_sampler = None
        self.quality_assessor = None
        self._initialize_components()
        
        # Performance tracking
        self.optimization_stats = {
            'total_structures_generated': 0,
            'total_optimization_time': 0.0,
            'adaptive_adjustments': 0,
            'quality_improvements': 0,
            'early_stops': 0,
            'convergence_detections': 0,
            'phase4_enabled': True,
            'start_time': time.time()
        }
        
        logger.info("Phase 4 Optimization Manager initialized")
        logger.info(f"Adaptive sampling: {'enabled' if enable_adaptive_sampling else 'disabled'}")
        logger.info(f"Quality assessment: {'enabled' if enable_quality_assessment else 'disabled'}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load Phase 4 configuration."""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded Phase 4 config from: {self.config_path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Phase 4 configuration."""
        return {
            'adaptive_sampling': {
                'enabled': True,
                'min_steps': 50,
                'max_steps': 500,
                'initial_steps': 200,
                'quality_threshold': 0.85,
                'early_stopping': {'enabled': True, 'patience': 20}
            },
            'quality_assessment': {
                'enabled': True,
                'thresholds': {
                    'min_overall_quality': 0.7,
                    'min_geometric_quality': 0.6,
                    'require_valid_cell': True
                }
            },
            'monitoring': {
                'enabled': True,
                'log_adaptive_decisions': True,
                'generate_optimization_report': True
            }
        }
    
    def _initialize_components(self):
        """Initialize adaptive sampling and quality assessment components."""
        # Initialize adaptive sampler
        if self.enable_adaptive_sampling and self.config.get('adaptive_sampling', {}).get('enabled', True):
            adaptive_config = AdaptiveSamplingConfig(
                min_steps=self.config['adaptive_sampling'].get('min_steps', 50),
                max_steps=self.config['adaptive_sampling'].get('max_steps', 500),
                initial_steps=self.config['adaptive_sampling'].get('initial_steps', 200),
                quality_threshold=self.config['adaptive_sampling'].get('quality_threshold', 0.85),
                enable_early_stopping=self.config['adaptive_sampling'].get('early_stopping', {}).get('enabled', True),
                early_stop_patience=self.config['adaptive_sampling'].get('early_stopping', {}).get('patience', 20)
            )
            self.adaptive_sampler = AdaptiveSampler(adaptive_config)
            logger.info("Adaptive sampler initialized")
        
        # Initialize quality assessor
        if self.enable_quality_assessment and self.config.get('quality_assessment', {}).get('enabled', True):
            quality_thresholds = QualityThresholds(
                min_overall_quality=self.config['quality_assessment']['thresholds'].get('min_overall_quality', 0.7),
                min_geometric_quality=self.config['quality_assessment']['thresholds'].get('min_geometric_quality', 0.6),
                require_valid_cell=self.config['quality_assessment']['thresholds'].get('require_valid_cell', True),
                require_no_overlaps=self.config['quality_assessment']['thresholds'].get('require_no_overlaps', True)
            )
            self.quality_assessor = StructureQualityAssessor(quality_thresholds)
            logger.info("Quality assessor initialized")
    
    def optimize_sampling_parameters(self, 
                                   current_batch: int,
                                   total_batches: int,
                                   current_loss: float,
                                   generated_structures: List[Dict],
                                   current_steps: int) -> Tuple[int, Dict[str, Any]]:
        """Optimize sampling parameters based on current generation state."""
        optimization_start = time.time()
        optimization_info = {
            'phase4_active': True,
            'adaptive_sampling_active': self.adaptive_sampler is not None,
            'quality_assessment_active': self.quality_assessor is not None,
            'original_steps': current_steps,
            'optimization_decisions': []
        }
        
        optimized_steps = current_steps
        
        if self.adaptive_sampler:
            # Assess structure quality for adaptive decisions
            quality_scores = []
            if generated_structures and self.quality_assessor:
                for i, structure in enumerate(generated_structures):
                    try:
                        quality_metrics = self.quality_assessor.assess_structure_quality(
                            structure, 
                            structure_id=f"batch_{current_batch}_struct_{i}"
                        )
                        quality_scores.append(quality_metrics.overall_quality)
                    except Exception as e:
                        logger.warning(f"Quality assessment failed for structure {i}: {e}")
                        quality_scores.append(0.5)  # Default poor quality
            
            # Create convergence metrics
            avg_quality = np.mean(quality_scores) if quality_scores else 0.5
            convergence_metrics = create_quality_metrics(
                structure_data={'quality_scores': quality_scores},
                loss_value=current_loss,
                step=current_batch
            )
            convergence_metrics.quality_score = avg_quality
            
            # Adapt sampling steps
            optimized_steps = self.adaptive_sampler.adapt_sampling_steps(
                convergence_metrics,
                target_quality=self.adaptive_sampler.config.quality_threshold
            )
            
            # Check for progressive refinement
            if self.adaptive_sampler.should_use_progressive_refinement(current_batch, total_batches):
                stage = min(2, current_batch // (total_batches // 3))  # 3 stages
                progressive_steps = self.adaptive_sampler.get_progressive_stage_steps(stage)
                optimized_steps = min(optimized_steps, progressive_steps)
                
                optimization_info['optimization_decisions'].append({
                    'type': 'progressive_refinement',
                    'stage': stage,
                    'progressive_steps': progressive_steps
                })
            
            # Log optimization decisions
            if optimized_steps != current_steps:
                decision_type = 'increase' if optimized_steps > current_steps else 'decrease'
                optimization_info['optimization_decisions'].append({
                    'type': 'step_adjustment',
                    'direction': decision_type,
                    'reason': f'quality_score={avg_quality:.3f}, convergence_detected={convergence_metrics.convergence_rate > 0.9}',
                    'quality_score': avg_quality,
                    'convergence_rate': convergence_metrics.convergence_rate
                })
                
                self.optimization_stats['adaptive_adjustments'] += 1
                
                if self.config.get('monitoring', {}).get('log_adaptive_decisions', True):
                    logger.info(f"Phase 4: Adapting steps from {current_steps} to {optimized_steps} "
                               f"(quality: {avg_quality:.3f}, batch: {current_batch}/{total_batches})")
        
        # Update optimization statistics
        optimization_time = time.time() - optimization_start
        self.optimization_stats['total_optimization_time'] += optimization_time
        
        optimization_info.update({
            'optimized_steps': optimized_steps,
            'step_change': optimized_steps - current_steps,
            'optimization_time_ms': round(optimization_time * 1000, 2),
            'average_quality': np.mean(quality_scores) if quality_scores else None,
            'quality_assessment_count': len(quality_scores)
        })
        
        return optimized_steps, optimization_info
    
    def filter_structures_by_quality(self, 
                                   structures: List[Dict],
                                   structure_ids: Optional[List[str]] = None) -> Tuple[List[Dict], List[QualityMetrics]]:
        """Filter structures based on quality assessment."""
        if not self.quality_assessor:
            return structures, []
        
        filtered_structures = []
        quality_metrics_list = []
        
        for i, structure in enumerate(structures):
            structure_id = structure_ids[i] if structure_ids else f"struct_{i}"
            
            try:
                quality_metrics = self.quality_assessor.assess_structure_quality(
                    structure, structure_id=structure_id
                )
                quality_metrics_list.append(quality_metrics)
                
                # Check if structure should be kept
                if self.quality_assessor.should_keep_structure(quality_metrics):
                    filtered_structures.append(structure)
                    self.optimization_stats['quality_improvements'] += 1
                else:
                    logger.debug(f"Filtered out structure {structure_id} "
                               f"(quality: {quality_metrics.overall_quality:.3f})")
            
            except Exception as e:
                logger.warning(f"Quality filtering failed for structure {structure_id}: {e}")
                # Keep structure if assessment fails
                filtered_structures.append(structure)
        
        if len(filtered_structures) != len(structures):
            filter_rate = (len(structures) - len(filtered_structures)) / len(structures) * 100
            logger.info(f"Phase 4: Filtered {len(structures) - len(filtered_structures)} structures "
                       f"({filter_rate:.1f}% filter rate)")
        
        return filtered_structures, quality_metrics_list
    
    def update_generation_stats(self, structures_generated: int, batch_time: float):
        """Update generation statistics."""
        self.optimization_stats['total_structures_generated'] += structures_generated
        
        # Update component stats
        if self.adaptive_sampler:
            self.adaptive_sampler.update_sampling_stats(structures_generated)
            sampler_stats = self.adaptive_sampler.get_optimization_summary()
            self.optimization_stats.update({
                'early_stops': sampler_stats.get('early_stops_triggered', 0),
                'convergence_detections': sampler_stats.get('convergence_detections', 0)
            })
    
    def get_phase4_summary(self) -> Dict[str, Any]:
        """Get comprehensive Phase 4 optimization summary."""
        total_time = time.time() - self.optimization_stats['start_time']
        
        summary = {
            'phase4_optimization_summary': {
                'total_runtime_seconds': round(total_time, 2),
                'total_structures_generated': self.optimization_stats['total_structures_generated'],
                'total_optimization_time_seconds': round(self.optimization_stats['total_optimization_time'], 3),
                'optimization_overhead_percent': round(
                    self.optimization_stats['total_optimization_time'] / total_time * 100, 2
                ) if total_time > 0 else 0,
                'adaptive_adjustments_made': self.optimization_stats['adaptive_adjustments'],
                'quality_improvements': self.optimization_stats['quality_improvements'],
                'early_stops_triggered': self.optimization_stats['early_stops'],
                'convergence_detections': self.optimization_stats['convergence_detections']
            }
        }
        
        # Add adaptive sampler summary
        if self.adaptive_sampler:
            summary['adaptive_sampling_summary'] = self.adaptive_sampler.get_optimization_summary()
        
        # Add quality assessor summary
        if self.quality_assessor:
            summary['quality_assessment_summary'] = self.quality_assessor.get_quality_summary()
        
        return summary
    
    def save_optimization_report(self, output_path: Path):
        """Save comprehensive Phase 4 optimization report."""
        report = self.get_phase4_summary()
        report.update({
            'configuration': self.config,
            'components_enabled': {
                'adaptive_sampling': self.adaptive_sampler is not None,
                'quality_assessment': self.quality_assessor is not None
            },
            'timestamp': time.time()
        })
        
        # Save main report
        report_path = output_path / 'phase4_optimization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Phase 4 optimization report saved to: {report_path}")
        
        # Save individual component reports
        if self.adaptive_sampler:
            self.adaptive_sampler.save_optimization_report(output_path)
        
        # Log summary
        summary = report['phase4_optimization_summary']
        logger.info(f"Phase 4 Summary: {summary['total_structures_generated']} structures, "
                   f"{summary['adaptive_adjustments_made']} adaptive adjustments, "
                   f"{summary['optimization_overhead_percent']:.1f}% overhead")


# Factory function for easy integration
def create_phase4_manager(
    config_path: Optional[Path] = None,
    enable_adaptive_sampling: bool = True,
    enable_quality_assessment: bool = True,
    min_steps: int = 50,
    max_steps: int = 500,
    quality_threshold: float = 0.85
) -> Phase4IntegrationManager:
    """Factory function to create Phase 4 optimization manager."""
    
    # Create temporary config if none provided
    if config_path is None:
        temp_config = {
            'adaptive_sampling': {
                'enabled': enable_adaptive_sampling,
                'min_steps': min_steps,
                'max_steps': max_steps,
                'initial_steps': 200,
                'quality_threshold': quality_threshold,
                'early_stopping': {'enabled': True, 'patience': 20}
            },
            'quality_assessment': {
                'enabled': enable_quality_assessment,
                'thresholds': {
                    'min_overall_quality': 0.7,
                    'require_valid_cell': True
                }
            },
            'monitoring': {
                'enabled': True,
                'log_adaptive_decisions': True
            }
        }
        
        manager = Phase4IntegrationManager(
            enable_adaptive_sampling=enable_adaptive_sampling,
            enable_quality_assessment=enable_quality_assessment
        )
        manager.config = temp_config
        manager._initialize_components()
        return manager
    
    return Phase4IntegrationManager(
        config_path=config_path,
        enable_adaptive_sampling=enable_adaptive_sampling,
        enable_quality_assessment=enable_quality_assessment
    )


# Add numpy import for compatibility
try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    class np:
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0
        
        @staticmethod
        def std(x):
            if not x:
                return 0
            mean_val = sum(x) / len(x)
            return (sum((xi - mean_val) ** 2 for xi in x) / len(x)) ** 0.5
        
        @staticmethod
        def diff(x):
            return [x[i+1] - x[i] for i in range(len(x)-1)]
