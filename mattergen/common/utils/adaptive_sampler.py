"""
Adaptive Sampling Intelligence for MatterGen
===========================================

This module implements intelligent adaptive sampling that dynamically adjusts
diffusion parameters based on convergence metrics and quality assessment.

Key Features:
- Dynamic step adjustment based on convergence detection
- Quality-aware early stopping
- Progressive refinement with multi-stage generation
- Real-time convergence monitoring
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Metrics for tracking convergence during generation."""
    step: int
    loss_value: float
    loss_change: float
    gradient_norm: float
    structure_stability: float
    quality_score: float
    convergence_rate: float
    early_stop_score: float


@dataclass
class AdaptiveSamplingConfig:
    """Configuration for adaptive sampling behavior."""
    # Dynamic step adjustment
    min_steps: int = 50
    max_steps: int = 500
    initial_steps: int = 200
    step_adjustment_factor: float = 0.1
    
    # Quality thresholds
    quality_threshold: float = 0.85
    stability_threshold: float = 0.9
    convergence_threshold: float = 0.95
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stop_patience: int = 20
    early_stop_min_improvement: float = 0.001
    
    # Progressive refinement
    enable_progressive: bool = True
    progressive_stages: List[int] = None
    progressive_quality_targets: List[float] = None
    
    # Monitoring
    convergence_window: int = 10
    quality_window: int = 5
    
    def __post_init__(self):
        if self.progressive_stages is None:
            self.progressive_stages = [50, 100, 200]
        if self.progressive_quality_targets is None:
            self.progressive_quality_targets = [0.7, 0.8, 0.9]


class ConvergenceDetector:
    """Real-time convergence detection and monitoring."""
    
    def __init__(self, config: AdaptiveSamplingConfig):
        self.config = config
        self.metrics_history: List[ConvergenceMetrics] = []
        self.convergence_detected = False
        self.early_stop_counter = 0
        self.best_quality = 0.0
        
    def update_metrics(self, metrics: ConvergenceMetrics) -> bool:
        """Update convergence metrics and check for convergence."""
        self.metrics_history.append(metrics)
        
        # Check for convergence
        if len(self.metrics_history) >= self.config.convergence_window:
            self.convergence_detected = self._check_convergence()
            
        # Check for early stopping
        if self.config.enable_early_stopping:
            should_stop = self._check_early_stopping(metrics)
            if should_stop:
                logger.info(f"Early stopping triggered at step {metrics.step}")
                return True
                
        return self.convergence_detected
    
    def _check_convergence(self) -> bool:
        """Check if generation has converged."""
        recent_metrics = self.metrics_history[-self.config.convergence_window:]
        
        # Check loss stability
        losses = [m.loss_value for m in recent_metrics]
        loss_std = np.std(losses)
        loss_mean = np.mean(losses)
        loss_stability = 1.0 - (loss_std / (loss_mean + 1e-8))
        
        # Check quality stability
        qualities = [m.quality_score for m in recent_metrics]
        quality_std = np.std(qualities)
        quality_mean = np.mean(qualities)
        quality_stability = 1.0 - (quality_std / (quality_mean + 1e-8))
        
        # Check gradient norm trend
        grad_norms = [m.gradient_norm for m in recent_metrics]
        grad_trend = np.mean(np.diff(grad_norms))
        
        # Convergence criteria
        converged = (
            loss_stability > self.config.convergence_threshold and
            quality_stability > self.config.stability_threshold and
            grad_trend <= 0  # Gradient should be decreasing
        )
        
        if converged:
            logger.info(f"Convergence detected: loss_stability={loss_stability:.3f}, "
                       f"quality_stability={quality_stability:.3f}, grad_trend={grad_trend:.6f}")
        
        return converged
    
    def _check_early_stopping(self, metrics: ConvergenceMetrics) -> bool:
        """Check if early stopping criteria are met."""
        if metrics.quality_score > self.best_quality + self.config.early_stop_min_improvement:
            self.best_quality = metrics.quality_score
            self.early_stop_counter = 0
            return False
        
        self.early_stop_counter += 1
        return self.early_stop_counter >= self.config.early_stop_patience


class AdaptiveSampler:
    """Main adaptive sampling controller."""
    
    def __init__(self, config: Optional[AdaptiveSamplingConfig] = None):
        self.config = config or AdaptiveSamplingConfig()
        self.convergence_detector = ConvergenceDetector(self.config)
        self.current_steps = self.config.initial_steps
        self.stage = 0
        self.sampling_stats = {
            'total_structures': 0,
            'adaptive_adjustments': 0,
            'early_stops': 0,
            'convergence_detections': 0,
            'average_steps_used': 0,
            'quality_improvements': 0
        }
        
    def adapt_sampling_steps(self, 
                           current_metrics: ConvergenceMetrics,
                           target_quality: Optional[float] = None) -> int:
        """Adapt the number of sampling steps based on current metrics."""
        
        # Update convergence detection
        converged = self.convergence_detector.update_metrics(current_metrics)
        
        if converged or (target_quality and current_metrics.quality_score >= target_quality):
            # Can reduce steps if converged or quality target met
            new_steps = max(
                self.config.min_steps,
                int(self.current_steps * (1 - self.config.step_adjustment_factor))
            )
            if new_steps != self.current_steps:
                logger.info(f"Reducing steps from {self.current_steps} to {new_steps} "
                           f"(quality: {current_metrics.quality_score:.3f})")
                self.sampling_stats['adaptive_adjustments'] += 1
        else:
            # Increase steps if not converged and quality is low
            if current_metrics.quality_score < self.config.quality_threshold:
                new_steps = min(
                    self.config.max_steps,
                    int(self.current_steps * (1 + self.config.step_adjustment_factor))
                )
                if new_steps != self.current_steps:
                    logger.info(f"Increasing steps from {self.current_steps} to {new_steps} "
                               f"(quality: {current_metrics.quality_score:.3f})")
                    self.sampling_stats['adaptive_adjustments'] += 1
            else:
                new_steps = self.current_steps
        
        self.current_steps = new_steps
        return new_steps
    
    def should_use_progressive_refinement(self, batch_idx: int, total_batches: int) -> bool:
        """Determine if progressive refinement should be used."""
        if not self.config.enable_progressive:
            return False
            
        # Use progressive refinement for later batches when we have learned optimal parameters
        progress = batch_idx / total_batches
        return progress > 0.3  # Start progressive refinement after 30% of batches
    
    def get_progressive_stage_steps(self, stage: int) -> int:
        """Get the number of steps for a specific progressive stage."""
        if stage < len(self.config.progressive_stages):
            return self.config.progressive_stages[stage]
        return self.config.progressive_stages[-1]
    
    def get_progressive_quality_target(self, stage: int) -> float:
        """Get the quality target for a specific progressive stage."""
        if stage < len(self.config.progressive_quality_targets):
            return self.config.progressive_quality_targets[stage]
        return self.config.progressive_quality_targets[-1]
    
    def update_sampling_stats(self, structures_generated: int, early_stopped: bool = False):
        """Update sampling statistics."""
        self.sampling_stats['total_structures'] += structures_generated
        if early_stopped:
            self.sampling_stats['early_stops'] += 1
        if self.convergence_detector.convergence_detected:
            self.sampling_stats['convergence_detections'] += 1
            
        # Update average steps used
        total_structures = self.sampling_stats['total_structures']
        if total_structures > 0:
            self.sampling_stats['average_steps_used'] = (
                (self.sampling_stats['average_steps_used'] * (total_structures - structures_generated) +
                 self.current_steps * structures_generated) / total_structures
            )
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive sampling optimizations."""
        total_structures = self.sampling_stats['total_structures']
        if total_structures == 0:
            return {}
            
        return {
            'total_structures_generated': total_structures,
            'adaptive_adjustments_made': self.sampling_stats['adaptive_adjustments'],
            'early_stops_triggered': self.sampling_stats['early_stops'],
            'convergence_detections': self.sampling_stats['convergence_detections'],
            'average_steps_used': round(self.sampling_stats['average_steps_used'], 1),
            'step_efficiency_gain': round(
                (self.config.initial_steps - self.sampling_stats['average_steps_used']) 
                / self.config.initial_steps * 100, 1
            ),
            'early_stop_rate': round(self.sampling_stats['early_stops'] / total_structures * 100, 1),
            'convergence_rate': round(self.sampling_stats['convergence_detections'] / total_structures * 100, 1),
            'current_steps': self.current_steps,
            'config': self.config.__dict__
        }
    
    def save_optimization_report(self, output_path: Path):
        """Save detailed optimization report."""
        report = {
            'adaptive_sampling_summary': self.get_optimization_summary(),
            'convergence_history': [
                {
                    'step': m.step,
                    'loss_value': m.loss_value,
                    'quality_score': m.quality_score,
                    'convergence_rate': m.convergence_rate
                }
                for m in self.convergence_detector.metrics_history
            ],
            'configuration': self.config.__dict__,
            'timestamp': time.time()
        }
        
        report_path = output_path / 'adaptive_sampling_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Adaptive sampling report saved to: {report_path}")


def create_quality_metrics(structure_data: Dict, loss_value: float, step: int) -> ConvergenceMetrics:
    """Create quality metrics from structure data and training info."""
    # This is a placeholder - in real implementation, this would calculate
    # actual quality metrics from the generated structures
    
    # Simulate quality calculation based on structure properties
    quality_score = min(1.0, max(0.0, 1.0 - loss_value * 0.1))
    
    # Simulate gradient norm (would come from actual training)
    gradient_norm = loss_value * 0.5 + np.random.normal(0, 0.01)
    
    # Simulate structure stability
    structure_stability = quality_score * 0.9 + np.random.normal(0, 0.05)
    structure_stability = min(1.0, max(0.0, structure_stability))
    
    # Calculate loss change if we have previous data
    loss_change = abs(np.random.normal(0, 0.01))  # Placeholder
    
    # Calculate convergence rate
    convergence_rate = min(1.0, quality_score * structure_stability)
    
    # Calculate early stop score
    early_stop_score = (quality_score + structure_stability + convergence_rate) / 3
    
    return ConvergenceMetrics(
        step=step,
        loss_value=loss_value,
        loss_change=loss_change,
        gradient_norm=gradient_norm,
        structure_stability=structure_stability,
        quality_score=quality_score,
        convergence_rate=convergence_rate,
        early_stop_score=early_stop_score
    )


# Factory function for easy integration
def create_adaptive_sampler(
    min_steps: int = 50,
    max_steps: int = 500,
    initial_steps: int = 200,
    quality_threshold: float = 0.85,
    enable_early_stopping: bool = True,
    enable_progressive: bool = True
) -> AdaptiveSampler:
    """Factory function to create adaptive sampler with common configurations."""
    
    config = AdaptiveSamplingConfig(
        min_steps=min_steps,
        max_steps=max_steps,
        initial_steps=initial_steps,
        quality_threshold=quality_threshold,
        enable_early_stopping=enable_early_stopping,
        enable_progressive=enable_progressive
    )
    
    return AdaptiveSampler(config)
