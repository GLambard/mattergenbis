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
- Modular component architecture for standalone usage

Architecture:
- Component-based design with clear interfaces
- Standalone convergence detection
- Independent quality metrics integration
- Pluggable sampling strategies
- Optional adaptive parameter adjustment
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Protocol, Union
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# CORE INTERFACES AND PROTOCOLS
# ============================================================================

class SamplingStrategy(Protocol):
    """Protocol for sampling strategies that can be plugged into the adaptive sampler."""
    
    def adjust_steps(self, metrics: 'ConvergenceMetrics', current_steps: int) -> int:
        """Adjust sampling steps based on current metrics."""
        ...
    
    def should_continue(self, metrics: 'ConvergenceMetrics', iteration: int, max_iterations: int) -> bool:
        """Determine if sampling should continue."""
        ...


class QualityAssessmentInterface(Protocol):
    """Protocol for quality assessment components."""
    
    def assess_quality(self, structures: List[Any]) -> Tuple[float, Dict[str, Any]]:
        """Assess quality of structures and return score and detailed metrics."""
        ...
    
    def filter_structures(self, structures: List[Any], threshold: float) -> List[Any]:
        """Filter structures based on quality threshold."""
        ...


class ConvergenceDetectorInterface(Protocol):
    """Protocol for convergence detection components."""
    
    def update_metrics(self, metrics: 'ConvergenceMetrics') -> bool:
        """Update with new metrics and return convergence status."""
        ...
    
    def reset(self) -> None:
        """Reset convergence detection state."""
        ...


# ============================================================================
# DATA STRUCTURES
# ============================================================================

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
    
    # Additional context
    batch_id: Optional[int] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AdaptiveSamplingConfig:
    """Configuration for adaptive sampling behavior with modular components."""
    
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
    
    # Iteration control
    max_iterations: int = 5
    
    # Modular component settings
    sampling_strategy: str = "adaptive"  # "adaptive", "conservative", "aggressive", "custom"
    convergence_strategy: str = "standard"  # "standard", "strict", "lenient", "custom"
    enable_quality_integration: bool = True
    enable_enterprise_callbacks: bool = False
    
    def __post_init__(self):
        if self.progressive_stages is None:
            self.progressive_stages = [50, 100, 200]
        if self.progressive_quality_targets is None:
            self.progressive_quality_targets = [0.7, 0.8, 0.9]


@dataclass
class SamplingResult:
    """Result from an adaptive sampling operation."""
    new_guidance_factor: float
    should_continue: bool
    adapted_steps: int
    quality_score: float
    converged: bool
    iteration: int
    metrics: ConvergenceMetrics
    recommendations: Dict[str, Any]
    performance_stats: Dict[str, Any]


# ============================================================================
# MODULAR COMPONENTS
# ============================================================================

class BaseSamplingStrategy(ABC):
    """Base class for sampling strategies."""
    
    def __init__(self, config: AdaptiveSamplingConfig):
        self.config = config
    
    @abstractmethod
    def adjust_steps(self, metrics: ConvergenceMetrics, current_steps: int) -> int:
        """Adjust sampling steps based on current metrics."""
        pass
    
    @abstractmethod
    def should_continue(self, metrics: ConvergenceMetrics, iteration: int, max_iterations: int) -> bool:
        """Determine if sampling should continue."""
        pass


class AdaptiveSamplingStrategy(BaseSamplingStrategy):
    """Standard adaptive sampling strategy with dynamic adjustment."""
    
    def adjust_steps(self, metrics: ConvergenceMetrics, current_steps: int) -> int:
        if metrics.quality_score >= self.config.quality_threshold:
            # Reduce steps if quality target met
            new_steps = max(
                self.config.min_steps,
                int(current_steps * (1 - self.config.step_adjustment_factor))
            )
        elif metrics.quality_score < self.config.quality_threshold * 0.8:
            # Increase steps if quality is significantly low
            new_steps = min(
                self.config.max_steps,
                int(current_steps * (1 + self.config.step_adjustment_factor))
            )
        else:
            new_steps = current_steps
        
        return new_steps
    
    def should_continue(self, metrics: ConvergenceMetrics, iteration: int, max_iterations: int) -> bool:
        if iteration >= max_iterations:
            return False
        
        # Continue if quality is improving or below threshold
        if metrics.quality_score < self.config.quality_threshold:
            return True
        
        # Stop if converged
        return metrics.convergence_rate < self.config.convergence_threshold


class ConservativeSamplingStrategy(BaseSamplingStrategy):
    """Conservative strategy that makes smaller adjustments."""
    
    def adjust_steps(self, metrics: ConvergenceMetrics, current_steps: int) -> int:
        # More conservative step adjustment
        conservative_factor = self.config.step_adjustment_factor * 0.5
        
        if metrics.quality_score >= self.config.quality_threshold:
            new_steps = max(
                self.config.min_steps,
                int(current_steps * (1 - conservative_factor))
            )
        elif metrics.quality_score < self.config.quality_threshold * 0.7:
            new_steps = min(
                self.config.max_steps,
                int(current_steps * (1 + conservative_factor))
            )
        else:
            new_steps = current_steps
        
        return new_steps
    
    def should_continue(self, metrics: ConvergenceMetrics, iteration: int, max_iterations: int) -> bool:
        # More conservative stopping criteria
        return (iteration < max_iterations and 
                metrics.quality_score < self.config.quality_threshold * 1.1)


class AggressiveSamplingStrategy(BaseSamplingStrategy):
    """Aggressive strategy that makes larger adjustments and stops early."""
    
    def adjust_steps(self, metrics: ConvergenceMetrics, current_steps: int) -> int:
        # More aggressive step adjustment
        aggressive_factor = self.config.step_adjustment_factor * 2.0
        
        if metrics.quality_score >= self.config.quality_threshold:
            new_steps = max(
                self.config.min_steps,
                int(current_steps * (1 - aggressive_factor))
            )
        elif metrics.quality_score < self.config.quality_threshold * 0.9:
            new_steps = min(
                self.config.max_steps,
                int(current_steps * (1 + aggressive_factor))
            )
        else:
            new_steps = current_steps
        
        return new_steps
    
    def should_continue(self, metrics: ConvergenceMetrics, iteration: int, max_iterations: int) -> bool:
        # More aggressive stopping (stop early if quality is good)
        return (iteration < max_iterations and 
                metrics.quality_score < self.config.quality_threshold * 0.95)


class BaseConvergenceDetector(ABC):
    """Base class for convergence detection strategies."""
    
    def __init__(self, config: AdaptiveSamplingConfig):
        self.config = config
    
    @abstractmethod
    def update_metrics(self, metrics: ConvergenceMetrics) -> bool:
        """Update metrics and return convergence status."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass


class StandardConvergenceDetector(BaseConvergenceDetector):
    """Standard convergence detection implementation."""
    
    def __init__(self, config: AdaptiveSamplingConfig):
        super().__init__(config)
        self.metrics_history: List[ConvergenceMetrics] = []
        self.convergence_detected = False
        self.early_stop_counter = 0
        self.best_quality = 0.0
    
    def reset(self) -> None:
        """Reset detector state."""
        self.metrics_history.clear()
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


class StrictConvergenceDetector(BaseConvergenceDetector):
    """Strict convergence detector that requires more stringent criteria."""
    
    def __init__(self, config: AdaptiveSamplingConfig):
        super().__init__(config)
        self.metrics_history: List[ConvergenceMetrics] = []
        self.convergence_detected = False
        self.early_stop_counter = 0
        self.best_quality = 0.0
        self.stability_counter = 0
    
    def reset(self) -> None:
        super().reset()
        self.stability_counter = 0
    
    def update_metrics(self, metrics: ConvergenceMetrics) -> bool:
        """Update with stricter convergence criteria."""
        self.metrics_history.append(metrics)
        
        # Require longer stability period
        stability_window = self.config.convergence_window * 2
        
        if len(self.metrics_history) >= stability_window:
            recent_metrics = self.metrics_history[-stability_window:]
            
            # Check for sustained quality improvement
            qualities = [m.quality_score for m in recent_metrics]
            quality_trend = np.polyfit(range(len(qualities)), qualities, 1)[0]
            
            # Require positive trend or high stable quality
            if quality_trend >= 0 and np.mean(qualities) >= self.config.quality_threshold * 1.1:
                self.stability_counter += 1
            else:
                self.stability_counter = 0
            
            # Require sustained stability
            self.convergence_detected = self.stability_counter >= 3
        
        return self.convergence_detected


class LenientConvergenceDetector(BaseConvergenceDetector):
    """Lenient convergence detector that stops early with less strict criteria."""
    
    def __init__(self, config: AdaptiveSamplingConfig):
        super().__init__(config)
        self.metrics_history: List[ConvergenceMetrics] = []
        self.convergence_detected = False
    
    def reset(self) -> None:
        super().reset()
    
    def update_metrics(self, metrics: ConvergenceMetrics) -> bool:
        """Update with lenient convergence criteria."""
        self.metrics_history.append(metrics)
        
        # Simple quality-based convergence
        if metrics.quality_score >= self.config.quality_threshold * 0.9:
            self.convergence_detected = True
            logger.info(f"Lenient convergence: quality {metrics.quality_score:.3f} meets threshold")
        
        return self.convergence_detected


# ============================================================================
# QUALITY ASSESSMENT ADAPTERS
# ============================================================================

class QualityAssessmentAdapter:
    """Adapter for integrating external quality assessment systems."""
    
    def __init__(self, quality_assessor: Optional[QualityAssessmentInterface] = None):
        self.quality_assessor = quality_assessor
        self.enabled = quality_assessor is not None
    
    def assess_quality(self, structures: List[Any]) -> Tuple[float, Dict[str, Any]]:
        """Assess quality using external assessor or provide default."""
        if self.enabled and self.quality_assessor:
            return self.quality_assessor.assess_quality(structures)
        else:
            # Default quality assessment
            return 0.8, {'method': 'default', 'structures_count': len(structures)}
    
    def filter_structures(self, structures: List[Any], threshold: float) -> List[Any]:
        """Filter structures using external assessor or pass-through."""
        if self.enabled and self.quality_assessor:
            return self.quality_assessor.filter_structures(structures, threshold)
        else:
            # Default: pass through all structures
            return structures


# ============================================================================
# ENTERPRISE INTEGRATION CALLBACKS
# ============================================================================

class EnterpriseCallbackAdapter:
    """Adapter for enterprise monitoring and analytics callbacks."""
    
    def __init__(self):
        self.callbacks: List[callable] = []
        self.enabled = False
    
    def register_callback(self, callback: callable):
        """Register an enterprise callback."""
        self.callbacks.append(callback)
        self.enabled = True
    
    def on_adaptation_start(self, config: AdaptiveSamplingConfig, initial_params: Dict[str, Any]):
        """Notify callbacks of adaptation start."""
        for callback in self.callbacks:
            try:
                callback('adaptation_start', config=config, params=initial_params)
            except Exception as e:
                logger.warning(f"Enterprise callback failed: {e}")
    
    def on_adaptation_step(self, iteration: int, metrics: ConvergenceMetrics, result: SamplingResult):
        """Notify callbacks of adaptation step."""
        for callback in self.callbacks:
            try:
                callback('adaptation_step', iteration=iteration, metrics=metrics, result=result)
            except Exception as e:
                logger.warning(f"Enterprise callback failed: {e}")
    
    def on_adaptation_complete(self, final_stats: Dict[str, Any]):
        """Notify callbacks of adaptation completion."""
        for callback in self.callbacks:
            try:
                callback('adaptation_complete', stats=final_stats)
            except Exception as e:
                logger.warning(f"Enterprise callback failed: {e}")


# ============================================================================
# COMPONENT FACTORIES
# ============================================================================

def create_sampling_strategy(strategy_name: str, config: AdaptiveSamplingConfig) -> BaseSamplingStrategy:
    """Factory function to create sampling strategies."""
    strategies = {
        'adaptive': AdaptiveSamplingStrategy,
        'conservative': ConservativeSamplingStrategy,
        'aggressive': AggressiveSamplingStrategy
    }
    
    strategy_class = strategies.get(strategy_name, AdaptiveSamplingStrategy)
    return strategy_class(config)


def create_convergence_detector(detector_name: str, config: AdaptiveSamplingConfig) -> BaseConvergenceDetector:
    """Factory function to create convergence detectors."""
    detectors = {
        'standard': StandardConvergenceDetector,
        'strict': StrictConvergenceDetector,
        'lenient': LenientConvergenceDetector
    }
    
    detector_class = detectors.get(detector_name, StandardConvergenceDetector)
    return detector_class(config)


# ============================================================================
# MAIN ADAPTIVE SAMPLER CLASS
# ============================================================================


class ModularAdaptiveSampler:
    """
    Modular adaptive sampling controller with pluggable components.
    
    This class provides a flexible architecture where different strategies
    and components can be plugged in based on requirements.
    """
    
    def __init__(self, 
                 config: Optional[AdaptiveSamplingConfig] = None,
                 sampling_strategy: Optional[BaseSamplingStrategy] = None,
                 convergence_detector: Optional[BaseConvergenceDetector] = None,
                 quality_assessor: Optional[QualityAssessmentInterface] = None,
                 enterprise_callbacks: Optional[EnterpriseCallbackAdapter] = None):
        
        self.config = config or AdaptiveSamplingConfig()
        
        # Initialize modular components
        self.sampling_strategy = sampling_strategy or create_sampling_strategy(
            self.config.sampling_strategy, self.config
        )
        self.convergence_detector = convergence_detector or create_convergence_detector(
            self.config.convergence_strategy, self.config
        )
        self.quality_adapter = QualityAssessmentAdapter(quality_assessor)
        self.enterprise_callbacks = enterprise_callbacks or EnterpriseCallbackAdapter()
        
        # Initialize state
        self.current_steps = self.config.initial_steps
        self.stage = 0
        self.sampling_stats = {
            'total_structures': 0,
            'adaptive_adjustments': 0,
            'early_stops': 0,
            'convergence_detections': 0,
            'average_steps_used': 0,
            'quality_improvements': 0,
            'component_usage': {
                'sampling_strategy': type(self.sampling_strategy).__name__,
                'convergence_detector': type(self.convergence_detector).__name__,
                'quality_integration': self.quality_adapter.enabled,
                'enterprise_integration': self.enterprise_callbacks.enabled
            }
        }
    
    def adapt_parameters(self, structures, quality_score: float, current_guidance_factor: float, iteration: int) -> SamplingResult:
        """
        Adapt sampling parameters using modular components.
        
        Args:
            structures: List of generated structures
            quality_score: Average quality score of current structures
            current_guidance_factor: Current guidance factor value
            iteration: Current adaptation iteration
            
        Returns:
            SamplingResult with adapted parameters and recommendations
        """
        # Create metrics for convergence detection
        metrics = ConvergenceMetrics(
            step=iteration,
            loss_value=1.0 - quality_score,
            loss_change=0.0,  # Would be calculated from previous iteration
            gradient_norm=0.0,  # Not available in this context
            structure_stability=quality_score,
            quality_score=quality_score,
            convergence_rate=0.0,  # Calculated by convergence detector
            early_stop_score=quality_score,
            batch_id=iteration,
            metadata={'structures_count': len(structures), 'guidance_factor': current_guidance_factor}
        )
        
        # Enterprise callback: adaptation step start
        if self.enterprise_callbacks.enabled:
            self.enterprise_callbacks.on_adaptation_step(iteration, metrics, None)
        
        # Update convergence detection using modular detector
        converged = self.convergence_detector.update_metrics(metrics)
        
        # Adapt sampling steps using pluggable strategy
        new_steps = self.sampling_strategy.adjust_steps(metrics, self.current_steps)
        if new_steps != self.current_steps:
            logger.info(f"Strategy '{type(self.sampling_strategy).__name__}' adjusted steps: {self.current_steps} → {new_steps}")
            self.sampling_stats['adaptive_adjustments'] += 1
        
        # Adjust guidance factor using modular quality assessment
        new_guidance_factor = self._adapt_guidance_factor(
            current_guidance_factor, quality_score, converged
        )
        
        # Determine if we should continue using strategy
        should_continue = self.sampling_strategy.should_continue(
            metrics, iteration, self.config.max_iterations
        )
        
        # Update stats
        self.update_sampling_stats(len(structures), early_stopped=converged)
        self.current_steps = new_steps
        
        # Create comprehensive result
        result = SamplingResult(
            new_guidance_factor=new_guidance_factor,
            should_continue=should_continue,
            adapted_steps=new_steps,
            quality_score=quality_score,
            converged=converged,
            iteration=iteration,
            metrics=metrics,
            recommendations=self._generate_recommendations(metrics, converged),
            performance_stats=self._get_performance_stats()
        )
        
        # Enterprise callback: adaptation step complete
        if self.enterprise_callbacks.enabled:
            self.enterprise_callbacks.on_adaptation_step(iteration, metrics, result)
        
        return result
    
    def _adapt_guidance_factor(self, current_factor: float, quality_score: float, converged: bool) -> float:
        """Adapt guidance factor based on quality and convergence."""
        if quality_score < self.config.quality_threshold:
            # Increase guidance if quality is low
            new_factor = min(current_factor * 1.1, 2.0)
            if new_factor != current_factor:
                logger.info(f"Increasing guidance factor: {current_factor:.3f} → {new_factor:.3f} (low quality: {quality_score:.3f})")
        elif converged or quality_score > 0.8:
            # Reduce guidance if converged or quality is high
            new_factor = max(current_factor * 0.9, 0.5)
            if new_factor != current_factor:
                logger.info(f"Reducing guidance factor: {current_factor:.3f} → {new_factor:.3f} (quality: {quality_score:.3f})")
        else:
            new_factor = current_factor
        
        return new_factor
    
    def _generate_recommendations(self, metrics: ConvergenceMetrics, converged: bool) -> Dict[str, Any]:
        """Generate recommendations based on current state."""
        recommendations = {
            'action': 'continue',
            'reason': 'normal_progress',
            'suggestions': []
        }
        
        if converged:
            recommendations['action'] = 'stop'
            recommendations['reason'] = 'converged'
            recommendations['suggestions'].append('Generation has converged')
        elif metrics.quality_score < self.config.quality_threshold * 0.8:
            recommendations['action'] = 'adjust'
            recommendations['reason'] = 'low_quality'
            recommendations['suggestions'].append('Consider increasing sampling steps or guidance')
        elif metrics.quality_score > self.config.quality_threshold * 1.2:
            recommendations['action'] = 'optimize'
            recommendations['reason'] = 'high_quality'
            recommendations['suggestions'].append('Consider reducing sampling steps for efficiency')
        
        return recommendations
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            'current_steps': self.current_steps,
            'total_adjustments': self.sampling_stats['adaptive_adjustments'],
            'convergence_rate': self.sampling_stats['convergence_detections'],
            'component_info': self.sampling_stats['component_usage']
        }
    
    def reset(self):
        """Reset sampler state for new generation session."""
        self.convergence_detector.reset()
        self.current_steps = self.config.initial_steps
        self.stage = 0
        # Don't reset stats - keep cumulative across sessions
    
    # Legacy compatibility methods
    def adapt_sampling_steps(self, current_metrics: ConvergenceMetrics, target_quality: Optional[float] = None) -> int:
        """Legacy compatibility: adapt sampling steps."""
        return self.sampling_strategy.adjust_steps(current_metrics, self.current_steps)
    
    def should_use_progressive_refinement(self, batch_idx: int, total_batches: int) -> bool:
        """Legacy compatibility: progressive refinement check."""
        if not self.config.enable_progressive:
            return False
        progress = batch_idx / total_batches
        return progress > 0.3
    
    def update_sampling_stats(self, structures_generated: int, early_stopped: bool = False):
        """Update sampling statistics."""
        self.sampling_stats['total_structures'] += structures_generated
        if early_stopped:
            self.sampling_stats['early_stops'] += 1
        if hasattr(self.convergence_detector, 'convergence_detected') and self.convergence_detector.convergence_detected:
            self.sampling_stats['convergence_detections'] += 1
        
        # Update average steps used
        total_structures = self.sampling_stats['total_structures']
        if total_structures > 0:
            self.sampling_stats['average_steps_used'] = (
                (self.sampling_stats['average_steps_used'] * (total_structures - structures_generated) +
                 self.current_steps * structures_generated) / total_structures
            )
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        total_structures = self.sampling_stats['total_structures']
        if total_structures == 0:
            return {'status': 'no_data', 'message': 'No structures processed yet'}
        
        summary = {
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
            'modular_components': self.sampling_stats['component_usage'],
            'configuration': self.config.__dict__
        }
        
        return summary
    
    def save_optimization_report(self, output_path: Path):
        """Save detailed optimization report."""
        report = {
            'adaptive_sampling_summary': self.get_optimization_summary(),
            'convergence_history': [
                {
                    'step': m.step,
                    'loss_value': m.loss_value,
                    'quality_score': m.quality_score,
                    'convergence_rate': m.convergence_rate,
                    'timestamp': m.timestamp,
                    'metadata': m.metadata
                }
                for m in getattr(self.convergence_detector, 'metrics_history', [])
            ],
            'component_configuration': {
                'sampling_strategy': {
                    'type': type(self.sampling_strategy).__name__,
                    'config': self.config.__dict__
                },
                'convergence_detector': {
                    'type': type(self.convergence_detector).__name__,
                    'config': self.config.__dict__
                },
                'quality_integration': self.quality_adapter.enabled,
                'enterprise_integration': self.enterprise_callbacks.enabled
            },
            'timestamp': time.time()
        }
        
        report_path = output_path / 'modular_adaptive_sampling_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Modular adaptive sampling report saved to: {report_path}")


# Legacy class for backward compatibility
class AdaptiveSampler(ModularAdaptiveSampler):
    """Legacy adaptive sampler class for backward compatibility."""
    
    def __init__(self, config: Optional[AdaptiveSamplingConfig] = None):
        # Initialize with default components for backward compatibility
        super().__init__(
            config=config,
            sampling_strategy=None,  # Will use default
            convergence_detector=None,  # Will use default
            quality_assessor=None,
            enterprise_callbacks=None
        )
    
    def adapt_parameters(self, structures, quality_score: float, current_guidance_factor: float, iteration: int) -> Dict[str, Any]:
        """
        Legacy compatibility method that returns a dictionary instead of SamplingResult.
        
        This method maintains backward compatibility with existing code that expects
        a dictionary return value from adapt_parameters().
        """
        # Call the parent modular method
        result = super().adapt_parameters(structures, quality_score, current_guidance_factor, iteration)
        
        # Convert SamplingResult to legacy dictionary format
        return {
            'new_guidance_factor': result.new_guidance_factor,
            'should_continue': result.should_continue,
            'adapted_steps': result.adapted_steps,
            'quality_score': result.quality_score,
            'converged': result.converged,
            'iteration': result.iteration,
            # Additional fields for enhanced compatibility
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'performance_stats': result.performance_stats
        }# ============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# ============================================================================

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


def create_adaptive_sampler(
    min_steps: int = 50,
    max_steps: int = 500,
    initial_steps: int = 200,
    quality_threshold: float = 0.85,
    enable_early_stopping: bool = True,
    enable_progressive: bool = True,
    sampling_strategy: str = "adaptive",
    convergence_strategy: str = "standard",
    quality_assessor: Optional[QualityAssessmentInterface] = None,
    enterprise_callbacks: Optional[EnterpriseCallbackAdapter] = None,
    modular: bool = False
) -> Union[AdaptiveSampler, ModularAdaptiveSampler]:
    """
    Factory function to create adaptive sampler with flexible configurations.
    
    Args:
        modular: If True, returns ModularAdaptiveSampler with enhanced capabilities.
                If False, returns legacy AdaptiveSampler for backward compatibility.
    """
    
    config = AdaptiveSamplingConfig(
        min_steps=min_steps,
        max_steps=max_steps,
        initial_steps=initial_steps,
        quality_threshold=quality_threshold,
        enable_early_stopping=enable_early_stopping,
        enable_progressive=enable_progressive,
        sampling_strategy=sampling_strategy,
        convergence_strategy=convergence_strategy,
        enable_quality_integration=quality_assessor is not None,
        enable_enterprise_callbacks=enterprise_callbacks is not None
    )
    
    if modular:
        return ModularAdaptiveSampler(
            config=config,
            sampling_strategy=None,  # Will be created by factory
            convergence_detector=None,  # Will be created by factory
            quality_assessor=quality_assessor,
            enterprise_callbacks=enterprise_callbacks
        )
    else:
        # Legacy mode - return simple AdaptiveSampler
        return AdaptiveSampler(config)


def create_standalone_quality_adapter(quality_assessor: QualityAssessmentInterface) -> QualityAssessmentAdapter:
    """Create a standalone quality adapter for use without adaptive sampling."""
    return QualityAssessmentAdapter(quality_assessor)


def create_enterprise_callback_adapter() -> EnterpriseCallbackAdapter:
    """Create an enterprise callback adapter for monitoring integration."""
    return EnterpriseCallbackAdapter()


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

class SamplingPresets:
    """Predefined configurations for common use cases."""
    
    @staticmethod
    def conservative() -> AdaptiveSamplingConfig:
        """Conservative sampling preset - slower but more stable."""
        return AdaptiveSamplingConfig(
            min_steps=100,
            max_steps=800,
            initial_steps=300,
            step_adjustment_factor=0.05,
            quality_threshold=0.9,
            convergence_threshold=0.98,
            sampling_strategy="conservative",
            convergence_strategy="strict"
        )
    
    @staticmethod
    def aggressive() -> AdaptiveSamplingConfig:
        """Aggressive sampling preset - faster but may be less stable."""
        return AdaptiveSamplingConfig(
            min_steps=25,
            max_steps=300,
            initial_steps=100,
            step_adjustment_factor=0.2,
            quality_threshold=0.75,
            convergence_threshold=0.9,
            sampling_strategy="aggressive",
            convergence_strategy="lenient"
        )
    
    @staticmethod
    def balanced() -> AdaptiveSamplingConfig:
        """Balanced sampling preset - good compromise between speed and stability."""
        return AdaptiveSamplingConfig(
            min_steps=50,
            max_steps=500,
            initial_steps=200,
            step_adjustment_factor=0.1,
            quality_threshold=0.85,
            convergence_threshold=0.95,
            sampling_strategy="adaptive",
            convergence_strategy="standard"
        )
    
    @staticmethod
    def production() -> AdaptiveSamplingConfig:
        """Production sampling preset - optimized for real-world deployment."""
        return AdaptiveSamplingConfig(
            min_steps=75,
            max_steps=600,
            initial_steps=250,
            step_adjustment_factor=0.08,
            quality_threshold=0.88,
            stability_threshold=0.92,
            convergence_threshold=0.96,
            enable_early_stopping=True,
            early_stop_patience=15,
            sampling_strategy="adaptive",
            convergence_strategy="standard",
            enable_quality_integration=True,
            enable_enterprise_callbacks=True
        )


# ============================================================================
# EXAMPLE USAGE AND INTEGRATION HELPERS
# ============================================================================

def create_full_adaptive_pipeline(
    sampling_preset: str = "balanced",
    quality_assessor: Optional[QualityAssessmentInterface] = None,
    enterprise_monitoring: bool = False
) -> ModularAdaptiveSampler:
    """
    Create a complete adaptive sampling pipeline with all components.
    
    Args:
        sampling_preset: One of "conservative", "aggressive", "balanced", "production"
        quality_assessor: External quality assessment component
        enterprise_monitoring: Enable enterprise monitoring callbacks
    
    Returns:
        Fully configured ModularAdaptiveSampler
    """
    
    # Get preset configuration
    preset_configs = {
        "conservative": SamplingPresets.conservative(),
        "aggressive": SamplingPresets.aggressive(),
        "balanced": SamplingPresets.balanced(),
        "production": SamplingPresets.production()
    }
    
    config = preset_configs.get(sampling_preset, SamplingPresets.balanced())
    
    # Create enterprise callbacks if requested
    enterprise_callbacks = None
    if enterprise_monitoring:
        enterprise_callbacks = create_enterprise_callback_adapter()
        
        # Register default enterprise callbacks
        def default_enterprise_callback(event_type: str, **kwargs):
            logger.info(f"Enterprise event: {event_type} - {kwargs}")
        
        enterprise_callbacks.register_callback(default_enterprise_callback)
    
    # Create modular sampler
    sampler = ModularAdaptiveSampler(
        config=config,
        quality_assessor=quality_assessor,
        enterprise_callbacks=enterprise_callbacks
    )
    
    # Log configuration
    logger.info(f"Created adaptive pipeline with preset: {sampling_preset}")
    logger.info(f"Components: {sampler.sampling_stats['component_usage']}")
    
    return sampler


# Backward compatibility alias
ConvergenceDetector = StandardConvergenceDetector
