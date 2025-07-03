# Modular Adaptive Sampling Architecture
## MatterGen Phase 4+ Enhancement Guide

### Overview

The MatterGen adaptive sampling system has been redesigned with a modular architecture that supports:

- **Component Independence**: Features can be enabled/disabled independently
- **Pluggable Strategies**: Different sampling and convergence strategies
- **Quality Integration**: External quality assessment systems
- **Enterprise Monitoring**: Optional monitoring and analytics callbacks
- **Backward Compatibility**: Legacy API support for existing code

### Architecture Components

#### 1. Core Interfaces

```python
# Sampling strategy interface
class SamplingStrategy(Protocol):
    def adjust_steps(self, metrics: ConvergenceMetrics, current_steps: int) -> int
    def should_continue(self, metrics: ConvergenceMetrics, iteration: int, max_iterations: int) -> bool

# Quality assessment interface  
class QualityAssessmentInterface(Protocol):
    def assess_quality(self, structures: List[Any]) -> Tuple[float, Dict[str, Any]]
    def filter_structures(self, structures: List[Any], threshold: float) -> List[Any]

# Convergence detection interface
class ConvergenceDetectorInterface(Protocol):
    def update_metrics(self, metrics: ConvergenceMetrics) -> bool
    def reset(self) -> None
```

#### 2. Modular Components

##### Sampling Strategies
- **AdaptiveSamplingStrategy**: Default balanced approach
- **ConservativeSamplingStrategy**: Smaller adjustments, more stable
- **AggressiveSamplingStrategy**: Larger adjustments, faster convergence

##### Convergence Detectors
- **StandardConvergenceDetector**: Default implementation
- **StrictConvergenceDetector**: Requires longer stability periods
- **LenientConvergenceDetector**: Early stopping with relaxed criteria

##### Quality Assessment Adapters
- **QualityAssessmentAdapter**: Integrates external quality systems
- Supports both Phase 4 integrated and standalone modes

##### Enterprise Integration
- **EnterpriseCallbackAdapter**: Optional monitoring callbacks
- Supports real-time analytics and dashboard integration

### Usage Patterns

#### 1. Legacy Compatible Usage

```python
# Simple usage - backward compatible
from mattergen.common.utils.adaptive_sampler import create_adaptive_sampler

sampler = create_adaptive_sampler(
    quality_threshold=0.85,
    sampling_strategy="adaptive"
)

result = sampler.adapt_parameters(
    structures=structures,
    quality_score=quality_score, 
    current_guidance_factor=guidance_factor,
    iteration=iteration
)
```

#### 2. Modular Configuration

```python
# Advanced modular usage
from mattergen.common.utils.adaptive_sampler import (
    ModularAdaptiveSampler, SamplingPresets, 
    create_sampling_strategy, create_convergence_detector
)

config = SamplingPresets.production()

sampler = ModularAdaptiveSampler(
    config=config,
    sampling_strategy=create_sampling_strategy("aggressive", config),
    convergence_detector=create_convergence_detector("strict", config),
    quality_assessor=my_quality_assessor,
    enterprise_callbacks=my_callbacks
)
```

#### 3. Full Pipeline Creation

```python
# Complete pipeline with all features
from mattergen.common.utils.adaptive_sampler import create_full_adaptive_pipeline

sampler = create_full_adaptive_pipeline(
    sampling_preset="production",
    quality_assessor=quality_assessor,
    enterprise_monitoring=True
)
```

#### 4. Standalone Component Usage

```python
# Use components independently
detector = create_convergence_detector("standard", config)
strategy = create_sampling_strategy("conservative", config)

# Use without full adaptive sampling
converged = detector.update_metrics(metrics)
new_steps = strategy.adjust_steps(metrics, current_steps)
```

### Configuration Presets

#### Conservative
- Slower adjustments, more stable
- Higher quality thresholds
- Strict convergence criteria
- Best for: Production environments, critical applications

#### Balanced (Default)
- Moderate adjustments
- Standard quality thresholds  
- Normal convergence criteria
- Best for: General usage, development

#### Aggressive  
- Fast adjustments, early stopping
- Lower quality thresholds
- Lenient convergence criteria
- Best for: Rapid prototyping, experimentation

#### Production
- Optimized for real-world deployment
- Enterprise integration enabled
- Comprehensive monitoring
- Best for: Production deployments

### CLI Integration

The modular system integrates seamlessly with the existing CLI:

#### Standalone Features (No Phase 4 Required)
```bash
# Quality metrics only
python multi_gpu_inference.py \
  --enable_quality_metrics true \
  --enable_phase4_features false

# Enterprise monitoring only  
python multi_gpu_inference.py \
  --enable_enterprise_monitoring true \
  --enable_phase4_features false

# Both standalone
python multi_gpu_inference.py \
  --enable_quality_metrics true \
  --enable_enterprise_monitoring true \
  --enable_phase4_features false
```

#### Full Phase 4 Integration
```bash
# Complete adaptive system
python multi_gpu_inference.py \
  --enable_phase4_features true \
  --enable_adaptive_sampling true \
  --enable_quality_metrics true \
  --enable_enterprise_monitoring true
```

### Quality Assessment Integration

#### External Quality Assessor
```python
class CustomQualityAssessor:
    def assess_quality(self, structures):
        # Custom quality assessment logic
        quality_score = self.calculate_quality(structures)
        report = self.generate_report(structures)
        return quality_score, report
    
    def filter_structures(self, structures, threshold):
        # Custom filtering logic
        return [s for s in structures if self.meets_threshold(s, threshold)]

# Integrate with sampler
sampler = ModularAdaptiveSampler(
    quality_assessor=CustomQualityAssessor()
)
```

### Enterprise Monitoring Integration

#### Custom Callbacks
```python
def enterprise_callback(event_type, **kwargs):
    if event_type == 'adaptation_start':
        dashboard.log_start(kwargs['config'])
    elif event_type == 'adaptation_step':
        metrics = kwargs['metrics']
        result = kwargs['result'] 
        dashboard.update_metrics(metrics, result)
    elif event_type == 'adaptation_complete':
        dashboard.log_completion(kwargs['stats'])

# Register callback
enterprise_callbacks = EnterpriseCallbackAdapter()
enterprise_callbacks.register_callback(enterprise_callback)

sampler = ModularAdaptiveSampler(
    enterprise_callbacks=enterprise_callbacks
)
```

### Migration Guide

#### From Legacy AdaptiveSampler
```python
# Old way
sampler = AdaptiveSampler(config)

# New way (backward compatible)
sampler = create_adaptive_sampler(modular=False)  # Legacy mode
# OR
sampler = create_adaptive_sampler(modular=True)   # Enhanced mode
```

#### Adding Quality Assessment
```python
# Before: No quality integration
sampler = AdaptiveSampler(config)

# After: With quality integration
sampler = ModularAdaptiveSampler(
    config=config,
    quality_assessor=my_assessor
)
```

#### Enabling Enterprise Features
```python
# Before: No enterprise integration
sampler = AdaptiveSampler(config)

# After: With enterprise monitoring
sampler = create_full_adaptive_pipeline(
    enterprise_monitoring=True
)
```

### Best Practices

#### 1. Choose Appropriate Presets
- **Development**: Use "balanced" preset
- **Experimentation**: Use "aggressive" preset  
- **Production**: Use "production" preset
- **Critical Systems**: Use "conservative" preset

#### 2. Modular Feature Usage
- Enable only features you need
- Use standalone mode for simple quality assessment
- Use full Phase 4 for adaptive optimization
- Add enterprise monitoring for production deployments

#### 3. Custom Component Development
- Implement interfaces for custom strategies
- Use adapters for external system integration
- Register callbacks for monitoring integration
- Follow protocol definitions for compatibility

#### 4. Performance Considerations
- Modular components have minimal overhead
- Standalone usage avoids adaptive sampling overhead
- Enterprise callbacks are optional and lightweight
- Quality assessment can be externalized

### Troubleshooting

#### Common Issues

1. **"Feature requires Phase 4" Error**
   - Solution: Set `--enable_phase4_features true` or use standalone mode

2. **"Component not available" Warning**
   - Solution: Install required dependencies or disable feature

3. **"Callback failed" Warning**
   - Solution: Check callback implementation and error handling

#### Debug Information
```python
# Enable detailed logging
import logging
logging.getLogger('mattergen.common.utils.adaptive_sampler').setLevel(logging.DEBUG)

# Check component status
print(sampler.sampling_stats['component_usage'])
print(sampler.get_optimization_summary())
```

### Future Extensions

The modular architecture supports future enhancements:

- Custom machine learning strategies
- Advanced quality prediction models
- Real-time dashboard integration
- Multi-objective optimization
- Custom convergence criteria
- Distributed sampling coordination

### Example Scripts

See `examples/modular_adaptive_sampler_demo.py` for comprehensive usage examples including:

- Basic usage patterns
- Modular configuration
- Quality integration
- Enterprise callbacks
- Preset comparisons
- Standalone component usage
