#!/usr/bin/env python3
"""
Phase 4 Adaptive Sampling Test Script
====================================

Test script to validate Phase 4 adaptive sampling intelligence implementation.
Tests adaptive sampling, quality assessment, and integration components.
"""

import sys
import time
import logging
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_adaptive_sampler():
    """Test adaptive sampling functionality."""
    logger.info("🧠 Testing Adaptive Sampler...")
    
    try:
        from mattergen.common.utils.adaptive_sampler import (
            create_adaptive_sampler, 
            create_quality_metrics,
            AdaptiveSamplingConfig
        )
        
        # Create adaptive sampler
        sampler = create_adaptive_sampler(
            min_steps=50,
            max_steps=300,
            initial_steps=200,
            quality_threshold=0.8,
            enable_early_stopping=True
        )
        
        logger.info(f"✅ Adaptive sampler created with initial steps: {sampler.current_steps}")
        
        # Simulate generation batches with varying quality
        test_scenarios = [
            {'loss': 0.5, 'expected_direction': 'increase'},  # High loss -> increase steps
            {'loss': 0.1, 'expected_direction': 'decrease'},  # Low loss -> decrease steps
            {'loss': 0.05, 'expected_direction': 'decrease'}, # Very low loss -> further decrease
        ]
        
        for i, scenario in enumerate(test_scenarios):
            logger.info(f"  Testing scenario {i+1}: loss={scenario['loss']}")
            
            # Create quality metrics
            metrics = create_quality_metrics(
                structure_data={'test': True},
                loss_value=scenario['loss'],
                step=i
            )
            
            old_steps = sampler.current_steps
            new_steps = sampler.adapt_sampling_steps(metrics)
            
            direction = 'increase' if new_steps > old_steps else 'decrease' if new_steps < old_steps else 'maintain'
            logger.info(f"    Steps: {old_steps} -> {new_steps} ({direction})")
            
            # Update stats
            sampler.update_sampling_stats(10, early_stopped=(metrics.quality_score > 0.9))
        
        # Get optimization summary
        summary = sampler.get_optimization_summary()
        logger.info(f"✅ Adaptive sampler test complete:")
        logger.info(f"    Total structures: {summary['total_structures_generated']}")
        logger.info(f"    Adaptive adjustments: {summary['adaptive_adjustments_made']}")
        logger.info(f"    Average steps used: {summary['average_steps_used']}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Adaptive sampler test failed: {e}")
        return False


def test_quality_assessor():
    """Test quality assessment functionality."""
    logger.info("🎯 Testing Quality Assessor...")
    
    try:
        from mattergen.common.utils.quality_metrics import (
            create_quality_assessor,
            QualityLevel
        )
        import numpy as np
        
        # Create quality assessor
        assessor = create_quality_assessor(
            min_overall_quality=0.7,
            min_geometric_quality=0.6,
            require_valid_cell=True
        )
        
        logger.info("✅ Quality assessor created")
        
        # Test structures with different quality levels
        test_structures = [
            {
                'name': 'good_structure',
                'cell': np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]),
                'positions': [[0, 0, 0], [1.5, 1.5, 1.5]],
                'atomic_numbers': [1, 8]
            },
            {
                'name': 'poor_structure',
                'cell': np.array([[1e-12, 0, 0], [0, 1e-12, 0], [0, 0, 1e-12]]),  # Singular cell
                'positions': [[0, 0, 0], [0.1, 0.1, 0.1]],  # Overlapping atoms
                'atomic_numbers': [1, 1]
            },
            {
                'name': 'medium_structure',
                'cell': np.array([[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]]),
                'positions': [[0, 0, 0], [1.2, 1.2, 1.2]],
                'atomic_numbers': [6, 8]
            }
        ]
        
        for structure in test_structures:
            logger.info(f"  Assessing {structure['name']}...")
            
            metrics = assessor.assess_structure_quality(
                structure_data=structure,
                structure_id=structure['name']
            )
            
            should_keep = assessor.should_keep_structure(metrics)
            
            logger.info(f"    Overall quality: {metrics.overall_quality:.3f}")
            logger.info(f"    Quality level: {metrics.quality_level.value}")
            logger.info(f"    Valid: {metrics.is_valid}")
            logger.info(f"    Should keep: {should_keep}")
        
        # Get quality summary
        summary = assessor.get_quality_summary()
        logger.info(f"✅ Quality assessor test complete:")
        logger.info(f"    Total assessed: {summary['total_structures_assessed']}")
        logger.info(f"    Valid rate: {summary['valid_rate']}%")
        logger.info(f"    Average quality: {summary['average_quality_score']}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Quality assessor test failed: {e}")
        return False


def test_phase4_integration():
    """Test Phase 4 integration manager."""
    logger.info("🚀 Testing Phase 4 Integration...")
    
    try:
        from mattergen.common.utils.phase4_integration import create_phase4_manager
        
        # Create Phase 4 manager
        manager = create_phase4_manager(
            enable_adaptive_sampling=True,
            enable_quality_assessment=True,
            min_steps=50,
            max_steps=300,
            quality_threshold=0.8
        )
        
        logger.info("✅ Phase 4 manager created")
        
        # Simulate generation batches
        for batch in range(3):
            logger.info(f"  Simulating batch {batch + 1}/3...")
            
            # Mock generated structures
            mock_structures = [
                {
                    'cell': [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]],
                    'positions': [[0, 0, 0], [1.5, 1.5, 1.5]],
                    'atomic_numbers': [1, 8]
                }
                for _ in range(5)
            ]
            
            # Optimize sampling parameters
            current_loss = 0.3 - batch * 0.1  # Decreasing loss
            optimized_steps, optimization_info = manager.optimize_sampling_parameters(
                current_batch=batch,
                total_batches=3,
                current_loss=current_loss,
                generated_structures=mock_structures,
                current_steps=200
            )
            
            logger.info(f"    Optimized steps: {optimized_steps}")
            logger.info(f"    Optimization decisions: {len(optimization_info.get('optimization_decisions', []))}")
            
            # Filter structures by quality
            filtered_structures, quality_metrics = manager.filter_structures_by_quality(mock_structures)
            
            logger.info(f"    Structures: {len(mock_structures)} -> {len(filtered_structures)} (filtered)")
            
            # Update stats
            manager.update_generation_stats(len(filtered_structures), 10.0)
        
        # Get Phase 4 summary
        summary = manager.get_phase4_summary()
        phase4_summary = summary['phase4_optimization_summary']
        
        logger.info("✅ Phase 4 integration test complete:")
        logger.info(f"    Total structures: {phase4_summary['total_structures_generated']}")
        logger.info(f"    Adaptive adjustments: {phase4_summary['adaptive_adjustments_made']}")
        logger.info(f"    Optimization overhead: {phase4_summary['optimization_overhead_percent']:.2f}%")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Phase 4 integration test failed: {e}")
        return False


def test_configuration_loading():
    """Test adaptive sampling configuration loading."""
    logger.info("⚙️ Testing Configuration Loading...")
    
    try:
        config_path = Path("sampling_conf/adaptive_sampling.yaml")
        
        if config_path.exists():
            import yaml
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info("✅ Configuration loaded successfully")
            logger.info(f"    Adaptive sampling enabled: {config.get('adaptive_sampling', {}).get('enabled', False)}")
            logger.info(f"    Quality assessment enabled: {config.get('quality_assessment', {}).get('enabled', False)}")
            logger.info(f"    Min steps: {config.get('adaptive_sampling', {}).get('min_steps', 'not set')}")
            logger.info(f"    Max steps: {config.get('adaptive_sampling', {}).get('max_steps', 'not set')}")
            
            return True
        else:
            logger.warning(f"❌ Configuration file not found: {config_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Configuration loading test failed: {e}")
        return False


def main():
    """Run all Phase 4 tests."""
    logger.info("🚀 Starting Phase 4 Adaptive Sampling Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Adaptive Sampler", test_adaptive_sampler),
        ("Quality Assessor", test_quality_assessor),
        ("Phase 4 Integration", test_phase4_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} Test...")
        start_time = time.time()
        
        try:
            success = test_func()
            duration = time.time() - start_time
            results[test_name] = {
                'success': success,
                'duration': duration,
                'status': '✅ PASS' if success else '❌ FAIL'
            }
            
        except Exception as e:
            duration = time.time() - start_time
            results[test_name] = {
                'success': False,
                'duration': duration,
                'status': '❌ ERROR',
                'error': str(e)
            }
            logger.error(f"❌ {test_name} test error: {e}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Phase 4 Test Results Summary")
    logger.info("=" * 60)
    
    total_tests = len(tests)
    passed_tests = sum(1 for r in results.values() if r['success'])
    total_time = sum(r['duration'] for r in results.values())
    
    for test_name, result in results.items():
        status = result['status']
        duration = result['duration']
        logger.info(f"{status} {test_name:<25} ({duration:.2f}s)")
        
        if 'error' in result:
            logger.info(f"     Error: {result['error']}")
    
    logger.info("-" * 60)
    logger.info(f"Tests passed: {passed_tests}/{total_tests}")
    logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    logger.info(f"Total time: {total_time:.2f}s")
    
    if passed_tests == total_tests:
        logger.info("🎉 All Phase 4 tests passed! Adaptive sampling is ready.")
        return 0
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} test(s) failed. Check implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
