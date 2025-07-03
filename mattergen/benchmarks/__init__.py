"""
MatterGen Benchmarks Module
===========================

This module contains comprehensive benchmarking and validation tools for MatterGen,
including crystalline accuracy validation, performance benchmarking, and debugging utilities.

Available modules:
- crystalline_accuracy_validator: Comprehensive crystalline structure validation
- real_crystalline_baseline: Real crystalline data baseline validation
- comprehensive_benchmark_suite: Complete benchmark suite runner
- comprehensive_benchmark: Individual benchmark operations

Available scripts:
- debug_imports.py: Import debugging utilities script
- debug_multi_gpu.py: Multi-GPU debugging tools script
"""

from .crystalline_accuracy_validator import (
    CrystallineAccuracyValidator,
    CrystallineMetrics,
    create_comprehensive_benchmark_suite
)

from .real_crystalline_baseline import (
    create_real_baseline_validator,
    validate_against_icsd_standards
)

from .comprehensive_benchmark_suite import ComprehensiveBenchmarkRunner

from .real_crystalline_baseline import RealCrystallineBaseline

from .comprehensive_benchmark import run_benchmark_test

__all__ = [
    'CrystallineAccuracyValidator',
    'CrystallineMetrics', 
    'create_comprehensive_benchmark_suite',
    'create_real_baseline_validator',
    'validate_against_icsd_standards',
    'ComprehensiveBenchmarkRunner',
    'RealCrystallineBaseline',
    'run_benchmark_test'
]
