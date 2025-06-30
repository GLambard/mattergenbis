#!/usr/bin/env python3
"""Quick Phase 4.2 validation test."""

print("=== Phase 4.2 Quick Test ===")

# Test imports
print("1. Testing imports...")
try:
    from mattergen.common.utils.advanced_quality_metrics import AdvancedQualityMetrics
    from mattergen.common.utils.quality_reporting import QualityReporter
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")

# Test basic initialization
print("2. Testing initialization...")
try:
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test AdvancedQualityMetrics
        metrics = AdvancedQualityMetrics(
            enable_ml_prediction=False,
            enable_trend_analysis=True,
            output_dir=temp_dir
        )
        print("✅ AdvancedQualityMetrics initialized")
        
        # Test QualityReporter  
        reporter = QualityReporter(
            output_dir=temp_dir,
            enable_visualizations=False
        )
        print("✅ QualityReporter initialized")
        
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("=== Test Complete ===")
