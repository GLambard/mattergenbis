#!/usr/bin/env python3
"""Simple import test for Phase 4.2 modules."""

import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test Phase 4.2 imports."""
    print("Testing AdvancedQualityMetrics import...")
    try:
        from mattergen.common.utils.advanced_quality_metrics import AdvancedQualityMetrics
        print("✅ AdvancedQualityMetrics imported successfully")
    except Exception as e:
        print(f"❌ AdvancedQualityMetrics import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nTesting QualityReporter import...")
    try:
        from mattergen.common.utils.quality_reporting import QualityReporter
        print("✅ QualityReporter imported successfully")
    except Exception as e:
        print(f"❌ QualityReporter import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    print(f"\nTest result: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
