#!/usr/bin/env python3
print("Starting Phase 4.2 import test...")

import sys
import os
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

print("Testing basic imports...")
try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

print("Testing project imports...")
try:
    from mattergen.common.utils.advanced_quality_metrics import AdvancedQualityMetrics
    print("✅ AdvancedQualityMetrics imported successfully")
except Exception as e:
    print(f"❌ AdvancedQualityMetrics import failed: {e}")

try:
    from mattergen.common.utils.quality_reporting import AdvancedQualityReporter
    print("✅ AdvancedQualityReporter imported successfully")
except Exception as e:
    print(f"❌ AdvancedQualityReporter import failed: {e}")

try:
    from mattergen.common.utils.quality_reporting import QualityReporter  
    print("✅ QualityReporter imported successfully")
except Exception as e:
    print(f"❌ QualityReporter import failed: {e}")

print("Import test complete!")
