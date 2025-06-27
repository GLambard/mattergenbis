#!/bin/bash
# Production Multi-GPU Test Launcher
# ===================================

echo "🚀 MatterGen Production Multi-GPU Test"
echo "======================================="
echo ""
echo "This script will run a comprehensive production test of the multi-GPU"
echo "optimization capabilities with a substantial number of structures."
echo ""

# Check if user wants to run the test
read -p "🎯 How many structures per test configuration? (default: 200): " STRUCTURES
STRUCTURES=${STRUCTURES:-200}

read -p "🔧 Maximum number of GPUs to test? (default: 8): " MAX_GPUS
MAX_GPUS=${MAX_GPUS:-8}

read -p "📁 Output directory name? (default: production_test_$(date +%Y%m%d_%H%M)): " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-"production_test_$(date +%Y%m%d_%H%M)"}

echo ""
echo "📊 Test Configuration:"
echo "   Structures per test: $STRUCTURES"
echo "   Max GPUs: $MAX_GPUS"
echo "   Output directory: $OUTPUT_DIR"
echo ""
echo "⏱️  Estimated test duration: 30-60 minutes"
echo "💾 Estimated disk usage: 1-5 GB"
echo ""

read -p "🚀 Proceed with production test? (y/N): " CONFIRM
if [[ $CONFIRM != [yY] ]]; then
    echo "❌ Test cancelled"
    exit 1
fi

echo ""
echo "🏃‍♂️ Starting production test..."
echo "📝 Logs will be displayed in real-time"
echo "⏹️  Press Ctrl+C to interrupt if needed"
echo ""

# Make script executable and run
chmod +x production_multi_gpu_test.py

# Run the test with specified parameters
python production_multi_gpu_test.py \
    --structures $STRUCTURES \
    --max_gpus $MAX_GPUS \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 Production test completed successfully!"
    echo "📁 Check results in: $OUTPUT_DIR"
    echo ""
    echo "🔍 Quick results summary:"
    if [ -f "$OUTPUT_DIR/test_*/production_test_results.json" ]; then
        echo "📊 Detailed results available in JSON format"
    fi
    echo ""
    echo "🚀 Your multi-GPU optimization is production-ready!"
else
    echo "❌ Production test failed (exit code: $EXIT_CODE)"
    echo "📋 Check the logs above for error details"
    echo "🔧 Common issues:"
    echo "   • Insufficient GPU memory (reduce batch size)"
    echo "   • Missing dependencies"
    echo "   • GPU driver issues"
fi

echo ""
echo "📈 To monitor GPU usage during future runs:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "🔄 To run tests again:"
echo "   ./run_production_test.sh"
echo ""

exit $EXIT_CODE
