#!/bin/bash
# Real-time GPU monitoring for production tests
# ============================================

echo "📊 Real-time GPU Monitoring for MatterGen Production Test"
echo "=========================================================="
echo ""
echo "This will monitor GPU usage in real-time during the production test."
echo "Open this in a separate terminal while running the production test."
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Create log file with timestamp
LOG_FILE="gpu_production_monitoring_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Logging to: $LOG_FILE"
echo ""

# Header for log file
echo "Timestamp,GPU_ID,GPU_Util_%,Memory_Used_MB,Memory_Total_MB,Memory_Util_%,Temp_C,Power_W,Processes" > "$LOG_FILE"

# Function to display GPU stats
display_gpu_stats() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - GPU Status:"
    echo "┌─────┬──────────┬─────────────┬────────────┬─────────┬───────┬─────────┐"
    echo "│ GPU │ Util %   │ Memory      │ Memory %   │ Temp °C │ Power │ Procs   │"
    echo "├─────┼──────────┼─────────────┼────────────┼─────────┼───────┼─────────┤"
    
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    while IFS=', ' read -r gpu_id util mem_used mem_total temp power; do
        # Calculate memory percentage
        mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "0")
        
        # Count processes on this GPU
        procs=$(nvidia-smi --query-compute-apps=gpu_uuid,name --format=csv,noheader 2>/dev/null | wc -l)
        
        # Log to file
        echo "$(date '+%Y-%m-%d %H:%M:%S'),$gpu_id,$util,$mem_used,$mem_total,$mem_percent,$temp,$power,$procs" >> "$LOG_FILE"
        
        # Display in table format
        printf "│ %-3s │ %-8s │ %5s/%5s │ %-10s │ %-7s │ %-5s │ %-7s │\n" \
               "$gpu_id" "$util%" "$mem_used" "$mem_total" "${mem_percent}%" "${temp}°C" "${power}W" "$procs"
    done
    
    echo "└─────┴──────────┴─────────────┴────────────┴─────────┴───────┴─────────┘"
}

# Function to show running processes
show_processes() {
    echo ""
    echo "🔄 Active GPU Processes:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,name,used_memory --format=csv,noheader 2>/dev/null | \
    while IFS=', ' read -r gpu_uuid pid name mem; do
        gpu_id=$(nvidia-smi --query-gpu=gpu_uuid,index --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f2 | tr -d ' ')
        echo "   GPU $gpu_id: PID $pid - $name (${mem} MB)"
    done
    
    if [ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) -eq 0 ]; then
        echo "   No active GPU processes"
    fi
}

# Function to show summary stats
show_summary() {
    echo ""
    echo "📈 Summary Statistics:"
    
    # Average utilization
    avg_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | \
               awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
    
    # Total memory used vs available
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{used+=$1; total+=$2} END {
        printf "   Average GPU Utilization: %s%%\n", "'$avg_util'"
        printf "   Total Memory Used: %.1f GB / %.1f GB (%.1f%%)\n", used/1024, total/1024, (used*100)/total
    }'
    
    # Running time
    if [ ! -f "/tmp/monitor_start_time" ]; then
        date +%s > /tmp/monitor_start_time
    fi
    start_time=$(cat /tmp/monitor_start_time)
    current_time=$(date +%s)
    duration=$((current_time - start_time))
    hours=$((duration / 3600))
    minutes=$(((duration % 3600) / 60))
    seconds=$((duration % 60))
    printf "   Monitoring Duration: %02d:%02d:%02d\n" $hours $minutes $seconds
}

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Monitoring stopped"
    echo "📝 Complete log saved to: $LOG_FILE"
    rm -f /tmp/monitor_start_time
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Make sure NVIDIA drivers are installed."
    exit 1
fi

# Check if bc is available for calculations
if ! command -v bc &> /dev/null; then
    echo "⚠️  'bc' not found. Installing for calculations..."
    # Try to install bc
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y bc
    elif command -v yum &> /dev/null; then
        sudo yum install -y bc
    else
        echo "⚠️  Please install 'bc' for percentage calculations"
    fi
fi

# Main monitoring loop
while true; do
    clear
    echo "📊 MatterGen Production Test - GPU Monitoring"
    echo "=============================================="
    echo "Log file: $LOG_FILE"
    echo ""
    
    display_gpu_stats
    show_processes
    show_summary
    
    echo ""
    echo "⏱️  Updates every 3 seconds | Press Ctrl+C to stop"
    
    sleep 3
done
