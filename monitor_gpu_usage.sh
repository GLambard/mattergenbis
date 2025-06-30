#!/bin/bash
# GPU Monitoring Script for MatterGen Benchmarks

echo "🔍 Monitoring GPU Usage During MatterGen Benchmarks"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Create log file with timestamp
LOG_FILE="gpu_usage_$(date +%Y%m%d_%H%M%S).log"

echo "Timestamp,GPU_ID,GPU_Name,Memory_Used_MB,Memory_Total_MB,GPU_Util_%,Temperature_C,Power_W" > $LOG_FILE

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get GPU stats and format for CSV
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    while IFS=', ' read -r gpu_id gpu_name mem_used mem_total util temp power; do
        echo "$timestamp,$gpu_id,$gpu_name,$mem_used,$mem_total,$util,$temp,$power" >> $LOG_FILE
        printf "GPU %s: %3s%% util | %5s/%5s MB | %2s°C | %5s W | %s\n" \
               "$gpu_id" "$util" "$mem_used" "$mem_total" "$temp" "$power" "$gpu_name"
    done
    
    echo "----------------------------------------"
    sleep 2
done
