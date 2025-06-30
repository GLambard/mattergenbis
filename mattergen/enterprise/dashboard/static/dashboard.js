// MatterGen Enterprise Dashboard JavaScript

class MatterGenDashboard {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.lastUpdate = null;
        this.updateInterval = 5000; // 5 seconds
        
        this.initializeSocket();
        this.initializeCharts();
        this.setupEventHandlers();
        
        console.log('MatterGen Dashboard initialized');
    }

    initializeSocket() {
        // Initialize Socket.IO connection
        this.socket = io();
        
        // Connection events
        this.socket.on('connect', () => {
            console.log('Connected to dashboard server');
            this.updateConnectionStatus('connected');
            this.requestInitialData();
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from dashboard server');
            this.updateConnectionStatus('disconnected');
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.updateConnectionStatus('error');
        });

        // Data update events
        this.socket.on('metrics_update', (data) => {
            this.handleMetricsUpdate(data);
        });

        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            this.showAlert('Error: ' + error.message, 'danger');
        });

        this.socket.on('status', (data) => {
            console.log('Status:', data.message);
        });
    }

    initializeCharts() {
        // Initialize Chart.js charts
        const chartConfig = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        displayFormats: {
                            minute: 'HH:mm',
                            hour: 'HH:mm'
                        }
                    }
                },
                y: {
                    beginAtZero: true
                }
            }
        };

        // Throughput chart
        const throughputCtx = document.getElementById('throughput-chart');
        this.charts.throughput = new Chart(throughputCtx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Structures/sec',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                ...chartConfig,
                scales: {
                    ...chartConfig.scales,
                    y: {
                        ...chartConfig.scales.y,
                        title: {
                            display: true,
                            text: 'Structures per Second'
                        }
                    }
                }
            }
        });

        // Quality chart
        const qualityCtx = document.getElementById('quality-chart');
        this.charts.quality = new Chart(qualityCtx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Quality Score',
                    data: [],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                ...chartConfig,
                scales: {
                    ...chartConfig.scales,
                    y: {
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Quality Score'
                        }
                    }
                }
            }
        });

        // System metrics chart
        const systemCtx = document.getElementById('system-chart');
        this.charts.system = new Chart(systemCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'CPU Usage (%)',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Memory Usage (%)',
                        data: [],
                        borderColor: '#f39c12',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...chartConfig,
                scales: {
                    ...chartConfig.scales,
                    y: {
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Usage (%)'
                        }
                    }
                }
            }
        });
    }

    setupEventHandlers() {
        // Manual refresh button (if added)
        document.addEventListener('DOMContentLoaded', () => {
            // Add refresh functionality if needed
        });

        // Window visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('Dashboard hidden, reducing update frequency');
            } else {
                console.log('Dashboard visible, resuming normal updates');
                this.requestUpdate();
            }
        });
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        
        statusElement.className = 'badge';
        
        switch (status) {
            case 'connected':
                statusElement.classList.add('bg-success');
                statusElement.textContent = 'Connected';
                break;
            case 'disconnected':
                statusElement.classList.add('bg-warning');
                statusElement.textContent = 'Disconnected';
                break;
            case 'error':
                statusElement.classList.add('bg-danger');
                statusElement.textContent = 'Error';
                break;
        }
    }

    requestInitialData() {
        // Request initial data load
        this.requestUpdate();
        
        // Fetch historical data for charts
        this.fetchHistoricalData();
    }

    requestUpdate() {
        if (this.socket && this.socket.connected) {
            this.socket.emit('request_update');
        }
    }

    handleMetricsUpdate(data) {
        console.log('Received metrics update:', data);
        this.lastUpdate = new Date();
        
        // Update status cards
        this.updateStatusCards(data);
        
        // Update charts
        this.updateCharts(data);
        
        // Update GPU status
        this.updateGpuStatus(data);
        
        // Update activity table
        this.updateActivityTable(data);
    }

    updateStatusCards(data) {
        // Generation rate
        const genRate = data.recent_generation_summary?.avg_throughput || 0;
        document.getElementById('generation-rate').textContent = genRate.toFixed(2);
        
        // Quality score
        const quality = data.recent_quality_summary?.avg_quality || 0;
        document.getElementById('quality-score').textContent = quality.toFixed(3);
        
        // GPU utilization
        const gpuUtil = this.calculateAverageGpuUtilization(data);
        document.getElementById('gpu-utilization').textContent = gpuUtil.toFixed(1);
        
        // Active batches
        const activeBatches = data.recent_generation_summary?.total_batches || 0;
        document.getElementById('active-batches').textContent = activeBatches;
    }

    updateCharts(data) {
        const now = new Date();
        
        // Update throughput chart
        if (data.recent_generation_summary?.avg_throughput !== undefined) {
            this.addDataPoint(
                this.charts.throughput,
                0,
                now,
                data.recent_generation_summary.avg_throughput
            );
        }
        
        // Update quality chart
        if (data.recent_quality_summary?.avg_quality !== undefined) {
            this.addDataPoint(
                this.charts.quality,
                0,
                now,
                data.recent_quality_summary.avg_quality
            );
        }
        
        // Update system chart
        if (data.latest_system) {
            this.addDataPoint(
                this.charts.system,
                0,
                now,
                data.latest_system.cpu_usage
            );
            this.addDataPoint(
                this.charts.system,
                1,
                now,
                data.latest_system.memory_usage
            );
        }
    }

    addDataPoint(chart, datasetIndex, timestamp, value) {
        const dataset = chart.data.datasets[datasetIndex];
        
        dataset.data.push({
            x: timestamp,
            y: value
        });
        
        // Keep only last 50 points
        if (dataset.data.length > 50) {
            dataset.data.shift();
        }
        
        chart.update('none');
    }

    updateGpuStatus(data) {
        const gpuContainer = document.getElementById('gpu-status');
        
        if (!data.latest_system?.gpu_metrics) {
            gpuContainer.innerHTML = '<p class="text-muted">No GPU data available</p>';
            return;
        }
        
        let html = '';
        for (const [gpuId, metrics] of Object.entries(data.latest_system.gpu_metrics)) {
            const utilization = metrics.utilization || 0;
            const memory = metrics.memory_used || 0;
            const temperature = metrics.temperature || 0;
            
            html += `
                <div class="gpu-card mb-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <h6 class="mb-0">GPU ${gpuId}</h6>
                        <span class="badge bg-light text-dark">${utilization.toFixed(1)}%</span>
                    </div>
                    <div class="progress mt-2" style="height: 6px;">
                        <div class="progress-bar" style="width: ${utilization}%"></div>
                    </div>
                    <div class="gpu-metrics mt-2">
                        <div class="gpu-metric">
                            <div class="value">${memory.toFixed(0)}MB</div>
                            <div class="label">Memory</div>
                        </div>
                        <div class="gpu-metric">
                            <div class="value">${temperature.toFixed(0)}°C</div>
                            <div class="label">Temp</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        gpuContainer.innerHTML = html;
    }

    updateActivityTable(data) {
        const tableBody = document.getElementById('activity-table');
        
        if (!data.recent_generation || data.recent_generation.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No recent activity</td></tr>';
            return;
        }
        
        let html = '';
        data.recent_generation.slice(-10).reverse().forEach(batch => {
            const time = new Date(batch.timestamp * 1000).toLocaleTimeString();
            const qualityClass = this.getQualityClass(batch.quality_score);
            
            html += `
                <tr>
                    <td>${time}</td>
                    <td>#${batch.batch_id}</td>
                    <td>GPU ${batch.gpu_id}</td>
                    <td>${batch.structures_generated}</td>
                    <td>${batch.batch_duration.toFixed(2)}s</td>
                    <td>
                        <span class="status-indicator ${qualityClass}"></span>
                        ${batch.quality_score.toFixed(3)}
                    </td>
                    <td>${batch.throughput.toFixed(2)}/s</td>
                </tr>
            `;
        });
        
        tableBody.innerHTML = html;
    }

    calculateAverageGpuUtilization(data) {
        if (!data.latest_system?.gpu_metrics) return 0;
        
        const utilizations = Object.values(data.latest_system.gpu_metrics)
            .map(metrics => metrics.utilization || 0);
        
        return utilizations.length > 0 
            ? utilizations.reduce((a, b) => a + b, 0) / utilizations.length 
            : 0;
    }

    getQualityClass(score) {
        if (score >= 0.8) return 'status-excellent';
        if (score >= 0.6) return 'status-good';
        if (score >= 0.4) return 'status-fair';
        return 'status-poor';
    }

    fetchHistoricalData() {
        // Fetch historical data for initial chart population
        fetch('/api/metrics/history?metric=throughput&hours=1')
            .then(response => response.json())
            .then(data => {
                if (data.data) {
                    data.data.forEach(point => {
                        this.addDataPoint(
                            this.charts.throughput,
                            0,
                            new Date(point.timestamp * 1000),
                            point.value
                        );
                    });
                }
            })
            .catch(error => console.error('Error fetching throughput history:', error));
        
        fetch('/api/metrics/history?metric=quality&hours=1')
            .then(response => response.json())
            .then(data => {
                if (data.data) {
                    data.data.forEach(point => {
                        this.addDataPoint(
                            this.charts.quality,
                            0,
                            new Date(point.timestamp * 1000),
                            point.value
                        );
                    });
                }
            })
            .catch(error => console.error('Error fetching quality history:', error));
    }

    showAlert(message, type = 'info') {
        const alertToast = document.getElementById('alert-toast');
        const alertMessage = document.getElementById('alert-message');
        
        alertMessage.textContent = message;
        
        // Update toast styling based on type
        alertToast.className = `toast show bg-${type}`;
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            alertToast.classList.add('hide');
        }, 5000);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new MatterGenDashboard();
});
