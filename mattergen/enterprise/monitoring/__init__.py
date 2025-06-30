"""
Enterprise Monitoring System
===========================

Real-time monitoring and metrics collection for MatterGen production deployments.
Provides comprehensive monitoring of:
- Structure generation performance
- GPU utilization and health
- Quality metrics trends
- System resource usage
- Error rates and alerting
"""

import time
import threading
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class SystemMetrics:
    """System-level metrics snapshot."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_metrics: Dict[int, Dict[str, float]]  # {gpu_id: {metric: value}}
    disk_usage: float
    network_io: Dict[str, float]


@dataclass
class GenerationMetrics:
    """Structure generation metrics."""
    timestamp: float
    batch_id: int
    gpu_id: int
    structures_generated: int
    batch_duration: float
    quality_score: float
    throughput: float  # structures/second
    memory_peak: float
    error_count: int


@dataclass
class QualityMetrics:
    """Quality assessment metrics."""
    timestamp: float
    batch_id: int
    overall_quality: float
    quality_distribution: Dict[str, float]
    improvement_rate: float
    trend: str  # "improving", "stable", "degrading"
    convergence_status: str


class MetricsCollector:
    """
    Collects and stores metrics from various MatterGen components.
    
    Features:
    - Real-time metric collection
    - Configurable retention policies
    - Thread-safe operations
    - Automatic aggregation
    - Alert threshold monitoring
    """
    
    def __init__(self, 
                 retention_hours: int = 24,
                 max_points_per_metric: int = 10000):
        self.retention_hours = retention_hours
        self.max_points_per_metric = max_points_per_metric
        
        # Metric storage
        self.metrics = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.system_metrics = deque(maxlen=max_points_per_metric)
        self.generation_metrics = deque(maxlen=max_points_per_metric)
        self.quality_metrics = deque(maxlen=max_points_per_metric)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Auto-cleanup
        self._cleanup_thread = None
        self._running = False
        
        logger.info(f"Metrics collector initialized (retention: {retention_hours}h)")
    
    def start(self):
        """Start the metrics collector and cleanup thread."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="MetricsCleanup"
            )
            self._cleanup_thread.start()
            logger.info("Metrics collector started")
    
    def stop(self):
        """Stop the metrics collector."""
        with self._lock:
            self._running = False
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5)
            logger.info("Metrics collector stopped")
    
    def add_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add a metric point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        
        with self._lock:
            self.metrics[name].append(point)
    
    def add_system_metrics(self, metrics: SystemMetrics):
        """Add system metrics snapshot."""
        with self._lock:
            self.system_metrics.append(metrics)
    
    def add_generation_metrics(self, metrics: GenerationMetrics):
        """Add generation metrics."""
        with self._lock:
            self.generation_metrics.append(metrics)
            
            # Check for performance alerts
            self._check_performance_alerts(metrics)
    
    def add_quality_metrics(self, metrics: QualityMetrics):
        """Add quality metrics."""
        with self._lock:
            self.quality_metrics.append(metrics)
            
            # Check for quality alerts
            self._check_quality_alerts(metrics)
    
    def get_metric_history(self, 
                          name: str, 
                          hours: int = 1) -> List[MetricPoint]:
        """Get metric history for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            if name not in self.metrics:
                return []
            
            return [
                point for point in self.metrics[name]
                if point.timestamp >= cutoff_time
            ]
    
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        with self._lock:
            current_time = time.time()
            
            # Latest system metrics
            latest_system = self.system_metrics[-1] if self.system_metrics else None
            
            # Recent generation metrics (last 5 minutes)
            recent_cutoff = current_time - 300
            recent_generation = [
                m for m in self.generation_metrics
                if m.timestamp >= recent_cutoff
            ]
            
            # Recent quality metrics
            recent_quality = [
                m for m in self.quality_metrics
                if m.timestamp >= recent_cutoff
            ]
            
            # Calculate aggregates
            total_structures = sum(m.structures_generated for m in recent_generation)
            avg_throughput = (
                sum(m.throughput for m in recent_generation) / len(recent_generation)
                if recent_generation else 0.0
            )
            avg_quality = (
                sum(m.overall_quality for m in recent_quality) / len(recent_quality)
                if recent_quality else 0.0
            )
            
            return {
                "timestamp": current_time,
                "system": asdict(latest_system) if latest_system else None,
                "generation": {
                    "total_structures_5min": total_structures,
                    "avg_throughput": avg_throughput,
                    "active_batches": len(recent_generation),
                    "total_batches": len(self.generation_metrics)
                },
                "quality": {
                    "avg_quality_5min": avg_quality,
                    "quality_assessments": len(recent_quality),
                    "total_assessments": len(self.quality_metrics)
                },
                "alerts": {
                    "active_count": len(self.alert_callbacks),
                    "last_check": current_time
                }
            }
    
    def get_performance_analytics(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance analytics for the specified period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            # Filter metrics to time period
            generation_data = [
                m for m in self.generation_metrics
                if m.timestamp >= cutoff_time
            ]
            quality_data = [
                m for m in self.quality_metrics
                if m.timestamp >= cutoff_time
            ]
            system_data = [
                m for m in self.system_metrics
                if m.timestamp >= cutoff_time
            ]
            
            if not generation_data:
                return {"error": "No data available for the specified period"}
            
            # Calculate analytics
            total_structures = sum(m.structures_generated for m in generation_data)
            total_time = hours * 3600
            overall_throughput = total_structures / total_time if total_time > 0 else 0
            
            # Quality trends
            quality_scores = [m.overall_quality for m in quality_data]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # GPU utilization
            gpu_utilization = {}
            if system_data:
                for system_metric in system_data:
                    for gpu_id, gpu_metrics in system_metric.gpu_metrics.items():
                        if gpu_id not in gpu_utilization:
                            gpu_utilization[gpu_id] = []
                        gpu_utilization[gpu_id].append(gpu_metrics.get('utilization', 0))
            
            avg_gpu_utilization = {
                gpu_id: sum(utils) / len(utils) if utils else 0
                for gpu_id, utils in gpu_utilization.items()
            }
            
            return {
                "period_hours": hours,
                "total_structures": total_structures,
                "overall_throughput": overall_throughput,
                "quality_metrics": {
                    "avg_quality": avg_quality,
                    "quality_samples": len(quality_scores),
                    "min_quality": min(quality_scores) if quality_scores else 0,
                    "max_quality": max(quality_scores) if quality_scores else 0
                },
                "system_metrics": {
                    "gpu_utilization": avg_gpu_utilization,
                    "samples": len(system_data)
                },
                "performance_score": self._calculate_performance_score(
                    overall_throughput, avg_quality, avg_gpu_utilization
                )
            }
    
    def register_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Register an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def _check_performance_alerts(self, metrics: GenerationMetrics):
        """Check for performance-related alerts."""
        # Low throughput alert
        if metrics.throughput < 0.1:  # Less than 0.1 structures/second
            self._trigger_alert("low_throughput", {
                "gpu_id": metrics.gpu_id,
                "batch_id": metrics.batch_id,
                "throughput": metrics.throughput,
                "threshold": 0.1
            })
        
        # High error rate alert
        if metrics.error_count > 0:
            self._trigger_alert("generation_errors", {
                "gpu_id": metrics.gpu_id,
                "batch_id": metrics.batch_id,
                "error_count": metrics.error_count
            })
    
    def _check_quality_alerts(self, metrics: QualityMetrics):
        """Check for quality-related alerts."""
        # Quality degradation alert
        if metrics.overall_quality < 0.5:  # Quality below 50%
            self._trigger_alert("quality_degradation", {
                "batch_id": metrics.batch_id,
                "quality": metrics.overall_quality,
                "threshold": 0.5,
                "trend": metrics.trend
            })
        
        # Convergence issues
        if metrics.trend == "degrading":
            self._trigger_alert("quality_trend_degrading", {
                "batch_id": metrics.batch_id,
                "quality": metrics.overall_quality,
                "improvement_rate": metrics.improvement_rate
            })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger an alert to all registered callbacks."""
        alert_data = {
            "type": alert_type,
            "timestamp": time.time(),
            "data": data
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _calculate_performance_score(self, 
                                   throughput: float,
                                   quality: float,
                                   gpu_utilization: Dict[int, float]) -> float:
        """Calculate overall performance score (0-100)."""
        # Normalize throughput (assume max 10 structures/second)
        throughput_score = min(throughput / 10.0, 1.0) * 30
        
        # Quality score (0-1 -> 0-40 points)
        quality_score = quality * 40
        
        # GPU utilization score (average utilization -> 0-30 points)
        avg_utilization = (
            sum(gpu_utilization.values()) / len(gpu_utilization)
            if gpu_utilization else 0
        )
        utilization_score = avg_utilization * 30
        
        return throughput_score + quality_score + utilization_score
    
    def _cleanup_worker(self):
        """Background worker to clean up old metrics."""
        while self._running:
            try:
                cutoff_time = time.time() - (self.retention_hours * 3600)
                
                with self._lock:
                    # Clean up metric points
                    for name, points in self.metrics.items():
                        while points and points[0].timestamp < cutoff_time:
                            points.popleft()
                    
                    # Clean up system metrics
                    while (self.system_metrics and 
                           self.system_metrics[0].timestamp < cutoff_time):
                        self.system_metrics.popleft()
                    
                    # Clean up generation metrics
                    while (self.generation_metrics and 
                           self.generation_metrics[0].timestamp < cutoff_time):
                        self.generation_metrics.popleft()
                    
                    # Clean up quality metrics
                    while (self.quality_metrics and 
                           self.quality_metrics[0].timestamp < cutoff_time):
                        self.quality_metrics.popleft()
                
                # Sleep for 5 minutes before next cleanup
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                time.sleep(60)  # Wait a minute before retrying


class SystemMonitor:
    """
    System resource monitoring for MatterGen deployments.
    
    Monitors:
    - CPU usage
    - Memory usage
    - GPU utilization and memory
    - Disk space
    - Network I/O
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self, interval_seconds: int = 10):
        """Start system monitoring with specified interval."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_worker,
            args=(interval_seconds,),
            daemon=True,
            name="SystemMonitor"
        )
        self._monitor_thread.start()
        logger.info(f"System monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=15)
        logger.info("System monitoring stopped")
    
    def _monitor_worker(self, interval: int):
        """Background worker for system monitoring."""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_collector.add_system_metrics(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        import psutil
        
        current_time = time.time()
        
        # CPU and memory
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        # GPU metrics (if nvidia-ml-py is available)
        gpu_metrics = self._collect_gpu_metrics()
        
        return SystemMetrics(
            timestamp=current_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_metrics=gpu_metrics,
            disk_usage=disk_usage,
            network_io=network_io
        )
    
    def _collect_gpu_metrics(self) -> Dict[int, Dict[str, float]]:
        """Collect GPU metrics using nvidia-ml-py."""
        gpu_metrics = {}
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_usage = (memory_info.used / memory_info.total) * 100
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0.0
                
                gpu_metrics[i] = {
                    "utilization": utilization.gpu,
                    "memory_utilization": utilization.memory,
                    "memory_usage_percent": memory_usage,
                    "memory_used_mb": memory_info.used / 1024 / 1024,
                    "memory_total_mb": memory_info.total / 1024 / 1024,
                    "temperature": temp,
                    "power_watts": power
                }
                
        except ImportError:
            logger.warning("pynvml not available, GPU metrics disabled")
        except Exception as e:
            logger.error(f"GPU metrics collection failed: {e}")
        
        return gpu_metrics


# Global metrics collector instance
_global_metrics_collector = None

def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the global metrics collector instance."""
    return _global_metrics_collector

def initialize_monitoring(retention_hours: int = 24) -> MetricsCollector:
    """Initialize the global monitoring system."""
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector(retention_hours=retention_hours)
        _global_metrics_collector.start()
        
        # Start system monitoring
        system_monitor = SystemMonitor(_global_metrics_collector)
        system_monitor.start_monitoring(interval_seconds=10)
        
        logger.info("Enterprise monitoring system initialized")
    
    return _global_metrics_collector

def shutdown_monitoring():
    """Shutdown the global monitoring system."""
    global _global_metrics_collector
    
    if _global_metrics_collector:
        _global_metrics_collector.stop()
        _global_metrics_collector = None
        logger.info("Enterprise monitoring system shutdown")
