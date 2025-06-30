"""
Enterprise Analytics System
===========================

Advanced analytics and reporting for MatterGen enterprise deployments.
Provides deep insights into:
- Performance trends and optimization opportunities
- Quality pattern analysis
- Resource utilization optimization
- Predictive analytics for maintenance
- Cost optimization recommendations
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceInsight:
    """Performance insight data structure."""
    category: str  # "efficiency", "quality", "resource", "cost"
    severity: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    impact: float  # 0-1 scale
    recommendation: str
    metrics: Dict[str, float]
    timestamp: float


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    metric_name: str
    timeframe: str
    trend_direction: str  # "improving", "stable", "declining"
    change_rate: float  # percentage change
    confidence: float  # 0-1 confidence score
    forecast: List[float]  # predicted future values
    anomalies: List[Dict[str, Any]]


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation."""
    category: str
    priority: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    expected_improvement: float  # percentage
    implementation_effort: str  # "low", "medium", "high"
    cost_benefit_ratio: float
    steps: List[str]


class AnalyticsEngine:
    """
    Advanced analytics engine for MatterGen enterprise monitoring.
    
    Features:
    - Performance trend analysis
    - Anomaly detection
    - Predictive analytics
    - Optimization recommendations
    - Cost analysis
    - Resource utilization insights
    """
    
    def __init__(self, metrics_collector=None):
        self.metrics_collector = metrics_collector
        self.insights_history = []
        self.trend_cache = {}
        self.anomaly_models = {}
        
        logger.info("Analytics engine initialized")
    
    def analyze_performance_trends(self, hours: int = 24) -> List[TrendAnalysis]:
        """Analyze performance trends over specified time period."""
        if not self.metrics_collector:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        trends = []
        
        # Analyze throughput trends
        throughput_trend = self._analyze_metric_trend(
            "throughput", cutoff_time, current_time
        )
        if throughput_trend:
            trends.append(throughput_trend)
        
        # Analyze quality trends
        quality_trend = self._analyze_metric_trend(
            "quality", cutoff_time, current_time
        )
        if quality_trend:
            trends.append(quality_trend)
        
        # Analyze efficiency trends
        efficiency_trend = self._analyze_efficiency_trend(cutoff_time, current_time)
        if efficiency_trend:
            trends.append(efficiency_trend)
        
        # Analyze resource utilization trends
        resource_trends = self._analyze_resource_trends(cutoff_time, current_time)
        trends.extend(resource_trends)
        
        return trends
    
    def detect_anomalies(self, hours: int = 6) -> List[Dict[str, Any]]:
        """Detect anomalies in recent metrics."""
        if not self.metrics_collector:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        anomalies = []
        
        # Check for throughput anomalies
        throughput_anomalies = self._detect_throughput_anomalies(cutoff_time)
        anomalies.extend(throughput_anomalies)
        
        # Check for quality anomalies
        quality_anomalies = self._detect_quality_anomalies(cutoff_time)
        anomalies.extend(quality_anomalies)
        
        # Check for system resource anomalies
        resource_anomalies = self._detect_resource_anomalies(cutoff_time)
        anomalies.extend(resource_anomalies)
        
        return anomalies
    
    def generate_insights(self) -> List[PerformanceInsight]:
        """Generate performance insights and recommendations."""
        insights = []
        
        # Performance efficiency insights
        efficiency_insights = self._analyze_efficiency_insights()
        insights.extend(efficiency_insights)
        
        # Quality optimization insights
        quality_insights = self._analyze_quality_insights()
        insights.extend(quality_insights)
        
        # Resource utilization insights
        resource_insights = self._analyze_resource_insights()
        insights.extend(resource_insights)
        
        # Cost optimization insights
        cost_insights = self._analyze_cost_insights()
        insights.extend(cost_insights)
        
        # Store insights for tracking
        self.insights_history.extend(insights)
        
        return insights
    
    def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze recent performance data
        insights = self.generate_insights()
        
        # GPU utilization optimization
        gpu_rec = self._generate_gpu_optimization_recommendation()
        if gpu_rec:
            recommendations.append(gpu_rec)
        
        # Batch size optimization
        batch_rec = self._generate_batch_optimization_recommendation()
        if batch_rec:
            recommendations.append(batch_rec)
        
        # Quality enhancement recommendations
        quality_recs = self._generate_quality_recommendations()
        recommendations.extend(quality_recs)
        
        # Resource allocation recommendations
        resource_recs = self._generate_resource_recommendations()
        recommendations.extend(resource_recs)
        
        # Sort by priority and impact
        recommendations.sort(
            key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.priority],
                x.expected_improvement
            ),
            reverse=True
        )
        
        return recommendations
    
    def calculate_performance_score(self) -> Dict[str, float]:
        """Calculate overall performance score."""
        if not self.metrics_collector:
            return {"overall": 0.0, "components": {}}
        
        # Get recent metrics
        current_time = time.time()
        recent_cutoff = current_time - 3600  # Last hour
        
        recent_generation = [
            m for m in self.metrics_collector.generation_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        recent_quality = [
            m for m in self.metrics_collector.quality_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        recent_system = [
            m for m in self.metrics_collector.system_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        scores = {}
        
        # Throughput score (0-100)
        if recent_generation:
            avg_throughput = np.mean([m.throughput for m in recent_generation])
            max_theoretical = 50.0  # Adjust based on your system
            scores["throughput"] = min(100, (avg_throughput / max_theoretical) * 100)
        else:
            scores["throughput"] = 0
        
        # Quality score (0-100)
        if recent_quality:
            avg_quality = np.mean([m.overall_quality for m in recent_quality])
            scores["quality"] = avg_quality * 100
        else:
            scores["quality"] = 0
        
        # Resource efficiency score (0-100)
        if recent_system:
            avg_gpu_util = self._calculate_avg_gpu_utilization(recent_system)
            avg_cpu_util = np.mean([m.cpu_usage for m in recent_system])
            
            # Optimal utilization is around 80-90%
            gpu_efficiency = max(0, 100 - abs(85 - avg_gpu_util))
            cpu_efficiency = max(0, 100 - abs(75 - avg_cpu_util))
            
            scores["resource_efficiency"] = (gpu_efficiency + cpu_efficiency) / 2
        else:
            scores["resource_efficiency"] = 0
        
        # Stability score (0-100) - based on error rates
        if recent_generation:
            total_errors = sum(m.error_count for m in recent_generation)
            total_batches = len(recent_generation)
            error_rate = total_errors / total_batches if total_batches > 0 else 0
            scores["stability"] = max(0, 100 - (error_rate * 100))
        else:
            scores["stability"] = 100
        
        # Calculate overall score
        weights = {
            "throughput": 0.3,
            "quality": 0.3,
            "resource_efficiency": 0.25,
            "stability": 0.15
        }
        
        overall_score = sum(
            scores.get(component, 0) * weight
            for component, weight in weights.items()
        )
        
        return {
            "overall": overall_score,
            "components": scores,
            "timestamp": current_time
        }
    
    def _analyze_metric_trend(self, 
                             metric_type: str, 
                             start_time: float, 
                             end_time: float) -> Optional[TrendAnalysis]:
        """Analyze trend for a specific metric type."""
        
        if metric_type == "throughput":
            data = [
                (m.timestamp, m.throughput)
                for m in self.metrics_collector.generation_metrics
                if start_time <= m.timestamp <= end_time
            ]
        elif metric_type == "quality":
            data = [
                (m.timestamp, m.overall_quality)
                for m in self.metrics_collector.quality_metrics
                if start_time <= m.timestamp <= end_time
            ]
        else:
            return None
        
        if len(data) < 5:  # Need minimum data points
            return None
        
        timestamps, values = zip(*data)
        
        # Calculate trend
        trend_direction, change_rate = self._calculate_trend(values)
        confidence = self._calculate_trend_confidence(values)
        
        # Simple forecast (linear projection)
        forecast = self._generate_simple_forecast(values, periods=5)
        
        # Detect anomalies in the data
        anomalies = self._detect_value_anomalies(timestamps, values)
        
        return TrendAnalysis(
            metric_name=metric_type,
            timeframe=f"{(end_time - start_time) / 3600:.1f}h",
            trend_direction=trend_direction,
            change_rate=change_rate,
            confidence=confidence,
            forecast=forecast,
            anomalies=anomalies
        )
    
    def _calculate_trend(self, values: List[float]) -> Tuple[str, float]:
        """Calculate trend direction and change rate."""
        if len(values) < 2:
            return "stable", 0.0
        
        # Use linear regression to determine trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        
        # Calculate percentage change
        change_rate = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
        
        if abs(change_rate) < 2:
            return "stable", change_rate
        elif change_rate > 0:
            return "improving", change_rate
        else:
            return "declining", change_rate
    
    def _calculate_trend_confidence(self, values: List[float]) -> float:
        """Calculate confidence in trend analysis."""
        if len(values) < 3:
            return 0.0
        
        # Calculate R-squared for linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2
        
        return r_squared
    
    def _generate_simple_forecast(self, values: List[float], periods: int = 5) -> List[float]:
        """Generate simple linear forecast."""
        if len(values) < 2:
            return [values[0] if values else 0] * periods
        
        # Simple linear extrapolation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Fit linear trend
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        intercept = np.mean(y) - slope * np.mean(x)
        
        # Generate forecast
        forecast_x = np.arange(len(values), len(values) + periods)
        forecast = [slope * xi + intercept for xi in forecast_x]
        
        return forecast
    
    def _detect_value_anomalies(self, 
                               timestamps: List[float], 
                               values: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in value series."""
        if len(values) < 10:
            return []
        
        anomalies = []
        
        # Simple statistical anomaly detection
        mean_val = np.mean(values)
        std_val = np.std(values)
        threshold = 2.5  # Standard deviations
        
        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
            
            if z_score > threshold:
                anomalies.append({
                    "timestamp": timestamp,
                    "value": value,
                    "z_score": z_score,
                    "type": "statistical_outlier",
                    "severity": "high" if z_score > 3.5 else "medium"
                })
        
        return anomalies
    
    def _analyze_efficiency_insights(self) -> List[PerformanceInsight]:
        """Analyze efficiency-related insights."""
        insights = []
        
        if not self.metrics_collector.generation_metrics:
            return insights
        
        # Recent performance data
        recent_cutoff = time.time() - 3600
        recent_generation = [
            m for m in self.metrics_collector.generation_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_generation:
            return insights
        
        # Batch duration analysis
        durations = [m.batch_duration for m in recent_generation]
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        
        if max_duration > avg_duration * 2:
            insights.append(PerformanceInsight(
                category="efficiency",
                severity="medium",
                title="Inconsistent Batch Processing Times",
                description=f"Some batches take {max_duration/avg_duration:.1f}x longer than average",
                impact=0.3,
                recommendation="Review batch size and GPU memory allocation",
                metrics={"avg_duration": avg_duration, "max_duration": max_duration},
                timestamp=time.time()
            ))
        
        return insights
    
    def _calculate_avg_gpu_utilization(self, system_metrics: List) -> float:
        """Calculate average GPU utilization."""
        total_util = 0
        count = 0
        
        for metrics in system_metrics:
            for gpu_id, gpu_metrics in metrics.gpu_metrics.items():
                total_util += gpu_metrics.get('utilization', 0)
                count += 1
        
        return total_util / count if count > 0 else 0
    
    def _analyze_quality_insights(self) -> List[PerformanceInsight]:
        """Analyze quality-related insights."""
        insights = []
        
        if not self.metrics_collector or not self.metrics_collector.quality_metrics:
            return insights
        
        # Recent quality data
        recent_cutoff = time.time() - 3600
        recent_quality = [
            m for m in self.metrics_collector.quality_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_quality:
            return insights
        
        # Quality trend analysis
        quality_scores = [m.overall_quality for m in recent_quality]
        
        try:
            import numpy as np
            avg_quality = np.mean(quality_scores)
        except ImportError:
            avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality < 0.7:
            insights.append(PerformanceInsight(
                category="quality",
                severity="medium" if avg_quality > 0.5 else "high",
                title="Low Quality Scores Detected",
                description=f"Average quality score is {avg_quality:.3f}, below recommended threshold",
                impact=0.4,
                recommendation="Review generation parameters and consider parameter tuning",
                metrics={"avg_quality": avg_quality, "num_batches": len(recent_quality)},
                timestamp=time.time()
            ))
        
        return insights
    
    def _analyze_resource_insights(self) -> List[PerformanceInsight]:
        """Analyze resource utilization insights."""
        insights = []
        
        if not self.metrics_collector or not self.metrics_collector.system_metrics:
            return insights
        
        # Recent system data
        recent_cutoff = time.time() - 3600
        recent_system = [
            m for m in self.metrics_collector.system_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_system:
            return insights
        
        # GPU utilization analysis
        avg_gpu_util = self._calculate_avg_gpu_utilization(recent_system)
        
        if avg_gpu_util < 50:
            insights.append(PerformanceInsight(
                category="resource",
                severity="medium",
                title="Low GPU Utilization",
                description=f"Average GPU utilization is {avg_gpu_util:.1f}%, indicating underutilization",
                impact=0.3,
                recommendation="Consider increasing batch size or using multiple GPUs",
                metrics={"avg_gpu_utilization": avg_gpu_util},
                timestamp=time.time()
            ))
        
        return insights
    
    def _analyze_cost_insights(self) -> List[PerformanceInsight]:
        """Analyze cost optimization insights."""
        insights = []
        
        # Placeholder for cost analysis - could integrate with cloud cost APIs
        # For now, focus on efficiency metrics that impact cost
        
        if not self.metrics_collector or not self.metrics_collector.generation_metrics:
            return insights
        
        recent_cutoff = time.time() - 3600
        recent_generation = [
            m for m in self.metrics_collector.generation_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_generation:
            return insights
        
        # Throughput efficiency
        throughputs = [m.throughput for m in recent_generation]
        try:
            import numpy as np
            avg_throughput = np.mean(throughputs)
        except ImportError:
            avg_throughput = sum(throughputs) / len(throughputs)
        
        if avg_throughput < 1.0:  # Less than 1 structure per second
            insights.append(PerformanceInsight(
                category="cost",
                severity="medium",
                title="Low Generation Efficiency",
                description=f"Average throughput is {avg_throughput:.2f} structures/sec, impacting cost efficiency",
                impact=0.3,
                recommendation="Optimize batch size and model compilation settings",
                metrics={"avg_throughput": avg_throughput},
                timestamp=time.time()
            ))
        
        return insights
    
    def _detect_throughput_anomalies(self, cutoff_time: float) -> List[Dict[str, Any]]:
        """Detect throughput anomalies."""
        anomalies = []
        
        if not self.metrics_collector.generation_metrics:
            return anomalies
        
        recent_metrics = [
            m for m in self.metrics_collector.generation_metrics
            if m.timestamp >= cutoff_time
        ]
        
        if len(recent_metrics) < 5:
            return anomalies
        
        throughputs = [m.throughput for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        return self._detect_value_anomalies(timestamps, throughputs)
    
    def _detect_quality_anomalies(self, cutoff_time: float) -> List[Dict[str, Any]]:
        """Detect quality anomalies."""
        anomalies = []
        
        if not self.metrics_collector.quality_metrics:
            return anomalies
        
        recent_metrics = [
            m for m in self.metrics_collector.quality_metrics
            if m.timestamp >= cutoff_time
        ]
        
        if len(recent_metrics) < 5:
            return anomalies
        
        quality_scores = [m.overall_quality for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        return self._detect_value_anomalies(timestamps, quality_scores)
    
    def _detect_resource_anomalies(self, cutoff_time: float) -> List[Dict[str, Any]]:
        """Detect resource usage anomalies."""
        anomalies = []
        
        if not self.metrics_collector.system_metrics:
            return anomalies
        
        recent_metrics = [
            m for m in self.metrics_collector.system_metrics
            if m.timestamp >= cutoff_time
        ]
        
        if len(recent_metrics) < 5:
            return anomalies
        
        # Check CPU usage anomalies
        cpu_usage = [m.cpu_usage for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        cpu_anomalies = self._detect_value_anomalies(timestamps, cpu_usage)
        for anomaly in cpu_anomalies:
            anomaly['metric'] = 'cpu_usage'
        anomalies.extend(cpu_anomalies)
        
        return anomalies
    
    def _analyze_efficiency_trend(self, start_time: float, end_time: float) -> Optional[TrendAnalysis]:
        """Analyze efficiency trends."""
        if not self.metrics_collector.generation_metrics:
            return None
        
        recent_metrics = [
            m for m in self.metrics_collector.generation_metrics
            if start_time <= m.timestamp <= end_time
        ]
        
        if len(recent_metrics) < 5:
            return None
        
        # Calculate efficiency as structures per unit time
        efficiencies = [m.throughput for m in recent_metrics]
        
        trend_direction, change_rate = self._calculate_trend(efficiencies)
        confidence = self._calculate_trend_confidence(efficiencies)
        forecast = self._generate_simple_forecast(efficiencies, periods=5)
        
        timestamps = [m.timestamp for m in recent_metrics]
        anomalies = self._detect_value_anomalies(timestamps, efficiencies)
        
        return TrendAnalysis(
            metric_name="efficiency",
            timeframe=f"{(end_time - start_time) / 3600:.1f}h",
            trend_direction=trend_direction,
            change_rate=change_rate,
            confidence=confidence,
            forecast=forecast,
            anomalies=anomalies
        )
    
    def _analyze_resource_trends(self, start_time: float, end_time: float) -> List[TrendAnalysis]:
        """Analyze resource utilization trends."""
        trends = []
        
        if not self.metrics_collector.system_metrics:
            return trends
        
        recent_metrics = [
            m for m in self.metrics_collector.system_metrics
            if start_time <= m.timestamp <= end_time
        ]
        
        if len(recent_metrics) < 5:
            return trends
        
        # CPU trend
        cpu_values = [m.cpu_usage for m in recent_metrics]
        trend_direction, change_rate = self._calculate_trend(cpu_values)
        confidence = self._calculate_trend_confidence(cpu_values)
        
        trends.append(TrendAnalysis(
            metric_name="cpu_usage",
            timeframe=f"{(end_time - start_time) / 3600:.1f}h",
            trend_direction=trend_direction,
            change_rate=change_rate,
            confidence=confidence,
            forecast=self._generate_simple_forecast(cpu_values, periods=5),
            anomalies=[]
        ))
        
        return trends
    
    def _generate_gpu_optimization_recommendation(self) -> Optional[OptimizationRecommendation]:
        """Generate GPU optimization recommendation."""
        if not self.metrics_collector.system_metrics:
            return None
        
        recent_cutoff = time.time() - 3600
        recent_system = [
            m for m in self.metrics_collector.system_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_system:
            return None
        
        avg_gpu_util = self._calculate_avg_gpu_utilization(recent_system)
        
        if avg_gpu_util < 60:
            return OptimizationRecommendation(
                category="gpu_optimization",
                priority="medium",
                title="Increase GPU Utilization",
                description=f"Current GPU utilization is {avg_gpu_util:.1f}%, which is below optimal levels",
                expected_improvement=25.0,
                implementation_effort="low",
                cost_benefit_ratio=3.0,
                steps=[
                    "Increase batch size to better utilize GPU memory",
                    "Enable model compilation for better performance",
                    "Consider using multiple GPUs for parallel processing"
                ]
            )
        
        return None
    
    def _generate_batch_optimization_recommendation(self) -> Optional[OptimizationRecommendation]:
        """Generate batch size optimization recommendation."""
        if not self.metrics_collector.generation_metrics:
            return None
        
        recent_cutoff = time.time() - 3600
        recent_generation = [
            m for m in self.metrics_collector.generation_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_generation:
            return None
        
        try:
            import numpy as np
            avg_throughput = np.mean([m.throughput for m in recent_generation])
        except ImportError:
            throughputs = [m.throughput for m in recent_generation]
            avg_throughput = sum(throughputs) / len(throughputs)
        
        if avg_throughput < 2.0:
            return OptimizationRecommendation(
                category="batch_optimization",
                priority="high",
                title="Optimize Batch Size",
                description=f"Current throughput is {avg_throughput:.2f} structures/sec, which can be improved",
                expected_improvement=40.0,
                implementation_effort="low",
                cost_benefit_ratio=4.0,
                steps=[
                    "Experiment with larger batch sizes",
                    "Monitor GPU memory usage to find optimal size",
                    "Enable batch size auto-tuning if available"
                ]
            )
        
        return None
    
    def _generate_quality_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if not self.metrics_collector.quality_metrics:
            return recommendations
        
        recent_cutoff = time.time() - 3600
        recent_quality = [
            m for m in self.metrics_collector.quality_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_quality:
            return recommendations
        
        try:
            import numpy as np
            avg_quality = np.mean([m.overall_quality for m in recent_quality])
        except ImportError:
            quality_scores = [m.overall_quality for m in recent_quality]
            avg_quality = sum(quality_scores) / len(quality_scores)
        
        if avg_quality < 0.8:
            recommendations.append(OptimizationRecommendation(
                category="quality_improvement",
                priority="high" if avg_quality < 0.6 else "medium",
                title="Improve Generation Quality",
                description=f"Average quality score is {avg_quality:.3f}, below optimal threshold",
                expected_improvement=20.0,
                implementation_effort="medium",
                cost_benefit_ratio=2.5,
                steps=[
                    "Adjust sampling parameters for better convergence",
                    "Increase guidance factors if using conditional generation",
                    "Review and tune diffusion parameters"
                ]
            ))
        
        return recommendations
    
    def _generate_resource_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate resource optimization recommendations."""
        recommendations = []
        
        if not self.metrics_collector.system_metrics:
            return recommendations
        
        recent_cutoff = time.time() - 3600
        recent_system = [
            m for m in self.metrics_collector.system_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_system:
            return recommendations
        
        try:
            import numpy as np
            avg_memory = np.mean([m.memory_usage for m in recent_system])
        except ImportError:
            memory_usage = [m.memory_usage for m in recent_system]
            avg_memory = sum(memory_usage) / len(memory_usage)
        
        if avg_memory > 85:
            recommendations.append(OptimizationRecommendation(
                category="memory_optimization",
                priority="high",
                title="Optimize Memory Usage",
                description=f"Memory usage is {avg_memory:.1f}%, approaching system limits",
                expected_improvement=15.0,
                implementation_effort="medium",
                cost_benefit_ratio=2.0,
                steps=[
                    "Reduce batch size to lower memory pressure",
                    "Enable gradient checkpointing if available",
                    "Consider distributed processing across multiple nodes"
                ]
            ))
        
        return recommendations


def create_analytics_engine(metrics_collector=None) -> AnalyticsEngine:
    """Create an analytics engine instance."""
    return AnalyticsEngine(metrics_collector=metrics_collector)
