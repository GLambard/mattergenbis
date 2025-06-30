"""
Enterprise Integration Manager
=============================

Manages integration between monitoring, dashboard, analytics, and main MatterGen pipeline.
Provides unified enterprise feature management and coordination.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from .monitoring import MetricsCollector
from .analytics import AnalyticsEngine

logger = logging.getLogger(__name__)


class EnterpriseManager:
    """
    Central manager for all MatterGen enterprise features.
    
    Features:
    - Unified monitoring and analytics coordination
    - Dashboard server management
    - Pipeline integration hooks
    - Configuration management
    - Feature flag control
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 enable_monitoring: bool = True,
                 enable_dashboard: bool = True,
                 enable_analytics: bool = True):
        
        self.config = config or {}
        self.enable_monitoring = enable_monitoring
        self.enable_dashboard = enable_dashboard
        self.enable_analytics = enable_analytics
        
        # Core components
        self.metrics_collector = None
        self.analytics_engine = None
        self.dashboard_server = None
        
        # State management
        self.running = False
        self.components_started = set()
        
        # Integration hooks
        self.generation_hooks = []
        self.quality_hooks = []
        self.system_hooks = []
        
        logger.info("Enterprise manager initialized")
    
    def initialize(self):
        """Initialize all enterprise components."""
        try:
            # Initialize metrics collector
            if self.enable_monitoring:
                self._initialize_monitoring()
            
            # Initialize analytics engine
            if self.enable_analytics and self.metrics_collector:
                self._initialize_analytics()
            
            # Initialize dashboard server
            if self.enable_dashboard and self.metrics_collector:
                self._initialize_dashboard()
            
            logger.info("Enterprise components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enterprise components: {e}")
            raise
    
    def start(self):
        """Start all enabled enterprise components."""
        if self.running:
            logger.warning("Enterprise manager already running")
            return
        
        self.running = True
        
        try:
            # Start metrics collector
            if self.metrics_collector:
                self.metrics_collector.start()
                self.components_started.add("monitoring")
            
            # Start dashboard server in background thread
            if self.dashboard_server:
                dashboard_thread = threading.Thread(
                    target=self._start_dashboard_background,
                    daemon=True,
                    name="DashboardServer"
                )
                dashboard_thread.start()
                self.components_started.add("dashboard")
            
            logger.info(f"Enterprise manager started with components: {self.components_started}")
            
        except Exception as e:
            logger.error(f"Failed to start enterprise components: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop all enterprise components."""
        self.running = False
        
        try:
            # Stop metrics collector
            if self.metrics_collector:
                self.metrics_collector.stop()
            
            # Stop dashboard server
            if self.dashboard_server:
                self.dashboard_server.stop()
            
            self.components_started.clear()
            logger.info("Enterprise manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping enterprise components: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all enterprise components."""
        return {
            "running": self.running,
            "components_started": list(self.components_started),
            "monitoring_enabled": self.enable_monitoring,
            "dashboard_enabled": self.enable_dashboard,
            "analytics_enabled": self.enable_analytics,
            "metrics_available": self.metrics_collector is not None,
            "analytics_available": self.analytics_engine is not None,
            "dashboard_available": self.dashboard_server is not None
        }
    
    # Integration hooks for main pipeline
    
    def on_generation_start(self, batch_id: int, gpu_id: int, batch_size: int):
        """Hook called when generation batch starts."""
        if not self.metrics_collector:
            return
        
        # Record generation start metrics
        self.metrics_collector.add_metric(
            "generation_batch_started",
            1.0,
            {"batch_id": str(batch_id), "gpu_id": str(gpu_id), "batch_size": str(batch_size)}
        )
        
        # Call registered hooks
        for hook in self.generation_hooks:
            try:
                hook("start", batch_id, gpu_id, batch_size)
            except Exception as e:
                logger.error(f"Error in generation hook: {e}")
    
    def on_generation_complete(self, 
                             batch_id: int, 
                             gpu_id: int, 
                             structures_generated: int,
                             batch_duration: float,
                             quality_score: float,
                             throughput: float,
                             memory_peak: float,
                             error_count: int = 0):
        """Hook called when generation batch completes."""
        if not self.metrics_collector:
            return
        
        # Import metrics classes to avoid circular imports
        from .monitoring import GenerationMetrics
        
        # Record generation metrics
        metrics = GenerationMetrics(
            timestamp=time.time(),
            batch_id=batch_id,
            gpu_id=gpu_id,
            structures_generated=structures_generated,
            batch_duration=batch_duration,
            quality_score=quality_score,
            throughput=throughput,
            memory_peak=memory_peak,
            error_count=error_count
        )
        
        self.metrics_collector.add_generation_metrics(metrics)
        
        # Call registered hooks
        for hook in self.generation_hooks:
            try:
                hook("complete", batch_id, gpu_id, structures_generated, 
                     batch_duration, quality_score, throughput, memory_peak, error_count)
            except Exception as e:
                logger.error(f"Error in generation hook: {e}")
    
    def on_quality_assessment(self,
                            batch_id: int,
                            overall_quality: float,
                            quality_distribution: Dict[str, float],
                            improvement_rate: float,
                            trend: str,
                            convergence_status: str):
        """Hook called when quality assessment completes."""
        if not self.metrics_collector:
            return
        
        # Import metrics classes to avoid circular imports
        from .monitoring import QualityMetrics
        
        # Record quality metrics
        metrics = QualityMetrics(
            timestamp=time.time(),
            batch_id=batch_id,
            overall_quality=overall_quality,
            quality_distribution=quality_distribution,
            improvement_rate=improvement_rate,
            trend=trend,
            convergence_status=convergence_status
        )
        
        self.metrics_collector.add_quality_metrics(metrics)
        
        # Call registered hooks
        for hook in self.quality_hooks:
            try:
                hook(batch_id, overall_quality, quality_distribution, 
                     improvement_rate, trend, convergence_status)
            except Exception as e:
                logger.error(f"Error in quality hook: {e}")
    
    def on_system_metrics(self,
                         cpu_usage: float,
                         memory_usage: float,
                         gpu_metrics: Dict[int, Dict[str, float]],
                         disk_usage: float,
                         network_io: Dict[str, float]):
        """Hook called with system metrics update."""
        if not self.metrics_collector:
            return
        
        # Import metrics classes to avoid circular imports
        from .monitoring import SystemMetrics
        
        # Record system metrics
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_metrics=gpu_metrics,
            disk_usage=disk_usage,
            network_io=network_io
        )
        
        self.metrics_collector.add_system_metrics(metrics)
        
        # Call registered hooks
        for hook in self.system_hooks:
            try:
                hook(cpu_usage, memory_usage, gpu_metrics, disk_usage, network_io)
            except Exception as e:
                logger.error(f"Error in system hook: {e}")
    
    # Analytics methods
    
    def get_performance_insights(self) -> List[Dict[str, Any]]:
        """Get performance insights from analytics engine."""
        if not self.analytics_engine:
            return []
        
        try:
            insights = self.analytics_engine.generate_insights()
            return [
                {
                    "category": insight.category,
                    "severity": insight.severity,
                    "title": insight.title,
                    "description": insight.description,
                    "impact": insight.impact,
                    "recommendation": insight.recommendation,
                    "metrics": insight.metrics,
                    "timestamp": insight.timestamp
                }
                for insight in insights
            ]
        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
            return []
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations."""
        if not self.analytics_engine:
            return []
        
        try:
            recommendations = self.analytics_engine.get_optimization_recommendations()
            return [
                {
                    "category": rec.category,
                    "priority": rec.priority,
                    "title": rec.title,
                    "description": rec.description,
                    "expected_improvement": rec.expected_improvement,
                    "implementation_effort": rec.implementation_effort,
                    "cost_benefit_ratio": rec.cost_benefit_ratio,
                    "steps": rec.steps
                }
                for rec in recommendations
            ]
        except Exception as e:
            logger.error(f"Error getting optimization recommendations: {e}")
            return []
    
    def get_performance_score(self) -> Dict[str, float]:
        """Get overall performance score."""
        if not self.analytics_engine:
            return {"overall": 0.0, "components": {}}
        
        try:
            return self.analytics_engine.calculate_performance_score()
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return {"overall": 0.0, "components": {}}
    
    # Hook registration methods
    
    def register_generation_hook(self, hook_func):
        """Register a generation event hook."""
        self.generation_hooks.append(hook_func)
    
    def register_quality_hook(self, hook_func):
        """Register a quality assessment hook."""
        self.quality_hooks.append(hook_func)
    
    def register_system_hook(self, hook_func):
        """Register a system metrics hook."""
        self.system_hooks.append(hook_func)
    
    # Private methods
    
    def _initialize_monitoring(self):
        """Initialize monitoring component."""
        try:
            self.metrics_collector = MetricsCollector(
                retention_hours=self.config.get("monitoring", {}).get("metrics_retention_hours", 24),
                max_points_per_metric=self.config.get("monitoring", {}).get("max_points_per_metric", 10000)
            )
            logger.info("Monitoring component initialized")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            raise
    
    def _initialize_analytics(self):
        """Initialize analytics component."""
        try:
            self.analytics_engine = AnalyticsEngine(
                metrics_collector=self.metrics_collector
            )
            logger.info("Analytics component initialized")
        except Exception as e:
            logger.error(f"Failed to initialize analytics: {e}")
            raise
    
    def _initialize_dashboard(self):
        """Initialize dashboard component."""
        try:
            # Check if dashboard dependencies are available
            from .dashboard import FLASK_AVAILABLE
            
            if not FLASK_AVAILABLE:
                logger.warning("Dashboard dependencies not available, skipping dashboard initialization")
                self.enable_dashboard = False
                return
            
            from .dashboard import create_dashboard
            
            dashboard_config = self.config.get("dashboard", {})
            self.dashboard_server = create_dashboard(
                metrics_collector=self.metrics_collector,
                host=dashboard_config.get("host", "localhost"),
                port=dashboard_config.get("port", 8080),
                debug=dashboard_config.get("debug", False)
            )
            logger.info("Dashboard component initialized")
            
        except ImportError as e:
            logger.warning(f"Dashboard dependencies not available: {e}")
            self.enable_dashboard = False
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            raise
    
    def _start_dashboard_background(self):
        """Start dashboard server in background."""
        try:
            if self.dashboard_server:
                logger.info("Starting dashboard server...")
                self.dashboard_server.start()
        except Exception as e:
            logger.error(f"Error starting dashboard server: {e}")


# Global enterprise manager instance
_enterprise_manager = None


def get_enterprise_manager() -> Optional[EnterpriseManager]:
    """Get the global enterprise manager instance."""
    return _enterprise_manager


def initialize_enterprise(config: Dict[str, Any] = None, **kwargs) -> EnterpriseManager:
    """Initialize the global enterprise manager."""
    global _enterprise_manager
    
    if _enterprise_manager is not None:
        logger.warning("Enterprise manager already initialized")
        return _enterprise_manager
    
    _enterprise_manager = EnterpriseManager(config=config, **kwargs)
    _enterprise_manager.initialize()
    
    return _enterprise_manager


def start_enterprise():
    """Start the global enterprise manager."""
    if _enterprise_manager:
        _enterprise_manager.start()
    else:
        logger.error("Enterprise manager not initialized")


def stop_enterprise():
    """Stop the global enterprise manager."""
    if _enterprise_manager:
        _enterprise_manager.stop()


def is_enterprise_enabled() -> bool:
    """Check if enterprise features are enabled and running."""
    return _enterprise_manager is not None and _enterprise_manager.running
