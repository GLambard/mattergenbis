"""
MatterGen Enterprise Features
============================

Production-grade enterprise capabilities for MatterGen including:
- Real-time monitoring and dashboards
- Advanced analytics and reporting
- Quality model management
- Integration APIs and connectors

Version: 4.3.0
Status: Active Development
"""

__version__ = "4.3.0"
__enterprise_features__ = [
    "real_time_monitoring",
    "quality_dashboard", 
    "performance_analytics",
    "model_management",
    "api_integration",
    "automated_reporting"
]

# Enterprise feature availability flags
ENTERPRISE_FEATURES_AVAILABLE = True

try:
    # Check for required enterprise dependencies
    import flask
    import websockets
    import prometheus_client
    WEB_FEATURES_AVAILABLE = True
except ImportError:
    WEB_FEATURES_AVAILABLE = False

try:
    import sqlalchemy
    import redis
    DATABASE_FEATURES_AVAILABLE = True
except ImportError:
    DATABASE_FEATURES_AVAILABLE = False

# Enterprise configuration
ENTERPRISE_CONFIG = {
    "monitoring": {
        "enabled": True,
        "metrics_retention_days": 30,
        "alert_thresholds": {
            "quality_degradation": 0.1,
            "performance_drop": 0.2,
            "gpu_utilization_low": 0.5
        }
    },
    "dashboard": {
        "enabled": True,
        "port": 8080,
        "host": "localhost",
        "auto_refresh_seconds": 5
    },
    "api": {
        "enabled": True,
        "port": 8000,
        "authentication": True,
        "rate_limiting": True
    }
}

def get_enterprise_status():
    """Get the status of enterprise features."""
    return {
        "version": __version__,
        "features_available": ENTERPRISE_FEATURES_AVAILABLE,
        "web_features": WEB_FEATURES_AVAILABLE,
        "database_features": DATABASE_FEATURES_AVAILABLE,
        "enabled_features": __enterprise_features__
    }

# Import key components for easy access
try:
    from .monitoring import MetricsCollector, SystemMetrics, GenerationMetrics, QualityMetrics
    from .integration import EnterpriseManager, get_enterprise_manager, initialize_enterprise
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from .analytics import AnalyticsEngine, PerformanceInsight
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

try:
    from .dashboard import DashboardServer, create_dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
