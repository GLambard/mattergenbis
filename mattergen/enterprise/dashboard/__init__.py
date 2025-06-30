"""
Enterprise Dashboard System
===========================

Real-time web dashboard for MatterGen enterprise monitoring.
Provides live visualization of:
- Structure generation performance
- GPU utilization and health
- Quality metrics trends
- System resource usage
- Historical analytics
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

logger = logging.getLogger(__name__)


class DashboardServer:
    """
    Real-time dashboard server for MatterGen monitoring.
    
    Features:
    - Live metrics visualization
    - WebSocket real-time updates
    - Historical trend analysis
    - Interactive charts and graphs
    - Mobile-responsive design
    - Alert notifications
    """
    
    def __init__(self, 
                 metrics_collector=None,
                 host: str = "localhost",
                 port: int = 8080,
                 debug: bool = False):
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and Flask-SocketIO required for dashboard")
        
        self.metrics_collector = metrics_collector
        self.host = host
        self.port = port
        self.debug = debug
        
        # Flask app setup
        self.app = Flask(__name__, 
                        template_folder=self._get_template_dir(),
                        static_folder=self._get_static_dir())
        self.app.config['SECRET_KEY'] = 'mattergen-enterprise-dashboard'
        
        # SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard state
        self.connected_clients = set()
        self.update_thread = None
        self.running = False
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_handlers()
        
        logger.info(f"Dashboard server initialized on {host}:{port}")
    
    def _get_template_dir(self) -> str:
        """Get templates directory path."""
        return str(Path(__file__).parent / "templates")
    
    def _get_static_dir(self) -> str:
        """Get static files directory path."""
        return str(Path(__file__).parent / "static")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route("/")
        def dashboard():
            """Main dashboard page."""
            return render_template("dashboard.html")
        
        @self.app.route("/api/status")
        def api_status():
            """API status endpoint."""
            return jsonify({
                "status": "online",
                "timestamp": datetime.now().isoformat(),
                "metrics_available": self.metrics_collector is not None,
                "connected_clients": len(self.connected_clients)
            })
        
        @self.app.route("/api/metrics/current")
        def api_current_metrics():
            """Get current metrics summary."""
            if not self.metrics_collector:
                return jsonify({"error": "Metrics collector not available"})
            
            try:
                summary = self.metrics_collector.get_current_metrics_summary()
                return jsonify(summary)
            except Exception as e:
                logger.error(f"Error getting current metrics: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/metrics/history")
        def api_metrics_history():
            """Get metrics history."""
            if not self.metrics_collector:
                return jsonify({"error": "Metrics collector not available"})
            
            metric_name = request.args.get("metric", "")
            hours = int(request.args.get("hours", 1))
            
            try:
                history = self.metrics_collector.get_metric_history(metric_name, hours)
                return jsonify({
                    "metric": metric_name,
                    "hours": hours,
                    "data": [
                        {
                            "timestamp": point.timestamp,
                            "value": point.value,
                            "labels": point.labels
                        }
                        for point in history
                    ]
                })
            except Exception as e:
                logger.error(f"Error getting metrics history: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/analytics/performance")
        def api_performance_analytics():
            """Get performance analytics."""
            if not self.metrics_collector:
                return jsonify({"error": "Metrics collector not available"})
            
            try:
                analytics = self._calculate_performance_analytics()
                return jsonify(analytics)
            except Exception as e:
                logger.error(f"Error calculating performance analytics: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _setup_socketio_handlers(self):
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.connected_clients.add(request.sid)
            logger.info(f"Client connected: {request.sid}")
            emit('status', {'message': 'Connected to MatterGen Dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            self.connected_clients.discard(request.sid)
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle manual update request."""
            try:
                self._send_metrics_update()
            except Exception as e:
                logger.error(f"Error sending update: {e}")
                emit('error', {'message': str(e)})
    
    def start(self):
        """Start the dashboard server."""
        if self.running:
            logger.warning("Dashboard server already running")
            return
        
        self.running = True
        
        # Start background update thread
        self.update_thread = threading.Thread(
            target=self._update_worker,
            daemon=True,
            name="DashboardUpdates"
        )
        self.update_thread.start()
        
        logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        
        # Run Flask-SocketIO server
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=self.debug,
            use_reloader=False
        )
    
    def stop(self):
        """Stop the dashboard server."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Dashboard server stopped")
    
    def _update_worker(self):
        """Background worker for real-time updates."""
        logger.info("Dashboard update worker started")
        
        while self.running:
            try:
                if self.connected_clients and self.metrics_collector:
                    self._send_metrics_update()
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in update worker: {e}")
                time.sleep(10)  # Longer delay on error
        
        logger.info("Dashboard update worker stopped")
    
    def _send_metrics_update(self):
        """Send metrics update to all connected clients."""
        if not self.metrics_collector:
            return
        
        try:
            # Get current metrics
            summary = self.metrics_collector.get_current_metrics_summary()
            
            # Add timestamp
            summary['dashboard_timestamp'] = datetime.now().isoformat()
            
            # Emit to all clients
            self.socketio.emit('metrics_update', summary)
            
        except Exception as e:
            logger.error(f"Error sending metrics update: {e}")
            self.socketio.emit('error', {'message': 'Failed to get metrics'})
    
    def _calculate_performance_analytics(self) -> Dict[str, Any]:
        """Calculate performance analytics."""
        if not self.metrics_collector:
            return {}
        
        current_time = time.time()
        
        # Get recent generation metrics (last hour)
        recent_generation = [
            m for m in self.metrics_collector.generation_metrics
            if m.timestamp >= current_time - 3600
        ]
        
        if not recent_generation:
            return {"message": "No recent generation data"}
        
        # Calculate analytics
        throughputs = [m.throughput for m in recent_generation]
        quality_scores = [m.quality_score for m in recent_generation]
        durations = [m.batch_duration for m in recent_generation]
        
        analytics = {
            "timeframe": "last_hour",
            "total_batches": len(recent_generation),
            "throughput": {
                "avg": sum(throughputs) / len(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
                "trend": self._calculate_trend(throughputs)
            },
            "quality": {
                "avg": sum(quality_scores) / len(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores),
                "trend": self._calculate_trend(quality_scores)
            },
            "efficiency": {
                "avg_duration": sum(durations) / len(durations),
                "total_structures": sum(m.structures_generated for m in recent_generation),
                "gpu_utilization": self._calculate_gpu_utilization()
            }
        }
        
        return analytics
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation based on first and last third
        first_third = values[:len(values)//3] or [values[0]]
        last_third = values[-len(values)//3:] or [values[-1]]
        
        avg_first = sum(first_third) / len(first_third)
        avg_last = sum(last_third) / len(last_third)
        
        change_ratio = (avg_last - avg_first) / avg_first
        
        if change_ratio > 0.05:
            return "improving"
        elif change_ratio < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _calculate_gpu_utilization(self) -> Dict[str, float]:
        """Calculate average GPU utilization."""
        if not self.metrics_collector.system_metrics:
            return {}
        
        recent_cutoff = time.time() - 300  # Last 5 minutes
        recent_system = [
            m for m in self.metrics_collector.system_metrics
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_system:
            return {}
        
        # Average GPU metrics across time
        gpu_utilization = {}
        for metrics in recent_system:
            for gpu_id, gpu_metrics in metrics.gpu_metrics.items():
                if gpu_id not in gpu_utilization:
                    gpu_utilization[gpu_id] = []
                gpu_utilization[gpu_id].append(gpu_metrics.get('utilization', 0))
        
        # Calculate averages
        return {
            str(gpu_id): sum(utils) / len(utils)
            for gpu_id, utils in gpu_utilization.items()
            if utils
        }


def create_dashboard(metrics_collector=None, **kwargs) -> DashboardServer:
    """Create a dashboard server instance."""
    return DashboardServer(metrics_collector=metrics_collector, **kwargs)
