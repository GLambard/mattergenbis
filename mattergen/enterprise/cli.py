"""
Enterprise CLI Commands
======================

Command-line interface for MatterGen enterprise features.
Provides commands for:
- Starting/stopping enterprise services
- Monitoring dashboard access
- Analytics reports
- Performance optimization
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cmd_start_enterprise(args):
    """Start enterprise monitoring and dashboard."""
    try:
        from mattergen.enterprise.integration import initialize_enterprise, start_enterprise
        
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Initialize enterprise components
        manager = initialize_enterprise(
            config=config,
            enable_monitoring=args.enable_monitoring,
            enable_dashboard=args.enable_dashboard,
            enable_analytics=args.enable_analytics
        )
        
        # Start services
        start_enterprise()
        
        print("✅ Enterprise services started successfully!")
        print(f"📊 Dashboard: http://{config.get('dashboard', {}).get('host', 'localhost')}:{config.get('dashboard', {}).get('port', 8080)}")
        print("🔄 Monitoring: Active")
        print("📈 Analytics: Active")
        
        if args.keep_alive:
            print("\n⏳ Keeping services alive... (Press Ctrl+C to stop)")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping enterprise services...")
                from mattergen.enterprise.integration import stop_enterprise
                stop_enterprise()
                print("✅ Enterprise services stopped")
        
    except ImportError as e:
        print(f"❌ Enterprise features not available: {e}")
        print("💡 Install enterprise dependencies with: pip install flask flask-socketio")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start enterprise services: {e}")
        sys.exit(1)


def cmd_stop_enterprise(args):
    """Stop enterprise services."""
    try:
        from mattergen.enterprise.integration import stop_enterprise
        
        stop_enterprise()
        print("✅ Enterprise services stopped")
        
    except Exception as e:
        print(f"❌ Failed to stop enterprise services: {e}")
        sys.exit(1)


def cmd_enterprise_status(args):
    """Get enterprise services status."""
    try:
        from mattergen.enterprise.integration import get_enterprise_manager
        
        manager = get_enterprise_manager()
        if not manager:
            print("❌ Enterprise services not initialized")
            sys.exit(1)
        
        status = manager.get_status()
        
        print("🏢 MatterGen Enterprise Status")
        print("=" * 40)
        print(f"Running: {'✅' if status['running'] else '❌'}")
        print(f"Components: {', '.join(status['components_started']) if status['components_started'] else 'None'}")
        print(f"Monitoring: {'✅' if status['monitoring_enabled'] else '❌'}")
        print(f"Dashboard: {'✅' if status['dashboard_enabled'] else '❌'}")
        print(f"Analytics: {'✅' if status['analytics_enabled'] else '❌'}")
        
    except Exception as e:
        print(f"❌ Failed to get enterprise status: {e}")
        sys.exit(1)


def cmd_performance_report(args):
    """Generate performance report."""
    try:
        from mattergen.enterprise.integration import get_enterprise_manager
        
        manager = get_enterprise_manager()
        if not manager:
            print("❌ Enterprise services not initialized")
            sys.exit(1)
        
        print("📊 MatterGen Performance Report")
        print("=" * 50)
        
        # Performance score
        score = manager.get_performance_score()
        print(f"\n🏆 Overall Performance Score: {score.get('overall', 0):.1f}/100")
        
        components = score.get('components', {})
        for component, component_score in components.items():
            status = "🟢" if component_score > 80 else "🟡" if component_score > 60 else "🔴"
            print(f"   {status} {component.replace('_', ' ').title()}: {component_score:.1f}/100")
        
        # Insights
        insights = manager.get_performance_insights()
        if insights:
            print(f"\n💡 Performance Insights ({len(insights)} items):")
            for insight in insights[:5]:  # Show top 5
                severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "💭"}
                emoji = severity_emoji.get(insight['severity'], "ℹ️")
                print(f"   {emoji} {insight['title']}")
                print(f"      {insight['description']}")
                print(f"      💡 {insight['recommendation']}")
                print()
        
        # Optimization recommendations
        recommendations = manager.get_optimization_recommendations()
        if recommendations:
            print(f"🔧 Optimization Recommendations ({len(recommendations)} items):")
            for rec in recommendations[:3]:  # Show top 3
                priority_emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "💭"}
                emoji = priority_emoji.get(rec['priority'], "ℹ️")
                print(f"   {emoji} {rec['title']} (Priority: {rec['priority']})")
                print(f"      {rec['description']}")
                print(f"      📈 Expected improvement: {rec['expected_improvement']:.1f}%")
                print(f"      🔨 Implementation effort: {rec['implementation_effort']}")
                print()
        
        # Save detailed report if requested
        if args.output:
            report_data = {
                "timestamp": time.time(),
                "performance_score": score,
                "insights": insights,
                "recommendations": recommendations
            }
            
            with open(args.output, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"💾 Detailed report saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Failed to generate performance report: {e}")
        sys.exit(1)


def cmd_analytics_dashboard(args):
    """Open analytics dashboard in browser."""
    try:
        import webbrowser
        
        host = args.host or "localhost"
        port = args.port or 8080
        url = f"http://{host}:{port}"
        
        print(f"🌐 Opening dashboard: {url}")
        webbrowser.open(url)
        
    except Exception as e:
        print(f"❌ Failed to open dashboard: {e}")
        sys.exit(1)


def cmd_export_metrics(args):
    """Export metrics data."""
    try:
        from mattergen.enterprise.integration import get_enterprise_manager
        
        manager = get_enterprise_manager()
        if not manager or not manager.metrics_collector:
            print("❌ Metrics collector not available")
            sys.exit(1)
        
        # Export current metrics summary
        summary = manager.metrics_collector.get_current_metrics_summary()
        
        output_file = args.output or f"mattergen_metrics_{int(time.time())}.json"
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Metrics exported to: {output_file}")
        
    except Exception as e:
        print(f"❌ Failed to export metrics: {e}")
        sys.exit(1)


def create_enterprise_parser() -> argparse.ArgumentParser:
    """Create the enterprise CLI parser."""
    parser = argparse.ArgumentParser(
        description="MatterGen Enterprise Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start enterprise services
  python -m mattergen.enterprise.cli start --keep-alive
  
  # Check status
  python -m mattergen.enterprise.cli status
  
  # Generate performance report
  python -m mattergen.enterprise.cli report --output report.json
  
  # Open dashboard
  python -m mattergen.enterprise.cli dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start enterprise services')
    start_parser.add_argument('--config', type=str, help='Configuration file path')
    start_parser.add_argument('--enable-monitoring', action='store_true', default=True, help='Enable monitoring')
    start_parser.add_argument('--enable-dashboard', action='store_true', default=True, help='Enable dashboard')
    start_parser.add_argument('--enable-analytics', action='store_true', default=True, help='Enable analytics')
    start_parser.add_argument('--keep-alive', action='store_true', help='Keep services running')
    start_parser.set_defaults(func=cmd_start_enterprise)
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop enterprise services')
    stop_parser.set_defaults(func=cmd_stop_enterprise)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get enterprise status')
    status_parser.set_defaults(func=cmd_enterprise_status)
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate performance report')
    report_parser.add_argument('--output', type=str, help='Output file for detailed report')
    report_parser.set_defaults(func=cmd_performance_report)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Open analytics dashboard')
    dashboard_parser.add_argument('--host', type=str, default='localhost', help='Dashboard host')
    dashboard_parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    dashboard_parser.set_defaults(func=cmd_analytics_dashboard)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export metrics data')
    export_parser.add_argument('--output', type=str, help='Output file path')
    export_parser.set_defaults(func=cmd_export_metrics)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_enterprise_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
