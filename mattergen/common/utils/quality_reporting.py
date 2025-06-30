"""
Advanced Quality Reporting - Phase 4.2
======================================

Comprehensive quality reporting and analysis system for MatterGen structures.

This module provides:
- Detailed quality reports with visualizations
- Comparative analysis across generations
- Quality improvement recommendations
- Export capabilities for various formats
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import csv

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .advanced_quality_metrics import (
    QualityPredictionResult, 
    QualityTrendData, 
    CrystallographicMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class QualityReportSection:
    """Individual section of a quality report."""
    title: str
    content: Dict[str, Any]
    visualizations: List[str] = None
    recommendations: List[str] = None


@dataclass
class ComprehensiveQualityReport:
    """Complete quality assessment report."""
    
    # Report metadata
    report_id: str
    generation_timestamp: datetime
    total_structures: int
    generation_parameters: Dict[str, Any]
    
    # Quality summary
    overall_quality_score: float
    quality_distribution: Dict[str, int]
    improvement_summary: str
    
    # Detailed sections
    executive_summary: QualityReportSection
    crystallographic_analysis: QualityReportSection
    trend_analysis: QualityReportSection
    ml_predictions: QualityReportSection
    recommendations: QualityReportSection
    
    # Technical details
    methodology: QualityReportSection
    appendix: QualityReportSection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(self.to_dict(), indent=2, default=json_serializer)


class AdvancedQualityReporter:
    """Advanced quality reporting system with visualization capabilities."""
    
    def __init__(self, 
                 output_dir: Path,
                 enable_visualizations: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_visualizations = enable_visualizations and MATPLOTLIB_AVAILABLE
        
        # Report templates and styling
        self.report_style = {
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'figure_size': (12, 8),
            'dpi': 300,
            'font_size': 10
        }
        
        # Historical data for trend analysis
        self.historical_reports = []
    
    def generate_comprehensive_report(self,
                                    quality_results: List[QualityPredictionResult],
                                    trend_data: List[QualityTrendData],
                                    generation_params: Dict[str, Any],
                                    ml_predictor=None) -> ComprehensiveQualityReport:
        """Generate a comprehensive quality assessment report."""
        
        report_id = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics(quality_results)
        
        # Generate report sections
        executive_summary = self._generate_executive_summary(overall_stats, quality_results)
        crystallographic_analysis = self._generate_crystallographic_analysis(quality_results)
        trend_analysis = self._generate_trend_analysis(trend_data)
        ml_predictions = self._generate_ml_predictions_section(quality_results, ml_predictor)
        recommendations = self._generate_recommendations_section(quality_results, overall_stats)
        methodology = self._generate_methodology_section()
        appendix = self._generate_appendix(quality_results, generation_params)
        
        # Create comprehensive report
        report = ComprehensiveQualityReport(
            report_id=report_id,
            generation_timestamp=datetime.now(),
            total_structures=len(quality_results),
            generation_parameters=generation_params,
            overall_quality_score=overall_stats['mean_quality'],
            quality_distribution=overall_stats['quality_distribution'],
            improvement_summary=overall_stats['improvement_summary'],
            executive_summary=executive_summary,
            crystallographic_analysis=crystallographic_analysis,
            trend_analysis=trend_analysis,
            ml_predictions=ml_predictions,
            recommendations=recommendations,
            methodology=methodology,
            appendix=appendix
        )
        
        # Generate visualizations if enabled
        if self.enable_visualizations:
            self._generate_visualizations(report, quality_results, trend_data)
        
        # Save report
        self._save_report(report)
        
        # Add to historical data
        self.historical_reports.append(report)
        
        return report
    
    def _calculate_overall_statistics(self, 
                                    quality_results: List[QualityPredictionResult]) -> Dict[str, Any]:
        """Calculate overall quality statistics."""
        
        if not quality_results:
            return {
                'mean_quality': 0.0,
                'std_quality': 0.0,
                'quality_distribution': {},
                'improvement_summary': "No data available"
            }
        
        qualities = [result.predicted_quality for result in quality_results]
        confidences = [result.confidence_score for result in quality_results]
        
        # Basic statistics
        mean_quality = sum(qualities) / len(qualities)
        std_quality = (sum((q - mean_quality) ** 2 for q in qualities) / len(qualities)) ** 0.5
        mean_confidence = sum(confidences) / len(confidences)
        
        # Quality distribution
        quality_bins = {
            'excellent': sum(1 for q in qualities if q >= 0.9),
            'good': sum(1 for q in qualities if 0.7 <= q < 0.9),
            'acceptable': sum(1 for q in qualities if 0.5 <= q < 0.7),
            'poor': sum(1 for q in qualities if q < 0.5)
        }
        
        # Risk factor analysis
        all_risk_factors = []
        for result in quality_results:
            all_risk_factors.extend(result.risk_factors)
        
        risk_frequency = {}
        for risk in all_risk_factors:
            risk_frequency[risk] = risk_frequency.get(risk, 0) + 1
        
        # Improvement summary
        excellent_rate = quality_bins['excellent'] / len(qualities)
        if excellent_rate > 0.2:
            improvement_summary = "Excellent generation quality achieved"
        elif excellent_rate > 0.1:
            improvement_summary = "Good generation quality with room for improvement"
        elif quality_bins['good'] / len(qualities) > 0.5:
            improvement_summary = "Acceptable quality, optimization recommended"
        else:
            improvement_summary = "Quality improvements needed"
        
        return {
            'mean_quality': mean_quality,
            'std_quality': std_quality,
            'mean_confidence': mean_confidence,
            'quality_distribution': quality_bins,
            'risk_frequency': risk_frequency,
            'improvement_summary': improvement_summary,
            'total_structures': len(quality_results)
        }
    
    def _generate_executive_summary(self, 
                                  overall_stats: Dict[str, Any],
                                  quality_results: List[QualityPredictionResult]) -> QualityReportSection:
        """Generate executive summary section."""
        
        content = {
            'key_metrics': {
                'total_structures_analyzed': overall_stats['total_structures'],
                'average_quality_score': round(overall_stats['mean_quality'], 3),
                'quality_consistency': round(1 - overall_stats['std_quality'], 3),
                'prediction_confidence': round(overall_stats['mean_confidence'], 3)
            },
            'quality_breakdown': overall_stats['quality_distribution'],
            'top_achievements': [
                f"{overall_stats['quality_distribution']['excellent']} structures achieved excellent quality",
                f"Average quality score: {overall_stats['mean_quality']:.1%}",
                f"Quality consistency: {(1 - overall_stats['std_quality']):.1%}"
            ],
            'main_findings': overall_stats['improvement_summary']
        }
        
        recommendations = [
            "Review detailed crystallographic analysis for specific insights",
            "Consider optimization strategies based on risk factor analysis",
            "Monitor quality trends for continuous improvement"
        ]
        
        return QualityReportSection(
            title="Executive Summary",
            content=content,
            recommendations=recommendations
        )
    
    def _generate_crystallographic_analysis(self, 
                                          quality_results: List[QualityPredictionResult]) -> QualityReportSection:
        """Generate crystallographic analysis section."""
        
        if not quality_results:
            return QualityReportSection(
                title="Crystallographic Analysis",
                content={"status": "No data available"}
            )
        
        # Aggregate crystallographic metrics
        all_metrics = [result.crystallographic_metrics for result in quality_results]
        
        # Calculate averages
        avg_metrics = {
            'space_group_probability': sum(m.space_group_probability for m in all_metrics) / len(all_metrics),
            'symmetry_precision': sum(m.symmetry_precision for m in all_metrics) / len(all_metrics),
            'lattice_stability': sum(m.lattice_stability for m in all_metrics) / len(all_metrics),
            'coordination_environment_score': sum(m.coordination_environment_score for m in all_metrics) / len(all_metrics),
            'bond_length_distribution_score': sum(m.bond_length_distribution_score for m in all_metrics) / len(all_metrics),
            'thermodynamic_quality': sum(m.thermodynamic_quality for m in all_metrics) / len(all_metrics),
            'electronic_quality': sum(m.electronic_quality for m in all_metrics) / len(all_metrics)
        }
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for metric, value in avg_metrics.items():
            if value > 0.8:
                strengths.append(f"High {metric.replace('_', ' ')}: {value:.2f}")
            elif value < 0.6:
                weaknesses.append(f"Low {metric.replace('_', ' ')}: {value:.2f}")
        
        content = {
            'average_metrics': avg_metrics,
            'structural_strengths': strengths,
            'areas_for_improvement': weaknesses,
            'detailed_breakdown': {
                'crystallographic_quality': round(avg_metrics.get('space_group_probability', 0), 3),
                'structural_stability': round(avg_metrics.get('lattice_stability', 0), 3),
                'chemical_reasonableness': round(avg_metrics.get('coordination_environment_score', 0), 3)
            }
        }
        
        recommendations = []
        if weaknesses:
            recommendations.append("Focus optimization on identified weak areas")
        if avg_metrics.get('lattice_stability', 0) < 0.7:
            recommendations.append("Consider lattice parameter optimization")
        if avg_metrics.get('symmetry_precision', 0) < 0.8:
            recommendations.append("Apply stricter symmetry constraints")
        
        return QualityReportSection(
            title="Crystallographic Analysis",
            content=content,
            recommendations=recommendations
        )
    
    def _generate_trend_analysis(self, 
                                trend_data: List[QualityTrendData]) -> QualityReportSection:
        """Generate trend analysis section."""
        
        if len(trend_data) < 2:
            return QualityReportSection(
                title="Trend Analysis",
                content={"status": "Insufficient data for trend analysis"}
            )
        
        # Calculate trend metrics
        qualities = [data.average_quality for data in trend_data]
        iterations = [data.iteration for data in trend_data]
        
        # Simple trend calculation
        quality_change = qualities[-1] - qualities[0]
        trend_direction = "improving" if quality_change > 0.05 else "stable" if abs(quality_change) <= 0.05 else "declining"
        
        # Convergence analysis
        recent_variance = sum((q - sum(qualities[-5:]) / 5) ** 2 for q in qualities[-5:]) / 5 if len(qualities) >= 5 else 0
        converged = recent_variance < 0.01 and qualities[-1] > 0.8
        
        content = {
            'trend_direction': trend_direction,
            'quality_change': round(quality_change, 3),
            'convergence_status': 'converged' if converged else 'improving',
            'stability_score': round(1 - recent_variance, 3),
            'iteration_range': f"{min(iterations)} - {max(iterations)}",
            'final_quality': round(qualities[-1], 3)
        }
        
        recommendations = []
        if trend_direction == "declining":
            recommendations.append("Investigate parameters causing quality decline")
        elif not converged:
            recommendations.append("Continue optimization for better convergence")
        if recent_variance > 0.05:
            recommendations.append("Focus on improving quality stability")
        
        return QualityReportSection(
            title="Trend Analysis",
            content=content,
            recommendations=recommendations
        )
    
    def _generate_ml_predictions_section(self, 
                                       quality_results: List[QualityPredictionResult],
                                       ml_predictor=None) -> QualityReportSection:
        """Generate ML predictions analysis section."""
        
        content = {
            'prediction_summary': {
                'total_predictions': len(quality_results),
                'average_confidence': round(sum(r.confidence_score for r in quality_results) / len(quality_results), 3) if quality_results else 0,
                'high_confidence_predictions': sum(1 for r in quality_results if r.confidence_score > 0.8),
            }
        }
        
        if ml_predictor and hasattr(ml_predictor, 'model_performance'):
            content['model_performance'] = ml_predictor.model_performance
            content['feature_importance'] = getattr(ml_predictor, 'feature_importance', {})
        
        # Risk factor analysis
        all_risk_factors = []
        for result in quality_results:
            all_risk_factors.extend(result.risk_factors)
        
        risk_frequency = {}
        for risk in all_risk_factors:
            risk_frequency[risk] = risk_frequency.get(risk, 0) + 1
        
        content['common_risk_factors'] = dict(sorted(risk_frequency.items(), key=lambda x: x[1], reverse=True)[:5])
        
        recommendations = []
        if content['prediction_summary']['average_confidence'] < 0.7:
            recommendations.append("Consider collecting more training data for better predictions")
        if risk_frequency:
            recommendations.append("Address most common risk factors systematically")
        
        return QualityReportSection(
            title="ML Predictions Analysis",
            content=content,
            recommendations=recommendations
        )
    
    def _generate_recommendations_section(self, 
                                        quality_results: List[QualityPredictionResult],
                                        overall_stats: Dict[str, Any]) -> QualityReportSection:
        """Generate recommendations section."""
        
        # Collect all improvement suggestions
        all_suggestions = []
        for result in quality_results:
            all_suggestions.extend(result.improvement_suggestions)
        
        # Count suggestion frequency
        suggestion_frequency = {}
        for suggestion in all_suggestions:
            suggestion_frequency[suggestion] = suggestion_frequency.get(suggestion, 0) + 1
        
        # Top recommendations
        top_recommendations = sorted(suggestion_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Priority recommendations based on overall stats
        priority_recommendations = []
        
        if overall_stats['mean_quality'] < 0.7:
            priority_recommendations.append("HIGH: Overall quality below target - comprehensive parameter review needed")
        
        if overall_stats['std_quality'] > 0.2:
            priority_recommendations.append("MEDIUM: High quality variance - improve consistency")
        
        if overall_stats['quality_distribution']['poor'] > overall_stats['total_structures'] * 0.2:
            priority_recommendations.append("HIGH: Too many poor quality structures - review generation parameters")
        
        content = {
            'priority_actions': priority_recommendations,
            'frequent_suggestions': [f"{suggestion} (mentioned {count} times)" for suggestion, count in top_recommendations],
            'quality_targets': {
                'target_average_quality': 0.85,
                'current_average_quality': round(overall_stats['mean_quality'], 3),
                'improvement_needed': round(0.85 - overall_stats['mean_quality'], 3),
                'target_excellent_rate': 0.25,
                'current_excellent_rate': round(overall_stats['quality_distribution']['excellent'] / overall_stats['total_structures'], 3)
            }
        }
        
        return QualityReportSection(
            title="Quality Improvement Recommendations",
            content=content
        )
    
    def _generate_methodology_section(self) -> QualityReportSection:
        """Generate methodology section."""
        
        content = {
            'quality_assessment_method': 'ML-enhanced crystallographic analysis',
            'metrics_calculated': [
                'Crystallographic quality (symmetry, space group)',
                'Thermodynamic stability indicators',
                'Electronic property estimates',
                'Structural integrity measures'
            ],
            'ml_model_details': {
                'primary_model': 'Random Forest Regressor',
                'feature_count': 14,
                'confidence_calculation': 'Prediction variance across ensemble'
            },
            'validation_approach': 'Cross-validation with historical data'
        }
        
        return QualityReportSection(
            title="Methodology",
            content=content
        )
    
    def _generate_appendix(self, 
                          quality_results: List[QualityPredictionResult],
                          generation_params: Dict[str, Any]) -> QualityReportSection:
        """Generate appendix with technical details."""
        
        content = {
            'generation_parameters': generation_params,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'structures_analyzed': len(quality_results),
                'prediction_method': 'ML-enhanced heuristic analysis',
                'confidence_threshold': 0.5
            }
        }
        
        # Sample detailed results
        if quality_results:
            sample_results = quality_results[:3]  # First 3 results as examples
            content['sample_detailed_results'] = [result.to_dict() for result in sample_results]
        
        return QualityReportSection(
            title="Technical Appendix",
            content=content
        )
    
    def _generate_visualizations(self, 
                               report: ComprehensiveQualityReport,
                               quality_results: List[QualityPredictionResult],
                               trend_data: List[QualityTrendData]):
        """Generate visualization plots for the report."""
        
        if not self.enable_visualizations:
            return
        
        # Quality distribution pie chart
        self._create_quality_distribution_plot(report.quality_distribution)
        
        # Quality trends over time
        if trend_data:
            self._create_quality_trend_plot(trend_data)
        
        # Crystallographic metrics radar chart
        if quality_results:
            self._create_crystallographic_radar_plot(quality_results)
    
    def _create_quality_distribution_plot(self, quality_distribution: Dict[str, int]):
        """Create quality distribution pie chart."""
        
        try:
            plt.figure(figsize=self.report_style['figure_size'])
            
            labels = list(quality_distribution.keys())
            sizes = list(quality_distribution.values())
            colors = self.report_style['color_palette'][:len(labels)]
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Structure Quality Distribution', fontsize=14, fontweight='bold')
            plt.axis('equal')
            
            output_path = self.output_dir / 'quality_distribution.png'
            plt.savefig(output_path, dpi=self.report_style['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Quality distribution plot saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create quality distribution plot: {e}")
    
    def _create_quality_trend_plot(self, trend_data: List[QualityTrendData]):
        """Create quality trend line plot."""
        
        try:
            plt.figure(figsize=self.report_style['figure_size'])
            
            iterations = [data.iteration for data in trend_data]
            qualities = [data.average_quality for data in trend_data]
            
            plt.plot(iterations, qualities, 'b-', linewidth=2, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Average Quality')
            plt.title('Quality Trend Over Time', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            output_path = self.output_dir / 'quality_trend.png'
            plt.savefig(output_path, dpi=self.report_style['dpi'], bbox_inches='tight')
            plt.close()
            
            logger.info(f"Quality trend plot saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create quality trend plot: {e}")
    
    def _create_crystallographic_radar_plot(self, quality_results: List[QualityPredictionResult]):
        """Create radar chart for crystallographic metrics."""
        
        try:
            # This would create a radar chart showing average crystallographic metrics
            # Implementation would depend on specific requirements and available data
            logger.info("Radar plot creation would be implemented here")
            
        except Exception as e:
            logger.warning(f"Failed to create crystallographic radar plot: {e}")
    
    def _save_report(self, report: ComprehensiveQualityReport):
        """Save report to various formats."""
        
        # Save as JSON
        json_path = self.output_dir / f"{report.report_id}.json"
        with open(json_path, 'w') as f:
            f.write(report.to_json())
        
        # Save as CSV summary
        self._save_csv_summary(report)
        
        logger.info(f"Report saved to {json_path}")
    
    def _save_csv_summary(self, report: ComprehensiveQualityReport):
        """Save report summary as CSV."""
        
        csv_path = self.output_dir / f"{report.report_id}_summary.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Metric', 'Value'])
            
            # Key metrics
            writer.writerow(['Report ID', report.report_id])
            writer.writerow(['Total Structures', report.total_structures])
            writer.writerow(['Overall Quality Score', report.overall_quality_score])
            writer.writerow(['Generation Timestamp', report.generation_timestamp.isoformat()])
            
            # Quality distribution
            writer.writerow(['', ''])  # Empty row
            writer.writerow(['Quality Distribution', ''])
            for quality_level, count in report.quality_distribution.items():
                writer.writerow([f'{quality_level.title()} Quality', count])
    
    # Simplified interface methods for easy integration
    def generate_report_from_quality_data(self, 
                                         quality_data: Dict[str, Any],
                                         generation_metadata: Dict[str, Any]) -> ComprehensiveQualityReport:
        """
        Simplified interface for generating reports from quality data.
        
        This method provides a simpler interface that matches the usage in generate.py.
        """
        # Convert quality_data to expected format for internal methods
        mock_quality_results = []  # Empty for now, can be populated from quality_data
        mock_trend_data = []       # Empty for now, can be populated from quality_data
        
        # Create a basic report with provided data
        report_id = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report = ComprehensiveQualityReport(
            report_id=report_id,
            generation_timestamp=datetime.now(),
            total_structures=generation_metadata.get('total_structures', 0),
            generation_parameters=generation_metadata,
            overall_quality_score=quality_data.get('overall_quality', 0.0),
            quality_distribution=quality_data.get('quality_distribution', {}),
            improvement_summary="Phase 4.2 quality analysis complete",
            executive_summary=QualityReportSection(
                title="Executive Summary",
                content={
                    'overview': 'Advanced quality analysis using Phase 4.2 features',
                    'key_findings': quality_data
                }
            ),
            crystallographic_analysis=QualityReportSection(
                title="Crystallographic Analysis", 
                content=quality_data.get('crystallographic_metrics', {})
            ),
            trend_analysis=QualityReportSection(
                title="Trend Analysis",
                content=quality_data.get('trend_data', {})
            ),
            ml_predictions=QualityReportSection(
                title="ML Predictions",
                content=quality_data.get('ml_predictions', {})
            ),
            recommendations=QualityReportSection(
                title="Recommendations",
                content={'recommendations': ['Phase 4.2 analysis recommendations available']}
            ),
            methodology=QualityReportSection(
                title="Methodology",
                content={'methods': 'Advanced ML-based quality assessment'}
            ),
            appendix=QualityReportSection(
                title="Appendix",
                content=generation_metadata
            )
        )
        
        return report
    
    def save_report(self, report: ComprehensiveQualityReport) -> str:
        """Save report and return the path."""
        # Save as JSON by default
        json_path = self.output_dir / f"{report.report_id}.json"
        
        # Convert report to dictionary for JSON serialization
        report_dict = asdict(report)
        
        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        return str(json_path)
    
    def generate_visualizations(self, quality_data: Dict[str, Any]) -> List[str]:
        """Generate visualizations from quality data."""
        viz_paths = []
        
        if not self.enable_visualizations:
            return viz_paths
        
        try:
            # Create a simple quality distribution plot
            if 'quality_distribution' in quality_data:
                viz_path = self._create_quality_distribution_plot(quality_data['quality_distribution'])
                if viz_path:
                    viz_paths.append(viz_path)
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        return viz_paths
    
    def _create_quality_distribution_plot(self, quality_dist: Dict) -> Optional[str]:
        """Create a simple quality distribution plot."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            if isinstance(quality_dist, dict) and quality_dist:
                keys = list(quality_dist.keys())
                values = list(quality_dist.values())
                plt.bar(keys, values, color='skyblue', alpha=0.7)
                plt.xlabel('Quality Metrics')
                plt.ylabel('Values')
                plt.title('Quality Distribution')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                viz_path = self.output_dir / f"quality_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return str(viz_path)
        except Exception as e:
            logger.warning(f"Plot creation failed: {e}")
            plt.close()
        
        return None


def create_quality_reporter(output_dir: Path, 
                          enable_visualizations: bool = True) -> AdvancedQualityReporter:
    """Create advanced quality reporter instance."""
    return AdvancedQualityReporter(output_dir, enable_visualizations)


# Alias for backward compatibility and consistent interface
QualityReporter = AdvancedQualityReporter
