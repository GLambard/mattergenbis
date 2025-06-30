"""
Advanced Quality Enhancement - Phase 4.2
========================================

ML-based quality prediction and advanced crystallographic analysis for MatterGen.

This module provides:
- Machine learning-based structure quality prediction
- Advanced crystallographic metrics and analysis
- Quality trend tracking and analysis
- Predictive quality modeling
- Enhanced quality reporting
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from datetime import datetime
from collections import defaultdict, deque
import warnings

# Suppress potential sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CrystallographicMetrics:
    """Advanced crystallographic quality metrics."""
    
    # Basic structural properties
    space_group_probability: float = 0.0
    symmetry_precision: float = 0.0
    lattice_stability: float = 0.0
    coordination_environment_score: float = 0.0
    
    # Bond analysis
    bond_length_distribution_score: float = 0.0
    bond_angle_distribution_score: float = 0.0
    bond_valence_sum_score: float = 0.0
    
    # Thermodynamic indicators
    formation_energy_estimate: float = 0.0
    stability_prediction: float = 0.0
    cohesive_energy_estimate: float = 0.0
    
    # Electronic properties prediction
    band_gap_estimate: float = 0.0
    metallic_character_score: float = 0.0
    
    # Defect analysis
    vacancy_formation_tendency: float = 0.0
    interstitial_tolerance: float = 0.0
    
    # Overall scores
    crystallographic_quality: float = 0.0
    thermodynamic_quality: float = 0.0
    electronic_quality: float = 0.0
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to feature vector for ML models."""
        return np.array([
            self.space_group_probability,
            self.symmetry_precision,
            self.lattice_stability,
            self.coordination_environment_score,
            self.bond_length_distribution_score,
            self.bond_angle_distribution_score,
            self.bond_valence_sum_score,
            self.formation_energy_estimate,
            self.stability_prediction,
            self.cohesive_energy_estimate,
            self.band_gap_estimate,
            self.metallic_character_score,
            self.vacancy_formation_tendency,
            self.interstitial_tolerance,
        ])


@dataclass
class QualityPredictionResult:
    """Result of ML-based quality prediction."""
    
    predicted_quality: float
    confidence_score: float
    quality_breakdown: Dict[str, float]
    risk_factors: List[str]
    improvement_suggestions: List[str]
    crystallographic_metrics: CrystallographicMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'predicted_quality': self.predicted_quality,
            'confidence_score': self.confidence_score,
            'quality_breakdown': self.quality_breakdown,
            'risk_factors': self.risk_factors,
            'improvement_suggestions': self.improvement_suggestions,
            'crystallographic_metrics': self.crystallographic_metrics.__dict__
        }


@dataclass
@dataclass
class QualityTrendData:
    """Data structure for tracking quality trends over time."""
    
    timestamp: datetime
    iteration: int
    batch_id: str
    structure_count: int
    average_quality: float
    quality_distribution: Dict[str, float]
    convergence_rate: float
    improvement_rate: float
    
    # Advanced metrics
    quality_variance: float = 0.0
    outlier_count: int = 0
    trend_direction: str = "stable"  # "improving", "degrading", "stable"
    prediction_accuracy: float = 0.0


class MLQualityPredictor:
    """Machine learning-based quality prediction system."""
    
    def __init__(self, 
                 model_type: str = "random_forest",
                 enable_training: bool = False):
        self.model_type = model_type
        self.enable_training = enable_training
        
        # Initialize models
        self.quality_predictor = None
        self.risk_classifier = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Training data storage
        self.training_features = []
        self.training_targets = []
        self.feature_importance = {}
        
        # Model performance tracking
        self.model_performance = {
            'mse': 0.0,
            'r2_score': 0.0,
            'prediction_accuracy': 0.0,
            'last_training_date': None
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. ML prediction will use heuristics.")
            return
        
        if self.model_type == "random_forest":
            self.quality_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.quality_predictor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            logger.warning(f"Unknown model type: {self.model_type}. Using random forest.")
            self.quality_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
    
    def predict_quality(self, 
                       crystallographic_metrics: CrystallographicMetrics,
                       structure_context: Optional[Dict] = None) -> QualityPredictionResult:
        """Predict structure quality using ML model."""
        
        # Extract features
        features = crystallographic_metrics.to_feature_vector()
        
        if SKLEARN_AVAILABLE and self.quality_predictor is not None:
            try:
                # Use trained model if available
                if hasattr(self.quality_predictor, 'predict'):
                    features_scaled = self.scaler.transform(features.reshape(1, -1))
                    predicted_quality = self.quality_predictor.predict(features_scaled)[0]
                    
                    # Calculate confidence based on model uncertainty
                    confidence_score = self._calculate_prediction_confidence(features_scaled)
                else:
                    # Fallback to heuristic prediction
                    predicted_quality, confidence_score = self._heuristic_prediction(crystallographic_metrics)
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}. Using heuristic fallback.")
                predicted_quality, confidence_score = self._heuristic_prediction(crystallographic_metrics)
        else:
            # Use heuristic prediction
            predicted_quality, confidence_score = self._heuristic_prediction(crystallographic_metrics)
        
        # Generate quality breakdown
        quality_breakdown = self._generate_quality_breakdown(crystallographic_metrics)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(crystallographic_metrics)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            crystallographic_metrics, risk_factors
        )
        
        return QualityPredictionResult(
            predicted_quality=predicted_quality,
            confidence_score=confidence_score,
            quality_breakdown=quality_breakdown,
            risk_factors=risk_factors,
            improvement_suggestions=improvement_suggestions,
            crystallographic_metrics=crystallographic_metrics
        )
    
    def _heuristic_prediction(self, 
                            metrics: CrystallographicMetrics) -> Tuple[float, float]:
        """Heuristic quality prediction when ML is not available."""
        
        # Weighted combination of key metrics
        weights = {
            'crystallographic': 0.3,
            'thermodynamic': 0.3,
            'electronic': 0.2,
            'structural': 0.2
        }
        
        crystallographic_score = np.mean([
            metrics.space_group_probability,
            metrics.symmetry_precision,
            metrics.coordination_environment_score
        ])
        
        thermodynamic_score = np.mean([
            metrics.lattice_stability,
            metrics.stability_prediction,
            1.0 - abs(metrics.formation_energy_estimate) / 10.0  # Normalize energy
        ])
        
        electronic_score = np.mean([
            metrics.band_gap_estimate / 10.0,  # Normalize band gap
            1.0 - metrics.metallic_character_score  # Higher for insulators
        ])
        
        structural_score = np.mean([
            metrics.bond_length_distribution_score,
            metrics.bond_angle_distribution_score,
            metrics.bond_valence_sum_score
        ])
        
        predicted_quality = (
            weights['crystallographic'] * crystallographic_score +
            weights['thermodynamic'] * thermodynamic_score +
            weights['electronic'] * electronic_score +
            weights['structural'] * structural_score
        )
        
        # Confidence based on consistency of metrics
        metric_values = [crystallographic_score, thermodynamic_score, 
                        electronic_score, structural_score]
        confidence_score = 1.0 - np.std(metric_values)
        
        return np.clip(predicted_quality, 0.0, 1.0), np.clip(confidence_score, 0.0, 1.0)
    
    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate prediction confidence based on model uncertainty."""
        if not SKLEARN_AVAILABLE or self.model_type != "random_forest":
            return 0.8  # Default confidence
        
        try:
            # For random forest, use prediction variance across trees
            tree_predictions = np.array([
                tree.predict(features)[0] 
                for tree in self.quality_predictor.estimators_
            ])
            
            prediction_variance = np.var(tree_predictions)
            confidence = 1.0 / (1.0 + prediction_variance)
            
            return np.clip(confidence, 0.0, 1.0)
        except:
            return 0.8
    
    def _generate_quality_breakdown(self, 
                                  metrics: CrystallographicMetrics) -> Dict[str, float]:
        """Generate detailed quality breakdown by category."""
        return {
            'crystallographic_quality': metrics.crystallographic_quality,
            'thermodynamic_quality': metrics.thermodynamic_quality,
            'electronic_quality': metrics.electronic_quality,
            'structural_integrity': np.mean([
                metrics.bond_length_distribution_score,
                metrics.bond_angle_distribution_score,
                metrics.coordination_environment_score
            ]),
            'symmetry_score': np.mean([
                metrics.space_group_probability,
                metrics.symmetry_precision
            ]),
            'stability_score': np.mean([
                metrics.lattice_stability,
                metrics.stability_prediction
            ])
        }
    
    def _identify_risk_factors(self, 
                              metrics: CrystallographicMetrics) -> List[str]:
        """Identify potential quality risk factors."""
        risk_factors = []
        
        if metrics.lattice_stability < 0.5:
            risk_factors.append("Low lattice stability detected")
        
        if metrics.bond_length_distribution_score < 0.6:
            risk_factors.append("Irregular bond length distribution")
        
        if metrics.coordination_environment_score < 0.6:
            risk_factors.append("Unusual coordination environments")
        
        if metrics.symmetry_precision < 0.7:
            risk_factors.append("Low symmetry precision")
        
        if abs(metrics.formation_energy_estimate) > 5.0:
            risk_factors.append("High formation energy indicates instability")
        
        if metrics.vacancy_formation_tendency > 0.8:
            risk_factors.append("High tendency for vacancy formation")
        
        return risk_factors
    
    def _generate_improvement_suggestions(self, 
                                        metrics: CrystallographicMetrics,
                                        risk_factors: List[str]) -> List[str]:
        """Generate suggestions for quality improvement."""
        suggestions = []
        
        if "Low lattice stability" in str(risk_factors):
            suggestions.append("Consider adjusting lattice parameters for better stability")
        
        if "Irregular bond length" in str(risk_factors):
            suggestions.append("Optimize atomic positions to regularize bond lengths")
        
        if "Unusual coordination" in str(risk_factors):
            suggestions.append("Review coordination environments for chemical reasonableness")
        
        if "Low symmetry precision" in str(risk_factors):
            suggestions.append("Apply symmetry constraints to improve precision")
        
        if "High formation energy" in str(risk_factors):
            suggestions.append("Consider alternative compositions or structures")
        
        if len(suggestions) == 0:
            suggestions.append("Structure quality is good - consider fine-tuning optimization")
        
        return suggestions
    
    def add_training_data(self, 
                         features: np.ndarray, 
                         quality_target: float):
        """Add training data for model improvement."""
        if self.enable_training:
            self.training_features.append(features)
            self.training_targets.append(quality_target)
    
    def retrain_model(self) -> bool:
        """Retrain the ML model with accumulated data."""
        if not SKLEARN_AVAILABLE or not self.enable_training:
            return False
        
        if len(self.training_features) < 10:
            logger.warning("Insufficient training data for retraining")
            return False
        
        try:
            X = np.array(self.training_features)
            y = np.array(self.training_targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.quality_predictor.fit(X_scaled, y)
            
            # Evaluate performance
            y_pred = self.quality_predictor.predict(X_scaled)
            self.model_performance['mse'] = mean_squared_error(y, y_pred)
            self.model_performance['r2_score'] = r2_score(y, y_pred)
            self.model_performance['last_training_date'] = datetime.now()
            
            # Feature importance
            if hasattr(self.quality_predictor, 'feature_importances_'):
                feature_names = [
                    'space_group_prob', 'symmetry_precision', 'lattice_stability',
                    'coordination_score', 'bond_length_score', 'bond_angle_score',
                    'bond_valence_score', 'formation_energy', 'stability_pred',
                    'cohesive_energy', 'band_gap', 'metallic_character',
                    'vacancy_tendency', 'interstitial_tolerance'
                ]
                self.feature_importance = dict(zip(
                    feature_names, 
                    self.quality_predictor.feature_importances_
                ))
            
            logger.info(f"Model retrained. R² score: {self.model_performance['r2_score']:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return False
    
    def save_model(self, model_path: Path):
        """Save trained model to disk."""
        if not SKLEARN_AVAILABLE:
            return
        
        model_data = {
            'quality_predictor': self.quality_predictor,
            'scaler': self.scaler,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path) -> bool:
        """Load trained model from disk."""
        if not SKLEARN_AVAILABLE or not model_path.exists():
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.quality_predictor = model_data['quality_predictor']
            self.scaler = model_data['scaler']
            self.model_performance = model_data['model_performance']
            self.feature_importance = model_data.get('feature_importance', {})
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_batch(self, structures: List[Dict]) -> Dict[str, Any]:
        """
        Predict quality for a batch of structures.
        
        Args:
            structures: List of structure dictionaries
            
        Returns:
            Dictionary with batch prediction results
        """
        if not structures:
            return {'mean_predicted_quality': 0.0, 'predictions': [], 'error': 'No structures provided'}
        
        try:
            predictions = []
            for structure in structures:
                # Calculate crystallographic metrics for the structure
                metrics = calculate_crystallographic_metrics(structure)
                
                # Predict quality
                prediction_result = self.predict_quality(metrics)
                predictions.append(prediction_result.predicted_quality)
            
            # Calculate batch statistics
            mean_quality = np.mean(predictions) if predictions else 0.0
            std_quality = np.std(predictions) if len(predictions) > 1 else 0.0
            
            return {
                'mean_predicted_quality': float(mean_quality),
                'std_predicted_quality': float(std_quality),
                'predictions': predictions,
                'prediction_count': len(predictions),
                'quality_range': {
                    'min': float(np.min(predictions)) if predictions else 0.0,
                    'max': float(np.max(predictions)) if predictions else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {
                'mean_predicted_quality': 0.0,
                'predictions': [],
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_available': self.quality_predictor is not None,
            'sklearn_available': SKLEARN_AVAILABLE,
            'model_type': type(self.quality_predictor).__name__ if self.quality_predictor else None,
            'performance': self.model_performance,
            'feature_importance': self.feature_importance
        }


class QualityTrendAnalyzer:
    """Analyzes quality trends over time for insights and predictions."""
    
    def __init__(self, max_history_length: int = 1000):
        self.max_history_length = max_history_length
        self.quality_history = deque(maxlen=max_history_length)
        self.trend_analysis = {}
    
    def add_quality_data(self, trend_data: QualityTrendData):
        """Add new quality data point for trend analysis."""
        self.quality_history.append(trend_data)
    
    def update(self, quality_score: float, structure_count: int, batch_id: int) -> Dict[str, Any]:
        """
        Update trend analysis with new quality data.
        
        Args:
            quality_score: Quality score for this batch
            structure_count: Number of structures in this batch
            batch_id: Batch identifier
            
        Returns:
            Updated trend analysis results
        """
        # Create trend data object
        trend_data = QualityTrendData(
            timestamp=datetime.now(),
            iteration=batch_id,
            batch_id=str(batch_id),
            structure_count=structure_count,
            average_quality=quality_score,
            quality_distribution={},
            convergence_rate=0.0,
            improvement_rate=0.0
        )
        
        # Add to history
        self.add_quality_data(trend_data)
        
        # Return current trend analysis
        return self.analyze_trends()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get trend analysis summary."""
        if not self.quality_history:
            return {'status': 'no_data', 'message': 'No trend data available'}
        
        return {
            'total_data_points': len(self.quality_history),
            'latest_quality': self.quality_history[-1].average_quality if self.quality_history else 0.0,
            'trend_analysis': self.analyze_trends(),
            'data_range': {
                'start': self.quality_history[0].timestamp.isoformat() if self.quality_history else None,
                'end': self.quality_history[-1].timestamp.isoformat() if self.quality_history else None
            }
        }
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze quality trends and generate insights."""
        if len(self.quality_history) < 5:
            return {"status": "insufficient_data", "message": "Need at least 5 data points"}
        
        # Extract time series data
        qualities = [data.average_quality for data in self.quality_history]
        iterations = [data.iteration for data in self.quality_history]
        
        # Trend analysis
        trend_analysis = {
            'current_quality': qualities[-1],
            'quality_trend': self._calculate_trend(qualities),
            'improvement_rate': self._calculate_improvement_rate(qualities),
            'convergence_status': self._assess_convergence(qualities),
            'quality_stability': self._calculate_stability(qualities),
            'predictions': self._predict_future_quality(qualities, iterations)
        }
        
        return trend_analysis
    
    def _calculate_trend(self, qualities: List[float]) -> str:
        """Calculate overall trend direction."""
        if len(qualities) < 3:
            return "insufficient_data"
        
        recent_avg = np.mean(qualities[-3:])
        earlier_avg = np.mean(qualities[-6:-3]) if len(qualities) >= 6 else np.mean(qualities[:-3])
        
        diff = recent_avg - earlier_avg
        
        if diff > 0.05:
            return "strongly_improving"
        elif diff > 0.01:
            return "improving"
        elif diff > -0.01:
            return "stable"
        elif diff > -0.05:
            return "degrading"
        else:
            return "strongly_degrading"
    
    def _calculate_improvement_rate(self, qualities: List[float]) -> float:
        """Calculate rate of quality improvement."""
        if len(qualities) < 2:
            return 0.0
        
        # Linear regression on recent data
        recent_qualities = qualities[-min(10, len(qualities)):]
        x = np.arange(len(recent_qualities))
        coeffs = np.polyfit(x, recent_qualities, 1)
        
        return coeffs[0]  # Slope indicates improvement rate
    
    def _assess_convergence(self, qualities: List[float]) -> Dict[str, Any]:
        """Assess convergence status."""
        if len(qualities) < 10:
            return {"status": "insufficient_data"}
        
        recent_window = qualities[-10:]
        variance = np.var(recent_window)
        mean_quality = np.mean(recent_window)
        
        # Convergence criteria
        converged = variance < 0.01 and mean_quality > 0.8
        
        return {
            "status": "converged" if converged else "still_improving",
            "variance": variance,
            "mean_quality": mean_quality,
            "convergence_score": max(0, 1 - variance) * mean_quality
        }
    
    def _calculate_stability(self, qualities: List[float]) -> float:
        """Calculate quality stability score."""
        if len(qualities) < 5:
            return 0.0
        
        # Moving average stability
        window_size = min(5, len(qualities))
        moving_vars = []
        
        for i in range(window_size, len(qualities) + 1):
            window = qualities[i-window_size:i]
            moving_vars.append(np.var(window))
        
        avg_variance = np.mean(moving_vars)
        stability_score = 1.0 / (1.0 + avg_variance)
        
        return stability_score
    
    def _predict_future_quality(self, 
                               qualities: List[float], 
                               iterations: List[int]) -> Dict[str, Any]:
        """Predict future quality based on trends."""
        if len(qualities) < 5:
            return {"status": "insufficient_data"}
        
        # Simple polynomial extrapolation
        try:
            x = np.array(iterations[-10:])  # Recent iterations
            y = np.array(qualities[-10:])   # Recent qualities
            
            # Fit polynomial (degree 2)
            coeffs = np.polyfit(x - x[0], y, min(2, len(x) - 1))
            poly = np.poly1d(coeffs)
            
            # Predict next few iterations
            future_iterations = [max(iterations) + i for i in range(1, 6)]
            future_qualities = [
                max(0, min(1, poly(it - x[0]))) 
                for it in future_iterations
            ]
            
            return {
                "status": "predicted",
                "future_iterations": future_iterations,
                "predicted_qualities": future_qualities,
                "confidence": 0.8 - min(0.6, np.var(y))  # Lower confidence with higher variance
            }
            
        except Exception as e:
            logger.warning(f"Quality prediction failed: {e}")
            return {"status": "prediction_failed"}


def create_advanced_quality_assessor(config: Optional[Dict] = None) -> Tuple[MLQualityPredictor, QualityTrendAnalyzer]:
    """Create advanced quality assessment system."""
    
    if config is None:
        config = {}
    
    # ML predictor configuration
    ml_config = config.get('ml_predictor', {})
    predictor = MLQualityPredictor(
        model_type=ml_config.get('model_type', 'random_forest'),
        enable_training=ml_config.get('enable_training', False)
    )
    
    # Trend analyzer configuration
    trend_config = config.get('trend_analyzer', {})
    analyzer = QualityTrendAnalyzer(
        max_history_length=trend_config.get('max_history_length', 1000)
    )
    
    return predictor, analyzer


# Example crystallographic metrics calculator (placeholder)
def calculate_crystallographic_metrics(structure_data: Any) -> CrystallographicMetrics:
    """
    Calculate advanced crystallographic metrics for a structure.
    
    This is a placeholder implementation. In a real system, this would
    analyze the actual crystal structure data.
    """
    
    # Mock calculation - replace with real analysis
    metrics = CrystallographicMetrics()
    
    # Simulate some realistic values
    metrics.space_group_probability = np.random.uniform(0.7, 0.95)
    metrics.symmetry_precision = np.random.uniform(0.8, 0.98)
    metrics.lattice_stability = np.random.uniform(0.6, 0.9)
    metrics.coordination_environment_score = np.random.uniform(0.7, 0.95)
    
    metrics.bond_length_distribution_score = np.random.uniform(0.65, 0.9)
    metrics.bond_angle_distribution_score = np.random.uniform(0.7, 0.92)
    metrics.bond_valence_sum_score = np.random.uniform(0.6, 0.88)
    
    metrics.formation_energy_estimate = np.random.uniform(-3.0, 1.0)
    metrics.stability_prediction = np.random.uniform(0.6, 0.9)
    metrics.cohesive_energy_estimate = np.random.uniform(-8.0, -2.0)
    
    metrics.band_gap_estimate = np.random.uniform(0.0, 5.0)
    metrics.metallic_character_score = np.random.uniform(0.0, 0.3)
    
    metrics.vacancy_formation_tendency = np.random.uniform(0.2, 0.7)
    metrics.interstitial_tolerance = np.random.uniform(0.3, 0.8)
    
    # Calculate overall scores
    metrics.crystallographic_quality = np.mean([
        metrics.space_group_probability,
        metrics.symmetry_precision,
        metrics.coordination_environment_score
    ])
    
    metrics.thermodynamic_quality = np.mean([
        metrics.lattice_stability,
        metrics.stability_prediction,
        max(0, 1 - abs(metrics.formation_energy_estimate) / 5.0)
    ])
    
    metrics.electronic_quality = np.mean([
        min(1, metrics.band_gap_estimate / 3.0),
        1 - metrics.metallic_character_score
    ])
    
    return metrics


class AdvancedQualityMetrics:
    """
    Main interface class for advanced quality metrics and analysis.
    
    This class provides a unified interface for:
    - ML-based quality prediction
    - Advanced crystallographic analysis  
    - Quality trend tracking
    - Comprehensive quality assessment
    """
    
    def __init__(
        self,
        enable_ml_prediction: bool = False,
        enable_trend_analysis: bool = True,
        model_path: Optional[str] = None,
        output_dir: str = "./quality_analysis"
    ):
        """Initialize advanced quality metrics system."""
        self.enable_ml_prediction = enable_ml_prediction
        self.enable_trend_analysis = enable_trend_analysis
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.ml_predictor = None
        self.trend_analyzer = None
        self.results_history = []
        
        if enable_ml_prediction and SKLEARN_AVAILABLE:
            self.ml_predictor = MLQualityPredictor()
            if model_path:
                # Try to load the model if a path is provided
                self.ml_predictor.load_model(Path(model_path))
            logger.info("ML quality predictor initialized")
        elif enable_ml_prediction:
            logger.warning("ML prediction requested but sklearn not available")
            
        if enable_trend_analysis:
            self.trend_analyzer = QualityTrendAnalyzer()
            logger.info("Quality trend analyzer initialized")
    
    def analyze_structures(
        self, 
        structures: List[Dict], 
        batch_id: int = 0,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality analysis on structures.
        
        Args:
            structures: List of structure dictionaries
            batch_id: Batch identifier for tracking
            metadata: Additional metadata for analysis
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            'batch_id': batch_id,
            'structure_count': len(structures),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'crystallographic_metrics': {},
            'quality_predictions': {},
            'trend_data': {}
        }
        
        # Analyze each structure
        structure_metrics = []
        for i, structure in enumerate(structures):
            metrics = calculate_crystallographic_metrics(structure)
            structure_metrics.append(metrics)
        
        # Aggregate crystallographic metrics
        if structure_metrics:
            results['crystallographic_metrics'] = self._aggregate_crystallographic_metrics(structure_metrics)
        
        # ML-based quality prediction if enabled
        if self.enable_ml_prediction and self.ml_predictor:
            try:
                prediction_results = self.ml_predictor.predict_batch(structures)
                results['quality_predictions'] = prediction_results
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                results['quality_predictions'] = {'error': str(e)}
        
        # Update trend analysis if enabled
        if self.enable_trend_analysis and self.trend_analyzer:
            try:
                overall_quality = results['crystallographic_metrics'].get('overall_quality', 0.0)
                trend_data = self.trend_analyzer.update(
                    quality_score=overall_quality,
                    structure_count=len(structures),
                    batch_id=batch_id
                )
                results['trend_data'] = trend_data
            except Exception as e:
                logger.warning(f"Trend analysis failed: {e}")
                results['trend_data'] = {'error': str(e)}
        
        # Store results
        self.results_history.append(results)
        
        # Save results to disk
        results_file = self.output_dir / f"analysis_batch_{batch_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def predict_quality(self, structures: List[Dict]) -> Dict[str, Any]:
        """Predict quality using ML models."""
        if not self.enable_ml_prediction or not self.ml_predictor:
            return {'error': 'ML prediction not enabled or not available'}
        
        try:
            return self.ml_predictor.predict_batch(structures)
        except Exception as e:
            logger.error(f"Quality prediction failed: {e}")
            return {'error': str(e)}
    
    def update_trends(
        self, 
        quality_score: float, 
        structures: List[Dict], 
        iteration: int
    ) -> Dict[str, Any]:
        """Update quality trend analysis."""
        if not self.enable_trend_analysis or not self.trend_analyzer:
            return {'error': 'Trend analysis not enabled'}
        
        try:
            return self.trend_analyzer.update(
                quality_score=quality_score,
                structure_count=len(structures),
                batch_id=iteration
            )
        except Exception as e:
            logger.error(f"Trend update failed: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_results(self) -> Dict[str, Any]:
        """Get comprehensive results from all analyses."""
        return {
            'total_batches': len(self.results_history),
            'total_structures': sum(r.get('structure_count', 0) for r in self.results_history),
            'batch_results': self.results_history,
            'trend_summary': self.trend_analyzer.get_summary() if self.trend_analyzer else {},
            'ml_model_info': self.ml_predictor.get_model_info() if self.ml_predictor else {}
        }
    
    def _aggregate_crystallographic_metrics(self, metrics_list: List[CrystallographicMetrics]) -> Dict[str, float]:
        """Aggregate crystallographic metrics across structures."""
        if not metrics_list:
            return {}
        
        # Calculate averages for all metrics
        aggregated = {}
        
        # Get all metric attributes
        sample_metrics = metrics_list[0]
        for attr_name in dir(sample_metrics):
            if not attr_name.startswith('_') and hasattr(sample_metrics, attr_name):
                attr_value = getattr(sample_metrics, attr_name)
                if isinstance(attr_value, (int, float)):
                    values = [getattr(m, attr_name) for m in metrics_list]
                    aggregated[f'avg_{attr_name}'] = np.mean(values)
                    aggregated[f'std_{attr_name}'] = np.std(values)
                    aggregated[f'min_{attr_name}'] = np.min(values)
                    aggregated[f'max_{attr_name}'] = np.max(values)
        
        # Calculate overall quality score
        if 'avg_crystallographic_quality' in aggregated:
            aggregated['overall_quality'] = aggregated['avg_crystallographic_quality']
        else:
            # Fallback calculation
            key_metrics = ['avg_space_group_probability', 'avg_symmetry_precision', 'avg_lattice_stability']
            available_metrics = [aggregated.get(metric, 0.0) for metric in key_metrics if metric in aggregated]
            aggregated['overall_quality'] = np.mean(available_metrics) if available_metrics else 0.0
        
        return aggregated