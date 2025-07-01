"""
Structure Quality Metrics for MatterGen
======================================

This module implements comprehensive quality assessment for generated crystal structures,
including validation, scoring, and quality-based filtering capabilities.

Key Features:
- Multi-metric quality scoring (geometric, physical, chemical)
- Real-time quality assessment during generation
- Quality-based filtering and validation
- Detailed quality reporting and analytics
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
from enum import Enum

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality level classifications."""
    EXCELLENT = "excellent"  # > 0.9
    GOOD = "good"           # 0.8 - 0.9
    ACCEPTABLE = "acceptable"  # 0.7 - 0.8
    POOR = "poor"           # 0.5 - 0.7
    INVALID = "invalid"     # < 0.5


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a crystal structure."""
    # Geometric quality
    bond_length_score: float
    bond_angle_score: float
    symmetry_score: float
    volume_score: float
    
    # Physical quality
    stability_score: float
    energy_score: float
    density_score: float
    
    # Chemical quality
    composition_score: float
    oxidation_state_score: float
    coordination_score: float
    
    # Overall scores
    geometric_quality: float
    physical_quality: float
    chemical_quality: float
    overall_quality: float
    quality_level: QualityLevel
    
    # Validation flags
    is_valid: bool
    has_singular_cell: bool
    has_overlapping_atoms: bool
    has_unrealistic_bonds: bool
    
    # Metadata
    structure_id: Optional[str] = None
    generation_step: Optional[int] = None
    computation_time: Optional[float] = None


@dataclass
class QualityThresholds:
    """Configurable quality thresholds for filtering."""
    # Minimum thresholds for acceptance
    min_overall_quality: float = 0.7
    min_geometric_quality: float = 0.6
    min_physical_quality: float = 0.6
    min_chemical_quality: float = 0.6
    
    # Individual metric thresholds
    min_bond_length_score: float = 0.5
    min_stability_score: float = 0.5
    min_symmetry_score: float = 0.4
    
    # Validation requirements
    require_valid_cell: bool = True
    require_no_overlaps: bool = True
    require_realistic_bonds: bool = True
    
    # Quality level targets
    target_excellent_rate: float = 0.1  # 10% excellent structures
    target_good_rate: float = 0.3       # 30% good structures
    target_acceptable_rate: float = 0.5  # 50% acceptable structures


class StructureQualityAssessor:
    """Main quality assessment engine for crystal structures."""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        self.assessment_stats = {
            'total_assessed': 0,
            'valid_structures': 0,
            'quality_distribution': {level.value: 0 for level in QualityLevel},
            'average_quality': 0.0,
            'assessment_time': 0.0
        }
        
    def assess_structure_quality(self, 
                               structure_data: Dict,
                               structure_id: Optional[str] = None,
                               generation_step: Optional[int] = None) -> QualityMetrics:
        """Assess the quality of a single crystal structure."""
        start_time = time.time()
        
        try:
            # Extract structure properties
            cell_matrix = structure_data.get('cell', np.eye(3))
            positions = structure_data.get('positions', [])
            atomic_numbers = structure_data.get('atomic_numbers', [])
            
            # Perform quality assessments
            geometric_metrics = self._assess_geometric_quality(cell_matrix, positions)
            physical_metrics = self._assess_physical_quality(cell_matrix, positions, atomic_numbers)
            chemical_metrics = self._assess_chemical_quality(atomic_numbers, positions)
            
            # Calculate overall scores
            geometric_quality = np.mean(list(geometric_metrics.values()))
            physical_quality = np.mean(list(physical_metrics.values()))
            chemical_quality = np.mean(list(chemical_metrics.values()))
            overall_quality = (geometric_quality + physical_quality + chemical_quality) / 3
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_quality)
            
            # Validation checks
            validation_results = self._perform_validation_checks(cell_matrix, positions)
            
            # Create quality metrics object
            metrics = QualityMetrics(
                # Geometric
                bond_length_score=geometric_metrics['bond_length'],
                bond_angle_score=geometric_metrics['bond_angle'],
                symmetry_score=geometric_metrics['symmetry'],
                volume_score=geometric_metrics['volume'],
                
                # Physical
                stability_score=physical_metrics['stability'],
                energy_score=physical_metrics['energy'],
                density_score=physical_metrics['density'],
                
                # Chemical
                composition_score=chemical_metrics['composition'],
                oxidation_state_score=chemical_metrics['oxidation_state'],
                coordination_score=chemical_metrics['coordination'],
                
                # Overall
                geometric_quality=geometric_quality,
                physical_quality=physical_quality,
                chemical_quality=chemical_quality,
                overall_quality=overall_quality,
                quality_level=quality_level,
                
                # Validation
                is_valid=validation_results['is_valid'],
                has_singular_cell=validation_results['has_singular_cell'],
                has_overlapping_atoms=validation_results['has_overlapping_atoms'],
                has_unrealistic_bonds=validation_results['has_unrealistic_bonds'],
                
                # Metadata
                structure_id=structure_id,
                generation_step=generation_step,
                computation_time=time.time() - start_time
            )
            
            # Update statistics
            self._update_assessment_stats(metrics)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Quality assessment failed for structure {structure_id}: {e}")
            # Return default poor quality metrics
            return self._create_failed_assessment_metrics(structure_id, generation_step)
    
    def _assess_geometric_quality(self, cell_matrix: np.ndarray, positions: List) -> Dict[str, float]:
        """Assess geometric quality aspects."""
        metrics = {}
        
        # Bond length quality (realistic bond lengths)
        bond_lengths = self._calculate_bond_lengths(positions)
        metrics['bond_length'] = self._score_bond_lengths(bond_lengths)
        
        # Bond angle quality (realistic bond angles)
        bond_angles = self._calculate_bond_angles(positions)
        metrics['bond_angle'] = self._score_bond_angles(bond_angles)
        
        # Symmetry quality (crystal symmetry preservation)
        metrics['symmetry'] = self._score_symmetry(cell_matrix, positions)
        
        # Volume quality (reasonable unit cell volume)
        cell_volume = np.abs(np.linalg.det(cell_matrix))
        metrics['volume'] = self._score_volume(cell_volume, len(positions))
        
        return metrics
    
    def _assess_physical_quality(self, cell_matrix: np.ndarray, positions: List, atomic_numbers: List) -> Dict[str, float]:
        """Assess physical quality aspects."""
        metrics = {}
        
        # Structural stability (based on energy considerations)
        metrics['stability'] = self._score_stability(cell_matrix, positions, atomic_numbers)
        
        # Energy quality (reasonable formation energy)
        metrics['energy'] = self._score_energy(atomic_numbers, positions)
        
        # Density quality (realistic crystal density)
        density = self._calculate_density(cell_matrix, atomic_numbers)
        metrics['density'] = self._score_density(density, atomic_numbers)
        
        return metrics
    
    def _assess_chemical_quality(self, atomic_numbers: List, positions: List) -> Dict[str, float]:
        """Assess chemical quality aspects."""
        metrics = {}
        
        # Composition quality (reasonable elemental composition)
        metrics['composition'] = self._score_composition(atomic_numbers)
        
        # Oxidation state quality (realistic oxidation states)
        metrics['oxidation_state'] = self._score_oxidation_states(atomic_numbers, positions)
        
        # Coordination quality (reasonable coordination environments)
        metrics['coordination'] = self._score_coordination(atomic_numbers, positions)
        
        return metrics
    
    def _perform_validation_checks(self, cell_matrix: np.ndarray, positions: List) -> Dict[str, bool]:
        """Perform basic validation checks on the structure."""
        results = {}
        
        # Check for singular cell matrix
        det = np.abs(np.linalg.det(cell_matrix))
        results['has_singular_cell'] = det < 1e-10
        
        # Check for overlapping atoms
        results['has_overlapping_atoms'] = self._check_atom_overlaps(positions)
        
        # Check for unrealistic bonds
        results['has_unrealistic_bonds'] = self._check_unrealistic_bonds(positions)
        
        # Overall validity
        results['is_valid'] = not any([
            results['has_singular_cell'] and self.thresholds.require_valid_cell,
            results['has_overlapping_atoms'] and self.thresholds.require_no_overlaps,
            results['has_unrealistic_bonds'] and self.thresholds.require_realistic_bonds
        ])
        
        return results
    
    def _calculate_bond_lengths(self, positions: List) -> List[float]:
        """Calculate bond lengths between atoms."""
        if len(positions) < 2:
            return []
        
        bond_lengths = []
        positions = np.array(positions)
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                bond_lengths.append(distance)
        
        return bond_lengths
    
    def _score_bond_lengths(self, bond_lengths: List[float]) -> float:
        """Score bond lengths based on chemical reasonableness."""
        if not bond_lengths:
            return 0.5
        
        # Typical bond length ranges (Angstroms)
        min_reasonable = 0.5
        max_reasonable = 5.0
        optimal_range = (1.0, 3.0)
        
        reasonable_count = 0
        optimal_count = 0
        
        for length in bond_lengths:
            if min_reasonable <= length <= max_reasonable:
                reasonable_count += 1
                if optimal_range[0] <= length <= optimal_range[1]:
                    optimal_count += 1
        
        if len(bond_lengths) == 0:
            return 0.5
        
        reasonable_ratio = reasonable_count / len(bond_lengths)
        optimal_ratio = optimal_count / len(bond_lengths)
        
        # Score based on both reasonable and optimal ratios
        score = 0.5 * reasonable_ratio + 0.5 * optimal_ratio
        return min(1.0, max(0.0, score))
    
    def _score_bond_angles(self, bond_angles: List[float]) -> float:
        """Score bond angles based on chemical reasonableness."""
        # Placeholder implementation - would analyze actual bond angles
        # For now, return a reasonable default
        return 0.8
    
    def _score_symmetry(self, cell_matrix: np.ndarray, positions: List) -> float:
        """Score crystal symmetry preservation."""
        # Placeholder implementation - would analyze crystal symmetry
        # For now, check if cell matrix is reasonable
        det = np.abs(np.linalg.det(cell_matrix))
        if det < 1e-10:
            return 0.0
        
        # Check for reasonable cell parameters
        cell_params = np.diag(cell_matrix)
        if np.any(cell_params <= 0):
            return 0.0
        
        return 0.75  # Default reasonable symmetry score
    
    def _score_volume(self, volume: float, num_atoms: int) -> float:
        """Score unit cell volume reasonableness."""
        if num_atoms == 0:
            return 0.0
        
        # Typical volume per atom (Angstrom^3)
        volume_per_atom = volume / num_atoms
        typical_range = (10, 50)  # Typical range for crystalline materials
        
        if typical_range[0] <= volume_per_atom <= typical_range[1]:
            return 1.0
        elif volume_per_atom < typical_range[0]:
            return max(0.0, volume_per_atom / typical_range[0])
        else:
            return max(0.0, typical_range[1] / volume_per_atom)
    
    def _score_stability(self, cell_matrix: np.ndarray, positions: List, atomic_numbers: List) -> float:
        """Score structural stability."""
        # Placeholder implementation - would calculate formation energy
        # For now, return based on basic structural checks
        if len(positions) == 0:
            return 0.0
        
        # Basic stability indicators
        det = np.abs(np.linalg.det(cell_matrix))
        if det < 1e-10:
            return 0.0
        
        return 0.7  # Default stability score
    
    def _score_energy(self, atomic_numbers: List, positions: List) -> float:
        """Score energy reasonableness."""
        # Placeholder implementation
        return 0.75
    
    def _calculate_density(self, cell_matrix: np.ndarray, atomic_numbers: List) -> float:
        """Calculate crystal density."""
        if len(atomic_numbers) == 0:
            return 0.0
        
        volume = np.abs(np.linalg.det(cell_matrix))
        if volume < 1e-10:
            return 0.0
        
        # Approximate atomic masses (simplified)
        atomic_masses = {1: 1.0, 6: 12.0, 8: 16.0, 26: 56.0}  # H, C, O, Fe
        total_mass = sum(atomic_masses.get(num, 50.0) for num in atomic_numbers)
        
        # Convert to g/cm^3 (approximate conversion)
        density = total_mass / volume * 1.66  # Rough conversion factor
        return density
    
    def _score_density(self, density: float, atomic_numbers: List) -> float:
        """Score density reasonableness."""
        # Typical density ranges for crystals (g/cm^3)
        typical_range = (1.0, 15.0)
        optimal_range = (2.0, 8.0)
        
        if optimal_range[0] <= density <= optimal_range[1]:
            return 1.0
        elif typical_range[0] <= density <= typical_range[1]:
            return 0.7
        else:
            return 0.3
    
    def _score_composition(self, atomic_numbers: List) -> float:
        """Score elemental composition reasonableness."""
        if not atomic_numbers:
            return 0.0
        
        # Check for reasonable elemental diversity
        unique_elements = set(atomic_numbers)
        if len(unique_elements) == 1:
            return 0.6  # Single element is okay but not optimal
        elif len(unique_elements) <= 4:
            return 1.0  # Good diversity
        else:
            return 0.8  # Too many elements might be unrealistic
    
    def _score_oxidation_states(self, atomic_numbers: List, positions: List) -> float:
        """Score oxidation state reasonableness."""
        # Placeholder implementation
        return 0.75
    
    def _score_coordination(self, atomic_numbers: List, positions: List) -> float:
        """Score coordination environment reasonableness."""
        # Placeholder implementation
        return 0.8
    
    def _check_atom_overlaps(self, positions: List) -> bool:
        """Check for overlapping atoms."""
        if len(positions) < 2:
            return False
        
        positions = np.array(positions)
        min_distance = 0.5  # Minimum reasonable distance (Angstroms)
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < min_distance:
                    return True
        
        return False
    
    def _check_unrealistic_bonds(self, positions: List) -> bool:
        """Check for unrealistic bond lengths."""
        bond_lengths = self._calculate_bond_lengths(positions)
        
        # Check for extremely short or long bonds
        for length in bond_lengths:
            if length < 0.3 or length > 6.0:  # Unrealistic range
                return True
        
        return False
    
    def _determine_quality_level(self, overall_quality: float) -> QualityLevel:
        """Determine quality level based on overall score."""
        if overall_quality >= 0.9:
            return QualityLevel.EXCELLENT
        elif overall_quality >= 0.8:
            return QualityLevel.GOOD
        elif overall_quality >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif overall_quality >= 0.5:
            return QualityLevel.POOR
        else:
            return QualityLevel.INVALID
    
    def _create_failed_assessment_metrics(self, structure_id: Optional[str], generation_step: Optional[int]) -> QualityMetrics:
        """Create metrics for failed assessment."""
        return QualityMetrics(
            bond_length_score=0.0,
            bond_angle_score=0.0,
            symmetry_score=0.0,
            volume_score=0.0,
            stability_score=0.0,
            energy_score=0.0,
            density_score=0.0,
            composition_score=0.0,
            oxidation_state_score=0.0,
            coordination_score=0.0,
            geometric_quality=0.0,
            physical_quality=0.0,
            chemical_quality=0.0,
            overall_quality=0.0,
            quality_level=QualityLevel.INVALID,
            is_valid=False,
            has_singular_cell=True,
            has_overlapping_atoms=True,
            has_unrealistic_bonds=True,
            structure_id=structure_id,
            generation_step=generation_step,
            computation_time=0.0
        )
    
    def _update_assessment_stats(self, metrics: QualityMetrics):
        """Update assessment statistics."""
        self.assessment_stats['total_assessed'] += 1
        if metrics.is_valid:
            self.assessment_stats['valid_structures'] += 1
        
        # Update quality distribution
        self.assessment_stats['quality_distribution'][metrics.quality_level.value] += 1
        
        # Update average quality
        total = self.assessment_stats['total_assessed']
        current_avg = self.assessment_stats['average_quality']
        self.assessment_stats['average_quality'] = (
            (current_avg * (total - 1) + metrics.overall_quality) / total
        )
        
        # Update assessment time
        if metrics.computation_time:
            current_time = self.assessment_stats['assessment_time']
            self.assessment_stats['assessment_time'] = (
                (current_time * (total - 1) + metrics.computation_time) / total
            )
    
    def should_keep_structure(self, metrics: QualityMetrics) -> bool:
        """Determine if a structure should be kept based on quality thresholds."""
        if not metrics.is_valid:
            return False
        
        return (
            metrics.overall_quality >= self.thresholds.min_overall_quality and
            metrics.geometric_quality >= self.thresholds.min_geometric_quality and
            metrics.physical_quality >= self.thresholds.min_physical_quality and
            metrics.chemical_quality >= self.thresholds.min_chemical_quality and
            metrics.bond_length_score >= self.thresholds.min_bond_length_score and
            metrics.stability_score >= self.thresholds.min_stability_score and
            metrics.symmetry_score >= self.thresholds.min_symmetry_score
        )
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of quality assessment statistics."""
        total = self.assessment_stats['total_assessed']
        if total == 0:
            return {}
        
        valid_rate = self.assessment_stats['valid_structures'] / total
        quality_rates = {
            level: count / total 
            for level, count in self.assessment_stats['quality_distribution'].items()
        }
        
        return {
            'total_structures_assessed': total,
            'valid_structures': self.assessment_stats['valid_structures'],
            'valid_rate': round(valid_rate * 100, 1),
            'average_quality_score': round(self.assessment_stats['average_quality'], 3),
            'average_assessment_time_ms': round(self.assessment_stats['assessment_time'] * 1000, 2),
            'quality_distribution': {
                level: {
                    'count': count,
                    'percentage': round(rate * 100, 1)
                }
                for level, (count, rate) in zip(
                    self.assessment_stats['quality_distribution'].keys(),
                    zip(self.assessment_stats['quality_distribution'].values(), quality_rates.values())
                )
            },
            'thresholds': self.thresholds.__dict__
        }
    
    def assess_structures(self, structures: List[Dict]) -> Tuple[float, Dict]:
        """Assess quality for a batch of structures."""
        if not structures:
            return 0.0, {}
        
        total_quality = 0.0
        quality_scores = []
        structure_reports = {}
        
        for i, structure in enumerate(structures):
            try:
                metrics = self.assess_structure_quality(structure, structure_id=f"batch_struct_{i}")
                quality_scores.append(metrics.overall_quality)
                total_quality += metrics.overall_quality
                
                structure_reports[f"structure_{i}"] = {
                    'overall_quality': metrics.overall_quality,
                    'quality_level': metrics.quality_level.value,
                    'is_valid': metrics.is_valid,
                    'geometric_quality': metrics.geometric_quality,
                    'physical_quality': metrics.physical_quality,
                    'chemical_quality': metrics.chemical_quality
                }
            except Exception as e:
                logger.warning(f"Failed to assess structure {i}: {e}")
                quality_scores.append(0.0)
        
        average_quality = total_quality / len(structures) if structures else 0.0
        
        report = {
            'average_quality': average_quality,
            'quality_scores': quality_scores,
            'num_structures': len(structures),
            'structure_reports': structure_reports,
            'quality_distribution': {
                'excellent': sum(1 for q in quality_scores if q >= 0.9),
                'good': sum(1 for q in quality_scores if 0.7 <= q < 0.9),
                'acceptable': sum(1 for q in quality_scores if 0.5 <= q < 0.7),
                'poor': sum(1 for q in quality_scores if q < 0.5)
            }
        }
        
        return average_quality, report
    
    def filter_structures(self, structures: List[Dict]) -> List[Dict]:
        """Filter structures based on quality thresholds."""
        if not structures:
            return []
        
        filtered = []
        for i, structure in enumerate(structures):
            try:
                metrics = self.assess_structure_quality(structure, structure_id=f"filter_struct_{i}")
                
                # Check if structure meets quality thresholds
                if (metrics.overall_quality >= self.thresholds.min_overall_quality and
                    metrics.is_valid and
                    (not self.thresholds.require_valid_cell or not metrics.has_singular_cell) and
                    (not self.thresholds.require_no_overlaps or not metrics.has_overlapping_atoms)):
                    filtered.append(structure)
                    
            except Exception as e:
                logger.warning(f"Failed to filter structure {i}: {e}")
                # Skip invalid structures
                continue
        
        return filtered


# Factory function for easy integration
def create_quality_assessor(
    min_overall_quality: float = 0.7,
    min_geometric_quality: float = 0.6,
    require_valid_cell: bool = True,
    require_no_overlaps: bool = True
) -> StructureQualityAssessor:
    """Factory function to create quality assessor with common configurations."""
    
    thresholds = QualityThresholds(
        min_overall_quality=min_overall_quality,
        min_geometric_quality=min_geometric_quality,
        require_valid_cell=require_valid_cell,
        require_no_overlaps=require_no_overlaps
    )
    
    return StructureQualityAssessor(thresholds)
