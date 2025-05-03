"""
Cultural Adaptation Module
Adjusts psychological analysis based on cultural norms and communication patterns
"""

from typing import Dict, Any, List
import numpy as np
from dataclasses import dataclass
from enum import Enum

class CulturalDimension(Enum):
    INDIVIDUALISM = "individualism"
    COLLECTIVISM = "collectivism"
    HIGH_CONTEXT = "high_context"
    LOW_CONTEXT = "low_context"
    POWER_DISTANCE = "power_distance"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"

@dataclass
class CulturalProfile:
    name: str
    dimensions: Dict[CulturalDimension, float]
    communication_patterns: Dict[str, float]
    adjustment_factors: Dict[str, float]

class CulturalAdapter:
    def __init__(self, config: Dict[str, Any]):
        # Initialize cultural profiles
        self.profiles = {
            'western': CulturalProfile(
                name='western',
                dimensions={
                    CulturalDimension.INDIVIDUALISM: 0.8,
                    CulturalDimension.COLLECTIVISM: 0.2,
                    CulturalDimension.HIGH_CONTEXT: 0.3,
                    CulturalDimension.LOW_CONTEXT: 0.7,
                    CulturalDimension.POWER_DISTANCE: 0.4,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.5
                },
                communication_patterns={
                    'directness': 0.8,
                    'explicitness': 0.7,
                    'emotional_expression': 0.6,
                    'conflict_style': 0.7
                },
                adjustment_factors={
                    'assertiveness': 1.2,
                    'emotionality': 1.1,
                    'independence': 1.3
                }
            ),
            'eastern': CulturalProfile(
                name='eastern',
                dimensions={
                    CulturalDimension.INDIVIDUALISM: 0.2,
                    CulturalDimension.COLLECTIVISM: 0.8,
                    CulturalDimension.HIGH_CONTEXT: 0.7,
                    CulturalDimension.LOW_CONTEXT: 0.3,
                    CulturalDimension.POWER_DISTANCE: 0.7,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.6
                },
                communication_patterns={
                    'directness': 0.4,
                    'explicitness': 0.3,
                    'emotional_expression': 0.5,
                    'conflict_style': 0.4
                },
                adjustment_factors={
                    'assertiveness': 0.8,
                    'emotionality': 0.9,
                    'independence': 0.7
                }
            ),
            'middle_eastern': CulturalProfile(
                name='middle_eastern',
                dimensions={
                    CulturalDimension.INDIVIDUALISM: 0.3,
                    CulturalDimension.COLLECTIVISM: 0.7,
                    CulturalDimension.HIGH_CONTEXT: 0.8,
                    CulturalDimension.LOW_CONTEXT: 0.2,
                    CulturalDimension.POWER_DISTANCE: 0.8,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.7
                },
                communication_patterns={
                    'directness': 0.5,
                    'explicitness': 0.4,
                    'emotional_expression': 0.7,
                    'conflict_style': 0.5
                },
                adjustment_factors={
                    'assertiveness': 0.9,
                    'emotionality': 1.2,
                    'independence': 0.8
                }
            )
        }
        
        # Set active profile based on config
        self.active_profile = self.profiles.get(
            config.get('culture', 'western'),
            self.profiles['western']
        )

    def adjust_personality_scores(self, 
                                scores: Dict[str, float]) -> Dict[str, float]:
        """Adjust personality scores based on cultural context"""
        adjusted_scores = {}
        
        # Apply cultural adjustments to each trait
        for trait, score in scores.items():
            adjustment_factor = self._get_trait_adjustment(trait)
            adjusted_scores[trait] = self._apply_adjustment(score, adjustment_factor)
        
        return adjusted_scores

    def _get_trait_adjustment(self, trait: str) -> float:
        """Get adjustment factor for a specific trait"""
        # Map traits to cultural dimensions
        trait_mapping = {
            'extraversion': 'assertiveness',
            'neuroticism': 'emotionality',
            'openness': 'independence',
            'agreeableness': 'emotionality',
            'conscientiousness': 'assertiveness'
        }
        
        # Get base adjustment factor
        base_factor = self.active_profile.adjustment_factors.get(
            trait_mapping.get(trait, 'assertiveness'),
            1.0
        )
        
        # Apply dimension-based adjustments
        if trait in ['extraversion', 'openness']:
            base_factor *= self.active_profile.dimensions[CulturalDimension.INDIVIDUALISM]
        elif trait in ['agreeableness', 'conscientiousness']:
            base_factor *= self.active_profile.dimensions[CulturalDimension.COLLECTIVISM]
        
        return base_factor

    def _apply_adjustment(self, score: float, adjustment: float) -> float:
        """Apply adjustment factor to score with bounds checking"""
        adjusted = score * adjustment
        return max(0.0, min(1.0, adjusted))

    def adjust_communication_patterns(self, 
                                   patterns: Dict[str, float]) -> Dict[str, float]:
        """Adjust communication patterns based on cultural context"""
        adjusted_patterns = {}
        
        for pattern, score in patterns.items():
            cultural_norm = self.active_profile.communication_patterns.get(
                pattern, 0.5
            )
            adjusted_patterns[pattern] = self._normalize_to_cultural_norm(
                score, cultural_norm
            )
        
        return adjusted_patterns

    def _normalize_to_cultural_norm(self, 
                                  score: float, 
                                  cultural_norm: float) -> float:
        """Normalize score relative to cultural norm"""
        # Adjust score to be relative to cultural norm
        adjusted = score + (cultural_norm - 0.5)
        return max(0.0, min(1.0, adjusted))

    def generate_cultural_context(self, 
                                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cultural context for analysis results"""
        context = {
            'cultural_profile': self.active_profile.name,
            'dimensions': {
                dim.name: value 
                for dim, value in self.active_profile.dimensions.items()
            },
            'adjustments_applied': {},
            'interpretation_guidelines': []
        }
        
        # Document adjustments
        if 'personality' in analysis_results:
            context['adjustments_applied']['personality'] = {
                trait: self._get_trait_adjustment(trait)
                for trait in analysis_results['personality'].keys()
            }
        
        if 'communication' in analysis_results:
            context['adjustments_applied']['communication'] = {
                pattern: self.active_profile.communication_patterns.get(pattern, 0.5)
                for pattern in analysis_results['communication'].keys()
            }
        
        # Generate interpretation guidelines
        context['interpretation_guidelines'] = self._generate_guidelines(
            analysis_results
        )
        
        return context

    def _generate_guidelines(self, 
                           analysis_results: Dict[str, Any]) -> List[str]:
        """Generate cultural interpretation guidelines"""
        guidelines = []
        
        # Add general cultural context
        guidelines.append(
            f"Analysis interpreted within {self.active_profile.name} cultural context"
        )
        
        # Add dimension-specific guidelines
        if self.active_profile.dimensions[CulturalDimension.INDIVIDUALISM] > 0.7:
            guidelines.append(
                "Individual achievement and autonomy are highly valued"
            )
        elif self.active_profile.dimensions[CulturalDimension.COLLECTIVISM] > 0.7:
            guidelines.append(
                "Group harmony and interdependence are prioritized"
            )
        
        if self.active_profile.dimensions[CulturalDimension.HIGH_CONTEXT] > 0.7:
            guidelines.append(
                "Communication relies heavily on context and implicit meaning"
            )
        
        # Add trait-specific guidelines
        if 'personality' in analysis_results:
            for trait, score in analysis_results['personality'].items():
                if score > 0.7:
                    guidelines.append(
                        f"High {trait} scores should be interpreted considering "
                        f"cultural norms of {self.active_profile.name} context"
                    )
        
        return guidelines

    def validate_cultural_assumptions(self, 
                                    analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cultural assumptions in analysis"""
        validation = {
            'potential_biases': [],
            'cultural_considerations': [],
            'recommendations': []
        }
        
        # Check for potential cultural biases
        if 'personality' in analysis_results:
            for trait, score in analysis_results['personality'].items():
                adjustment = self._get_trait_adjustment(trait)
                if abs(adjustment - 1.0) > 0.3:
                    validation['potential_biases'].append(
                        f"Significant cultural adjustment ({adjustment:.2f}) "
                        f"applied to {trait} scores"
                    )
        
        # Add cultural considerations
        validation['cultural_considerations'] = self._get_cultural_considerations(
            analysis_results
        )
        
        # Generate recommendations
        validation['recommendations'] = self._generate_validation_recommendations(
            validation['potential_biases']
        )
        
        return validation

    def _get_cultural_considerations(self, 
                                   analysis_results: Dict[str, Any]) -> List[str]:
        """Get cultural considerations for analysis"""
        considerations = []
        
        # Add general cultural considerations
        considerations.append(
            f"Analysis conducted in {self.active_profile.name} cultural context"
        )
        
        # Add specific considerations based on cultural dimensions
        if self.active_profile.dimensions[CulturalDimension.HIGH_CONTEXT] > 0.7:
            considerations.append(
                "High-context communication patterns may affect interpretation"
            )
        
        if self.active_profile.dimensions[CulturalDimension.POWER_DISTANCE] > 0.7:
            considerations.append(
                "Hierarchical relationships may influence behavior patterns"
            )
        
        return considerations

    def _generate_validation_recommendations(self, 
                                          biases: List[str]) -> List[str]:
        """Generate recommendations for cultural validation"""
        recommendations = []
        
        # Add general recommendations
        recommendations.append(
            "Consider cultural context when interpreting results"
        )
        
        # Add specific recommendations for identified biases
        for bias in biases:
            recommendations.append(
                f"Review {bias} for potential cultural bias"
            )
        
        # Add cultural competence recommendations
        recommendations.append(
            "Consult with cultural experts when interpreting results"
        )
        
        return recommendations 