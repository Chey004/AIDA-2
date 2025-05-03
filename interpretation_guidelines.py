"""
Interpretation Guidelines
Scoring rubrics and contextual evaluation framework
"""

from typing import Dict, Any, List
import numpy as np

class InterpretationGuidelines:
    def __init__(self):
        self.scoring_rubrics = {
            'neuroticism': {
                'clinical_range': (8, 10),
                'healthy_range': (3, 7),
                'interpretation_tips': [
                    "Consider stress coping mechanisms",
                    "Evaluate emotional regulation strategies",
                    "Assess anxiety management techniques"
                ]
            },
            'dark_triad': {
                'clinical_range': (6, 10),
                'healthy_range': (0, 2),
                'interpretation_tips': [
                    "Verify with behavioral evidence",
                    "Consider situational factors",
                    "Evaluate impact on relationships"
                ]
            },
            'attachment_style': {
                'secure_range': (7, 10),
                'anxious_range': (4, 6),
                'avoidant_range': (0, 3),
                'interpretation_tips': [
                    "Assess relationship patterns",
                    "Evaluate trust development",
                    "Consider childhood experiences"
                ]
            }
        }
        
        self.cultural_contexts = {
            'western': {
                'individualism_weight': 1.2,
                'direct_communication': 1.1,
                'achievement_focus': 1.3
            },
            'eastern': {
                'collectivism_weight': 1.2,
                'indirect_communication': 1.1,
                'harmony_focus': 1.3
            },
            'middle_eastern': {
                'family_orientation': 1.2,
                'respect_hierarchy': 1.1,
                'relationship_focus': 1.3
            }
        }

    def generate_interpretation(self, report: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate contextual interpretation of analysis results"""
        primary_trait = self._identify_primary_trait(report)
        cultural_adjustment = self._apply_cultural_context(context['cultural_norms'])
        intervention = self._choose_intervention(report, context)
        
        interpretation = (
            f"Subject shows {primary_trait['trait']} tendencies "
            f"(score: {primary_trait['score']:.2f}, "
            f"{context['cultural_norms']} cultural lens applied). "
            f"Recommend {intervention} "
            f"with {context['assessment_purpose']} focus."
        )
        
        return interpretation

    def _identify_primary_trait(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Identify the most prominent personality trait"""
        traits = {
            'neuroticism': report.get('neuroticism', 0),
            'extraversion': report.get('extraversion', 0),
            'openness': report.get('openness', 0),
            'agreeableness': report.get('agreeableness', 0),
            'conscientiousness': report.get('conscientiousness', 0)
        }
        
        primary_trait = max(traits.items(), key=lambda x: x[1])
        return {
            'trait': primary_trait[0],
            'score': primary_trait[1]
        }

    def _apply_cultural_context(self, cultural_norm: str) -> Dict[str, float]:
        """Apply cultural context adjustments"""
        if cultural_norm not in self.cultural_contexts:
            return {}
        
        return self.cultural_contexts[cultural_norm]

    def _choose_intervention(self, report: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Select appropriate intervention based on analysis"""
        interventions = {
            'clinical': [
                "cognitive behavioral therapy",
                "mindfulness training",
                "emotional regulation techniques"
            ],
            'organizational': [
                "leadership development",
                "team building exercises",
                "communication skills training"
            ],
            'educational': [
                "learning style assessment",
                "study skills development",
                "motivation enhancement"
            ]
        }
        
        purpose = context['assessment_purpose']
        if purpose not in interventions:
            return "standard assessment"
        
        # Select intervention based on primary traits
        primary_trait = self._identify_primary_trait(report)
        trait_interventions = {
            'neuroticism': interventions[purpose][0],
            'extraversion': interventions[purpose][1],
            'openness': interventions[purpose][2]
        }
        
        return trait_interventions.get(primary_trait['trait'], interventions[purpose][0])

    def get_scoring_guidelines(self, trait: str) -> Dict[str, Any]:
        """Get scoring guidelines for specific trait"""
        if trait not in self.scoring_rubrics:
            return {}
        
        return self.scoring_rubrics[trait]

    def evaluate_trait_severity(self, trait: str, score: float) -> str:
        """Evaluate severity of trait score"""
        if trait not in self.scoring_rubrics:
            return "unknown"
        
        guidelines = self.scoring_rubrics[trait]
        if 'clinical_range' in guidelines:
            clinical_min, clinical_max = guidelines['clinical_range']
            if score >= clinical_min:
                return "clinical"
        
        if 'healthy_range' in guidelines:
            healthy_min, healthy_max = guidelines['healthy_range']
            if healthy_min <= score <= healthy_max:
                return "healthy"
        
        return "borderline" 