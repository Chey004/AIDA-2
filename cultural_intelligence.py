"""
Cultural Intelligence Layer
Adjusts psychological analysis based on cultural norms and communication patterns
"""

import numpy as np
from textblob import TextBlob

class CulturalIntelligence:
    def __init__(self):
        # Cultural profiles with adjustment factors
        self.cultural_profiles = {
            'east_asian': {
                'modesty_adjustment': 0.7,
                'collectivism_boost': 1.3,
                'indirectness_factor': 1.2,
                'harmony_weight': 1.4
            },
            'nordic': {
                'directness_factor': 1.2,
                'individualism_scaling': 0.9,
                'equality_bias': 1.1,
                'pragmatism_weight': 1.3
            },
            'middle_eastern': {
                'respect_hierarchy': 1.3,
                'group_orientation': 1.2,
                'indirectness_factor': 1.1,
                'relationship_weight': 1.4
            },
            'latin': {
                'emotional_expression': 1.3,
                'relationship_orientation': 1.2,
                'flexibility_factor': 1.1,
                'social_weight': 1.4
            }
        }
        
        # Communication pattern detectors
        self.pattern_detectors = {
            'modesty': self._detect_modesty_patterns,
            'directness': self._detect_directness,
            'collectivism': self._detect_collectivism,
            'emotional_expression': self._detect_emotional_expression
        }

    def adjust_for_cultural_norms(self, text, context):
        """Apply cultural adjustments to text analysis"""
        if context not in self.cultural_profiles:
            return text
            
        profile = self.cultural_profiles[context]
        adjustments = self._analyze_cultural_patterns(text)
        
        # Apply cultural adjustments
        adjusted_scores = {}
        for pattern, score in adjustments.items():
            if pattern in profile:
                adjusted_scores[pattern] = score * profile[pattern]
            else:
                adjusted_scores[pattern] = score
                
        return adjusted_scores

    def _analyze_cultural_patterns(self, text):
        """Analyze cultural communication patterns in text"""
        patterns = {}
        for pattern_name, detector in self.pattern_detectors.items():
            patterns[pattern_name] = detector(text)
        return patterns

    def _detect_modesty_patterns(self, text):
        """Detect modesty-related communication patterns"""
        modesty_indicators = [
            'maybe', 'perhaps', 'possibly', 'I think', 'in my opinion',
            'could be', 'might be', 'seems like'
        ]
        
        doc = nlp(text.lower())
        modesty_count = sum(1 for token in doc if token.text in modesty_indicators)
        total_words = len(doc)
        
        return modesty_count / max(1, total_words)

    def _detect_directness(self, text):
        """Detect directness in communication"""
        direct_indicators = [
            'definitely', 'certainly', 'absolutely', 'must', 'should',
            'need to', 'have to', 'will'
        ]
        
        doc = nlp(text.lower())
        direct_count = sum(1 for token in doc if token.text in direct_indicators)
        total_words = len(doc)
        
        return direct_count / max(1, total_words)

    def _detect_collectivism(self, text):
        """Detect collectivist communication patterns"""
        collectivist_indicators = [
            'we', 'our', 'us', 'together', 'team', 'group',
            'collaborate', 'cooperate', 'share'
        ]
        
        individualist_indicators = [
            'I', 'me', 'my', 'mine', 'myself', 'individual',
            'personal', 'private'
        ]
        
        doc = nlp(text.lower())
        collectivist_count = sum(1 for token in doc if token.text in collectivist_indicators)
        individualist_count = sum(1 for token in doc if token.text in individualist_indicators)
        total_words = len(doc)
        
        return (collectivist_count - individualist_count) / max(1, total_words)

    def _detect_emotional_expression(self, text):
        """Detect emotional expression patterns"""
        emotion_indicators = [
            'feel', 'emotion', 'passion', 'heart', 'soul',
            'excited', 'happy', 'sad', 'angry', 'frustrated'
        ]
        
        doc = nlp(text.lower())
        emotion_count = sum(1 for token in doc if token.text in emotion_indicators)
        total_words = len(doc)
        
        return emotion_count / max(1, total_words)

    def get_cultural_interpretation(self, scores, context):
        """Generate cultural context-aware interpretation"""
        if context not in self.cultural_profiles:
            return "Standard interpretation"
            
        profile = self.cultural_profiles[context]
        interpretation = []
        
        # Generate context-specific interpretations
        if profile.get('modesty_adjustment', 1) < 1:
            interpretation.append("Note: Responses may reflect cultural modesty norms")
        if profile.get('collectivism_boost', 1) > 1:
            interpretation.append("Consider collectivist cultural context in interpretation")
        if profile.get('indirectness_factor', 1) > 1:
            interpretation.append("Account for potential indirect communication style")
            
        return " | ".join(interpretation)

    def adjust_personality_scores(self, scores, context):
        """Adjust personality scores based on cultural context"""
        if context not in self.cultural_profiles:
            return scores
            
        profile = self.cultural_profiles[context]
        adjusted_scores = scores.copy()
        
        # Apply cultural adjustments to personality scores
        if 'modesty_adjustment' in profile:
            adjusted_scores['Extraversion'] *= profile['modesty_adjustment']
        if 'collectivism_boost' in profile:
            adjusted_scores['Agreeableness'] *= profile['collectivism_boost']
        if 'directness_factor' in profile:
            adjusted_scores['Openness'] *= profile['directness_factor']
            
        return adjusted_scores 