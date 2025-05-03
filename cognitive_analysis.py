"""
Cognitive Analysis Engine
Analysis of thought patterns and cognitive distortions
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import Dict, Any, List
import re
from collections import defaultdict

class CognitiveAnalyzer:
    def __init__(self):
        # Initialize BERT model for text analysis
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Define cognitive distortion patterns
        self.distortion_patterns = {
            'catastrophizing': [
                r'worst.*ever',
                r'disaster',
                r'ruin.*life',
                r'never.*recover'
            ],
            'polarized_thinking': [
                r'always|never',
                r'perfect|failure',
                r'best|worst',
                r'completely.*terrible'
            ],
            'overgeneralization': [
                r'every.*time',
                r'all.*people',
                r'nothing.*works',
                r'everyone.*hates'
            ],
            'personalization': [
                r'my.*fault',
                r'because.*of.*me',
                r'should.*have',
                r'if.*only.*I'
            ],
            'mind_reading': [
                r'they.*think',
                r'everyone.*knows',
                r'people.*judge',
                r'assume.*they'
            ]
        }
        
        # Initialize distortion classifiers
        self.distortion_classifiers = {
            name: nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            ) for name in self.distortion_patterns.keys()
        }
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for classifier in self.distortion_classifiers.values():
            for layer in classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def analyze_thought_patterns(self, text: str) -> Dict[str, Dict[str, float]]:
        """Analyze text for cognitive distortions"""
        # Get BERT embeddings
        inputs = self.tokenizer(text, return_tensors='pt',
                              truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Initialize results
        results = {
            'pattern_matches': {},
            'neural_predictions': {},
            'severity_scores': {}
        }
        
        # Pattern matching analysis
        for distortion, patterns in self.distortion_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.finditer(pattern, text.lower()))
            results['pattern_matches'][distortion] = len(matches)
        
        # Neural network predictions
        for distortion, classifier in self.distortion_classifiers.items():
            with torch.no_grad():
                score = torch.sigmoid(classifier(embeddings)).item()
                results['neural_predictions'][distortion] = score
        
        # Calculate severity scores
        for distortion in self.distortion_patterns.keys():
            pattern_score = min(results['pattern_matches'][distortion] / 5, 1.0)
            neural_score = results['neural_predictions'][distortion]
            results['severity_scores'][distortion] = (pattern_score + neural_score) / 2
        
        return results

    def identify_cognitive_biases(self, text: str) -> Dict[str, float]:
        """Identify cognitive biases in text"""
        biases = {
            'confirmation_bias': self._detect_confirmation_bias(text),
            'negativity_bias': self._detect_negativity_bias(text),
            'anchoring_bias': self._detect_anchoring_bias(text),
            'availability_bias': self._detect_availability_bias(text)
        }
        
        return biases

    def _detect_confirmation_bias(self, text: str) -> float:
        """Detect confirmation bias in text"""
        # Look for selective evidence and ignoring counter-evidence
        patterns = [
            r'proves.*point',
            r'ignore.*fact',
            r'only.*see',
            r'clear.*evidence'
        ]
        
        matches = sum(len(re.findall(pattern, text.lower())) 
                     for pattern in patterns)
        return min(matches / 3, 1.0)

    def _detect_negativity_bias(self, text: str) -> float:
        """Detect negativity bias in text"""
        # Count negative vs positive words
        negative_words = len(re.findall(r'\b(?:bad|terrible|awful|horrible)\b', 
                                      text.lower()))
        positive_words = len(re.findall(r'\b(?:good|great|excellent|wonderful)\b', 
                                      text.lower()))
        
        total_words = negative_words + positive_words
        if total_words == 0:
            return 0.0
            
        return negative_words / total_words

    def _detect_anchoring_bias(self, text: str) -> float:
        """Detect anchoring bias in text"""
        # Look for initial reference points and subsequent judgments
        patterns = [
            r'first.*thought',
            r'initial.*impression',
            r'based.*on',
            r'compared.*to'
        ]
        
        matches = sum(len(re.findall(pattern, text.lower())) 
                     for pattern in patterns)
        return min(matches / 2, 1.0)

    def _detect_availability_bias(self, text: str) -> float:
        """Detect availability bias in text"""
        # Look for recent or vivid examples
        patterns = [
            r'remember.*when',
            r'last.*time',
            r'just.*saw',
            r'heard.*about'
        ]
        
        matches = sum(len(re.findall(pattern, text.lower())) 
                     for pattern in patterns)
        return min(matches / 2, 1.0)

    def generate_cognitive_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from cognitive analysis"""
        insights = {
            'primary_distortions': [],
            'bias_impact': {},
            'recommendations': []
        }
        
        # Identify primary cognitive distortions
        severity_scores = analysis_results['severity_scores']
        primary_distortions = sorted(
            severity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        insights['primary_distortions'] = [
            {
                'distortion': distortion,
                'severity': severity,
                'description': self._get_distortion_description(distortion)
            }
            for distortion, severity in primary_distortions
        ]
        
        # Calculate bias impact
        bias_scores = self.identify_cognitive_biases(
            analysis_results.get('original_text', '')
        )
        insights['bias_impact'] = {
            bias: {
                'score': score,
                'impact': self._get_bias_impact(bias, score)
            }
            for bias, score in bias_scores.items()
        }
        
        # Generate recommendations
        insights['recommendations'] = self._generate_recommendations(
            primary_distortions,
            bias_scores
        )
        
        return insights

    def _get_distortion_description(self, distortion: str) -> str:
        """Get description of cognitive distortion"""
        descriptions = {
            'catastrophizing': "Tendency to expect the worst possible outcome",
            'polarized_thinking': "Viewing situations in black and white terms",
            'overgeneralization': "Making broad conclusions from single events",
            'personalization': "Taking excessive responsibility for events",
            'mind_reading': "Assuming knowledge of others' thoughts"
        }
        return descriptions.get(distortion, "Unknown distortion")

    def _get_bias_impact(self, bias: str, score: float) -> str:
        """Get impact description of cognitive bias"""
        if score < 0.3:
            return "Minimal impact on thinking"
        elif score < 0.6:
            return "Moderate influence on decision-making"
        else:
            return "Significant impact on judgment"

    def _generate_recommendations(self, 
                                primary_distortions: List[tuple],
                                bias_scores: Dict[str, float]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Add distortion-specific recommendations
        for distortion, severity in primary_distortions:
            if severity > 0.5:
                recommendations.append(
                    f"Practice challenging {distortion} patterns by "
                    f"identifying evidence for and against your thoughts"
                )
        
        # Add bias-specific recommendations
        for bias, score in bias_scores.items():
            if score > 0.5:
                recommendations.append(
                    f"Be aware of {bias} when making decisions and "
                    f"consider alternative perspectives"
                )
        
        return recommendations 