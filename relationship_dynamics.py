"""
Relationship Dynamics Analysis
Analysis of social interactions and relationship patterns
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import Dict, Any, List, Tuple
import networkx as nx
from collections import defaultdict

class RelationshipAnalyzer:
    def __init__(self):
        # Initialize BERT model for text analysis
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize relationship pattern classifiers
        self.pattern_classifiers = {
            'attachment_style': nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 3)  # Secure, Anxious, Avoidant
            ),
            'interaction_style': nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 4)  # Assertive, Passive, Aggressive, Passive-Aggressive
            )
        }
        
        # Initialize weights
        self._init_weights()
        
        # Define interaction patterns
        self.interaction_patterns = {
            'assertive': [
                r'express.*needs',
                r'respect.*boundaries',
                r'clear.*communication',
                r'confident.*manner'
            ],
            'passive': [
                r'avoid.*conflict',
                r'hard.*say.*no',
                r'put.*others.*first',
                r'keep.*opinions'
            ],
            'aggressive': [
                r'dominate.*conversation',
                r'disregard.*feelings',
                r'impose.*views',
                r'control.*situation'
            ],
            'passive_aggressive': [
                r'indirect.*communication',
                r'sarcastic.*remarks',
                r'withhold.*information',
                r'backhanded.*compliments'
            ]
        }

    def _init_weights(self):
        """Initialize model weights"""
        for classifier in self.pattern_classifiers.values():
            for layer in classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def analyze_relationships(self, text: str) -> Dict[str, Any]:
        """Analyze relationship patterns in text"""
        # Get BERT embeddings
        inputs = self.tokenizer(text, return_tensors='pt',
                              truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Initialize results
        results = {
            'attachment_style': {},
            'interaction_style': {},
            'pattern_matches': {},
            'relationship_quality': {}
        }
        
        # Predict attachment style
        with torch.no_grad():
            attachment_scores = torch.softmax(
                self.pattern_classifiers['attachment_style'](embeddings),
                dim=1
            ).squeeze()
            results['attachment_style'] = {
                'secure': attachment_scores[0].item(),
                'anxious': attachment_scores[1].item(),
                'avoidant': attachment_scores[2].item()
            }
        
        # Predict interaction style
        with torch.no_grad():
            interaction_scores = torch.softmax(
                self.pattern_classifiers['interaction_style'](embeddings),
                dim=1
            ).squeeze()
            results['interaction_style'] = {
                'assertive': interaction_scores[0].item(),
                'passive': interaction_scores[1].item(),
                'aggressive': interaction_scores[2].item(),
                'passive_aggressive': interaction_scores[3].item()
            }
        
        # Pattern matching analysis
        for style, patterns in self.interaction_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.finditer(pattern, text.lower()))
            results['pattern_matches'][style] = len(matches)
        
        # Calculate relationship quality indicators
        results['relationship_quality'] = self._calculate_quality_indicators(
            results['attachment_style'],
            results['interaction_style']
        )
        
        return results

    def _calculate_quality_indicators(self, 
                                   attachment_scores: Dict[str, float],
                                   interaction_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate relationship quality indicators"""
        return {
            'trust': attachment_scores['secure'] * interaction_scores['assertive'],
            'conflict_resolution': (
                attachment_scores['secure'] * 
                (1 - interaction_scores['aggressive'])
            ),
            'emotional_safety': (
                attachment_scores['secure'] * 
                (1 - attachment_scores['anxious'])
            ),
            'boundary_respect': (
                interaction_scores['assertive'] * 
                (1 - interaction_scores['aggressive'])
            )
        }

    def analyze_social_network(self, 
                             interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze social network structure and dynamics"""
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for interaction in interactions:
            source = interaction['source']
            target = interaction['target']
            weight = interaction.get('weight', 1.0)
            
            G.add_edge(source, target, weight=weight)
        
        # Calculate network metrics
        metrics = {
            'centrality': nx.degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'pagerank': nx.pagerank(G)
        }
        
        # Identify key roles
        roles = self._identify_social_roles(metrics)
        
        return {
            'network_metrics': metrics,
            'social_roles': roles,
            'community_structure': self._detect_communities(G)
        }

    def _identify_social_roles(self, 
                             metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Identify social roles based on network metrics"""
        roles = defaultdict(list)
        
        for node in metrics['centrality'].keys():
            # Calculate role scores
            centrality = metrics['centrality'][node]
            betweenness = metrics['betweenness'][node]
            closeness = metrics['closeness'][node]
            pagerank = metrics['pagerank'][node]
            
            # Identify roles based on metric combinations
            if centrality > 0.7 and betweenness > 0.7:
                roles['connector'].append(node)
            elif closeness > 0.7 and pagerank > 0.7:
                roles['influencer'].append(node)
            elif betweenness > 0.7 and pagerank < 0.3:
                roles['bridge'].append(node)
            elif centrality < 0.3 and betweenness < 0.3:
                roles['peripheral'].append(node)
        
        return dict(roles)

    def _detect_communities(self, G: nx.DiGraph) -> Dict[str, List[str]]:
        """Detect communities in the social network"""
        # Convert to undirected graph for community detection
        G_undirected = G.to_undirected()
        
        # Use Louvain method for community detection
        communities = nx.community.louvain_communities(G_undirected)
        
        # Format results
        return {
            f'community_{i}': list(community)
            for i, community in enumerate(communities)
        }

    def generate_relationship_insights(self, 
                                    analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from relationship analysis"""
        insights = {
            'primary_patterns': [],
            'relationship_strengths': [],
            'areas_for_growth': [],
            'recommendations': []
        }
        
        # Identify primary patterns
        attachment_scores = analysis_results['attachment_style']
        interaction_scores = analysis_results['interaction_style']
        
        primary_attachment = max(attachment_scores.items(), key=lambda x: x[1])
        primary_interaction = max(interaction_scores.items(), key=lambda x: x[1])
        
        insights['primary_patterns'] = [
            {
                'type': 'attachment',
                'style': primary_attachment[0],
                'score': primary_attachment[1],
                'description': self._get_attachment_description(primary_attachment[0])
            },
            {
                'type': 'interaction',
                'style': primary_interaction[0],
                'score': primary_interaction[1],
                'description': self._get_interaction_description(primary_interaction[0])
            }
        ]
        
        # Identify strengths and areas for growth
        quality_indicators = analysis_results['relationship_quality']
        
        for indicator, score in quality_indicators.items():
            if score > 0.7:
                insights['relationship_strengths'].append(
                    f"Strong {indicator.replace('_', ' ')}"
                )
            elif score < 0.3:
                insights['areas_for_growth'].append(
                    f"Need to improve {indicator.replace('_', ' ')}"
                )
        
        # Generate recommendations
        insights['recommendations'] = self._generate_recommendations(
            primary_attachment,
            primary_interaction,
            quality_indicators
        )
        
        return insights

    def _get_attachment_description(self, style: str) -> str:
        """Get description of attachment style"""
        descriptions = {
            'secure': "Comfortable with intimacy and independence",
            'anxious': "Seeks high levels of intimacy and approval",
            'avoidant': "Values independence and self-sufficiency"
        }
        return descriptions.get(style, "Unknown attachment style")

    def _get_interaction_description(self, style: str) -> str:
        """Get description of interaction style"""
        descriptions = {
            'assertive': "Communicates needs and boundaries effectively",
            'passive': "Tends to avoid conflict and prioritize others' needs",
            'aggressive': "May disregard others' feelings and boundaries",
            'passive_aggressive': "Expresses hostility indirectly"
        }
        return descriptions.get(style, "Unknown interaction style")

    def _generate_recommendations(self,
                                primary_attachment: Tuple[str, float],
                                primary_interaction: Tuple[str, float],
                                quality_indicators: Dict[str, float]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Attachment-based recommendations
        if primary_attachment[0] == 'anxious' and primary_attachment[1] > 0.7:
            recommendations.append(
                "Practice self-soothing techniques and develop "
                "confidence in your worth independent of others"
            )
        elif primary_attachment[0] == 'avoidant' and primary_attachment[1] > 0.7:
            recommendations.append(
                "Gradually increase comfort with emotional intimacy "
                "and vulnerability in safe relationships"
            )
        
        # Interaction-based recommendations
        if primary_interaction[0] == 'passive' and primary_interaction[1] > 0.7:
            recommendations.append(
                "Practice assertive communication and setting "
                "healthy boundaries"
            )
        elif primary_interaction[0] == 'aggressive' and primary_interaction[1] > 0.7:
            recommendations.append(
                "Develop active listening skills and practice "
                "considering others' perspectives"
            )
        
        # Quality indicator-based recommendations
        if quality_indicators['trust'] < 0.3:
            recommendations.append(
                "Work on building trust through consistent, "
                "reliable behavior and open communication"
            )
        if quality_indicators['conflict_resolution'] < 0.3:
            recommendations.append(
                "Learn and practice constructive conflict "
                "resolution techniques"
            )
        
        return recommendations 