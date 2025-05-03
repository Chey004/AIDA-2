"""
Advanced Visualization Suite
Interactive personality profiles and relationship mapping
"""

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import Dict, List, Any

class AdvancedVisualizer:
    def __init__(self):
        self.trait_labels = [
            'Openness', 'Conscientiousness', 'Extraversion',
            'Agreeableness', 'Neuroticism'
        ]
        
        self.relationship_metrics = {
            'interaction_frequency': 0.4,
            'emotional_connection': 0.3,
            'conflict_resolution': 0.3
        }

    def create_interactive_profile(self, analysis_results: Dict[str, float]) -> go.Figure:
        """Create interactive personality radar chart"""
        # Normalize scores to 0-1 range
        scores = [analysis_results.get(trait.lower(), 0) for trait in self.trait_labels]
        normalized_scores = self._normalize_scores(scores)
        
        fig = px.line_polar(
            r=normalized_scores,
            theta=self.trait_labels,
            line_close=True,
            template="plotly_dark"
        )
        
        fig.update_traces(
            fill='tosurface',
            line_color='rgb(0, 176, 246)',
            fillcolor='rgba(0, 176, 246, 0.3)'
        )
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Personality Profile Radar"
        )
        
        return fig

    def generate_relationship_graph(self, social_data: Dict[str, Any]) -> go.Figure:
        """Generate interactive relationship network graph"""
        G = nx.Graph()
        
        # Add nodes with attributes
        for person in social_data['nodes']:
            G.add_node(
                person['id'],
                name=person['name'],
                interactions=person['interactions'],
                lon=person['location']['lon'],
                lat=person['location']['lat']
            )
        
        # Add edges with weights
        for connection in social_data['connections']:
            weight = self._calculate_connection_weight(connection)
            G.add_edge(
                connection['source'],
                connection['target'],
                weight=weight
            )
        
        # Create scattergeo plot
        edge_traces = []
        for edge in G.edges():
            source = G.nodes[edge[0]]
            target = G.nodes[edge[1]]
            
            edge_trace = go.Scattergeo(
                lon=[source['lon'], target['lon']],
                lat=[source['lat'], target['lat']],
                mode='lines',
                line=dict(
                    width=G.edges[edge]['weight'] * 2,
                    color='rgba(0, 176, 246, 0.5)'
                ),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        node_trace = go.Scattergeo(
            lon=[n['lon'] for n in G.nodes.values()],
            lat=[n['lat'] for n in G.nodes.values()],
            text=[n['name'] for n in G.nodes.values()],
            mode='markers+text',
            marker=dict(
                size=[n['interactions'] * 2 for n in G.nodes.values()],
                color='rgb(0, 176, 246)',
                line=dict(width=2, color='rgb(255, 255, 255)')
            ),
            textposition='top center'
        )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title='Relationship Network',
            showlegend=False,
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)'
            )
        )
        
        return fig

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def _calculate_connection_weight(self, connection: Dict[str, Any]) -> float:
        """Calculate weighted connection strength"""
        weight = 0
        for metric, importance in self.relationship_metrics.items():
            if metric in connection:
                weight += connection[metric] * importance
        return weight 