"""
COMPREHENSIVE PSYCHOLOGICAL ANALYSIS SYSTEM (CPAS)
Version 1.2 - Integrates personality, cognitive, and relational analysis

Dependencies:
- spacy: NLP processing
- textblob: Sentiment analysis
- matplotlib: Visualization
- numpy: Numerical computations

Usage:
1. Install dependencies:
   pip install spacy textblob matplotlib numpy
   python -m spacy download en_core_web_md

2. Basic usage:
   analyzer = PersonalityAnalyzer(cultural_context="western")
   results = analyzer.analyze_text("Sample text")
   analyzer.visualize_results()

3. Ethical considerations:
   - Always obtain informed consent
   - Provide clear results explanation
   - Store data securely
   - Use as supplement, not diagnosis
"""

import spacy
from textblob import TextBlob
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import networkx as nx

# Load NLP model
nlp = spacy.load("en_core_web_md")

# Cultural adjustment coefficients
CULTURAL_WEIGHTS = {
    'western': {
        'IndividualPronouns': 1.0,
        'GroupPronouns': 1.0,
        'Directness': 1.0
    },
    'east_asian': {
        'IndividualPronouns': 0.7,
        'GroupPronouns': 1.3,
        'Directness': 0.8
    },
    'middle_eastern': {
        'IndividualPronouns': 0.8,
        'GroupPronouns': 1.2,
        'Directness': 0.9
    }
}

class PersonalityAnalyzer:
    def __init__(self, cultural_context='western'):
        """Initialize the psychological analyzer with cultural context"""
        # Core personality frameworks
        self.big_five = {
            'Openness': {'score': 0, 'keywords': ['creative', 'curious', 'imaginative']},
            'Conscientiousness': {'score': 0, 'keywords': ['organized', 'responsible', 'diligent']},
            'Extraversion': {'score': 0, 'keywords': ['outgoing', 'energetic', 'sociable']},
            'Agreeableness': {'score': 0, 'keywords': ['compassionate', 'cooperative', 'trusting']},
            'Neuroticism': {'score': 0, 'keywords': ['anxious', 'tense', 'moody']}
        }
        
        # Defense mechanism detection
        self.defense_mechanisms = {
            'projection': {'score': 0, 'keywords': ['they', 'them', 'their']},
            'rationalization': {'score': 0, 'keywords': ['but', 'however', 'because']},
            'displacement': {'score': 0, 'keywords': ['angry', 'frustrated', 'annoyed']},
            'denial': {'score': 0, 'keywords': ['not', 'never', 'no']}
        }
        
        # Attachment style analysis
        self.attachment_profiles = {
            'secure': {'score': 0, 'keywords': ['we', 'our', 'together', 'support']},
            'anxious': {'score': 0, 'keywords': ['worry', 'if', 'maybe', 'hope']},
            'avoidant': {'score': 0, 'keywords': ['I', 'me', 'alone', 'independent']}
        }
        
        # Leadership assessment
        self.leadership_dimensions = {
            'decisiveness': {'score': 0, 'keywords': ['decide', 'choose', 'commit']},
            'empathy': {'score': 0, 'keywords': ['understand', 'feel', 'care']},
            'vision': {'score': 0, 'keywords': ['future', 'goal', 'plan']}
        }
        
        # Cognitive distortion detection
        self.cognitive_distortions = {
            'catastrophizing': {'score': 0, 'keywords': ['worst', 'disaster', 'ruined']},
            'black_white': {'score': 0, 'keywords': ['always', 'never', 'everyone']},
            'mind_reading': {'score': 0, 'keywords': ['they think', 'they feel', 'probably']},
            'overgeneralization': {'score': 0, 'keywords': ['every time', 'all the time', 'never']}
        }
        
        # Initialize cultural context
        self.cultural_context = cultural_context
        self._apply_cultural_adjustment()
        
        # Ethical considerations
        self.ethical_guidelines = {
            'consent_obtained': False,
            'data_storage': 'secure',
            'purpose': 'self-reflection'
        }

    def _apply_cultural_adjustment(self):
        """Apply cultural context adjustments to scoring thresholds"""
        self.cultural_weights = {
            'western': {
                'individualism': 1.2,
                'certainty': 0.9,
                'directness': 1.1
            },
            'eastern': {
                'collectivism': 1.3,
                'modesty': 1.1,
                'harmony': 1.2
            },
            'middle_eastern': {
                'collectivism': 1.1,
                'respect': 1.2,
                'indirectness': 1.1
            }
        }
        
        if self.cultural_context in self.cultural_weights:
            weights = self.cultural_weights[self.cultural_context]
            
            # Adjust attachment patterns
            if 'collectivism' in weights:
                self.attachment_profiles['secure']['score'] *= weights['collectivism']
            if 'individualism' in weights:
                self.attachment_profiles['avoidant']['score'] *= weights['individualism']
            
            # Adjust leadership indicators
            if 'harmony' in weights:
                self.leadership_dimensions['empathy']['score'] *= weights['harmony']
            if 'directness' in weights:
                self.leadership_dimensions['decisiveness']['score'] *= weights['directness']

    def analyze_text(self, text):
        """Main analysis entry point"""
        doc = nlp(text)
        
        # Core analysis
        self._big_five_analysis(doc)
        self._defense_mechanism_detection(doc)
        self._attachment_style_evaluation(doc)
        self._leadership_assessment(doc)
        self._cognitive_distortion_check(doc)
        
        # Generate comprehensive report
        return self.generate_report()

    def _big_five_analysis(self, doc):
        """Analyze Big Five personality traits"""
        for trait, data in self.big_five.items():
            score = 0
            for keyword in data['keywords']:
                score += len([token for token in doc if token.text.lower() == keyword.lower()])
            self.big_five[trait]['score'] = score

    def _defense_mechanism_detection(self, doc):
        """Detect defense mechanisms in text"""
        for mechanism, data in self.defense_mechanisms.items():
            score = 0
            for keyword in data['keywords']:
                score += len([token for token in doc if token.text.lower() == keyword.lower()])
            self.defense_mechanisms[mechanism]['score'] = score

    def _attachment_style_evaluation(self, doc):
        """Evaluate attachment style patterns"""
        for style, data in self.attachment_profiles.items():
            score = 0
            for keyword in data['keywords']:
                score += len([token for token in doc if token.text.lower() == keyword.lower()])
            self.attachment_profiles[style]['score'] = score

    def _cognitive_distortion_check(self, doc):
        """Check for cognitive distortions"""
        for distortion, data in self.cognitive_distortions.items():
            score = 0
            for keyword in data['keywords']:
                if ' ' in keyword:
                    score += text.lower().count(keyword.lower())
                else:
                    score += len([token for token in doc if token.text.lower() == keyword.lower()])
            self.cognitive_distortions[distortion]['score'] = score

    def _leadership_assessment(self, doc):
        """Assess leadership dimensions"""
        for dimension, data in self.leadership_dimensions.items():
            score = 0
            for keyword in data['keywords']:
                score += len([token for token in doc if token.text.lower() == keyword.lower()])
            self.leadership_dimensions[dimension]['score'] = score

    def generate_report(self):
        """Generate comprehensive analysis report"""
        return {
            'personality': self._interpret_big_five(),
            'relationships': self._relationship_analysis(),
            'cognition': self._cognitive_evaluation(),
            'growth': self._development_recommendations()
        }

    def _interpret_big_five(self):
        """Interpret Big Five personality scores"""
        interpretation = {}
        for trait, data in self.big_five.items():
            score = data['score']
            if score > 7:
                interpretation[trait] = "High"
            elif score > 4:
                interpretation[trait] = "Moderate"
            else:
                interpretation[trait] = "Low"
        return interpretation

    def _relationship_analysis(self):
        """Analyze relationship patterns"""
        primary_attachment = max(self.attachment_profiles.items(), key=lambda x: x[1]['score'])[0]
        return {
            'primary_attachment': primary_attachment,
            'defense_mechanisms': {k: v['score'] for k, v in self.defense_mechanisms.items()},
            'recommendations': self._generate_relationship_recommendations()
        }

    def _cognitive_evaluation(self):
        """Evaluate cognitive patterns"""
        distortions = {k: v['score'] for k, v in self.cognitive_distortions.items()}
        primary_distortion = max(distortions.items(), key=lambda x: x[1])[0]
        return {
            'primary_distortion': primary_distortion,
            'distortion_scores': distortions,
            'recommendations': self._generate_cognitive_recommendations()
        }

    def _development_recommendations(self):
        """Generate personalized development recommendations"""
        recommendations = []
        
        # Personality-based recommendations
        if self.big_five['Neuroticism']['score'] > 7:
            recommendations.append("Practice stress management techniques")
        if self.big_five['Extraversion']['score'] < 4:
            recommendations.append("Consider social skills development")
        
        # Relationship-based recommendations
        if self.attachment_profiles['anxious']['score'] > 5:
            recommendations.append("Work on emotional regulation strategies")
        if self.defense_mechanisms['projection']['score'] > 3:
            recommendations.append("Practice self-reflection and ownership")
        
        # Cognitive-based recommendations
        if self.cognitive_distortions['black_white']['score'] > 3:
            recommendations.append("Practice cognitive flexibility exercises")
        if self.cognitive_distortions['catastrophizing']['score'] > 3:
            recommendations.append("Learn to challenge catastrophic thinking")
        
        return recommendations

    def visualize_results(self):
        """Generate comprehensive visualizations"""
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(20, 15))
        
        # Big Five Radar Chart
        ax1 = plt.subplot(2, 2, 1, polar=True)
        traits = list(self.big_five.keys())
        scores = [v['score'] for v in self.big_five.values()]
        angles = np.linspace(0, 2*np.pi, len(traits), endpoint=False)
        scores = np.concatenate((scores, [scores[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        ax1.plot(angles, scores, marker='o')
        ax1.fill(angles, scores, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(traits)
        ax1.set_title("Big Five Personality Profile")
        
        # Attachment Style Pie Chart
        ax2 = plt.subplot(2, 2, 2)
        styles = list(self.attachment_profiles.keys())
        scores = [v['score'] for v in self.attachment_profiles.values()]
        ax2.pie(scores, labels=styles, autopct='%1.1f%%')
        ax2.set_title("Attachment Style Distribution")
        
        # Defense Mechanisms Bar Chart
        ax3 = plt.subplot(2, 2, 3)
        mechanisms = list(self.defense_mechanisms.keys())
        scores = [v['score'] for v in self.defense_mechanisms.values()]
        ax3.barh(mechanisms, scores)
        ax3.set_title("Defense Mechanisms")
        
        # Cognitive Distortions Bar Chart
        ax4 = plt.subplot(2, 2, 4)
        distortions = list(self.cognitive_distortions.keys())
        scores = [v['score'] for v in self.cognitive_distortions.values()]
        ax4.barh(distortions, scores)
        ax4.set_title("Cognitive Distortions")
        
        plt.tight_layout()
        plt.show()

    def plot_relationship_network(self):
        """Generate relationship network visualization"""
        G = nx.Graph()
        
        # Add nodes for different aspects
        G.add_node("Personality")
        G.add_node("Relationships")
        G.add_node("Cognition")
        G.add_node("Leadership")
        
        # Add edges based on correlations
        G.add_edge("Personality", "Relationships", weight=self.big_five['Agreeableness']['score'])
        G.add_edge("Personality", "Cognition", weight=self.big_five['Neuroticism']['score'])
        G.add_edge("Relationships", "Leadership", weight=self.leadership_dimensions['empathy']['score'])
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=12, font_weight='bold')
        plt.title("Psychological Profile Network")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = PersonalityAnalyzer(cultural_context="western")
    
    # Get user consent
    print("\n=== Ethical Considerations ===")
    print("This analysis is for self-reflection and development purposes only.")
    print("Results should not be used for diagnostic or hiring decisions.")
    consent = input("\nDo you consent to participate in this analysis? (yes/no): ").lower()
    
    if consent == 'yes':
        # Sample analysis
        text = input("\nPlease provide your response for analysis: ")
        results = analyzer.analyze_text(text)
        
        # Display results
        print("\n=== Analysis Results ===")
        for category, data in results.items():
            print(f"\n{category.upper()}:")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"  {k}: {v}")
            else:
                for item in data:
                    print(f"  - {item}")
        
        # Generate visualizations
        analyzer.visualize_results()
        analyzer.plot_relationship_network()
    else:
        print("Analysis cancelled. Thank you for your time.") 