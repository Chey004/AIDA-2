import spacy
import numpy as np
from textblob import TextBlob
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Load NLP model
nlp = spacy.load("en_core_web_md")

class DeepPersonalityAnalyzer:
    def __init__(self):
        # Existing frameworks
        self.big_five = {
            'Openness': {
                'keywords': {
                    'curious': 1.5, 'imaginative': 1.5, 'creative': 1.5,
                    'new ideas': 2.0, 'explore': 1.5, 'innovative': 2.0
                },
                'score': 0
            },
            'Conscientiousness': {
                'keywords': {
                    'organized': 1.5, 'responsible': 1.5, 'plan': 1.5,
                    'detail': 1.5, 'systematic': 2.0, 'efficient': 1.5
                },
                'score': 0
            },
            'Extraversion': {
                'keywords': {
                    'social': 1.5, 'energetic': 1.5, 'people': 1.5,
                    'outgoing': 2.0, 'team': 1.5, 'communicate': 1.5
                },
                'score': 0
            },
            'Agreeableness': {
                'keywords': {
                    'trusting': 1.5, 'helpful': 1.5, 'cooperative': 1.5,
                    'support': 1.5, 'collaborate': 2.0, 'empathy': 2.0
                },
                'score': 0
            },
            'Neuroticism': {
                'keywords': {
                    'anxious': 1.5, 'tense': 1.5, 'worry': 1.5,
                    'stress': 1.5, 'nervous': 2.0, 'overwhelm': 2.0
                },
                'score': 0
            }
        }
        
        # Enhanced DISC with temporal and sentiment analysis
        self.disc_scores = {
            'Dominance': {'score': 0, 'temporal_weight': 2.0, 'sentiment_weight': 0.5},
            'Influence': {'score': 0, 'temporal_weight': 0.5, 'sentiment_weight': 2.0},
            'Steadiness': {'score': 0, 'temporal_weight': 0.5, 'sentiment_weight': 1.0},
            'Compliance': {'score': 0, 'temporal_weight': 1.5, 'sentiment_weight': 0.5}
        }
        
        # Enhanced personality types with weighted patterns
        self.personality_type = {
            'Type_A': {
                'keywords': {
                    'challenge': 1.5, 'deadline': 1.5, 'compete': 1.5,
                    'achieve': 2.0, 'goal': 1.5, 'success': 1.5
                },
                'score': 0
            },
            'Type_B': {
                'keywords': {
                    'creative': 1.5, 'flexible': 1.5, 'relaxed': 1.5,
                    'adapt': 2.0, 'balance': 1.5, 'flow': 1.5
                },
                'score': 0
            },
            'Type_C': {
                'keywords': {
                    'detail': 1.5, 'accurate': 1.5, 'systematic': 1.5,
                    'precise': 2.0, 'methodical': 1.5, 'thorough': 1.5
                },
                'score': 0
            },
            'Type_D': {
                'keywords': {
                    'worry': 1.5, 'stress': 1.5, 'anxious': 1.5,
                    'concern': 2.0, 'apprehensive': 1.5, 'doubt': 1.5
                },
                'score': 0
            }
        }
        
        # Shadow analysis system
        self.shadow_traits = {
            'Suppressed Anger': {
                'indicators': {
                    'should': 1.5, 'must': 1.5, 'always': 1.5, 'never': 1.5,
                    'have to': 2.0, 'need to': 2.0, 'obligated': 2.0
                },
                'score': 0
            },
            'Hidden Insecurity': {
                'indicators': {
                    'maybe': 1.5, 'perhaps': 1.5, 'not sure': 2.0, 'might': 1.5,
                    'possibly': 1.5, 'uncertain': 2.0, 'doubt': 2.0
                },
                'score': 0
            },
            'Perfectionism Mask': {
                'indicators': {
                    'perfect': 2.0, 'flawless': 2.0, 'never wrong': 2.0,
                    'best': 1.5, 'excellent': 1.5, 'ideal': 1.5
                },
                'score': 0
            },
            'Emotional Avoidance': {
                'indicators': {
                    'fine': 1.5, 'okay': 1.5, 'whatever': 2.0, 'I guess': 2.0,
                    'no problem': 1.5, 'it\'s nothing': 2.0
                },
                'score': 0
            }
        }

        # Defense mechanism detection
        self.defense_mechanisms = {
            'Projection': {
                'pattern': {
                    'they': 1.5, 'others': 1.5, 'people': 1.5,
                    'everyone': 2.0, 'some people': 2.0
                },
                'score': 0
            },
            'Rationalization': {
                'pattern': {
                    'because': 1.5, 'therefore': 1.5, 'thus': 1.5,
                    'reason': 1.5, 'explanation': 2.0
                },
                'score': 0
            },
            'Displacement': {
                'pattern': {
                    'but': 1.5, 'however': 1.5, 'although': 1.5,
                    'yet': 1.5, 'still': 1.5
                },
                'score': 0
            }
        }

        # Initialize counters
        self.pronouns = defaultdict(int)
        self.qualifiers = 0
        self.contradictions = 0
        
        # Response time analysis parameters
        self.temporal_thresholds = {
            'fast': 2.0,  # seconds
            'medium': 4.0,  # seconds
            'slow': 6.0  # seconds
        }
        
        # Sentiment analysis parameters
        self.sentiment_thresholds = {
            'positive': 0.3,
            'negative': -0.3,
            'strong_positive': 0.6,
            'strong_negative': -0.6
        }

    def analyze_response(self, question, response, response_time):
        """Analyze interview responses using enhanced frameworks"""
        doc = nlp(response.lower())
        
        # Traditional analysis
        self._analyze_big_five(doc)
        self._analyze_disc(question, response, response_time)
        self._analyze_personality_type(doc)
        
        # Deep psychological analysis
        self._detect_shadow_traits(doc)
        self._identify_defense_mechanisms(doc)
        self._analyze_linguistic_patterns(doc)

    def _analyze_big_five(self, doc):
        for trait in self.big_five:
            score = 0
            for keyword, weight in self.big_five[trait]['keywords'].items():
                if ' ' in keyword:
                    if keyword in doc.text:
                        score += weight
                else:
                    matches = [token.text for token in doc if token.text == keyword]
                    score += len(matches) * weight
            self.big_five[trait]['score'] += score

    def _analyze_disc(self, question, response, response_time):
        temporal_score = self._calculate_temporal_score(response_time)
        sentiment_score = self._calculate_sentiment_score(response)
        
        for trait in self.disc_scores:
            self.disc_scores[trait]['score'] += (
                temporal_score * self.disc_scores[trait]['temporal_weight'] +
                sentiment_score * self.disc_scores[trait]['sentiment_weight']
            )

    def _calculate_temporal_score(self, response_time):
        if response_time < self.temporal_thresholds['fast']:
            return 1.0
        elif response_time < self.temporal_thresholds['medium']:
            return 0.5
        elif response_time < self.temporal_thresholds['slow']:
            return -0.5
        else:
            return -1.0

    def _calculate_sentiment_score(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > self.sentiment_thresholds['strong_positive']:
            return 1.0
        elif polarity > self.sentiment_thresholds['positive']:
            return 0.5
        elif polarity < self.sentiment_thresholds['strong_negative']:
            return -1.0
        elif polarity < self.sentiment_thresholds['negative']:
            return -0.5
        else:
            return 0.0

    def _analyze_personality_type(self, doc):
        for p_type in self.personality_type:
            score = 0
            for keyword, weight in self.personality_type[p_type]['keywords'].items():
                if ' ' in keyword:
                    if keyword in doc.text:
                        score += weight
                else:
                    matches = [token.text for token in doc if token.text == keyword]
                    score += len(matches) * weight
            self.personality_type[p_type]['score'] += score

    def _detect_shadow_traits(self, doc):
        for trait, data in self.shadow_traits.items():
            score = 0
            for indicator, weight in data['indicators'].items():
                if ' ' in indicator:
                    if indicator in doc.text:
                        score += weight
                else:
                    matches = [token.text for token in doc if token.text == indicator]
                    score += len(matches) * weight
            self.shadow_traits[trait]['score'] += score

    def _identify_defense_mechanisms(self, doc):
        for mechanism, data in self.defense_mechanisms.items():
            score = 0
            for pattern, weight in data['pattern'].items():
                if ' ' in pattern:
                    if pattern in doc.text:
                        score += weight
                else:
                    matches = [token.text for token in doc if token.text == pattern]
                    score += len(matches) * weight
            self.defense_mechanisms[mechanism]['score'] += score

    def _analyze_linguistic_patterns(self, doc):
        # Pronoun analysis
        for token in doc:
            if token.pos_ == 'PRON':
                self.pronouns[token.text.lower()] += 1
                
        # Qualifier detection
        qualifier_words = ['maybe', 'perhaps', 'sort of', 'kind of', 'probably', 'possibly']
        self.qualifiers = len([token for token in doc if token.text in qualifier_words])
        
        # Contradiction detection
        self.contradictions = len(list(doc.sents)) - len([sent for sent in doc.sents 
                                if 'but' not in sent.text and 'however' not in sent.text])

    def generate_insights(self):
        strengths = []
        weaknesses = []
        hidden_aspects = []
        behavioral_patterns = {}

        # Big Five insights
        if self.big_five['Conscientiousness']['score'] > 8:
            strengths.append("Strong organizational skills and reliability")
            weaknesses.append("Potential perfectionism and difficulty adapting to chaos")
        if self.big_five['Neuroticism']['score'] > 6:
            hidden_aspects.append("Possible hidden anxiety masked by emotional control")

        # Shadow trait insights
        if self.shadow_traits['Suppressed Anger']['score'] > 3:
            hidden_aspects.append("Unexpressed frustration manifesting as passive-aggressive tendencies")
        if self.shadow_traits['Emotional Avoidance']['score'] > 2:
            weaknesses.append("Difficulty engaging with deep emotional experiences")

        # Defense mechanism insights
        if self.defense_mechanisms['Projection']['score'] > 1:
            hidden_aspects.append("Tendency to project personal feelings onto others")
        if self.qualifiers > 4:
            weaknesses.append("Overuse of qualifiers suggesting self-doubt")

        # Behavioral patterns
        behavioral_patterns = {
            'Self-Reference Ratio': self.pronouns['i']/(self.pronouns['they']+1),
            'Certainty Index': 1 - (self.qualifiers/len(list(doc.sents))),
            'Emotional Expression': sum(self.shadow_traits[trait]['score'] for trait in self.shadow_traits),
            'Defense Mechanism Score': sum(self.defense_mechanisms[mechanism]['score'] for mechanism in self.defense_mechanisms)
        }

        return {
            'Strengths': strengths,
            'Growth Areas': weaknesses,
            'Hidden Aspects': hidden_aspects,
            'Behavioral Patterns': behavioral_patterns
        }

    def visualize_results(self):
        # Create a 2x2 grid of visualizations
        plt.figure(figsize=(15, 12))
        
        # Big Five bar chart
        plt.subplot(2, 2, 1)
        traits = list(self.big_five.keys())
        scores = [v['score'] for v in self.big_five.values()]
        plt.barh(traits, scores)
        plt.title("Big Five Personality Traits")
        plt.xlabel("Score")
        
        # DISC pie chart with normalized values
        plt.subplot(2, 2, 2)
        disc_labels = list(self.disc_scores.keys())
        disc_values = [v['score'] for v in self.disc_scores.values()]
        
        # Normalize DISC scores to be positive
        min_score = min(disc_values)
        if min_score < 0:
            disc_values = [score - min_score + 1 for score in disc_values]  # Shift all values to be positive
        total = sum(disc_values)
        if total > 0:
            disc_values = [score/total for score in disc_values]  # Normalize to sum to 1
        else:
            disc_values = [1/len(disc_values) for _ in disc_values]  # Equal distribution if all zeros
        
        plt.pie(disc_values, labels=disc_labels, autopct='%1.1f%%')
        plt.title("DISC Profile")
        
        # Shadow Traits radar chart
        plt.subplot(2, 2, 3, polar=True)
        shadow_traits = list(self.shadow_traits.keys())
        shadow_scores = [v['score'] for v in self.shadow_traits.values()]
        
        # Normalize shadow scores to be positive
        min_shadow = min(shadow_scores)
        if min_shadow < 0:
            shadow_scores = [score - min_shadow + 1 for score in shadow_scores]
        
        angles = np.linspace(0, 2*np.pi, len(shadow_traits), endpoint=False)
        shadow_scores = np.concatenate((shadow_scores, [shadow_scores[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        plt.polar(angles, shadow_scores)
        plt.fill(angles, shadow_scores, alpha=0.25)
        plt.title("Shadow Traits Profile")
        plt.xticks(angles[:-1], shadow_traits)
        
        # Defense Mechanisms bar chart
        plt.subplot(2, 2, 4)
        mechanisms = list(self.defense_mechanisms.keys())
        mechanism_scores = [v['score'] for v in self.defense_mechanisms.values()]
        
        # Normalize defense mechanism scores to be positive
        min_mech = min(mechanism_scores)
        if min_mech < 0:
            mechanism_scores = [score - min_mech + 1 for score in mechanism_scores]
        
        plt.bar(mechanisms, mechanism_scores)
        plt.title("Defense Mechanisms")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

# Interview Simulation
questions = [
    "Describe how you handle tight deadlines?",
    "How do you react to unexpected challenges?",
    "What's your approach to teamwork?"
]

analyzer = DeepPersonalityAnalyzer()

for question in questions:
    start_time = time.time()
    response = input(f"\nInterview Question: {question}\nYour Answer: ")
    response_time = time.time() - start_time
    
    analyzer.analyze_response(question, response, response_time)

# Display analysis
print("\n=== Deep Personality Analysis ===")
analyzer.visualize_results()

# Generate and display insights
insights = analyzer.generate_insights()
print("\nPsychological Insights:")
for category, items in insights.items():
    if isinstance(items, list):
        print(f"\n{category}:")
        for item in items:
            print(f"- {item}")
    else:
        print(f"\n{category}:")
        for k, v in items.items():
            print(f"{k}: {v:.2f}")

# Determine primary personality type
primary_type = max(analyzer.personality_type, key=lambda x: analyzer.personality_type[x]['score'])
print(f"\nPrimary Personality Type: {primary_type.replace('_', ' ')}") 