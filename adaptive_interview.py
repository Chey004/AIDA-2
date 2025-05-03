"""
Adaptive Interview System
Real-time adaptation of interview questions based on response patterns
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

@dataclass
class Question:
    id: str
    text: str
    category: str
    dimension: str
    depth_level: int
    follow_up: bool

class ReinforcementLearningAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Initialize experience replay buffer
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state: np.ndarray, action: int, 
                reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def replay(self):
        """Train on experiences from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch])
        actions = torch.LongTensor([self.memory[i][1] for i in batch])
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch])
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch])
        dones = torch.FloatTensor([self.memory[i][4] for i in batch])
        
        # Compute Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AdaptiveInterviewSystem:
    def __init__(self, config: Dict[str, Any]):
        # Initialize question bank
        self.question_bank = self._initialize_question_bank()
        
        # Initialize BERT model for text analysis
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize RL agent
        self.state_size = 768 + 5  # BERT embedding + personality scores
        self.action_size = len(self.question_bank)
        self.rl_agent = ReinforcementLearningAgent(self.state_size, self.action_size)
        
        # Initialize tracking
        self.asked_questions = set()
        self.response_patterns = defaultdict(list)
        self.current_analysis = None

    def _initialize_question_bank(self) -> List[Question]:
        """Initialize question bank with categorized questions"""
        return [
            # Coping and Stress Management
            Question(
                id="q1",
                text="How do you typically handle stressful situations?",
                category="coping",
                dimension="neuroticism",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q2",
                text="Describe a time when you had to work in a team.",
                category="social",
                dimension="extraversion",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q3",
                text="How do you make important decisions?",
                category="cognitive",
                dimension="conscientiousness",
                depth_level=2,
                follow_up=True
            ),
            # Personality - Openness
            Question(
                id="q4",
                text="Describe your ideal creative project and how you would approach it.",
                category="personality",
                dimension="openness",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q5",
                text="How do you typically respond to new and unfamiliar situations?",
                category="personality",
                dimension="openness",
                depth_level=2,
                follow_up=True
            ),
            # Personality - Conscientiousness
            Question(
                id="q6",
                text="How do you organize your work and personal tasks?",
                category="personality",
                dimension="conscientiousness",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q7",
                text="Describe your approach to long-term planning and goal setting.",
                category="personality",
                dimension="conscientiousness",
                depth_level=2,
                follow_up=True
            ),
            # Personality - Extraversion
            Question(
                id="q8",
                text="How do you recharge after social interactions?",
                category="personality",
                dimension="extraversion",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q9",
                text="What role do you typically play in group settings?",
                category="personality",
                dimension="extraversion",
                depth_level=2,
                follow_up=True
            ),
            # Personality - Agreeableness
            Question(
                id="q10",
                text="How do you handle conflicts with others?",
                category="personality",
                dimension="agreeableness",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q11",
                text="Describe a time when you had to compromise in a difficult situation.",
                category="personality",
                dimension="agreeableness",
                depth_level=2,
                follow_up=True
            ),
            # Personality - Neuroticism
            Question(
                id="q12",
                text="How do you manage stress in challenging situations?",
                category="personality",
                dimension="neuroticism",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q13",
                text="What strategies do you use to maintain emotional balance?",
                category="personality",
                dimension="neuroticism",
                depth_level=2,
                follow_up=True
            ),
            # Dark Triad - Narcissism
            Question(
                id="q14",
                text="How do you react when others don't recognize your achievements?",
                category="dark_triad",
                dimension="narcissism",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q15",
                text="What makes you stand out from others?",
                category="dark_triad",
                dimension="narcissism",
                depth_level=2,
                follow_up=True
            ),
            # Dark Triad - Machiavellianism
            Question(
                id="q16",
                text="How do you approach situations where you need to influence others?",
                category="dark_triad",
                dimension="machiavellianism",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q17",
                text="What's your view on using strategic deception?",
                category="dark_triad",
                dimension="machiavellianism",
                depth_level=2,
                follow_up=True
            ),
            # Dark Triad - Psychopathy
            Question(
                id="q18",
                text="How do you handle situations where you need to make tough decisions?",
                category="dark_triad",
                dimension="psychopathy",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q19",
                text="What's your approach to risk-taking?",
                category="dark_triad",
                dimension="psychopathy",
                depth_level=2,
                follow_up=True
            ),
            # Ethical Considerations
            Question(
                id="q20",
                text="How would you handle a situation where you saw someone cheating?",
                category="ethics",
                dimension="integrity",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q21",
                text="What would you do if you discovered a colleague was taking credit for your work?",
                category="ethics",
                dimension="integrity",
                depth_level=2,
                follow_up=True
            ),
            # Emotional Intelligence
            Question(
                id="q22",
                text="How do you recognize and manage your own emotions?",
                category="emotional_intelligence",
                dimension="self_awareness",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q23",
                text="How do you handle the emotions of others in difficult situations?",
                category="emotional_intelligence",
                dimension="empathy",
                depth_level=2,
                follow_up=True
            ),
            # Problem Solving
            Question(
                id="q24",
                text="Describe a complex problem you solved and your approach.",
                category="problem_solving",
                dimension="analytical_thinking",
                depth_level=1,
                follow_up=True
            ),
            Question(
                id="q25",
                text="How do you approach problems with no clear solution?",
                category="problem_solving",
                dimension="creativity",
                depth_level=2,
                follow_up=True
            )
        ]

    def get_next_question(self, 
                         current_analysis: Dict[str, Any],
                         previous_answers: List[Dict[str, Any]]) -> Question:
        """Select next question based on current analysis and previous answers"""
        # Update current analysis
        self.current_analysis = current_analysis
        
        # Get state representation
        state = self._get_state_representation(
            current_analysis,
            previous_answers
        )
        
        # Select action using RL agent
        action = self.rl_agent.select_action(state)
        
        # Get question
        question = self.question_bank[action]
        
        # Update tracking
        self.asked_questions.add(question.id)
        
        return question

    def _get_state_representation(self,
                                current_analysis: Dict[str, Any],
                                previous_answers: List[Dict[str, Any]]) -> np.ndarray:
        """Get state representation for RL agent"""
        # Get BERT embedding of last answer
        if previous_answers:
            last_answer = previous_answers[-1]['text']
            inputs = self.tokenizer(last_answer, return_tensors='pt',
                                  truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.bert(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
        else:
            embedding = np.zeros(768)
        
        # Get personality scores
        personality_scores = np.array([
            current_analysis.get('extraversion', 0.5),
            current_analysis.get('neuroticism', 0.5),
            current_analysis.get('openness', 0.5),
            current_analysis.get('agreeableness', 0.5),
            current_analysis.get('conscientiousness', 0.5)
        ])
        
        # Concatenate features
        return np.concatenate([embedding, personality_scores])

    def update_with_answer(self, 
                         question: Question,
                         answer: Dict[str, Any]):
        """Update system with new answer"""
        # Store answer
        self.response_patterns[question.category].append(answer)
        
        # Calculate reward
        reward = self._calculate_reward(question, answer)
        
        # Get next state
        next_state = self._get_state_representation(
            self.current_analysis,
            list(self.response_patterns.values())
        )
        
        # Update RL agent
        self.rl_agent.remember(
            self._get_state_representation(
                self.current_analysis,
                list(self.response_patterns.values())[:-1]
            ),
            self.question_bank.index(question),
            reward,
            next_state,
            False
        )
        self.rl_agent.replay()

    def _calculate_reward(self, 
                        question: Question,
                        answer: Dict[str, Any]) -> float:
        """Calculate reward for question-answer pair"""
        # Information gain reward
        info_gain = self._calculate_information_gain(question, answer)
        
        # Diversity reward
        diversity = self._calculate_diversity_score(question)
        
        # Depth reward
        depth = self._calculate_depth_score(question)
        
        # Combine rewards with weights
        return (
            0.5 * info_gain +
            0.3 * diversity +
            0.2 * depth
        )

    def _calculate_information_gain(self,
                                  question: Question,
                                  answer: Dict[str, Any]) -> float:
        """Calculate information gain from answer"""
        # Get BERT embeddings
        inputs = self.tokenizer(answer['text'], return_tensors='pt',
                              truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        
        # Calculate similarity with previous answers
        if self.response_patterns[question.category]:
            prev_embeddings = torch.stack([
                self.bert(
                    self.tokenizer(a['text'], return_tensors='pt',
                                 truncation=True, max_length=512)['input_ids']
                ).last_hidden_state[:, 0, :]
                for a in self.response_patterns[question.category]
            ])
            similarity = torch.cosine_similarity(
                embedding, prev_embeddings
            ).mean().item()
        else:
            similarity = 0.0
        
        # Information gain is inverse of similarity
        return 1.0 - similarity

    def _calculate_diversity_score(self, question: Question) -> float:
        """Calculate diversity score for question"""
        # Count questions asked in same category
        category_count = sum(
            1 for q in self.asked_questions
            if self.question_bank[self.question_bank.index(
                Question(id=q, text="", category="", dimension="", depth_level=0, follow_up=False)
            )].category == question.category
        )
        
        # Diversity score is inverse of category frequency
        return 1.0 / (1.0 + category_count)

    def _calculate_depth_score(self, question: Question) -> float:
        """Calculate depth score for question"""
        # Depth score based on question's depth level
        return question.depth_level / 3.0  # Assuming max depth level is 3

    def optimize_question_order(self,
                              questions: List[Question],
                              diversity_weight: float = 0.3,
                              depth_weight: float = 0.7) -> List[Question]:
        """Optimize question order based on diversity and depth"""
        # Calculate scores for each question
        scores = []
        for question in questions:
            diversity_score = self._calculate_diversity_score(question)
            depth_score = self._calculate_depth_score(question)
            
            total_score = (
                diversity_weight * diversity_score +
                depth_weight * depth_score
            )
            scores.append((question, total_score))
        
        # Sort questions by score
        sorted_questions = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )
        
        return [q for q, _ in sorted_questions]

    def detect_contradictions(self,
                            answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect contradictions in answers"""
        contradictions = []
        
        # Group answers by category
        category_answers = defaultdict(list)
        for answer in answers:
            category_answers[answer['category']].append(answer)
        
        # Check for contradictions within categories
        for category, category_answers in category_answers.items():
            if len(category_answers) < 2:
                continue
            
            # Compare each answer with others in category
            for i, answer1 in enumerate(category_answers):
                for answer2 in category_answers[i+1:]:
                    contradiction_score = self._calculate_contradiction_score(
                        answer1, answer2
                    )
                    if contradiction_score > 0.7:
                        contradictions.append({
                            'answers': [answer1, answer2],
                            'category': category,
                            'score': contradiction_score
                        })
        
        return contradictions

    def _calculate_contradiction_score(self,
                                     answer1: Dict[str, Any],
                                     answer2: Dict[str, Any]) -> float:
        """Calculate contradiction score between two answers"""
        # Get BERT embeddings
        inputs1 = self.tokenizer(answer1['text'], return_tensors='pt',
                               truncation=True, max_length=512)
        inputs2 = self.tokenizer(answer2['text'], return_tensors='pt',
                               truncation=True, max_length=512)
        
        with torch.no_grad():
            embedding1 = self.bert(**inputs1).last_hidden_state[:, 0, :]
            embedding2 = self.bert(**inputs2).last_hidden_state[:, 0, :]
        
        # Calculate similarity
        similarity = torch.cosine_similarity(embedding1, embedding2).item()
        
        # Contradiction score is inverse of similarity
        return 1.0 - similarity

    def _analyze_answer(self, answer: Dict[str, Any]) -> Dict[str, float]:
        """Analyze answer for follow-up opportunities"""
        # Get BERT embedding
        inputs = self.tokenizer(answer['text'], return_tensors='pt',
                              truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        
        # Calculate clarity score (based on answer length and complexity)
        clarity = min(len(answer['text'].split()) / 50, 1.0)
        
        # Calculate depth score (based on semantic richness)
        depth = torch.norm(embedding).item() / 10.0
        
        # Calculate emotional tone
        emotional_tone = self._analyze_emotional_tone(answer['text'])
        
        # Calculate cognitive complexity
        cognitive_complexity = self._analyze_cognitive_complexity(answer['text'])
        
        # Calculate response consistency
        consistency = self._calculate_response_consistency(answer)
        
        return {
            'clarity': clarity,
            'depth': depth,
            'emotional_tone': emotional_tone,
            'cognitive_complexity': cognitive_complexity,
            'consistency': consistency
        }

    def _analyze_emotional_tone(self, text: str) -> Dict[str, float]:
        """Analyze emotional tone of the response"""
        # Get BERT embedding
        inputs = self.tokenizer(text, return_tensors='pt',
                              truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        
        # Define emotional tone dimensions
        emotional_dimensions = {
            'positive': ['happy', 'joy', 'excited', 'pleased', 'satisfied'],
            'negative': ['sad', 'angry', 'frustrated', 'disappointed', 'worried'],
            'neutral': ['calm', 'balanced', 'measured', 'objective', 'rational']
        }
        
        # Calculate tone scores
        tone_scores = {}
        for dimension, keywords in emotional_dimensions.items():
            # Get embeddings for keywords
            keyword_embeddings = []
            for keyword in keywords:
                keyword_inputs = self.tokenizer(keyword, return_tensors='pt')
                with torch.no_grad():
                    keyword_outputs = self.bert(**keyword_inputs)
                    keyword_embeddings.append(
                        keyword_outputs.last_hidden_state[:, 0, :]
                    )
            
            # Calculate similarity with response
            similarities = [
                torch.cosine_similarity(embedding, k_emb).item()
                for k_emb in keyword_embeddings
            ]
            tone_scores[dimension] = np.mean(similarities)
        
        return tone_scores

    def _analyze_cognitive_complexity(self, text: str) -> float:
        """Analyze cognitive complexity of the response"""
        # Split into sentences
        sentences = text.split('.')
        
        # Calculate complexity metrics
        metrics = {
            'sentence_length': np.mean([len(s.split()) for s in sentences]),
            'vocabulary_diversity': len(set(text.split())) / len(text.split()),
            'subordination_ratio': sum(1 for s in sentences if ',' in s) / len(sentences),
            'conceptual_density': len([w for w in text.split() if len(w) > 6]) / len(text.split())
        }
        
        # Combine metrics into complexity score
        weights = {
            'sentence_length': 0.3,
            'vocabulary_diversity': 0.2,
            'subordination_ratio': 0.3,
            'conceptual_density': 0.2
        }
        
        complexity = sum(
            score * weights[metric]
            for metric, score in metrics.items()
        )
        
        return min(complexity, 1.0)

    def _calculate_response_consistency(self, 
                                     answer: Dict[str, Any]) -> float:
        """Calculate consistency with previous responses"""
        if not self.response_patterns[answer['category']]:
            return 1.0  # First response in category
        
        # Get previous responses in same category
        prev_responses = self.response_patterns[answer['category']]
        
        # Calculate semantic consistency
        semantic_similarities = []
        for prev_response in prev_responses:
            # Get embeddings
            inputs1 = self.tokenizer(answer['text'], return_tensors='pt',
                                   truncation=True, max_length=512)
            inputs2 = self.tokenizer(prev_response['text'], return_tensors='pt',
                                   truncation=True, max_length=512)
            
            with torch.no_grad():
                embedding1 = self.bert(**inputs1).last_hidden_state[:, 0, :]
                embedding2 = self.bert(**inputs2).last_hidden_state[:, 0, :]
            
            # Calculate similarity
            similarity = torch.cosine_similarity(embedding1, embedding2).item()
            semantic_similarities.append(similarity)
        
        # Calculate emotional consistency
        emotional_similarities = []
        current_tone = self._analyze_emotional_tone(answer['text'])
        for prev_response in prev_responses:
            prev_tone = self._analyze_emotional_tone(prev_response['text'])
            tone_similarity = np.mean([
                abs(current_tone[dim] - prev_tone[dim])
                for dim in current_tone.keys()
            ])
            emotional_similarities.append(1.0 - tone_similarity)
        
        # Combine consistency scores
        semantic_consistency = np.mean(semantic_similarities)
        emotional_consistency = np.mean(emotional_similarities)
        
        return 0.7 * semantic_consistency + 0.3 * emotional_consistency

    def generate_follow_up_questions(self,
                                   question: Question,
                                   answer: Dict[str, Any]) -> List[Question]:
        """Generate follow-up questions based on answer analysis"""
        follow_ups = []
        
        # Check if follow-up is needed
        if not question.follow_up:
            return follow_ups
        
        # Analyze answer
        analysis = self._analyze_answer(answer)
        
        # Generate follow-ups based on analysis
        if analysis['clarity'] < 0.5:
            follow_ups.append(
                Question(
                    id=f"f_{question.id}_clarify",
                    text="Could you elaborate on that?",
                    category=question.category,
                    dimension=question.dimension,
                    depth_level=question.depth_level + 1,
                    follow_up=False
                )
            )
        
        if analysis['depth'] < 0.5:
            follow_ups.append(
                Question(
                    id=f"f_{question.id}_depth",
                    text="What led you to that perspective?",
                    category=question.category,
                    dimension=question.dimension,
                    depth_level=question.depth_level + 1,
                    follow_up=False
                )
            )
        
        # Add emotional tone follow-ups
        if analysis['emotional_tone']['negative'] > 0.7:
            follow_ups.append(
                Question(
                    id=f"f_{question.id}_emotion",
                    text="How do you typically cope with these feelings?",
                    category="emotional_intelligence",
                    dimension="coping",
                    depth_level=question.depth_level + 1,
                    follow_up=False
                )
            )
        
        # Add cognitive complexity follow-ups
        if analysis['cognitive_complexity'] < 0.3:
            follow_ups.append(
                Question(
                    id=f"f_{question.id}_complexity",
                    text="Could you provide more specific details about that?",
                    category=question.category,
                    dimension=question.dimension,
                    depth_level=question.depth_level + 1,
                    follow_up=False
                )
            )
        
        return follow_ups

"""
Example Usage Documentation

The AdaptiveInterviewSystem provides a sophisticated framework for conducting
adaptive psychological interviews. Here's how to use it:

1. Initialization:
```python
from adaptive_interview import AdaptiveInterviewSystem

# Initialize the system with configuration
config = {
    'culture': 'western',  # Options: 'western', 'eastern', 'middle_eastern'
    'interview_type': 'personality',  # Options: 'personality', 'clinical', 'research'
    'max_questions': 20,
    'min_confidence': 0.7
}

interview_system = AdaptiveInterviewSystem(config)
```

2. Conducting an Interview:
```python
# Initialize interview state
current_analysis = {
    'extraversion': 0.5,
    'neuroticism': 0.5,
    'openness': 0.5,
    'agreeableness': 0.5,
    'conscientiousness': 0.5
}
previous_answers = []

# Get first question
question = interview_system.get_next_question(current_analysis, previous_answers)
print(f"Question: {question.text}")

# Process answer
answer = {
    'text': "I typically handle stress by taking a step back and analyzing the situation...",
    'category': question.category,
    'dimension': question.dimension
}

# Update system with answer
interview_system.update_with_answer(question, answer)
previous_answers.append(answer)

# Generate follow-up questions if needed
follow_ups = interview_system.generate_follow_up_questions(question, answer)
for follow_up in follow_ups:
    print(f"Follow-up: {follow_up.text}")

# Get next question
next_question = interview_system.get_next_question(current_analysis, previous_answers)
```

3. Analyzing Responses:
```python
# Detect contradictions in answers
contradictions = interview_system.detect_contradictions(previous_answers)
for contradiction in contradictions:
    print(f"Contradiction detected in {contradiction['category']} with score {contradiction['score']}")

# Get detailed analysis of an answer
analysis = interview_system._analyze_answer(answer)
print(f"Clarity: {analysis['clarity']}")
print(f"Depth: {analysis['depth']}")
print(f"Emotional Tone: {analysis['emotional_tone']}")
print(f"Cognitive Complexity: {analysis['cognitive_complexity']}")
print(f"Consistency: {analysis['consistency']}")
```

4. Optimizing Question Order:
```python
# Get optimized question order
questions = interview_system.question_bank
optimized_order = interview_system.optimize_question_order(
    questions,
    diversity_weight=0.3,
    depth_weight=0.7
)

# Print optimized order
for i, question in enumerate(optimized_order):
    print(f"{i+1}. {question.text} (Category: {question.category}, Depth: {question.depth_level})")
```

5. Advanced Features:
```python
# Analyze emotional tone
tone_analysis = interview_system._analyze_emotional_tone(answer['text'])
print(f"Positive: {tone_analysis['positive']}")
print(f"Negative: {tone_analysis['negative']}")
print(f"Neutral: {tone_analysis['neutral']}")

# Analyze cognitive complexity
complexity = interview_system._analyze_cognitive_complexity(answer['text'])
print(f"Cognitive Complexity Score: {complexity}")

# Calculate response consistency
consistency = interview_system._calculate_response_consistency(answer)
print(f"Response Consistency: {consistency}")
```

Dependencies:
- torch
- transformers
- numpy
- dataclasses
- typing
- collections

The system uses BERT for semantic analysis and implements reinforcement learning
for adaptive question selection. It provides comprehensive analysis of responses
including emotional tone, cognitive complexity, and response consistency.
""" 