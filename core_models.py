"""
Core Psychological Models
Big Five and Dark Triad prediction using transformer-based architectures
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, Any, List, Optional
import torch.nn.functional as F
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self, config_path: Optional[str] = None):
        self.default_config = {
            'model_name': 'bert-base-uncased',
            'max_length': 512,
            'batch_size': 32,
            'learning_rate': 1e-5,
            'dropout_rate': 0.1,
            'hidden_size': 768,
            'intermediate_size': 256,
            'ensemble_weights': {
                'text': 0.4,
                'audio': 0.3,
                'video': 0.3
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = {**self.default_config, **json.load(f)}
        else:
            self.config = self.default_config

class CorePersonalityModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        try:
            # Use AutoModel for flexibility in model selection
            self.bert = AutoModel.from_pretrained(config.config['model_name'])
            self.tokenizer = AutoTokenizer.from_pretrained(config.config['model_name'])
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        # Dynamic token length based on model capabilities
        self.max_length = min(
            config.config['max_length'],
            self.tokenizer.model_max_length
        )
        
        # Enhanced prediction heads with layer normalization
        self.big_five_head = nn.Sequential(
            nn.LayerNorm(config.config['hidden_size']),
            nn.Linear(config.config['hidden_size'], config.config['intermediate_size']),
            nn.GELU(),
            nn.Dropout(config.config['dropout_rate']),
            nn.Linear(config.config['intermediate_size'], 5)  # OCEAN traits
        )
        
        self.dark_triad_head = nn.Sequential(
            nn.LayerNorm(config.config['hidden_size']),
            nn.Linear(config.config['hidden_size'], config.config['intermediate_size']),
            nn.GELU(),
            nn.Dropout(config.config['dropout_rate']),
            nn.Linear(config.config['intermediate_size'], 3)  # Dark Triad traits
        )
        
        # Initialize weights with better initialization
        self._init_weights()
        
        # Enable gradient checkpointing for memory efficiency
        self.bert.gradient_checkpointing_enable()

    def _init_weights(self):
        """Initialize model weights with improved initialization"""
        for module in [self.big_five_head, self.dark_triad_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(layer.bias)

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model with error handling"""
        try:
            # Get BERT embeddings with gradient checkpointing
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False  # Disable caching for memory efficiency
            )
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
            # Predict traits with error handling
            big_five_scores = self.big_five_head(pooled_output)
            dark_triad_scores = self.dark_triad_head(pooled_output)
            
            return {
                'big_five': big_five_scores,
                'dark_triad': dark_triad_scores
            }
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise

    def predict_big_five(self, text: str) -> Dict[str, float]:
        """Predict Big Five personality traits from text with improved error handling"""
        try:
            # Tokenize input with dynamic length
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding='max_length'
            )
            
            # Get predictions with memory optimization
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.forward(inputs['input_ids'], inputs['attention_mask'])
                big_five_scores = F.sigmoid(outputs['big_five']).squeeze()
            
            # Convert to dictionary with validation
            traits = ['Openness', 'Conscientiousness', 'Extraversion',
                     'Agreeableness', 'Neuroticism']
            
            results = {
                trait: float(score.item())
                for trait, score in zip(traits, big_five_scores)
            }
            
            # Validate scores
            if not all(0 <= score <= 1 for score in results.values()):
                raise ValueError("Invalid score range detected")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in Big Five prediction: {e}")
            raise

    def detect_dark_triad(self, text: str) -> Dict[str, float]:
        """Detect Dark Triad traits from text with improved error handling"""
        try:
            # Tokenize input with dynamic length
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding='max_length'
            )
            
            # Get predictions with memory optimization
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.forward(inputs['input_ids'], inputs['attention_mask'])
                dark_triad_scores = F.sigmoid(outputs['dark_triad']).squeeze()
            
            # Convert to dictionary with validation
            traits = ['Narcissism', 'Machiavellianism', 'Psychopathy']
            
            results = {
                trait: float(score.item())
                for trait, score in zip(traits, dark_triad_scores)
            }
            
            # Validate scores
            if not all(0 <= score <= 1 for score in results.values()):
                raise ValueError("Invalid score range detected")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in Dark Triad detection: {e}")
            raise

    def analyze_speech_patterns(self, audio_features: Dict[str, Any]) -> Dict[str, float]:
        """Analyze speech patterns for personality indicators"""
        # Extract relevant features
        mfccs = audio_features['mfccs']
        speech_features = audio_features['speech_features']
        
        # Calculate personality indicators
        indicators = {
            'extraversion': self._calculate_extraversion_score(speech_features),
            'neuroticism': self._calculate_neuroticism_score(speech_features),
            'psychopathy': self._calculate_psychopathy_score(mfccs)
        }
        
        return indicators

    def _calculate_extraversion_score(self, features: Dict[str, float]) -> float:
        """Calculate extraversion score from speech features"""
        # Higher speech rate and energy indicate extraversion
        speech_rate_score = min(features['speech_rate'] / 3.0, 1.0)
        energy_score = min(features['energy'] * 2, 1.0)
        
        return (speech_rate_score + energy_score) / 2

    def _calculate_neuroticism_score(self, features: Dict[str, float]) -> float:
        """Calculate neuroticism score from speech features"""
        # Higher pitch variation indicates neuroticism
        pitch_variation = min(features['pitch_std'] / 100, 1.0)
        return pitch_variation

    def _calculate_psychopathy_score(self, mfccs: np.ndarray) -> float:
        """Calculate psychopathy score from MFCC features"""
        # Psychopathy often associated with reduced emotional prosody
        mfcc_variation = np.std(mfccs, axis=1).mean()
        return 1.0 - min(mfcc_variation, 1.0)

class PersonalityEnsemble:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.weights = config.config['ensemble_weights']
        
        # Initialize models with error handling
        try:
            for modality in ['text', 'audio', 'video']:
                self.models[modality] = CorePersonalityModel(config)
        except Exception as e:
            logger.error(f"Failed to initialize ensemble models: {e}")
            raise

    def predict(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Make ensemble predictions from multi-modal data with improved error handling"""
        try:
            predictions = {}
            available_modalities = []
            
            # Get predictions from each modality with validation
            for modality, model in self.models.items():
                if modality in data and data[modality]:
                    try:
                        if modality == 'text':
                            pred = model.predict_big_five(data[modality])
                        elif modality == 'audio':
                            pred = model.analyze_speech_patterns(data[modality])
                        else:
                            pred = model.detect_dark_triad(data[modality])
                        
                        predictions[modality] = pred
                        available_modalities.append(modality)
                    except Exception as e:
                        logger.warning(f"Failed to process {modality} modality: {e}")
                        continue
            
            if not available_modalities:
                raise ValueError("No valid modalities available for prediction")
            
            # Combine predictions with dynamic weights
            combined = {}
            total_weight = sum(self.weights[m] for m in available_modalities)
            
            for trait in ['Openness', 'Conscientiousness', 'Extraversion',
                         'Agreeableness', 'Neuroticism']:
                scores = []
                for modality in available_modalities:
                    if trait in predictions[modality]:
                        scores.append(
                            predictions[modality][trait] * 
                            (self.weights[modality] / total_weight)
                        )
                
                if scores:
                    combined[trait] = sum(scores)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            raise 