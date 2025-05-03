"""
Data Ingestion Pipeline
Multi-modal data processing and feature extraction
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import librosa
import cv2
import mediapipe as mp
from spacy.lang.en import English

class DataIngestor:
    def __init__(self):
        # Initialize text processing
        self.nlp = English()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Initialize audio processing
        self.audio_params = {
            'sample_rate': 16000,
            'n_mfcc': 13,
            'hop_length': 512
        }
        
        # Initialize video processing
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Feature extraction pipelines
        self.processors = {
            'text': self._process_text,
            'audio': self._process_audio,
            'video': self._process_video
        }

    def ingest(self, data: Union[str, np.ndarray, bytes], 
              data_type: str = 'text') -> Dict[str, Any]:
        """Ingest and process multi-modal data"""
        if data_type not in self.processors:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        return self.processors[data_type](data)

    def _process_text(self, text: str) -> Dict[str, Any]:
        """Process text data and extract features"""
        # Tokenize and get BERT embeddings
        tokens = self.tokenizer(text, return_tensors='pt', 
                              truncation=True, max_length=512)
        embeddings = self.bert_model(**tokens).last_hidden_state
        
        # Extract linguistic features
        doc = self.nlp(text)
        linguistic_features = {
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents)),
            'avg_word_length': np.mean([len(word) for word in doc]),
            'readability_score': self._calculate_readability(text)
        }
        
        return {
            'embeddings': embeddings.detach().numpy(),
            'linguistic_features': linguistic_features,
            'tokens': tokens
        }

    def _process_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process audio data and extract features"""
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=self.audio_params['sample_rate'],
            n_mfcc=self.audio_params['n_mfcc'],
            hop_length=self.audio_params['hop_length']
        )
        
        # Extract prosodic features
        pitch = librosa.yin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        
        # Calculate speech features
        speech_features = {
            'pitch_mean': np.mean(pitch[~np.isnan(pitch)]),
            'pitch_std': np.std(pitch[~np.isnan(pitch)]),
            'energy': np.mean(np.abs(audio_data)),
            'speech_rate': len(mfccs[0]) / (len(audio_data) / 
                                          self.audio_params['sample_rate'])
        }
        
        return {
            'mfccs': mfccs,
            'speech_features': speech_features,
            'pitch_contour': pitch
        }

    def _process_video(self, video_data: np.ndarray) -> Dict[str, Any]:
        """Process video data and extract features"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        # Extract facial features
        facial_features = {
            'landmarks': [],
            'emotion_scores': self._detect_emotions(results),
            'gaze_direction': self._estimate_gaze(results)
        }
        
        if results.multi_face_landmarks:
            facial_features['landmarks'] = [
                (lm.x, lm.y, lm.z) 
                for lm in results.multi_face_landmarks[0].landmark
            ]
        
        return facial_features

    def _calculate_readability(self, text: str) -> float:
        """Calculate text readability score"""
        # Simplified Flesch-Kincaid score
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.0
            
        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = sum(
            self._count_syllables(word) for word in words
        ) / len(words)
        
        return 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        vowels = 'aeiouy'
        word = word.lower()
        count = 0
        
        if word[0] in vowels:
            count += 1
            
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
                
        if word.endswith('e'):
            count -= 1
            
        return max(1, count)

    def _detect_emotions(self, results) -> Dict[str, float]:
        """Detect emotions from facial landmarks"""
        if not results.multi_face_landmarks:
            return {
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'surprised': 0.0,
                'neutral': 1.0
            }
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Simplified emotion detection
        mouth_open = landmarks[13].y - landmarks[14].y
        eyebrow_raise = landmarks[105].y - landmarks[107].y
        
        if mouth_open > 0.1 and eyebrow_raise > 0.05:
            return {'surprised': 1.0}
        elif mouth_open > 0.05:
            return {'happy': 1.0}
        elif eyebrow_raise < -0.05:
            return {'angry': 1.0}
        elif mouth_open < -0.05:
            return {'sad': 1.0}
        else:
            return {'neutral': 1.0}

    def _estimate_gaze(self, results) -> Dict[str, float]:
        """Estimate gaze direction"""
        if not results.multi_face_landmarks:
            return {'x': 0.0, 'y': 0.0}
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Simplified gaze estimation using eye landmarks
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        return {
            'x': (left_eye.x + right_eye.x) / 2,
            'y': (left_eye.y + right_eye.y) / 2
        } 