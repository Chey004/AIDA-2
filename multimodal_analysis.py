"""
Multi-Modal Analysis System
Integrates facial expression, vocal pattern, and textual analysis
"""

import cv2
import numpy as np
from scipy.io import wavfile
import librosa
import mediapipe as mp
from textblob import TextBlob

class MultiModalAnalyzer:
    def __init__(self):
        # Initialize face detection
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Initialize audio processing parameters
        self.audio_params = {
            'sample_rate': 16000,
            'n_mfcc': 13,
            'hop_length': 512
        }
        
        # Emotion detection parameters
        self.emotion_weights = {
            'happy': 1.0,
            'sad': 1.0,
            'angry': 1.2,
            'surprised': 0.8,
            'neutral': 0.5
        }

    def analyze_video_interview(self, video_path):
        """Analyze video interview for multimodal insights"""
        # Extract frames and audio
        video_data = self._process_video(video_path)
        
        # Analyze facial expressions
        emotion_data = self._analyze_facial_expressions(video_data['frames'])
        
        # Analyze vocal patterns
        speech_features = self._analyze_vocal_patterns(video_data['audio'])
        
        # Analyze transcript
        text_analysis = self._analyze_transcript(video_data['transcript'])
        
        # Combine analyses
        return self._combine_analyses(emotion_data, speech_features, text_analysis)

    def _process_video(self, video_path):
        """Process video file into frames and audio"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        audio = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        
        # Extract audio
        sample_rate, audio_data = wavfile.read(video_path)
        audio = audio_data.astype(np.float32) / 32768.0
        
        return {
            'frames': frames,
            'audio': audio,
            'sample_rate': sample_rate
        }

    def _analyze_facial_expressions(self, frames):
        """Analyze facial expressions using MediaPipe"""
        emotion_scores = {
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'surprised': 0,
            'neutral': 0
        }
        
        total_frames = len(frames)
        for frame in frames:
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Extract facial landmarks
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Calculate emotion scores based on landmark positions
                emotion = self._detect_emotion_from_landmarks(landmarks)
                emotion_scores[emotion] += 1
        
        # Normalize scores
        for emotion in emotion_scores:
            emotion_scores[emotion] /= total_frames
            
        return emotion_scores

    def _detect_emotion_from_landmarks(self, landmarks):
        """Detect emotion from facial landmarks"""
        # Simplified emotion detection based on key facial points
        mouth_open = landmarks[13].y - landmarks[14].y
        eyebrow_raise = landmarks[105].y - landmarks[107].y
        
        if mouth_open > 0.1 and eyebrow_raise > 0.05:
            return 'surprised'
        elif mouth_open > 0.05:
            return 'happy'
        elif eyebrow_raise < -0.05:
            return 'angry'
        elif mouth_open < -0.05:
            return 'sad'
        else:
            return 'neutral'

    def _analyze_vocal_patterns(self, audio):
        """Analyze vocal patterns using librosa"""
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.audio_params['sample_rate'],
            n_mfcc=self.audio_params['n_mfcc'],
            hop_length=self.audio_params['hop_length']
        )
        
        # Calculate speech features
        features = {
            'pitch_variation': np.std(mfccs[0]),
            'speech_rate': len(mfccs[0]) / (len(audio) / self.audio_params['sample_rate']),
            'energy': np.mean(np.abs(audio)),
            'pitch_range': np.ptp(mfccs[0])
        }
        
        return features

    def _analyze_transcript(self, transcript):
        """Analyze transcript text"""
        blob = TextBlob(transcript)
        
        return {
            'sentiment': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'word_count': len(blob.words),
            'sentence_count': len(blob.sentences)
        }

    def _combine_analyses(self, emotion_data, speech_features, text_analysis):
        """Combine multimodal analyses into comprehensive profile"""
        # Calculate confidence scores
        confidence_scores = {
            'facial_confidence': np.mean(list(emotion_data.values())),
            'vocal_confidence': np.mean(list(speech_features.values())),
            'text_confidence': (text_analysis['word_count'] / 100) * 0.5
        }
        
        # Generate insights
        insights = {
            'emotional_state': self._interpret_emotions(emotion_data),
            'communication_style': self._interpret_communication(speech_features),
            'cognitive_patterns': self._interpret_cognition(text_analysis),
            'confidence_scores': confidence_scores
        }
        
        return insights

    def _interpret_emotions(self, emotion_data):
        """Interpret emotional patterns"""
        primary_emotion = max(emotion_data.items(), key=lambda x: x[1])[0]
        intensity = emotion_data[primary_emotion]
        
        return {
            'primary_emotion': primary_emotion,
            'intensity': intensity,
            'stability': 1 - np.std(list(emotion_data.values()))
        }

    def _interpret_communication(self, speech_features):
        """Interpret communication style from vocal patterns"""
        style = []
        
        if speech_features['pitch_variation'] > 0.5:
            style.append("expressive")
        if speech_features['speech_rate'] > 3.0:
            style.append("fast-paced")
        if speech_features['energy'] > 0.5:
            style.append("enthusiastic")
            
        return {
            'style': style,
            'confidence': np.mean(list(speech_features.values()))
        }

    def _interpret_cognition(self, text_analysis):
        """Interpret cognitive patterns from text"""
        patterns = []
        
        if text_analysis['sentiment'] > 0.3:
            patterns.append("positive outlook")
        if text_analysis['subjectivity'] > 0.5:
            patterns.append("subjective perspective")
        if text_analysis['word_count'] / text_analysis['sentence_count'] > 15:
            patterns.append("detailed thinking")
            
        return {
            'patterns': patterns,
            'complexity': text_analysis['word_count'] / text_analysis['sentence_count']
        } 