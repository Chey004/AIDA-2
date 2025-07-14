"""
Comprehensive Psychological Analysis System (CPAS)
Main application entry point
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import uvicorn
import torch
import logging
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import os
import json
import numpy as np
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

# Import system components
from core_models import CorePersonalityModel, ModelConfig
from personality_analyzer import PersonalityEnsemble
from cognitive_analysis import CognitiveAnalyzer
from relationship_dynamics import RelationshipAnalyzer
from cultural_adaptation import CulturalAdapter
from clinical_validation import ClinicalValidator
from system_upgrade import SystemUpgrader
from data_ingestion import DataProcessor
from multimodal_analysis import MultimodalAnalyzer
from interpretation_guidelines import InterpretationEngine
from enterprise_security import SecurityManager
from adaptive_interview import AdaptiveInterviewer
from advanced_visualization import VisualizationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log'),
    ]
)

logger = logging.getLogger(__name__)

# Initialize Sentry for error tracking
try:
    sentry_sdk.init(
        dsn="https://exampledsn@sentry.io/123456",
        traces_sample_rate=0.1,
        environment="production"
    )
except Exception as e:
    logger.warning(f"Sentry initialization failed: {e}")

# Create FastAPI app
app = FastAPI(
    title="Comprehensive Psychological Analysis System",
    description="Advanced AI system for psychological analysis and personality assessment",
    version="1.0.0",
)

# Add middleware
app.add_middleware(CORSMiddleware, 
                  allow_origins=["*"], 
                  allow_credentials=True,
                  allow_methods=["*"], 
                  allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security middleware
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit_per_minute=60):
        super().__init__(app)
        self.rate_limit = rate_limit_per_minute
        self.request_timestamps = {}
        
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = datetime.now()
        
        # Clean up old timestamps
        self.request_timestamps = {ip: times for ip, times in 
                                 self.request_timestamps.items() 
                                 if times[-1] > current_time - timedelta(minutes=1)}
        
        # Check rate limit
        if client_ip in self.request_timestamps:
            timestamps = self.request_timestamps[client_ip]
            if len(timestamps) >= self.rate_limit:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please try again later."}
                )
            self.request_timestamps[client_ip].append(current_time)
        else:
            self.request_timestamps[client_ip] = [current_time]
            
        return await call_next(request)

app.add_middleware(RateLimitMiddleware, rate_limit_per_minute=100)

# Initialize system components
try:
    # Load configuration
    config = ModelConfig(
        model_name="cpas-core-v1",
        embedding_dim=768,
        hidden_dim=512,
        num_layers=4,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=32,
        max_seq_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Initialize components
    core_model = CorePersonalityModel(config)
    personality_ensemble = PersonalityEnsemble(config)
    cognitive_analyzer = CognitiveAnalyzer()
    relationship_analyzer = RelationshipAnalyzer()
    cultural_adapter = CulturalAdapter(config)
    clinical_validator = ClinicalValidator(config)
    system_upgrader = SystemUpgrader({
        'current_version': '1.0.0',
        'improvement_threshold': 0.1,
        'validation_thresholds': {
            'accuracy': 0.85,
            'clinical_validity': 0.80,
            'cultural_sensitivity': 0.90
        }
    })
    data_processor = DataProcessor()
    multimodal_analyzer = MultimodalAnalyzer()
    interpretation_engine = InterpretationEngine()
    security_manager = SecurityManager()
    adaptive_interviewer = AdaptiveInterviewer()
    visualization_engine = VisualizationEngine()
    
    logger.info("System components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize system components: {str(e)}")
    raise

# Pydantic models with validation
class AnalysisRequest(BaseModel):
    text: Optional[str] = Field(None, max_length=10000)
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    cultural_context: str = Field("western", pattern="^(western|eastern|middle_eastern|african)$")
    analysis_type: str = Field("comprehensive", pattern="^(comprehensive|basic|advanced)$")

    @validator('text')
    def validate_text(cls, v):
        if v is not None and len(v.strip()) < 10:
            raise ValueError("Text must be at least 10 characters long")
        return v

    @validator('audio_path', 'video_path')
    def validate_paths(cls, v):
        if v is not None and not Path(v).exists():
            raise ValueError(f"File path does not exist: {v}")
        return v

class AnalysisResponse(BaseModel):
    personality_profile: Dict[str, float]
    cognitive_patterns: Dict[str, Any]
    relationship_dynamics: Dict[str, Any]
    cultural_insights: Dict[str, Any]
    clinical_indicators: Optional[Dict[str, Any]] = None
    recommendations: List[str]
    confidence_score: float
    analysis_timestamp: str
    model_version: str

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "Comprehensive Psychological Analysis System API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_personality(
    request: AnalysisRequest,
    api_key: str = Security(api_key_header)
):
    # Validate API key
    if api_key != os.getenv("API_KEY", "test_key"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Process input data
        if request.text:
            processed_text = data_processor.process_text(request.text)
        elif request.audio_path:
            processed_text = data_processor.process_audio(request.audio_path)
        elif request.video_path:
            processed_text = data_processor.process_video(request.video_path)
        else:
            raise HTTPException(status_code=400, detail="No input data provided")
        
        # Generate personality profile
        personality_scores = personality_ensemble.analyze(processed_text)
        
        # Analyze cognitive patterns
        cognitive_patterns = cognitive_analyzer.analyze(processed_text)
        
        # Analyze relationship dynamics
        relationship_dynamics = relationship_analyzer.analyze(processed_text)
        
        # Apply cultural adaptation
        cultural_insights = cultural_adapter.adjust_personality_scores(personality_scores)
        
        # Validate clinical indicators
        clinical_indicators = clinical_validator.validate(
            personality_scores, 
            cognitive_patterns
        )
        
        # Generate recommendations
        recommendations = interpretation_engine.generate_recommendations(
            personality_scores,
            cognitive_patterns,
            clinical_indicators
        )
        
        # Prepare response
        response = AnalysisResponse(
            personality_profile=personality_scores,
            cognitive_patterns=cognitive_patterns,
            relationship_dynamics=relationship_dynamics,
            cultural_insights=cultural_insights,
            clinical_indicators=clinical_indicators,
            recommendations=recommendations,
            confidence_score=0.85,  # This would be dynamically calculated
            analysis_timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "core_model": "operational",
            "personality_ensemble": "operational",
            "cognitive_analyzer": "operational",
            "relationship_analyzer": "operational",
            "cultural_adapter": "operational",
            "clinical_validator": "operational"
        },
        "system_load": 0.45,  # This would be dynamically calculated
        "memory_usage": 0.32,  # This would be dynamically calculated
        "timestamp": datetime.now().isoformat()
    }

@app.post("/feedback")
async def submit_feedback(
    feedback: Dict[str, Any],
    api_key: str = Security(api_key_header)
):
    # Validate API key
    if api_key != os.getenv("API_KEY", "test_key"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Process feedback for system improvement
        system_upgrader.process_feedback(feedback)
        
        return {"status": "success", "message": "Feedback received and processed"}
    
    except Exception as e:
        logger.error(f"Feedback processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")

@app.get("/models")
async def list_models(
    api_key: str = Security(api_key_header)
):
    # Validate API key
    if api_key != os.getenv("API_KEY", "test_key"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "models": [
            {
                "id": "cpas-core-v1",
                "name": "CPAS Core Model",
                "version": "1.0.0",
                "description": "Core personality analysis model",
                "capabilities": ["personality_analysis", "cognitive_assessment"]
            },
            {
                "id": "cpas-clinical-v1",
                "name": "CPAS Clinical Model",
                "version": "1.0.0",
                "description": "Clinical validation model",
                "capabilities": ["clinical_assessment", "risk_evaluation"]
            }
        ]
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=12000, reload=True)