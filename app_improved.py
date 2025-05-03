"""
Comprehensive Psychological Analysis System (CPAS)
Improved main application entry point
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import uvicorn
import torch
import logging
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import time
import os
import json
from cachetools import TTLCache
import jwt
from jwt.exceptions import InvalidTokenError
import redis
from redis import Redis
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from data_ingestion import DataIngestor
from core_models import CorePersonalityModel, PersonalityEnsemble, ModelConfig
from cognitive_analysis import CognitiveAnalyzer
from relationship_dynamics import RelationshipAnalyzer
from cultural_adaptation import CulturalAdapter
from clinical_validation import ClinicalValidator
from system_upgrade import SystemUpgrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)
MODEL_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction duration in seconds',
    ['model_type']
)
SYSTEM_METRICS = {
    'cpu_usage': Gauge('system_cpu_usage', 'CPU usage percentage'),
    'memory_usage': Gauge('system_memory_usage', 'Memory usage percentage'),
    'gpu_usage': Gauge('system_gpu_usage', 'GPU usage percentage'),
    'active_connections': Gauge('active_connections', 'Number of active connections')
}

# Security configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Cache configuration
CACHE_TTL = 3600  # 1 hour
cache = TTLCache(maxsize=1000, ttl=CACHE_TTL)

# Redis configuration
redis_client = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

# Initialize FastAPI app with better configuration
app = FastAPI(
    title="Comprehensive Psychological Analysis System",
    description="API for psychological analysis and behavioral prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Security middleware
class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Rate limiting
        client_ip = request.client.host
        rate_key = f"rate_limit:{client_ip}"
        current = redis_client.incr(rate_key)
        if current == 1:
            redis_client.expire(rate_key, 60)  # 1 minute window
        if current > 100:  # 100 requests per minute
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"}
            )
        
        response = await call_next(request)
        return response

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SecurityMiddleware)

# Initialize Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "development"),
    integrations=[
        FastApiIntegration(),
        RedisIntegration(),
        LoggingIntegration(level=logging.INFO)
    ],
    traces_sample_rate=1.0
)

# Initialize system components with error handling
try:
    config = ModelConfig("config/model_config.json")
    data_ingestor = DataIngestor()
    core_model = CorePersonalityModel(config)
    personality_ensemble = PersonalityEnsemble(config)
    cognitive_analyzer = CognitiveAnalyzer()
    relationship_analyzer = RelationshipAnalyzer()
    cultural_adapter = CulturalAdapter()
    clinical_validator = ClinicalValidator()
    system_upgrader = SystemUpgrader({
        'current_version': '1.0.0',
        'improvement_threshold': 0.1,
        'validation_thresholds': {
            'accuracy': 0.85,
            'f1_score': 0.80,
            'roc_auc': 0.85
        }
    })
except Exception as e:
    logger.error(f"Failed to initialize system components: {e}")
    raise

# Pydantic models with validation
class AnalysisRequest(BaseModel):
    text: Optional[str] = Field(None, max_length=10000)
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    cultural_context: str = Field("western", regex="^(western|eastern|middle_eastern|african)$")
    analysis_type: str = Field("comprehensive", regex="^(comprehensive|basic|advanced)$")

    @validator('text')
    def validate_text(cls, v):
        if v is not None and len(v.strip()) < 10:
            raise ValueError("Text must be at least 10 characters long")
        return v

class AnalysisResponse(BaseModel):
    personality_profile: Dict[str, float]
    cognitive_analysis: Dict[str, Any]
    relationship_patterns: Dict[str, Any]
    cultural_insights: Dict[str, Any]
    clinical_validation: Dict[str, float]
    timestamp: datetime
    request_id: str

class SystemStatus(BaseModel):
    version: str
    components: Dict[str, str]
    last_update: datetime
    performance_metrics: Dict[str, float]
    health_status: str

# Authentication and authorization
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Health check endpoint with detailed status
@app.get("/health", response_model=SystemStatus)
async def health_check():
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        
        # Check system metrics
        metrics = {
            "cpu_usage": 0.5,  # Placeholder, implement actual monitoring
            "memory_usage": 0.3,
            "gpu_usage": 0.2 if gpu_available else 0.0
        }
        
        return SystemStatus(
            version=system_upgrader.current_version,
            components={
                "core_model": "operational",
                "cognitive_analyzer": "operational",
                "relationship_analyzer": "operational",
                "cultural_adapter": "operational",
                "clinical_validator": "operational",
                "redis": "operational",
                "gpu": "available" if gpu_available else "unavailable"
            },
            last_update=datetime.utcnow(),
            performance_metrics=metrics,
            health_status="healthy"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemStatus(
            version=system_upgrader.current_version,
            components={"status": "degraded"},
            last_update=datetime.utcnow(),
            performance_metrics={},
            health_status="unhealthy"
        )

# Main analysis endpoint with caching and retry logic
@app.post("/analyze", response_model=AnalysisResponse)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def analyze_behavior(
    request: AnalysisRequest,
    api_key: str = Security(get_api_key)
):
    try:
        # Generate request ID
        request_id = f"req_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Check cache
        cache_key = f"analysis:{request_id}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Process input data
        processed_data = await data_ingestor.process_input(request)
        
        # Run analysis with timing
        start_time = time.time()
        
        personality_profile = await core_model.analyze(processed_data)
        MODEL_LATENCY.labels(model_type="personality").observe(time.time() - start_time)
        
        cognitive_analysis = await cognitive_analyzer.analyze(processed_data)
        relationship_patterns = await relationship_analyzer.analyze(processed_data)
        cultural_insights = await cultural_adapter.adapt(processed_data, request.cultural_context)
        clinical_validation = await clinical_validator.validate(processed_data)
        
        # Create response
        response = AnalysisResponse(
            personality_profile=personality_profile,
            cognitive_analysis=cognitive_analysis,
            relationship_patterns=relationship_patterns,
            cultural_insights=cultural_insights,
            clinical_validation=clinical_validation,
            timestamp=datetime.utcnow(),
            request_id=request_id
        )
        
        # Cache result
        cache[cache_key] = response
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# System status endpoint with detailed metrics
@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    try:
        # Get system metrics
        metrics = {
            "cpu_usage": 0.5,  # Placeholder, implement actual monitoring
            "memory_usage": 0.3,
            "gpu_usage": 0.2 if torch.cuda.is_available() else 0.0,
            "active_connections": len(cache)
        }
        
        return SystemStatus(
            version=system_upgrader.current_version,
            components={
                "core_model": "operational",
                "cognitive_analyzer": "operational",
                "relationship_analyzer": "operational",
                "cultural_adapter": "operational",
                "clinical_validator": "operational"
            },
            last_update=datetime.utcnow(),
            performance_metrics=metrics,
            health_status="healthy"
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Model validation endpoint with improved error handling
@app.post("/validate")
async def validate_model(api_key: str = Security(get_api_key)):
    try:
        results = await clinical_validator.run_validation()
        return results
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Background tasks
@app.on_event("startup")
async def startup_event():
    """Initialize system components and start background tasks"""
    logger.info("Starting CPAS system...")
    
    # Start system upgrade cycle in background
    asyncio.create_task(run_system_upgrade())
    
    # Start metrics collection
    asyncio.create_task(collect_system_metrics())

async def run_system_upgrade():
    """Background task for system upgrades"""
    while True:
        try:
            await system_upgrader.check_and_apply_updates()
            await asyncio.sleep(3600)  # Check every hour
        except Exception as e:
            logger.error(f"Error in system upgrade: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retrying

async def collect_system_metrics():
    """Collect and update system metrics"""
    while True:
        try:
            # Update system metrics (implement actual monitoring)
            SYSTEM_METRICS['cpu_usage'].set(0.5)
            SYSTEM_METRICS['memory_usage'].set(0.3)
            SYSTEM_METRICS['gpu_usage'].set(0.2 if torch.cuda.is_available() else 0.0)
            SYSTEM_METRICS['active_connections'].set(len(cache))
            
            await asyncio.sleep(60)  # Update every minute
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    uvicorn.run(
        "app_improved:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=int(os.getenv("WORKERS", 4)),
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*"
    ) 