"""
Minimal FastAPI app for AIDA-2
"""

import logging
import os
from pathlib import Path
import uvicorn
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)
file_handler = logging.FileHandler('logs/api.log')
logger.addHandler(file_handler)

logger.info("Starting minimal AIDA-2 API")

# Import FastAPI components
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    
    # Create FastAPI app
    app = FastAPI(
        title="AIDA-2 Minimal API",
        description="Minimal API for AIDA-2 system",
        version="1.0.0",
    )
    
    # Define Pydantic models
    class AnalysisRequest(BaseModel):
        text: str = Field(..., min_length=10)
        cultural_context: str = "western"
    
    class AnalysisResponse(BaseModel):
        personality_profile: Dict[str, float]
        recommendations: List[str]
        
    # Define API endpoints
    @app.get("/")
    async def root():
        return {
            "message": "AIDA-2 Minimal API",
            "version": "1.0.0",
            "status": "operational"
        }
    
    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze_personality(request: AnalysisRequest):
        try:
            # Simulate analysis
            logger.info(f"Analyzing text: {request.text[:50]}...")
            
            # Return mock response
            return AnalysisResponse(
                personality_profile={
                    "openness": 0.75,
                    "conscientiousness": 0.82,
                    "extraversion": 0.45,
                    "agreeableness": 0.68,
                    "neuroticism": 0.32
                },
                recommendations=[
                    "Consider exploring creative activities",
                    "Your conscientiousness is a strength in professional settings",
                    "You may benefit from more social interactions"
                ]
            )
        
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "components": {
                "api": "operational",
                "analysis_engine": "operational"
            }
        }
    
    logger.info("FastAPI app initialized successfully")
    
except Exception as e:
    logger.error(f"Error initializing FastAPI app: {str(e)}")
    raise

# Run the application
if __name__ == "__main__":
    uvicorn.run("minimal_api:app", host="0.0.0.0", port=12000, reload=True)