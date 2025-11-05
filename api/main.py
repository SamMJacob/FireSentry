"""
FireSentry FastAPI Application

RESTful API for fire risk prediction using trained models.
Provides endpoints for real-time fire risk assessment.

Author: FireSentry Team
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, date
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.train import FirePredictionTrainer
from features.pipeline import FeaturePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FireSentry API",
    description="Forest Fire Risk Prediction API for Uttarakhand",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and pipeline
trainer = None
pipeline = None

# Pydantic models
class PredictRequest(BaseModel):
    """Request model for fire risk prediction."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    
    class Config:
        schema_extra = {
            "example": {
                "lat": 30.0,
                "lon": 79.0,
                "date": "2024-04-15"
            }
        }

class PredictResponse(BaseModel):
    """Response model for fire risk prediction."""
    probability: float = Field(..., ge=0, le=1, description="Fire probability")
    risk_level: str = Field(..., description="Risk level category")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    features_used: int = Field(..., description="Number of features used")
    
    class Config:
        schema_extra = {
            "example": {
                "probability": 0.75,
                "risk_level": "High",
                "confidence": 0.85,
                "features_used": 12
            }
        }

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    timestamp: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize model and pipeline on startup."""
    global trainer, pipeline
    
    try:
        logger.info("Loading trained model...")
        trainer = FirePredictionTrainer()
        trainer.load_model()
        logger.info("Model loaded successfully")
        
        logger.info("Initializing feature pipeline...")
        pipeline = FeaturePipeline()
        logger.info("Feature pipeline initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        # Continue without model for health checks

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if trainer is not None else "unhealthy",
        model_loaded=trainer is not None,
        timestamp=datetime.now().isoformat()
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
async def predict_fire_risk(request: PredictRequest):
    """
    Predict fire risk for a given location and date.
    
    Args:
        request: Prediction request with lat, lon, and date
        
    Returns:
        Fire risk prediction with probability and risk level
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Parse date
        try:
            pred_date = datetime.strptime(request.date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Validate date range (should be within training data range)
        current_date = date.today()
        if pred_date > current_date:
            raise HTTPException(status_code=400, detail="Date cannot be in the future")
        
        # Validate coordinates (Uttarakhand bounding box)
        if not (28.7 <= request.lat <= 31.5 and 77.5 <= request.lon <= 81.0):
            raise HTTPException(
                status_code=400, 
                detail="Coordinates must be within Uttarakhand bounds (28.7-31.5°N, 77.5-81.0°E)"
            )
        
        # Build features for the location and date
        logger.info(f"Building features for ({request.lat}, {request.lon}) on {pred_date}")
        
        # Create a single-row DataFrame for feature extraction
        point_df = pd.DataFrame({
            'lat': [request.lat],
            'lon': [request.lon],
            'date': [pred_date]
        })
        
        # Calculate DTW for this point
        point_with_dtw = pipeline.dtw.calculate_dtw_batch(point_df, "data/raw/chirps")
        
        if len(point_with_dtw) == 0:
            raise HTTPException(status_code=500, detail="Failed to calculate DTW window")
        
        # Extract all features
        features = pipeline.extract_all_features(
            request.lat, request.lon, 
            point_with_dtw.iloc[0]['dtw_start'], 
            point_with_dtw.iloc[0]['dtw_end']
        )
        
        # Create feature DataFrame
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        predictions, probabilities = trainer.predict(feature_df)
        
        probability = float(probabilities[0])
        prediction = int(predictions[0])
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        elif probability < 0.8:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Calculate confidence (based on probability distance from 0.5)
        confidence = abs(probability - 0.5) * 2
        
        return PredictResponse(
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            features_used=len(trainer.feature_names)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(request: list[PredictRequest]):
    """
    Predict fire risk for multiple locations and dates.
    
    Args:
        request: List of prediction requests
        
    Returns:
        List of fire risk predictions
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 requests")
    
    results = []
    
    for req in request:
        try:
            # Use the single prediction logic
            response = await predict_fire_risk(req)
            results.append(response.dict())
        except Exception as e:
            results.append({
                "error": str(e),
                "lat": req.lat,
                "lon": req.lon,
                "date": req.date
            })
    
    return {"results": results}

# Model info endpoint
@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": trainer.model_artifacts.get('model_type', 'unknown'),
        "features_used": len(trainer.feature_names),
        "selected_features": trainer.feature_names,
        "training_results": trainer.training_results,
        "created_at": trainer.model_artifacts.get('created_at', 'unknown')
    }

# Feature importance endpoint
@app.get("/model/importance")
async def get_feature_importance():
    """Get feature importance scores."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if hasattr(trainer.msfs, 'get_feature_importance'):
        importance = trainer.msfs.get_feature_importance()
    else:
        importance = {}
    
    return {
        "feature_importance": importance,
        "feature_names": trainer.feature_names
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FireSentry API - Forest Fire Risk Prediction",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "feature_importance": "/model/importance",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



