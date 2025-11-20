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
        trainer.load_model("models/")  # Load from models/ directory
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
        # Convert date to datetime for DTW calculation (DTW expects datetime objects)
        point_df = pd.DataFrame({
            'lat': [request.lat],
            'lon': [request.lon],
            'date': [pd.to_datetime(pred_date)]  # Convert date to datetime
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
        
        # Create feature DataFrame with all required columns
        # Ensure all feature columns are present (even if NaN)
        feature_df = pd.DataFrame([features])
        
        # Ensure all expected feature columns are present
        expected_columns = pipeline.feature_columns
        for col in expected_columns:
            if col not in feature_df.columns:
                feature_df[col] = np.nan
        
        # Reorder columns to match expected order
        feature_df = feature_df[expected_columns]
        
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

# Detailed metrics endpoint
@app.get("/model/detailed_metrics")
async def get_detailed_metrics():
    """Get detailed evaluation metrics including confusion matrix and ROC curve."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        from sklearn.metrics import confusion_matrix, roc_curve
        from pathlib import Path as FilePath
        
        # Check if feature file exists
        feature_file = FilePath("data/processed/features_5000.parquet")
        if not feature_file.exists():
            raise HTTPException(status_code=404, detail="Training data not found")
        
        logger.info("Loading test data for detailed metrics...")
        
        # Load data
        df = pd.read_parquet(feature_file)
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['lat', 'lon', 'date', 'target', 'is_fire', 'dtw_start', 'dtw_end']]
        X = df[feature_columns].copy()
        
        # Handle target column
        if 'target' in df.columns:
            y = df['target'].copy()
        elif 'is_fire' in df.columns:
            y = df['is_fire'].copy()
        else:
            raise HTTPException(status_code=500, detail="No target column found")
        
        # Handle missing values
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        y = y.astype(int)
        
        # Apply temporal split (same as training)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            split_date = df['date'].quantile(0.8)
            test_mask = df['date'] >= split_date
            
            X_test = X[test_mask]
            y_test = y[test_mask]
        else:
            # Fallback to last 20% if no date column
            split_idx = int(len(X) * 0.8)
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
        
        logger.info(f"Test set size: {len(X_test)} samples")
        
        # Make predictions
        predictions, probabilities = trainer.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, probabilities)
        avg_precision = average_precision_score(y_test, probabilities)
        
        logger.info("Detailed metrics calculated successfully")
        
        # Handle NaN/Inf values for JSON serialization
        def clean_for_json(arr):
            """Replace NaN and Inf values with valid numbers."""
            arr = np.array(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            return arr.tolist()
        
        # Normalize confusion matrix safely
        cm_normalized = cm.astype('float')
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_normalized = cm_normalized / row_sums
        
        return {
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": clean_for_json(cm_normalized),
            "roc_curve": {
                "fpr": clean_for_json(fpr),
                "tpr": clean_for_json(tpr),
                "thresholds": clean_for_json(thresholds)
            },
            "precision_recall_curve": {
                "precision": clean_for_json(precision_curve),
                "recall": clean_for_json(recall_curve),
                "thresholds": clean_for_json(pr_thresholds),
                "average_precision": float(avg_precision) if not np.isnan(avg_precision) else 0.0
            },
            "test_samples": len(y_test),
            "class_distribution": {
                "negative": int((y_test == 0).sum()),
                "positive": int((y_test == 1).sum())
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating detailed metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate detailed metrics: {str(e)}")

# Auto-sklearn ensemble details endpoint
@app.get("/model/ensemble_details")
async def get_ensemble_details():
    """Get detailed information about Auto-sklearn ensemble models."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_type = trainer.model_artifacts.get('model_type', 'unknown')
        
        if model_type != 'autosklearn':
            return {
                "ensemble_available": False,
                "model_type": model_type,
                "message": "Ensemble details only available for Auto-sklearn models"
            }
        
        ensemble_info = {
            "ensemble_available": True,
            "model_type": "autosklearn",
            "models": [],
            "ensemble_statistics": {}
        }
        
        # Get ensemble models and weights
        try:
            models_with_weights = trainer.model.get_models_with_weights()
            logger.info(f"Found {len(models_with_weights)} models in ensemble")
            
            for idx, (weight, model) in enumerate(models_with_weights):
                model_info = {
                    "rank": idx + 1,
                    "weight": float(weight),
                    "model_id": str(model),
                    "estimator_name": "Unknown",
                    "hyperparameters": {}
                }
                
                # Try to extract model details
                try:
                    # Get the actual sklearn pipeline
                    if hasattr(model, 'steps'):
                        # It's a pipeline - dig into Auto-sklearn structure
                        for step_name, step_model in model.steps:
                            # Auto-sklearn uses 'classifier' for the main estimator
                            if step_name in ['estimator', 'classifier']:
                                # Check if it's a ClassifierChoice (Auto-sklearn wrapper)
                                if hasattr(step_model, 'choice') and hasattr(step_model.choice, 'estimator'):
                                    # Get the actual estimator inside
                                    actual_estimator = step_model.choice.estimator
                                    model_info["estimator_name"] = type(actual_estimator).__name__
                                    
                                    # Try to get parameters from the actual estimator
                                    if hasattr(actual_estimator, 'get_params'):
                                        params = actual_estimator.get_params()
                                        important_params = {k: v for k, v in params.items() 
                                                           if not k.startswith('_') and v is not None 
                                                           and not callable(v)}
                                        model_info["hyperparameters"] = {k: str(v) for k, v in important_params.items()}
                                else:
                                    # Regular sklearn estimator
                                    model_info["estimator_name"] = type(step_model).__name__
                                    if hasattr(step_model, 'get_params'):
                                        params = step_model.get_params()
                                        important_params = {k: v for k, v in params.items() 
                                                           if not k.startswith('_') and v is not None 
                                                           and not callable(v)}
                                        model_info["hyperparameters"] = {k: str(v) for k, v in important_params.items()}
                    else:
                        model_info["estimator_name"] = type(model).__name__
                        if hasattr(model, 'get_params'):
                            params = model.get_params()
                            important_params = {k: v for k, v in params.items() 
                                               if not k.startswith('_') and v is not None 
                                               and not callable(v)}
                            model_info["hyperparameters"] = {k: str(v) for k, v in important_params.items()}
                except Exception as e:
                    logger.warning(f"Could not extract details for model {idx}: {e}")
                    model_info["estimator_name"] = type(model).__name__
                    logger.warning(f"Model structure: {model}")
                
                ensemble_info["models"].append(model_info)
            
            # Calculate ensemble statistics
            total_weight = sum(m["weight"] for m in ensemble_info["models"])
            ensemble_info["ensemble_statistics"] = {
                "total_models": len(ensemble_info["models"]),
                "total_weight": float(total_weight),
                "top_model_weight": float(ensemble_info["models"][0]["weight"]) if ensemble_info["models"] else 0,
                "model_types_count": {}
            }
            
            # Count model types
            for model in ensemble_info["models"]:
                model_type = model["estimator_name"]
                if model_type in ensemble_info["ensemble_statistics"]["model_types_count"]:
                    ensemble_info["ensemble_statistics"]["model_types_count"][model_type] += 1
                else:
                    ensemble_info["ensemble_statistics"]["model_types_count"][model_type] = 1
            
        except AttributeError:
            logger.warning("Could not get ensemble details - model may not be Auto-sklearn")
            ensemble_info["models"].append({
                "rank": 1,
                "weight": 1.0,
                "estimator_name": type(trainer.model).__name__,
                "hyperparameters": {}
            })
        
        # Try to get leaderboard
        try:
            leaderboard = trainer.model.leaderboard()
            if leaderboard is not None and len(leaderboard) > 0:
                ensemble_info["leaderboard"] = leaderboard.head(10).to_dict('records')
        except:
            pass
        
        logger.info("Ensemble details retrieved successfully")
        return ensemble_info
        
    except Exception as e:
        logger.error(f"Error getting ensemble details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ensemble details: {str(e)}")

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
            "detailed_metrics": "/model/detailed_metrics",
            "ensemble_details": "/model/ensemble_details",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



