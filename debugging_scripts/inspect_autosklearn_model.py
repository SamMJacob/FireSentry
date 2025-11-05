#!/usr/bin/env python3
"""
Script to inspect Auto-sklearn model details and ensemble composition.
This will show which models Auto-sklearn selected and their weights.
"""

import joblib
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_autosklearn_model(model_path="models/model.joblib"):
    """Inspect the trained Auto-sklearn model."""
    
    try:
        # Load the model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        logger.info("=" * 60)
        logger.info("AUTO-SKLEARN MODEL INSPECTION")
        logger.info("=" * 60)
        
        # Get model type
        logger.info(f"Model type: {type(model).__name__}")
        
        # Check if it's an Auto-sklearn model
        if hasattr(model, 'get_models_with_weights'):
            logger.info("✅ This is an Auto-sklearn model")
            
            try:
                # Get ensemble information
                models_with_weights = model.get_models_with_weights()
                logger.info(f"📊 Ensemble size: {len(models_with_weights)}")
                
                if len(models_with_weights) > 0:
                    logger.info("\n🔍 ENSEMBLE COMPOSITION:")
                    logger.info("-" * 40)
                    
                    for i, (model_id, weight) in enumerate(models_with_weights, 1):
                        logger.info(f"{i:2d}. Model ID: {model_id}")
                        logger.info(f"    Weight: {weight:.4f}")
                        
                        # Try to get more details about this model
                        try:
                            individual_model = model.get_models()[model_id]
                            logger.info(f"    Type: {type(individual_model).__name__}")
                            
                            # Get model parameters if available
                            if hasattr(individual_model, 'get_params'):
                                params = individual_model.get_params()
                                # Show only key parameters
                                key_params = {k: v for k, v in params.items() 
                                            if k in ['n_estimators', 'max_depth', 'C', 'kernel', 'random_state']}
                                if key_params:
                                    logger.info(f"    Key params: {key_params}")
                            
                        except Exception as e:
                            logger.warning(f"    Could not get details for model {model_id}: {e}")
                        
                        logger.info("")
                
                else:
                    logger.warning("⚠️  Empty ensemble - no models found!")
                    
            except Exception as e:
                logger.error(f"❌ Could not get ensemble info: {e}")
        
        else:
            logger.info("❌ This is not an Auto-sklearn model")
            logger.info(f"Model type: {type(model).__name__}")
            
            # If it's a sklearn model, show its details
            if hasattr(model, 'get_params'):
                params = model.get_params()
                logger.info("\n🔍 MODEL PARAMETERS:")
                logger.info("-" * 40)
                for key, value in params.items():
                    logger.info(f"{key}: {value}")
        
        # Try to get leaderboard
        try:
            if hasattr(model, 'leaderboard'):
                logger.info("\n📈 LEADERBOARD:")
                logger.info("-" * 40)
                leaderboard = model.leaderboard()
                if len(leaderboard) > 0:
                    logger.info(leaderboard.head(10).to_string())
                else:
                    logger.warning("No leaderboard data available")
        except Exception as e:
            logger.warning(f"Could not get leaderboard: {e}")
        
        # Try to get performance summary
        try:
            if hasattr(model, 'sprint_statistics'):
                logger.info("\n📊 PERFORMANCE STATISTICS:")
                logger.info("-" * 40)
                stats = model.sprint_statistics()
                logger.info(stats)
        except Exception as e:
            logger.warning(f"Could not get performance statistics: {e}")
            
    except FileNotFoundError:
        logger.error(f"❌ Model file not found: {model_path}")
        logger.info("Make sure the model has been trained and saved.")
    except Exception as e:
        logger.error(f"❌ Error inspecting model: {e}")

def check_model_artifacts(artifacts_path="models/model_artifacts.joblib"):
    """Check saved model artifacts."""
    
    try:
        logger.info(f"\n🔍 CHECKING MODEL ARTIFACTS:")
        logger.info("-" * 40)
        
        artifacts = joblib.load(artifacts_path)
        logger.info(f"Artifacts keys: {list(artifacts.keys())}")
        
        if 'model_type' in artifacts:
            logger.info(f"Model type: {artifacts['model_type']}")
        
        if 'feature_names' in artifacts:
            logger.info(f"Selected features: {artifacts['feature_names']}")
        
        if 'training_results' in artifacts:
            logger.info(f"Training results keys: {list(artifacts['training_results'].keys())}")
            
    except FileNotFoundError:
        logger.warning(f"Artifacts file not found: {artifacts_path}")
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")

if __name__ == "__main__":
    logger.info("🔍 Auto-sklearn Model Inspector")
    logger.info("=" * 60)
    
    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("❌ Models directory not found!")
        logger.info("Make sure you've trained a model first.")
        exit(1)
    
    # Inspect the model
    inspect_autosklearn_model()
    
    # Check artifacts
    check_model_artifacts()
    
    logger.info("\n✅ Model inspection complete!")
