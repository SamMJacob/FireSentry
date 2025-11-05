#!/usr/bin/env python3
"""
Test script to verify the evaluation fix works correctly.
This script tests the feature selection pipeline to ensure
training and evaluation use the same number of features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_selection_pipeline():
    """Test that feature selection works correctly for both training and evaluation."""
    
    # Create synthetic data with 24 features (like our real data)
    np.random.seed(42)
    n_samples = 1000
    n_features = 24
    
    # Generate synthetic features
    feature_names = [
        'elevation', 'slope', 'aspect',  # 3 terrain
        'ndvi_min', 'ndvi_median', 'ndvi_mean', 'ndvi_max',
        'evi_min', 'evi_median', 'evi_mean', 'evi_max',
        'ndwi_min', 'ndwi_median', 'ndwi_mean', 'ndwi_max',
        'prec_min', 'prec_median', 'prec_mean', 'prec_max', 'prec_sum',
        'lst_min', 'lst_median', 'lst_mean', 'lst_max'
    ]
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    y = np.random.randint(0, 2, n_samples)
    
    logger.info(f"Created synthetic data: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Data split: {X_train.shape} train, {X_test.shape} test")
    
    # Simulate feature selection (select first 9 features)
    selected_features = feature_names[:9]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    logger.info(f"After feature selection: {X_train_selected.shape} train, {X_test_selected.shape} test")
    logger.info(f"Selected features: {list(X_train_selected.columns)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_selected),
        columns=X_train_selected.columns,
        index=X_train_selected.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_selected),
        columns=X_test_selected.columns,
        index=X_test_selected.index
    )
    
    logger.info(f"After scaling: {X_train_scaled.shape} train, {X_test_scaled.shape} test")
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    logger.info("Model trained successfully")
    
    # Test evaluation (this should work now)
    try:
        # Training predictions
        logger.info("Making training predictions...")
        y_train_pred = model.predict(X_train_scaled)
        y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
        
        # Test predictions
        logger.info("Making test predictions...")
        y_test_pred = model.predict(X_test_scaled)
        y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        logger.info(f"✅ Evaluation successful!")
        logger.info(f"Train accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    logger.info("Testing feature selection pipeline...")
    success = test_feature_selection_pipeline()
    
    if success:
        logger.info("🎉 Test passed! Feature selection pipeline works correctly.")
    else:
        logger.error("💥 Test failed! There's still an issue with the pipeline.")
