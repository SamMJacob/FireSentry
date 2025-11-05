#!/usr/bin/env python3
"""
Test Auto-sklearn Installation and Functionality
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_autosklearn():
    """Test Auto-sklearn installation and basic functionality."""
    
    print("🔍 Testing Auto-sklearn installation...")
    
    # Test 1: Import Auto-sklearn
    try:
        import autosklearn.classification
        from autosklearn.metrics import accuracy, precision, recall, f1_macro, roc_auc
        print("✅ Auto-sklearn import successful")
    except ImportError as e:
        print(f"❌ Auto-sklearn import failed: {e}")
        return False
    
    # Test 2: Create synthetic dataset
    print("📊 Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Test 3: Initialize Auto-sklearn
    try:
        print("🤖 Initializing Auto-sklearn classifier...")
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=60,  # 1 minute for quick test
            per_run_time_limit=10,      # 10 seconds per run
            memory_limit=3072,          # 3GB memory limit
            ensemble_size=1,            # Single model for speed
            seed=42
        )
        print("✅ Auto-sklearn classifier initialized")
    except Exception as e:
        print(f"❌ Auto-sklearn initialization failed: {e}")
        return False
    
    # Test 4: Train model
    try:
        print("🏋️ Training Auto-sklearn model...")
        automl.fit(X_train, y_train)
        print("✅ Auto-sklearn training completed")
    except Exception as e:
        print(f"❌ Auto-sklearn training failed: {e}")
        return False
    
    # Test 5: Make predictions
    try:
        print("🔮 Making predictions...")
        y_pred = automl.predict(X_test)
        y_pred_proba = automl.predict_proba(X_test)
        print("✅ Predictions successful")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    # Test 6: Evaluate performance
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        print(f"📈 Performance Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   AUC: {auc:.3f}")
        
        if accuracy > 0.7 and auc > 0.7:
            print("✅ Performance looks good!")
        else:
            print("⚠️ Performance might be low (expected for quick test)")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    
    # Test 7: Model persistence
    try:
        import joblib
        print("💾 Testing model persistence...")
        joblib.dump(automl, 'test_autosklearn_model.joblib')
        loaded_model = joblib.load('test_autosklearn_model.joblib')
        print("✅ Model persistence successful")
        
        # Clean up
        import os
        os.remove('test_autosklearn_model.joblib')
        
    except Exception as e:
        print(f"❌ Model persistence failed: {e}")
        return False
    
    # Test 8: Show model details
    try:
        print("🔍 Model details:")
        print(f"   Best model: {automl.get_models_with_weights()}")
        print(f"   Feature importance: {automl.get_feature_importance()}")
    except Exception as e:
        print(f"⚠️ Could not get model details: {e}")
    
    print("🎉 Auto-sklearn test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_autosklearn()
    if success:
        print("\n✅ Auto-sklearn is working correctly!")
        print("🚀 You can now run the full feature engineering pipeline!")
    else:
        print("\n❌ Auto-sklearn has issues that need to be resolved.")
        print("💡 Try installing missing dependencies or check memory limits.")
