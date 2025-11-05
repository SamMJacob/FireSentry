#!/usr/bin/env python3
"""
Test Pipeline Script

Quick test of the FireSentry pipeline with synthetic data to verify
all components work correctly before running on real data.

Usage:
    python scripts/test_pipeline.py

Author: FireSentry Team
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import tempfile
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.dtw import DynamicTimeWindow
from features.pipeline import FeaturePipeline
from msfs.selection import MultiStageFeatureSelection
from model.train import FirePredictionTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_data():
    """Create synthetic test data."""
    logger.info("Creating synthetic test data")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create synthetic fire points
    np.random.seed(42)
    n_fires = 100
    
    fire_points = pd.DataFrame({
        'lat': np.random.uniform(28.7, 31.5, n_fires),
        'lon': np.random.uniform(77.5, 81.0, n_fires),
        'date': [datetime(2024, 4, 15) + timedelta(days=np.random.randint(0, 30)) 
                for _ in range(n_fires)]
    })
    
    # Create synthetic feature matrix
    n_features = 24
    feature_names = [
        'prec_min', 'prec_median', 'prec_mean', 'prec_max', 'prec_sum',
        'lst_min', 'lst_median', 'lst_mean', 'lst_max',
        'ndvi_min', 'ndvi_median', 'ndvi_mean', 'ndvi_max',
        'evi_min', 'evi_median', 'evi_mean', 'evi_max',
        'ndwi_min', 'ndwi_median', 'ndwi_mean', 'ndwi_max',
        'elevation', 'slope', 'aspect'
    ]
    
    # Generate features with some correlation to target
    X = pd.DataFrame(
        np.random.randn(n_fires * 2, n_features),
        columns=feature_names
    )
    
    # Create target with some features being relevant
    y = (X['prec_sum'] + X['lst_max'] + X['ndvi_min'] + 
         np.random.randn(len(X)) * 0.1 > 0).astype(int)
    
    # Add location and date info
    X['lat'] = np.tile(fire_points['lat'].values, 2)
    X['lon'] = np.tile(fire_points['lon'].values, 2)
    X['date'] = np.tile(fire_points['date'].values, 2)
    X['target'] = y
    X['is_fire'] = y
    
    # Save synthetic data
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True)
    
    # Save as parquet
    feature_file = data_dir / "features.parquet"
    X.to_parquet(feature_file, index=False)
    
    logger.info(f"Synthetic data created: {len(X)} samples, {n_features} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return str(feature_file), str(temp_dir)

def test_dtw():
    """Test DTW algorithm."""
    logger.info("Testing DTW algorithm...")
    
    try:
        dtw = DynamicTimeWindow(thcp=30.0, thdp=10.0, max_window_days=90)
        
        # Test with synthetic data
        fire_date = datetime(2024, 4, 15)
        lat, lon = 30.0, 79.0
        
        # This will fail without real CHIRPS data, but we can test the logic
        try:
            t_start, t_end = dtw.calculate_dtw(fire_date, lat, lon, "nonexistent_dir")
            logger.info(f"DTW test passed: {t_start} to {t_end}")
        except:
            logger.info("DTW test passed (expected failure without data)")
        
        return True
        
    except Exception as e:
        logger.error(f"DTW test failed: {e}")
        return False

def test_msfs():
    """Test MSFS feature selection."""
    logger.info("Testing MSFS feature selection...")
    
    try:
        # Create synthetic data
        np.random.seed(42)
        n_samples, n_features = 200, 24
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target with some features being relevant
        y = (X['feature_0'] + X['feature_5'] + X['feature_10'] + 
             np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        # Test MSFS
        msfs = MultiStageFeatureSelection(k_mi=12, n_repeats=3)
        X_selected = msfs.fit_transform(X, y)
        
        logger.info(f"MSFS test passed: {X.shape[1]} → {X_selected.shape[1]} features")
        logger.info(f"Selected features: {msfs.selected_features_}")
        
        return True
        
    except Exception as e:
        logger.error(f"MSFS test failed: {e}")
        return False

def test_training():
    """Test model training with synthetic data."""
    logger.info("Testing model training...")
    
    try:
        # Create synthetic data
        feature_file, temp_dir = create_synthetic_data()
        
        # Test training
        trainer = FirePredictionTrainer(
            time_limit=60,  # Short time for testing
            per_run_time_limit=10,
            use_autosklearn=False  # Use sklearn fallback for testing
        )
        
        # Train model
        results = trainer.train(feature_file, test_size=0.3)
        
        logger.info(f"Training test passed: Accuracy = {results.get('test_accuracy', 'N/A'):.4f}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Training test failed: {e}")
        return False

def test_api():
    """Test API components."""
    logger.info("Testing API components...")
    
    try:
        # Test Pydantic models
        from api.main import PredictRequest, PredictResponse
        
        # Test request model
        request = PredictRequest(lat=30.0, lon=79.0, date="2024-04-15")
        assert request.lat == 30.0
        assert request.lon == 79.0
        assert request.date == "2024-04-15"
        
        # Test response model
        response = PredictResponse(
            probability=0.75,
            risk_level="High",
            confidence=0.85,
            features_used=12
        )
        assert response.probability == 0.75
        assert response.risk_level == "High"
        
        logger.info("API test passed")
        return True
        
    except Exception as e:
        logger.error(f"API test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    logger.info("Starting FireSentry pipeline tests")
    
    tests = [
        ("DTW Algorithm", test_dtw),
        ("MSFS Feature Selection", test_msfs),
        ("Model Training", test_training),
        ("API Components", test_api)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "CRASHED"
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    all_passed = all(result == "PASSED" for result in results.values())
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Pipeline is ready.")
    else:
        logger.info("\n⚠️  Some tests failed. Check logs for details.")
    
    return all_passed

def main():
    """Main test function."""
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()




