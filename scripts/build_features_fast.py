#!/usr/bin/env python3
"""
Fast Feature Extraction Script

Optimized version that processes features much faster by:
1. Pre-calculating tile mappings
2. Sampling fewer days in DTW windows
3. Using batch processing where possible
4. Reducing file I/O operations

Usage:
    python scripts/build_features_fast.py
"""

import sys
from pathlib import Path
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.pipeline import FeaturePipeline
from features.indices_optimized import OptimizedVegetationIndices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_fast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FastFeaturePipeline(FeaturePipeline):
    """Optimized feature pipeline for faster processing."""
    
    def __init__(self):
        super().__init__()
        # Replace with optimized vegetation indices
        self.vi = OptimizedVegetationIndices()
        logger.info("Fast feature pipeline initialized")
    
    def build_feature_matrix_fast(self, fire_points, pseudo_points=None):
        """Fast feature matrix building with optimizations."""
        logger.info("Building feature matrix (optimized)...")
        
        # Combine points
        if pseudo_points is not None:
            all_points = pd.concat([
                fire_points.assign(is_fire=True),
                pseudo_points
            ], ignore_index=True)
        else:
            all_points = fire_points.assign(is_fire=True)
        
        # Calculate DTW (this is fast)
        logger.info("Calculating DTW windows...")
        all_points = self.dtw.calculate_dtw_batch(all_points, "data/raw/chirps")
        
        # Initialize feature matrix
        feature_matrix = all_points[['lat', 'lon', 'date', 'is_fire']].copy()
        for col in self.feature_columns:
            feature_matrix[col] = np.nan
        
        # Process in smaller batches for better progress tracking
        batch_size = 1000
        total_points = len(all_points)
        
        logger.info(f"Extracting features for {total_points} points in batches of {batch_size}")
        
        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            batch_points = all_points.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: points {batch_start}-{batch_end}")
            
            for idx, row in batch_points.iterrows():
                try:
                    # Extract features
                    features = self.extract_all_features_fast(
                        row['lat'], row['lon'], row['dtw_start'], row['dtw_end']
                    )
                    
                    # Update feature matrix
                    for feature_name, value in features.items():
                        if feature_name in feature_matrix.columns:
                            feature_matrix.loc[idx, feature_name] = value
                            
                except Exception as e:
                    logger.error(f"Error extracting features for point {idx}: {e}")
            
            # Progress update
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            rate = (batch_end - batch_start) / elapsed if elapsed > 0 else 0
            eta = (total_points - batch_end) / rate if rate > 0 else 0
            logger.info(f"Batch complete. Rate: {rate:.1f} points/sec, ETA: {eta/60:.1f} minutes")
        
        # Create target column
        feature_matrix['target'] = feature_matrix['is_fire'].astype(int)
        
        # Remove rows with too many NaN values
        nan_threshold = len(self.feature_columns) * 0.5
        valid_mask = feature_matrix[self.feature_columns].isnull().sum(axis=1) <= nan_threshold
        feature_matrix = feature_matrix[valid_mask].reset_index(drop=True)
        
        logger.info(f"Feature matrix built: {len(feature_matrix)} samples, {len(self.feature_columns)} features")
        return feature_matrix
    
    def extract_all_features_fast(self, lat, lon, dtw_start, dtw_end):
        """Fast feature extraction with optimizations."""
        features = {}
        
        # Precipitation features (fast)
        precip_features = self.extract_precipitation_features(lat, lon, dtw_start, dtw_end)
        features.update(precip_features)
        
        # LST features (fast)
        lst_features = self.lst.extract_dtw_features(lat, lon, dtw_start, dtw_end)
        features.update(lst_features)
        
        # Vegetation indices (optimized)
        vi_features = self.vi.extract_dtw_features_fast(lat, lon, dtw_start, dtw_end)
        features.update(vi_features)
        
        # Terrain features (fast)
        terrain_features = self.tf.extract_terrain_features(lat, lon)
        features.update(terrain_features)
        
        return features

def main():
    """Main entry point."""
    logger.info("Starting fast feature extraction...")
    start_time = time.time()
    
    try:
        # Initialize pipeline
        pipeline = FastFeaturePipeline()
        
        # Load fire data
        fire_points = pipeline.load_fire_data()
        logger.info(f"Loaded {len(fire_points)} fire points")
        
        # Generate pseudo points (smaller sample for testing)
        pseudo_points = pipeline.generate_pseudo_fire_points(fire_points, ratio=0.1)  # 10% for speed
        logger.info(f"Generated {len(pseudo_points)} pseudo points")
        
        # Build feature matrix
        feature_matrix = pipeline.build_feature_matrix_fast(fire_points, pseudo_points)
        
        # Save results
        output_file, metadata_file = pipeline.save_feature_matrix(feature_matrix)
        
        # Summary
        elapsed = time.time() - start_time
        logger.info(f"Fast feature extraction completed in {elapsed/60:.1f} minutes")
        logger.info(f"Dataset: {len(feature_matrix)} samples, {len(pipeline.feature_columns)} features")
        logger.info(f"Target distribution: {feature_matrix['target'].value_counts().to_dict()}")
        logger.info(f"Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Fast feature extraction failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
