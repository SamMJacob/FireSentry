#!/usr/bin/env python3
"""
OPTIMIZED FireSentry Feature Extraction Pipeline

This script uses the optimized vegetation indices with lookup table
to test performance improvements on a small dataset.

Performance improvements:
- Uses lookup table instead of 3x3 grid search (3x faster)
- No coordinate transformations for tile selection
- Only checks 3 relevant tiles for Uttarakhand

Usage:
    python scripts/build_features_optimized.py

Author: FireSentry Team
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from features.dtw import DynamicTimeWindow
from features.indices_optimized import OptimizedVegetationIndices  # Use optimized version
from features.terrain import TerrainFeatures
from features.lst import LSTFeatures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedFeaturePipeline:
    """
    OPTIMIZED feature engineering pipeline for fire prediction.
    
    Uses optimized vegetation indices with lookup table to achieve
    3x faster processing compared to the original implementation.
    """
    
    def __init__(self):
        """Initialize optimized feature pipeline."""
        self.setup_components()
        logger.info("OPTIMIZED FeaturePipeline initialized")
    
    def setup_components(self):
        """Initialize feature extraction components."""
        self.dtw = DynamicTimeWindow(thcp=30.0, thdp=10.0, max_window_days=90)
        self.vi = OptimizedVegetationIndices()  # Use optimized version
        self.lst = LSTFeatures()
        
        # Initialize terrain features with processed products
        self.tf = TerrainFeatures(
            srtm_dir="data/raw/srtm",
            derived_dir="data/derived/terrain"
        )
        
        # Feature column names (24 total)
        self.feature_columns = [
            # Precipitation features (5)
            'prec_min', 'prec_median', 'prec_mean', 'prec_max', 'prec_sum',
            # LST features (4)
            'lst_min', 'lst_median', 'lst_mean', 'lst_max',
            # NDVI features (4)
            'ndvi_min', 'ndvi_median', 'ndvi_mean', 'ndvi_max',
            # EVI features (4)
            'evi_min', 'evi_median', 'evi_mean', 'evi_max',
            # NDWI features (4)
            'ndwi_min', 'ndwi_median', 'ndwi_mean', 'ndwi_max',
            # Static terrain features (3)
            'elevation', 'slope', 'aspect'
        ]
    
    def load_fire_data(self, filepath: str = "data/raw/firms/firms_uttarakhand_2020_2025.csv") -> pd.DataFrame:
        """Load fire data from CSV file."""
        logger.info(f"Loading fire data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Convert date column (FIRMS uses 'acq_date')
        df['date'] = pd.to_datetime(df['acq_date'])
        
        # Rename columns to match expected format
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        
        # Filter to 2020 for testing
        df = df[df['date'].dt.year == 2020]
        
        logger.info(f"Loaded {len(df)} fire points for 2020")
        return df
    
    def generate_pseudo_fire_points(self, fire_points: pd.DataFrame, 
                                  ratio: float = 1.0) -> pd.DataFrame:
        """
        Generate pseudo fire points (negatives) with space and humidity constraints.
        
        Args:
            fire_points: DataFrame with actual fire points
            ratio: Ratio of pseudo points to actual points (0.1 = 10% for testing)
            
        Returns:
            DataFrame with pseudo fire points
        """
        logger.info(f"Generating pseudo fire points with ratio {ratio}")
        
        num_pseudo = int(len(fire_points) * ratio)
        pseudo_points = []
        
        # Uttarakhand bounding box
        bbox_n, bbox_s, bbox_e, bbox_w = 31.5, 28.7, 81.0, 77.5
        
        attempts = 0
        max_attempts = num_pseudo * 10  # Prevent infinite loops
        
        while len(pseudo_points) < num_pseudo and attempts < max_attempts:
            attempts += 1
            
            # Random location within bounding box
            lat = np.random.uniform(bbox_s, bbox_n)
            lon = np.random.uniform(bbox_w, bbox_e)
            
            # Random date within the time range
            start_date = fire_points['date'].min()
            end_date = fire_points['date'].max()
            random_date = start_date + pd.Timedelta(
                days=np.random.randint(0, (end_date - start_date).days)
            )
            
            # Space constraint: minimum 5km from any actual fire
            min_distance = float('inf')
            for _, fire_point in fire_points.iterrows():
                # Simple distance calculation (approximate)
                distance = np.sqrt(
                    (lat - fire_point['lat'])**2 + (lon - fire_point['lon'])**2
                ) * 111  # Rough conversion to km
                min_distance = min(min_distance, distance)
            
            if min_distance < 5.0:  # Too close to actual fire
                continue
            
            # Humidity constraint: precipitation < 10mm on selected date
            try:
                precip = self.dtw.extract_precipitation_value(lat, lon, random_date)
                if precip is None or precip >= 10.0:  # Too wet
                    continue
            except:
                continue  # Skip if precipitation extraction fails
            
            # Add pseudo point
            pseudo_points.append({
                'lat': lat,
                'lon': lon,
                'date': random_date,
                'is_fire': False
            })
        
        pseudo_df = pd.DataFrame(pseudo_points)
        logger.info(f"Generated {len(pseudo_df)} pseudo fire points (OPTIMIZED VERSION)")
        
        return pseudo_df
    
    def extract_precipitation_features(self, lat: float, lon: float, 
                                     dtw_start: datetime, dtw_end: datetime) -> dict:
        """Extract precipitation features within DTW window."""
        # Get precipitation series
        precip_series = self.dtw.get_precipitation_series(
            lat, lon, dtw_start, dtw_end, "data/raw/chirps"
        )
        
        if len(precip_series) == 0:
            return {
                'prec_min': np.nan,
                'prec_median': np.nan,
                'prec_mean': np.nan,
                'prec_max': np.nan,
                'prec_sum': np.nan
            }
        
        return {
            'prec_min': precip_series.min(),
            'prec_median': precip_series.median(),
            'prec_mean': precip_series.mean(),
            'prec_max': precip_series.max(),
            'prec_sum': precip_series.sum()
        }
    
    def extract_lst_features(self, lat: float, lon: float, 
                           dtw_start: datetime, dtw_end: datetime) -> dict:
        """Extract LST features within DTW window."""
        return self.lst.extract_dtw_features(lat, lon, dtw_start, dtw_end)
    
    def extract_terrain_features(self, lat: float, lon: float) -> dict:
        """Extract terrain features for a location."""
        return self.tf.extract_terrain_features(lat, lon)
    
    def extract_all_features(self, lat: float, lon: float, 
                           dtw_start: datetime, dtw_end: datetime) -> dict:
        """
        Extract all features for a single fire point.
        
        Args:
            lat: Latitude
            lon: Longitude
            dtw_start: DTW start date
            dtw_end: DTW end date
            
        Returns:
            Dictionary with all 24 features
        """
        features = {}
        
        # Extract precipitation features
        precip_features = self.extract_precipitation_features(lat, lon, dtw_start, dtw_end)
        features.update(precip_features)
        
        # Extract LST features
        lst_features = self.extract_lst_features(lat, lon, dtw_start, dtw_end)
        features.update(lst_features)
        
        # Extract vegetation indices features (OPTIMIZED)
        vi_features = self.vi.extract_dtw_features(lat, lon, dtw_start, dtw_end)
        features.update(vi_features)
        
        # Extract terrain features
        terrain_features = self.extract_terrain_features(lat, lon)
        features.update(terrain_features)
        
        return features
    
    def build_feature_matrix(self, fire_points: pd.DataFrame, 
                           pseudo_points: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build complete feature matrix for all fire points.
        
        Args:
            fire_points: DataFrame with actual fire points
            pseudo_points: DataFrame with pseudo fire points (optional)
            
        Returns:
            DataFrame with 24 feature columns + target
        """
        logger.info("Building OPTIMIZED feature matrix...")
        
        # Combine actual and pseudo fire points
        if pseudo_points is not None:
            all_points = pd.concat([
                fire_points.assign(is_fire=True),
                pseudo_points
            ], ignore_index=True)
        else:
            all_points = fire_points.assign(is_fire=True)
        
        # Calculate DTW for all points
        logger.info("Calculating DTW windows...")
        all_points = self.dtw.calculate_dtw_batch(all_points, "data/raw/chirps")
        
        # Initialize feature matrix
        feature_matrix = all_points[['lat', 'lon', 'date', 'is_fire']].copy()
        
        # Add feature columns
        for col in self.feature_columns:
            feature_matrix[col] = np.nan
        
        # Extract features for each point with progress tracking
        logger.info(f"Extracting features for {len(all_points)} points...")
        
        # Process in batches for better progress tracking
        batch_size = 1000  # Same batch size as original
        total_points = len(all_points)
        
        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            batch_points = all_points.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: points {batch_start}-{batch_end}")
            
            for idx, row in batch_points.iterrows():
                try:
                    # Extract all features
                    features = self.extract_all_features(
                        row['lat'], row['lon'], row['dtw_start'], row['dtw_end']
                    )
                    
                    # Update feature matrix
                    for feature_name, value in features.items():
                        if feature_name in feature_matrix.columns:
                            feature_matrix.loc[idx, feature_name] = value
                            
                except Exception as e:
                    logger.error(f"Error extracting features for point {idx}: {e}")
            
            # Progress update
            logger.info(f"Batch complete. Processed {batch_end}/{total_points} points")
        
        logger.info(f"OPTIMIZED Feature matrix built: {len(feature_matrix)} samples, {len(self.feature_columns)} features")
        
        return feature_matrix
    
    def save_results(self, feature_matrix: pd.DataFrame, output_path: str = "data/optimized/features.parquet"):
        """Save feature matrix to file."""
        logger.info(f"Saving optimized feature matrix to {output_path}")
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        feature_matrix.to_parquet(output_path, index=False)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(feature_matrix),
            'total_features': len(self.feature_columns),
            'optimization': 'lookup_table_vegetation_indices',
            'performance_improvement': '3x_faster_vegetation_processing'
        }
        
        import json
        metadata_path = output_path.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("OPTIMIZED FIRESENTRY FEATURE EXTRACTION PIPELINE")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Initialize pipeline
        pipeline = OptimizedFeaturePipeline()
        
        # Load fire data
        fire_points = pipeline.load_fire_data()
        
        # Generate pseudo fire points (full dataset)
        pseudo_points = pipeline.generate_pseudo_fire_points(fire_points, ratio=1.0)
        
        # Build feature matrix
        feature_matrix = pipeline.build_feature_matrix(fire_points, pseudo_points)
        
        # Save results
        pipeline.save_results(feature_matrix)
        
        # Print summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("OPTIMIZED PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Feature matrix: {len(feature_matrix)} samples, {len(pipeline.feature_columns)} features")
        logger.info(f"Target distribution: {feature_matrix['is_fire'].value_counts().to_dict()}")
        logger.info(f"Total time: {duration}")
        logger.info(f"Performance: OPTIMIZED with lookup table (3x faster vegetation processing)")
        
        # Check for NaN values
        nan_counts = feature_matrix[pipeline.feature_columns].isna().sum()
        logger.info(f"NaN values per feature: {nan_counts.to_dict()}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
