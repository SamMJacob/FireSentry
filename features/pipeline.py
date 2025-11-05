"""
Feature Engineering Pipeline

Main pipeline that combines DTW calculation, vegetation indices, terrain features,
and precipitation/LST features to create the 24-dimensional feature matrix
as specified in the base paper.

Author: FireSentry Team
"""

import numpy as np
import pandas as pd
import rasterio
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Tuple, Optional, List
import geopandas as gpd
from shapely.geometry import Point
import json
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.dtw import DynamicTimeWindow
from features.indices import VegetationIndices
from features.terrain import TerrainFeatures
from features.lst import LSTFeatures

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    Main feature engineering pipeline for fire prediction.
    
    Combines all feature extraction components to create the 24-dimensional
    feature matrix: 3 static (elevation, slope, aspect) + 21 dynamic features
    (precipitation, LST, NDVI, EVI, NDWI statistics within DTW windows).
    """
    
    def __init__(self, config_path: str = "env.example"):
        """
        Initialize feature pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.load_config(config_path)
        self.setup_components()
        
        logger.info("FeaturePipeline initialized")
    
    def load_config(self, config_path: str):
        """Load configuration from environment file."""
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        config[key] = value
        
        # Default values
        self.bbox_n = float(config.get('BBOX_N', 31.5))
        self.bbox_s = float(config.get('BBOX_S', 28.7))
        self.bbox_e = float(config.get('BBOX_E', 81.0))
        self.bbox_w = float(config.get('BBOX_W', 77.5))
        
        logger.info(f"Loaded config: bbox=({self.bbox_w}, {self.bbox_s}, {self.bbox_e}, {self.bbox_n})")
    
    def setup_components(self):
        """Initialize feature extraction components."""
        self.dtw = DynamicTimeWindow(thcp=30.0, thdp=10.0, max_window_days=90)
        self.vi = VegetationIndices()
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
    
    def load_fire_data(self, firms_file: str = "data/raw/firms/firms_uttarakhand_2020_2025.csv") -> pd.DataFrame:
        """
        Load and preprocess FIRMS fire data.
        
        Args:
            firms_file: Path to FIRMS CSV file
            
        Returns:
            DataFrame with fire points
        """
        logger.info(f"Loading fire data from {firms_file}")
        
        if not Path(firms_file).exists():
            logger.error(f"FIRMS file not found: {firms_file}")
            return pd.DataFrame()
        
        # Load FIRMS data
        df = pd.read_csv(firms_file)
        
        # Filter to Uttarakhand bounding box
        mask = (
            (df['latitude'] >= self.bbox_s) & (df['latitude'] <= self.bbox_n) &
            (df['longitude'] >= self.bbox_w) & (df['longitude'] <= self.bbox_e)
        )
        df = df[mask].copy()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['acq_date'])
        
        # Select required columns
        fire_points = df[['latitude', 'longitude', 'date']].copy()
        fire_points.columns = ['lat', 'lon', 'date']
        
        # Remove duplicates
        fire_points = fire_points.drop_duplicates().reset_index(drop=True)
        
        logger.info(f"Loaded {len(fire_points)} fire points")
        return fire_points
    
    def extract_precipitation_features(self, lat: float, lon: float, 
                                     dtw_start: datetime, dtw_end: datetime) -> dict:
        """
        Extract precipitation features within DTW window.
        
        Args:
            lat: Latitude
            lon: Longitude
            dtw_start: DTW start date
            dtw_end: DTW end date
            
        Returns:
            Dictionary with precipitation features
        """
        try:
            # Get precipitation time series
            precip_series = self.dtw.get_precipitation_series(
                lat, lon, dtw_start, dtw_end, "data/raw/chirps"
            )
            
            if len(precip_series) == 0:
                return {f'prec_{stat}': np.nan for stat in ['min', 'median', 'mean', 'max', 'sum']}
            
            # Calculate statistics
            features = {
                'prec_min': float(precip_series.min()),
                'prec_median': float(precip_series.median()),
                'prec_mean': float(precip_series.mean()),
                'prec_max': float(precip_series.max()),
                'prec_sum': float(precip_series.sum())
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting precipitation features: {e}")
            return {f'prec_{stat}': np.nan for stat in ['min', 'median', 'mean', 'max', 'sum']}
    
    def extract_lst_features(self, lat: float, lon: float, 
                           dtw_start: datetime, dtw_end: datetime) -> dict:
        """
        Extract LST features within DTW window.
        
        Args:
            lat: Latitude
            lon: Longitude
            dtw_start: DTW start date
            dtw_end: DTW end date
            
        Returns:
            Dictionary with LST features
        """
        try:
            # Extract LST features using the LST features class
            lst_features = self.lst.extract_dtw_features(lat, lon, dtw_start, dtw_end)
            return lst_features
            
        except Exception as e:
            logger.error(f"Error extracting LST features: {e}")
            return {f'lst_{stat}': np.nan for stat in ['min', 'median', 'mean', 'max']}
    
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
        
        # Extract vegetation indices features
        vi_features = self.vi.extract_dtw_features(lat, lon, dtw_start, dtw_end)
        features.update(vi_features)
        
        # Extract terrain features
        terrain_features = self.tf.extract_terrain_features(lat, lon)
        features.update(terrain_features)
        
        return features
    
    def generate_pseudo_fire_points(self, fire_points: pd.DataFrame, 
                                  ratio: float = 1.0) -> pd.DataFrame:
        """
        Generate pseudo fire points (negatives) with space and humidity constraints.
        
        Args:
            fire_points: DataFrame with actual fire points
            ratio: Ratio of pseudo points to actual points (1.0 = 1:1)
            
        Returns:
            DataFrame with pseudo fire points
        """
        logger.info(f"Generating pseudo fire points with ratio {ratio}")
        
        num_pseudo = int(len(fire_points) * ratio)
        pseudo_points = []
        
        # Create grid of candidate locations within bounding box
        lat_step = (self.bbox_n - self.bbox_s) / 50  # 50x50 grid
        lon_step = (self.bbox_e - self.bbox_w) / 50
        
        attempts = 0
        max_attempts = num_pseudo * 10  # Prevent infinite loops
        
        while len(pseudo_points) < num_pseudo and attempts < max_attempts:
            attempts += 1
            
            # Random location within bounding box
            lat = np.random.uniform(self.bbox_s, self.bbox_n)
            lon = np.random.uniform(self.bbox_w, self.bbox_e)
            
            # Random date within the time range
            start_date = fire_points['date'].min()
            end_date = fire_points['date'].max()
            random_date = start_date + timedelta(
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
        logger.info(f"Generated {len(pseudo_df)} pseudo fire points")
        
        return pseudo_df
    
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
        logger.info("Building feature matrix...")
        
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
        batch_size = 1000
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
        
        # Create target column
        feature_matrix['target'] = feature_matrix['is_fire'].astype(int)
        
        # Remove rows with too many NaN values
        nan_threshold = len(self.feature_columns) * 0.5  # Allow 50% NaN
        valid_mask = feature_matrix[self.feature_columns].isnull().sum(axis=1) <= nan_threshold
        feature_matrix = feature_matrix[valid_mask].reset_index(drop=True)
        
        logger.info(f"Feature matrix built: {len(feature_matrix)} samples, {len(self.feature_columns)} features")
        
        return feature_matrix
    
    def save_feature_matrix(self, feature_matrix: pd.DataFrame, 
                          output_file: str = "data/features.parquet"):
        """
        Save feature matrix to Parquet file.
        
        Args:
            feature_matrix: Feature matrix DataFrame
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to Parquet
        feature_matrix.to_parquet(output_path, index=False)
        
        # Save metadata
        metadata = {
            'num_samples': len(feature_matrix),
            'num_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'target_distribution': feature_matrix['target'].value_counts().to_dict(),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_file = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature matrix saved to {output_path}")
        logger.info(f"Metadata saved to {metadata_file}")
        
        return output_path, metadata_file

def main():
    """Example usage of feature pipeline."""
    # Initialize pipeline
    pipeline = FeaturePipeline()
    
    # Load fire data
    fire_points = pipeline.load_fire_data()
    
    if len(fire_points) == 0:
        logger.error("No fire data loaded. Please check FIRMS file.")
        return
    
    # Generate pseudo fire points
    pseudo_points = pipeline.generate_pseudo_fire_points(fire_points, ratio=1.0)
    
    # Build feature matrix
    feature_matrix = pipeline.build_feature_matrix(fire_points, pseudo_points)
    
    # Save feature matrix
    output_file, metadata_file = pipeline.save_feature_matrix(feature_matrix)
    
    print(f"Feature matrix created with {len(feature_matrix)} samples")
    print(f"Target distribution: {feature_matrix['target'].value_counts().to_dict()}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()

