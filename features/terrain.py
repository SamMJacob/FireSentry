"""
Terrain Features Extraction

Extracts elevation, slope, and aspect from SRTM DEM data for fire points.
Computes terrain features using GDAL operations.

Author: FireSentry Team
"""

import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point, box
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Optional, List
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

class TerrainFeatures:
    """
    Terrain features extractor for SRTM DEM data.
    
    Extracts elevation, slope, and aspect for fire points from SRTM DEM
    and derived terrain products (slope, aspect).
    """
    
    def __init__(self, srtm_dir: str = "data/raw/srtm", 
                 derived_dir: str = "data/derived/terrain"):
        """
        Initialize terrain features extractor.
        
        Args:
            srtm_dir: Directory containing SRTM DEM files
            derived_dir: Directory for derived terrain products (slope, aspect)
        """
        self.srtm_dir = Path(srtm_dir)
        self.derived_dir = Path(derived_dir)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        
        # Terrain product filenames
        self.elevation_file = self.derived_dir / "elevation.tif"
        self.slope_file = self.derived_dir / "slope.tif"
        self.aspect_file = self.derived_dir / "aspect.tif"
        
        logger.info(f"TerrainFeatures initialized with SRTM directory: {self.srtm_dir}")
        logger.info(f"Derived products directory: {self.derived_dir}")
    
    def find_srtm_files(self) -> List[Path]:
        """
        Find all SRTM files in the SRTM directory.
        
        Returns:
            List of SRTM file paths
        """
        srtm_files = []
        
        if self.srtm_dir.exists():
            # Look for common SRTM file extensions
            for ext in ['*.tif', '*.tiff', '*.hgt', '*.dem']:
                srtm_files.extend(self.srtm_dir.glob(ext))
                srtm_files.extend(self.srtm_dir.glob(f"**/{ext}"))
        
        logger.info(f"Found {len(srtm_files)} SRTM files")
        return srtm_files
    
    def mosaic_srtm_files(self, output_file: Path, bbox: Tuple[float, float, float, float] = None):
        """
        Mosaic SRTM files into a single GeoTIFF.
        
        Args:
            output_file: Output mosaic file path
            bbox: Optional bounding box (west, south, east, north) for clipping
        """
        srtm_files = self.find_srtm_files()
        
        if not srtm_files:
            logger.error("No SRTM files found")
            return False
        
        try:
            if len(srtm_files) == 1:
                # Single file - just copy if needed
                if srtm_files[0] != output_file:
                    subprocess.run(['cp', str(srtm_files[0]), str(output_file)], check=True)
            else:
                # Multiple files - mosaic them
                cmd = ['gdal_merge.py', '-o', str(output_file)]
                
                if bbox:
                    cmd.extend(['-ul_lr', str(bbox[0]), str(bbox[3]), str(bbox[2]), str(bbox[1])])
                
                cmd.extend([str(f) for f in srtm_files])
                
                subprocess.run(cmd, check=True, capture_output=True)
            
            logger.info(f"SRTM mosaic created: {output_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating SRTM mosaic: {e}")
            return False
    
    def compute_slope(self, elevation_file: Path, output_file: Path):
        """
        Compute slope from elevation using GDAL.
        
        Args:
            elevation_file: Input elevation file
            output_file: Output slope file
        """
        try:
            cmd = [
                'gdaldem', 'slope',
                str(elevation_file),
                str(output_file),
                '-of', 'GTiff',
                '-co', 'COMPRESS=LZW'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Slope computed: {output_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error computing slope: {e}")
            return False
    
    def compute_aspect(self, elevation_file: Path, output_file: Path):
        """
        Compute aspect from elevation using GDAL.
        
        Args:
            elevation_file: Input elevation file
            output_file: Output aspect file
        """
        try:
            cmd = [
                'gdaldem', 'aspect',
                str(elevation_file),
                str(output_file),
                '-of', 'GTiff',
                '-co', 'COMPRESS=LZW'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Aspect computed: {output_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error computing aspect: {e}")
            return False
    
    def prepare_terrain_products(self, bbox: Tuple[float, float, float, float] = None):
        """
        Prepare all terrain products (elevation, slope, aspect).
        
        Args:
            bbox: Optional bounding box for clipping (west, south, east, north)
        """
        logger.info("Preparing terrain products...")
        
        # Create elevation mosaic if needed
        if not self.elevation_file.exists():
            if not self.mosaic_srtm_files(self.elevation_file, bbox):
                logger.error("Failed to create elevation mosaic")
                return False
        
        # Compute slope if needed
        if not self.slope_file.exists():
            if not self.compute_slope(self.elevation_file, self.slope_file):
                logger.error("Failed to compute slope")
                return False
        
        # Compute aspect if needed
        if not self.aspect_file.exists():
            if not self.compute_aspect(self.elevation_file, self.aspect_file):
                logger.error("Failed to compute aspect")
                return False
        
        logger.info("Terrain products prepared successfully")
        return True
    
    def extract_terrain_value(self, lat: float, lon: float, terrain_file: Path) -> Optional[float]:
        """
        Extract terrain value for a specific location.
        
        Args:
            lat: Latitude
            lon: Longitude
            terrain_file: Path to terrain raster file
            
        Returns:
            Terrain value or None if extraction fails
        """
        try:
            with rasterio.open(terrain_file) as src:
                # Get row, col for the point
                row, col = src.index(lon, lat)
                
                # Check bounds
                if 0 <= row < src.height and 0 <= col < src.width:
                    # Read the value
                    value = src.read(1)[row, col]
                    
                    # Handle NoData values
                    if value == src.nodata or np.isnan(value):
                        return None
                    
                    return float(value)
                else:
                    logger.warning(f"Point ({lat}, {lon}) outside raster bounds for {terrain_file}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error extracting terrain value from {terrain_file}: {e}")
            return None
    
    def extract_terrain_features(self, lat: float, lon: float) -> dict:
        """
        Extract all terrain features for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with elevation, slope, aspect values
        """
        features = {}
        
        # Extract elevation
        elevation = self.extract_terrain_value(lat, lon, self.elevation_file)
        features['elevation'] = elevation if elevation is not None else np.nan
        
        # Extract slope
        slope = self.extract_terrain_value(lat, lon, self.slope_file)
        features['slope'] = slope if slope is not None else np.nan
        
        # Extract aspect
        aspect = self.extract_terrain_value(lat, lon, self.aspect_file)
        features['aspect'] = aspect if aspect is not None else np.nan
        
        return features
    
    def extract_terrain_features_batch(self, fire_points: pd.DataFrame) -> pd.DataFrame:
        """
        Extract terrain features for multiple fire points.
        
        Args:
            fire_points: DataFrame with columns ['lat', 'lon']
            
        Returns:
            DataFrame with added terrain feature columns
        """
        results = fire_points.copy()
        
        # Initialize terrain feature columns
        terrain_columns = ['elevation', 'slope', 'aspect']
        for col in terrain_columns:
            results[col] = np.nan
        
        logger.info(f"Extracting terrain features for {len(fire_points)} fire points")
        
        for idx, row in fire_points.iterrows():
            try:
                # Extract features
                features = self.extract_terrain_features(row['lat'], row['lon'])
                
                # Update results
                for col in terrain_columns:
                    results.loc[idx, col] = features[col]
                
            except Exception as e:
                logger.error(f"Error extracting terrain features for point {idx}: {e}")
        
        return results
    
    def validate_terrain_products(self) -> dict:
        """
        Validate terrain products and return statistics.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'elevation_file_exists': self.elevation_file.exists(),
            'slope_file_exists': self.slope_file.exists(),
            'aspect_file_exists': self.aspect_file.exists(),
            'statistics': {}
        }
        
        # Get statistics for each product
        for product_name, file_path in [
            ('elevation', self.elevation_file),
            ('slope', self.slope_file),
            ('aspect', self.aspect_file)
        ]:
            if file_path.exists():
                try:
                    with rasterio.open(file_path) as src:
                        data = src.read(1)
                        valid_data = data[data != src.nodata]
                        
                        validation['statistics'][product_name] = {
                            'min': float(np.min(valid_data)),
                            'max': float(np.max(valid_data)),
                            'mean': float(np.mean(valid_data)),
                            'std': float(np.std(valid_data)),
                            'shape': data.shape,
                            'crs': str(src.crs),
                            'bounds': src.bounds
                        }
                except Exception as e:
                    logger.error(f"Error reading {product_name} statistics: {e}")
                    validation['statistics'][product_name] = {'error': str(e)}
            else:
                validation['statistics'][product_name] = {'error': 'File not found'}
        
        return validation

def main():
    """Example usage of terrain features extractor."""
    # Initialize extractor
    tf = TerrainFeatures()
    
    # Example location
    lat, lon = 30.0, 79.0
    
    # Prepare terrain products
    bbox = (77.5, 28.7, 81.0, 31.5)  # Uttarakhand bounding box
    if tf.prepare_terrain_products(bbox):
        print("Terrain products prepared successfully")
        
        # Extract features for single point
        features = tf.extract_terrain_features(lat, lon)
        print(f"Terrain features for ({lat}, {lon}):")
        print(f"Elevation: {features['elevation']:.1f} m")
        print(f"Slope: {features['slope']:.1f} degrees")
        print(f"Aspect: {features['aspect']:.1f} degrees")
        
        # Validate products
        validation = tf.validate_terrain_products()
        print(f"\nValidation results:")
        for product, stats in validation['statistics'].items():
            if 'error' not in stats:
                print(f"{product}: {stats['min']:.1f} to {stats['max']:.1f} "
                      f"(mean: {stats['mean']:.1f})")
    else:
        print("Failed to prepare terrain products")

if __name__ == "__main__":
    main()




