"""
Vegetation Indices Computation

Computes NDVI, EVI, and NDWI from MODIS Surface Reflectance bands.
Implements the vegetation index calculations as specified in the base paper.

Formulas:
- NDVI = (NIR - Red) / (NIR + Red)
- EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
- NDWI = (NIR - SWIR1) / (NIR + SWIR1)

Author: FireSentry Team
"""

import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Tuple, Optional, List
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

class VegetationIndices:
    """
    Vegetation indices calculator for MODIS Surface Reflectance data.
    
    Computes NDVI, EVI, and NDWI from MODIS MYD09GA bands and extracts
    statistical features (min, median, mean, max) within DTW windows.
    """
    
    def __init__(self, modis_sr_dir: str = "data/raw/modis_sr"):
        """
        Initialize vegetation indices calculator.
        
        Args:
            modis_sr_dir: Directory containing MODIS Surface Reflectance HDF files
        """
        self.modis_sr_dir = Path(modis_sr_dir)
        self._tile_cache = {}  # Cache tile mappings for faster processing
        
        # MODIS band mappings for MYD09GA
        self.band_mappings = {
            'red': 'sur_refl_b01',      # Band 1: 620-670nm
            'nir': 'sur_refl_b02',      # Band 2: 841-876nm  
            'blue': 'sur_refl_b03',     # Band 3: 459-479nm
            'swir1': 'sur_refl_b06',    # Band 6: 1628-1652nm
            'swir2': 'sur_refl_b07'     # Band 7: 2105-2155nm
        }
        
        logger.info(f"VegetationIndices initialized with MODIS SR directory: {self.modis_sr_dir}")
    
    def get_modis_tile_cached(self, lat: float, lon: float) -> str:
        """Get MODIS tile for coordinates (cached for performance)."""
        key = (round(lat, 4), round(lon, 4))
        if key not in self._tile_cache:
            # Simple tile calculation (approximate)
            h = int((lon + 180) / 10)
            v = int((90 - lat) / 10)
            self._tile_cache[key] = f"h{h:02d}v{v:02d}"
        return self._tile_cache[key]
    
    def find_modis_sr_band_files(self, lat: float, lon: float, date: datetime) -> dict:
        """
        Find MODIS Surface Reflectance band files for given location and date.
        
        Args:
            lat: Latitude
            lon: Longitude  
            date: Date to find files for
            
        Returns:
            Dictionary mapping band names to file paths, or empty dict if not found
        """
        # Calculate base tile (cached for performance)
        base_tile = self.get_modis_tile_cached(lat, lon)
        
        # Extract h and v from base tile
        h_base = int(base_tile[1:3])
        v_base = int(base_tile[4:6])
        
        # Search in a 3x3 grid around the base tile to handle boundary cases
        year = date.year
        doy = date.timetuple().tm_yday
        
        # Search in year directory
        year_dir = self.modis_sr_dir / str(year)
        if not year_dir.exists():
            return {}
        
        # Try tiles in a 3x3 grid around the calculated tile
        for h_offset in [-1, 0, 1]:
            for v_offset in [-1, 0, 1]:
                h = h_base + h_offset
                v = v_base + v_offset
                
                # Skip invalid tiles
                if h < 0 or h > 35 or v < 0 or v > 17:
                    continue
                
                tile = f"h{h:02d}v{v:02d}"
                
                # Check if any band files exist for this tile
                band_files = {}
                for band_name, band_key in self.band_mappings.items():
                    # Map band names to B01, B02, etc.
                    band_mapping = {
                        'red': 'B01',
                        'nir': 'B02', 
                        'blue': 'B03',
                        'swir1': 'B06',
                        'swir2': 'B07'
                    }
                    
                    band_file = f"MYD09GA.AA{year}{doy:03d}.{tile}.061.{band_mapping[band_name]}.tif"
                    filepath = year_dir / band_file
                    
                    if filepath.exists():
                        # Verify the point is actually in this tile
                        try:
                            with rasterio.open(filepath) as src:
                                from rasterio.warp import transform
                                from rasterio.crs import CRS
                                
                                # Transform coordinates
                                x, y = transform(
                                    CRS.from_epsg(4326),  # WGS84
                                    src.crs,              # MODIS Sinusoidal
                                    [lon], [lat]
                                )
                                
                                # Check if point is within bounds
                                bounds = src.bounds
                                if (bounds.left <= x[0] <= bounds.right and 
                                    bounds.bottom <= y[0] <= bounds.top):
                                    band_files[band_name] = filepath
                        except Exception:
                            continue
                
                # If we found all required bands for this tile, return them
                if len(band_files) >= 4:  # Need at least red, nir, blue, swir1
                    return band_files
        
        return {}
    
    def get_modis_tile(self, lat: float, lon: float) -> str:
        """
        Get MODIS tile identifier for given coordinates.
        
        Note: This uses a simple calculation that is approximate for
        MODIS Sinusoidal projection. The actual tile search uses a 3x3
        grid with point containment checks for accuracy.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            MODIS tile identifier (approximate, used as starting point for grid search)
        """
        # Simple MODIS tile calculation (approximate)
        # Due to Sinusoidal projection distortion, actual tiles may differ
        h = int((lon + 180) / 10)
        v = int((90 - lat) / 10)
        
        return f"h{h:02d}v{v:02d}"
    
    def extract_modis_bands(self, band_files: dict, lat: float, lon: float) -> dict:
        """
        Extract MODIS bands for a specific location from TIF files.
        
        Args:
            band_files: Dictionary mapping band names to file paths
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with band values
        """
        bands = {}
        
        try:
            for band_name, file_path in band_files.items():
                with rasterio.open(file_path) as src:
                    # Transform coordinates from WGS84 to raster CRS
                    from rasterio.warp import transform
                    from rasterio.crs import CRS
                    
                    # Transform from WGS84 to raster CRS
                    x, y = transform(
                        CRS.from_epsg(4326),  # WGS84
                        src.crs,              # MODIS Sinusoidal
                        [lon], [lat]
                    )
                    
                    # Get row, col for the transformed point
                    row, col = src.index(x[0], y[0])
                    
                    # Check bounds
                    if 0 <= row < src.height and 0 <= col < src.width:
                        # Read the value
                        value = src.read(1)[row, col]
                        
                        # Handle NoData values
                        if value == src.nodata or np.isnan(value):
                            bands[band_name] = np.nan
                        else:
                            # Scale factor for MODIS (0-10000 -> 0-1)
                            bands[band_name] = value / 10000.0
                    else:
                        bands[band_name] = np.nan
                        
            return bands
            
        except Exception as e:
            logger.error(f"Error extracting MODIS bands: {e}")
            return {band: np.nan for band in self.band_mappings.keys()}
    
    def calculate_ndvi(self, red: float, nir: float) -> float:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        Args:
            red: Red band reflectance (0-1)
            nir: NIR band reflectance (0-1)
            
        Returns:
            NDVI value (-1 to 1)
        """
        if np.isnan(red) or np.isnan(nir) or (nir + red) == 0:
            return np.nan
        
        return (nir - red) / (nir + red)
    
    def calculate_evi(self, red: float, nir: float, blue: float) -> float:
        """
        Calculate Enhanced Vegetation Index (EVI).
        
        Args:
            red: Red band reflectance (0-1)
            nir: NIR band reflectance (0-1)
            blue: Blue band reflectance (0-1)
            
        Returns:
            EVI value
        """
        if np.isnan(red) or np.isnan(nir) or np.isnan(blue):
            return np.nan
        
        denominator = nir + 6 * red - 7.5 * blue + 1
        if denominator == 0:
            return np.nan
        
        return 2.5 * (nir - red) / denominator
    
    def calculate_ndwi(self, nir: float, swir1: float) -> float:
        """
        Calculate Normalized Difference Water Index (NDWI).
        
        Args:
            nir: NIR band reflectance (0-1)
            swir1: SWIR1 band reflectance (0-1)
            
        Returns:
            NDWI value (-1 to 1)
        """
        if np.isnan(nir) or np.isnan(swir1) or (nir + swir1) == 0:
            return np.nan
        
        return (nir - swir1) / (nir + swir1)
    
    def calculate_indices_for_date(self, lat: float, lon: float, date: datetime) -> dict:
        """
        Calculate all vegetation indices for a specific location and date.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date
            
        Returns:
            Dictionary with NDVI, EVI, NDWI values
        """
        # Find MODIS SR band files
        band_files = self.find_modis_sr_band_files(lat, lon, date)
        if not band_files:
            logger.warning(f"No MODIS SR files found for ({lat}, {lon}, {date.date()})")
            return {'ndvi': np.nan, 'evi': np.nan, 'ndwi': np.nan}
        
        # Extract bands
        bands = self.extract_modis_bands(band_files, lat, lon)
        
        # Calculate indices
        ndvi = self.calculate_ndvi(bands['red'], bands['nir'])
        evi = self.calculate_evi(bands['red'], bands['nir'], bands['blue'])
        ndwi = self.calculate_ndwi(bands['nir'], bands['swir1'])
        
        return {
            'ndvi': ndvi,
            'evi': evi,
            'ndwi': ndwi,
            'bands': bands  # Include raw band values for debugging
        }
    
    def calculate_indices_series(self, lat: float, lon: float, 
                               start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Calculate vegetation indices time series for a location and date range.
        OPTIMIZED: Sample every 3 days instead of daily to reduce processing time.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with date index and vegetation indices
        """
        # OPTIMIZATION: Sample every 3 days instead of daily (3x faster)
        dates = pd.date_range(start_date, end_date, freq='3D')
        indices_data = []
        
        for date in dates:
            indices = self.calculate_indices_for_date(lat, lon, date)
            indices_data.append({
                'date': date,
                'ndvi': indices['ndvi'],
                'evi': indices['evi'],
                'ndwi': indices['ndwi']
            })
        
        df = pd.DataFrame(indices_data)
        df.set_index('date', inplace=True)
        
        return df
    
    def extract_dtw_features(self, lat: float, lon: float, 
                           dtw_start: datetime, dtw_end: datetime) -> dict:
        """
        Extract vegetation index features within DTW window.
        
        Args:
            lat: Latitude
            lon: Longitude
            dtw_start: DTW start date
            dtw_end: DTW end date
            
        Returns:
            Dictionary with statistical features for each index
        """
        # Get time series
        series_df = self.calculate_indices_series(lat, lon, dtw_start, dtw_end)
        
        features = {}
        
        for index_name in ['ndvi', 'evi', 'ndwi']:
            if index_name in series_df.columns:
                values = series_df[index_name].dropna()
                
                if len(values) > 0:
                    features[f'{index_name}_min'] = float(values.min())
                    features[f'{index_name}_median'] = float(values.median())
                    features[f'{index_name}_mean'] = float(values.mean())
                    features[f'{index_name}_max'] = float(values.max())
                else:
                    # All NaN values
                    features[f'{index_name}_min'] = np.nan
                    features[f'{index_name}_median'] = np.nan
                    features[f'{index_name}_mean'] = np.nan
                    features[f'{index_name}_max'] = np.nan
            else:
                features[f'{index_name}_min'] = np.nan
                features[f'{index_name}_median'] = np.nan
                features[f'{index_name}_mean'] = np.nan
                features[f'{index_name}_max'] = np.nan
        
        return features
    
    def extract_dtw_features_batch(self, fire_points: pd.DataFrame) -> pd.DataFrame:
        """
        Extract vegetation index features for multiple fire points.
        
        Args:
            fire_points: DataFrame with columns ['lat', 'lon', 'dtw_start', 'dtw_end']
            
        Returns:
            DataFrame with added vegetation index feature columns
        """
        results = fire_points.copy()
        
        # Initialize feature columns
        feature_columns = []
        for index_name in ['ndvi', 'evi', 'ndwi']:
            for stat in ['min', 'median', 'mean', 'max']:
                col_name = f'{index_name}_{stat}'
                feature_columns.append(col_name)
                results[col_name] = np.nan
        
        logger.info(f"Extracting vegetation indices for {len(fire_points)} fire points")
        
        for idx, row in fire_points.iterrows():
            try:
                # Convert dates if needed
                if isinstance(row['dtw_start'], str):
                    dtw_start = pd.to_datetime(row['dtw_start'])
                else:
                    dtw_start = row['dtw_start']
                
                if isinstance(row['dtw_end'], str):
                    dtw_end = pd.to_datetime(row['dtw_end'])
                else:
                    dtw_end = row['dtw_end']
                
                # Extract features
                features = self.extract_dtw_features(
                    row['lat'], row['lon'], dtw_start, dtw_end
                )
                
                # Update results
                for col_name in feature_columns:
                    results.loc[idx, col_name] = features[col_name]
                
            except Exception as e:
                logger.error(f"Error extracting vegetation indices for point {idx}: {e}")
        
        return results

def main():
    """Example usage of vegetation indices calculator."""
    # Initialize calculator
    vi = VegetationIndices()
    
    # Example location and date
    lat, lon = 30.0, 79.0
    date = datetime(2024, 4, 15)
    
    # Calculate indices for single date
    indices = vi.calculate_indices_for_date(lat, lon, date)
    print(f"Vegetation indices for ({lat}, {lon}) on {date.date()}:")
    print(f"NDVI: {indices['ndvi']:.3f}")
    print(f"EVI: {indices['evi']:.3f}")
    print(f"NDWI: {indices['ndwi']:.3f}")
    
    # Calculate time series
    start_date = datetime(2024, 4, 1)
    end_date = datetime(2024, 4, 30)
    series = vi.calculate_indices_series(lat, lon, start_date, end_date)
    print(f"\nTime series statistics:")
    print(series.describe())

if __name__ == "__main__":
    main()


