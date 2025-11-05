"""
LST Features Extraction

Extracts Land Surface Temperature (LST) features from MODIS LST data.
Computes LST statistics within DTW windows for fire prediction.

Author: FireSentry Team
"""

import numpy as np
import pandas as pd
import rasterio
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

class LSTFeatures:
    """
    LST features extractor for MODIS LST data.
    
    Extracts LST values from MODIS MOD11A1 LST TIF files and computes
    statistical features within DTW windows.
    """
    
    def __init__(self, modis_lst_dir: str = "data/raw/modis_lst"):
        """
        Initialize LST features extractor.
        
        Args:
            modis_lst_dir: Directory containing MODIS LST TIF files
        """
        self.modis_lst_dir = Path(modis_lst_dir)
        
        logger.info(f"LSTFeatures initialized with directory: {self.modis_lst_dir}")
    
    def get_modis_tile(self, lat: float, lon: float) -> str:
        """
        Get MODIS tile identifier for given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            MODIS tile identifier (e.g., 'h24v05')
        """
        # MODIS tile calculation
        # Tiles are 10° x 10° starting from (0,0) at (-180, 90)
        h = int((lon + 180) / 10)
        v = int((90 - lat) / 10)
        
        return f"h{h:02d}v{v:02d}"
    
    def find_modis_lst_file(self, lat: float, lon: float, date: datetime) -> Optional[Path]:
        """
        Find MODIS LST file for given location and date.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date
            
        Returns:
            Path to MODIS LST file or None if not found
        """
        # Calculate base tile
        base_tile = self.get_modis_tile(lat, lon)
        
        # Extract h and v from base tile
        h_base = int(base_tile[1:3])
        v_base = int(base_tile[4:6])
        
        # Search in a 3x3 grid around the base tile to handle boundary cases
        year = date.year
        doy = date.timetuple().tm_yday
        
        # Search in year directory
        year_dir = self.modis_lst_dir / str(year)
        if not year_dir.exists():
            return None
        
        # Try tiles in a 3x3 grid around the calculated tile
        for h_offset in [-1, 0, 1]:
            for v_offset in [-1, 0, 1]:
                h = h_base + h_offset
                v = v_base + v_offset
                
                # Skip invalid tiles
                if h < 0 or h > 35 or v < 0 or v > 17:
                    continue
                
                tile = f"h{h:02d}v{v:02d}"
                filename = f"MOD11A1.AA{year}{doy:03d}.{tile}.061.LST_Day_1km.tif"
                filepath = year_dir / filename
                
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
                                return filepath
                    except Exception:
                        continue
        
        return None
    
    def extract_lst_value(self, lat: float, lon: float, date: datetime) -> Optional[float]:
        """
        Extract LST value for a specific location and date.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date
            
        Returns:
            LST value in Kelvin, or None if extraction fails
        """
        try:
            # Find MODIS LST file
            lst_file = self.find_modis_lst_file(lat, lon, date)
            if not lst_file:
                logger.debug(f"No MODIS LST file found for ({lat}, {lon}, {date.date()})")
                return None
            
            # Read raster and extract value
            with rasterio.open(lst_file) as src:
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
                        return None
                    
                    # MODIS LST values are in Kelvin * 0.02 scale factor
                    # Convert to actual Kelvin
                    lst_kelvin = value * 0.02
                    
                    # Check for reasonable LST range (200-350 K)
                    if 200 <= lst_kelvin <= 350:
                        return float(lst_kelvin)
                    else:
                        logger.debug(f"LST value out of range: {lst_kelvin} K")
                        return None
                else:
                    logger.debug(f"Point ({lat}, {lon}) outside LST raster bounds")
                    return None
                    
        except Exception as e:
            logger.debug(f"Error extracting LST for ({lat}, {lon}, {date}): {e}")
            return None
    
    def get_lst_series(self, lat: float, lon: float, 
                      start_date: datetime, end_date: datetime) -> pd.Series:
        """
        Get LST time series for a location and date range.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date
            end_date: End date
            
        Returns:
            Pandas Series with date index and LST values in Kelvin
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        lst_values = []
        
        for date in dates:
            lst = self.extract_lst_value(lat, lon, date)
            lst_values.append(lst)
        
        return pd.Series(lst_values, index=dates, name='lst')
    
    def extract_dtw_features(self, lat: float, lon: float, 
                           dtw_start: datetime, dtw_end: datetime) -> dict:
        """
        Extract LST features within DTW window.
        
        Args:
            lat: Latitude
            lon: Longitude
            dtw_start: DTW start date
            dtw_end: DTW end date
            
        Returns:
            Dictionary with statistical features for LST
        """
        # Get time series
        series = self.get_lst_series(lat, lon, dtw_start, dtw_end)
        
        # Remove NaN values
        valid_values = series.dropna()
        
        if len(valid_values) > 0:
            features = {
                'lst_min': float(valid_values.min()),
                'lst_median': float(valid_values.median()),
                'lst_mean': float(valid_values.mean()),
                'lst_max': float(valid_values.max())
            }
        else:
            # All NaN values
            features = {
                'lst_min': np.nan,
                'lst_median': np.nan,
                'lst_mean': np.nan,
                'lst_max': np.nan
            }
        
        return features
    
    def extract_dtw_features_batch(self, fire_points: pd.DataFrame) -> pd.DataFrame:
        """
        Extract LST features for multiple fire points.
        
        Args:
            fire_points: DataFrame with columns ['lat', 'lon', 'dtw_start', 'dtw_end']
            
        Returns:
            DataFrame with added LST feature columns
        """
        results = fire_points.copy()
        
        # Initialize LST feature columns
        lst_columns = ['lst_min', 'lst_median', 'lst_mean', 'lst_max']
        for col in lst_columns:
            results[col] = np.nan
        
        logger.info(f"Extracting LST features for {len(fire_points)} fire points")
        
        for idx, row in fire_points.iterrows():
            try:
                # Extract LST features
                lst_features = self.extract_dtw_features(
                    row['lat'], row['lon'], row['dtw_start'], row['dtw_end']
                )
                
                # Update results
                for feature_name, value in lst_features.items():
                    results.loc[idx, feature_name] = value
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx}/{len(fire_points)} LST extractions")
                    
            except Exception as e:
                logger.error(f"Error extracting LST features for point {idx}: {e}")
        
        return results
