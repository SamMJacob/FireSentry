"""
Dynamic Time Window (DTW) Algorithm Implementation

Implements the DTW algorithm from the base paper to determine the critical drying period
preceding each fire event based on cumulative precipitation thresholds.

Algorithm:
1. Start from fire date and walk backward through time
2. Accumulate precipitation until cumulative threshold (Thcp=30mm) is reached
3. Use daily threshold (Thdp=10mm) to identify significant precipitation days
4. Return time window [t_start, t_end] where t_end is fire date

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

logger = logging.getLogger(__name__)

class DynamicTimeWindow:
    """
    Dynamic Time Window calculator for fire prediction.
    
    Determines the critical drying period preceding fire events by analyzing
    precipitation patterns and identifying when cumulative precipitation
    dropped below fire-prone thresholds.
    """
    
    def __init__(self, 
                 thcp: float = 30.0, 
                 thdp: float = 10.0, 
                 max_window_days: int = 90):
        """
        Initialize DTW calculator.
        
        Args:
            thcp: Cumulative precipitation threshold (mm) - default 30mm
            thdp: Daily precipitation threshold (mm) - default 10mm  
            max_window_days: Maximum lookback window (days) - default 90
        """
        self.thcp = thcp
        self.thdp = thdp
        self.max_window_days = max_window_days
        self._warned_points = set()  # Track points we've warned about
        
        logger.info(f"DTW initialized: Thcp={thcp}mm, Thdp={thdp}mm, max_window={max_window_days}days")
    
    def extract_precipitation_value(self, 
                                  lat: float, 
                                  lon: float, 
                                  date: datetime, 
                                  chirps_dir: str = "data/raw/chirps") -> Optional[float]:
        """
        Extract precipitation value for a specific location and date from CHIRPS data.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date to extract precipitation for
            chirps_dir: Directory containing CHIRPS GeoTIFF files
            
        Returns:
            Precipitation value in mm, or None if file not found
        """
        try:
            # Construct filename
            year = date.year
            doy = date.timetuple().tm_yday
            filename = f"chirps-v2.0.{year}.{doy:03d}.tif"
            filepath = Path(chirps_dir) / str(year) / filename
            
            if not filepath.exists():
                logger.warning(f"CHIRPS file not found: {filepath}")
                return 0.0
            
            # Read raster and extract value
            with rasterio.open(filepath) as src:
                # Get row, col for the point
                row, col = src.index(lon, lat)
                
                # Check bounds
                if 0 <= row < src.height and 0 <= col < src.width:
                    # Read the value
                    value = src.read(1)[row, col]
                    
                    # Handle NoData values
                    if value == src.nodata or np.isnan(value):
                        return 0.0  # Treat as no precipitation
                    
                    return float(value)
                else:
                    # Only warn once per point to avoid spam
                    point_key = (round(lat, 6), round(lon, 6))
                    if point_key not in self._warned_points:
                        logger.warning(f"Point ({lat}, {lon}) outside raster bounds")
                        self._warned_points.add(point_key)
                    return 0.0  # Treat as no precipitation
                    
        except Exception as e:
            logger.error(f"Error extracting precipitation for ({lat}, {lon}, {date}): {e}")
            return 0.0  # Treat as no precipitation
    
    def calculate_dtw(self, 
                     fire_date: datetime, 
                     lat: float, 
                     lon: float, 
                     chirps_dir: str = "data/raw/chirps") -> Tuple[datetime, datetime]:
        """
        Calculate Dynamic Time Window for a fire event.
        
        Args:
            fire_date: Date of fire event
            lat: Latitude of fire location
            lon: Longitude of fire location
            chirps_dir: Directory containing CHIRPS data
            
        Returns:
            Tuple of (t_start, t_end) where t_end = fire_date
        """
        t_end = fire_date
        t_current = t_end - timedelta(days=1)
        cumulative_precip = 0.0
        days_back = 0
        
        logger.debug(f"Calculating DTW for fire at ({lat}, {lon}) on {fire_date.date()}")
        
        while days_back < self.max_window_days:
            # Extract precipitation for current day
            precip = self.extract_precipitation_value(lat, lon, t_current, chirps_dir)
            
            if precip is None:
                # If we can't get precipitation data, continue to next day
                t_current -= timedelta(days=1)
                days_back += 1
                continue
            
            # Check if precipitation meets daily threshold
            if precip >= self.thdp:
                cumulative_precip += precip
                
                # Check if cumulative threshold is reached
                if cumulative_precip >= self.thcp:
                    t_start = t_current + timedelta(days=1)
                    window_length = (t_end - t_start).days
                    
                    logger.debug(f"DTW found: {t_start.date()} to {t_end.date()} "
                               f"({window_length} days, {cumulative_precip:.1f}mm cumulative)")
                    
                    return t_start, t_end
            
            # Move to previous day
            t_current -= timedelta(days=1)
            days_back += 1
        
        # If we reach max window without finding threshold
        t_start = t_end - timedelta(days=self.max_window_days)
        window_length = (t_end - t_start).days
        
        logger.debug(f"DTW max window reached: {t_start.date()} to {t_end.date()} "
                   f"({window_length} days, {cumulative_precip:.1f}mm cumulative)")
        
        return t_start, t_end
    
    def calculate_dtw_batch(self, 
                           fire_points: pd.DataFrame, 
                           chirps_dir: str = "data/raw/chirps") -> pd.DataFrame:
        """
        Calculate DTW for multiple fire points.
        
        Args:
            fire_points: DataFrame with columns ['lat', 'lon', 'date']
            chirps_dir: Directory containing CHIRPS data
            
        Returns:
            DataFrame with added columns ['dtw_start', 'dtw_end', 'dtw_length']
        """
        results = fire_points.copy()
        dtw_starts = []
        dtw_ends = []
        dtw_lengths = []
        
        logger.info(f"Calculating DTW for {len(fire_points)} fire points")
        
        for idx, row in fire_points.iterrows():
            try:
                # Convert date to datetime if needed
                if isinstance(row['date'], str):
                    fire_date = pd.to_datetime(row['date'])
                else:
                    fire_date = row['date']
                
                # Calculate DTW
                t_start, t_end = self.calculate_dtw(
                    fire_date, row['lat'], row['lon'], chirps_dir
                )
                
                dtw_starts.append(t_start)
                dtw_ends.append(t_end)
                dtw_lengths.append((t_end - t_start).days)
                
            except Exception as e:
                logger.error(f"Error calculating DTW for point {idx}: {e}")
                dtw_starts.append(None)
                dtw_ends.append(None)
                dtw_lengths.append(None)
        
        results['dtw_start'] = dtw_starts
        results['dtw_end'] = dtw_ends
        results['dtw_length'] = dtw_lengths
        
        # Remove rows where DTW calculation failed
        valid_mask = results['dtw_start'].notna()
        valid_count = valid_mask.sum()
        invalid_count = len(results) - valid_count
        
        logger.info(f"DTW calculation complete: {valid_count} successful, {invalid_count} failed")
        
        return results[valid_mask].reset_index(drop=True)
    
    def get_precipitation_series(self, 
                               lat: float, 
                               lon: float, 
                               start_date: datetime, 
                               end_date: datetime,
                               chirps_dir: str = "data/raw/chirps") -> pd.Series:
        """
        Get precipitation time series for a location and date range.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date
            end_date: End date
            chirps_dir: Directory containing CHIRPS data
            
        Returns:
            Pandas Series with date index and precipitation values
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        precip_values = []
        
        for date in dates:
            precip = self.extract_precipitation_value(lat, lon, date, chirps_dir)
            precip_values.append(precip if precip is not None else 0.0)
        
        return pd.Series(precip_values, index=dates, name='precipitation')
    
    def validate_dtw(self, 
                    lat: float, 
                    lon: float, 
                    fire_date: datetime,
                    chirps_dir: str = "data/raw/chirps") -> dict:
        """
        Validate DTW calculation by returning detailed information.
        
        Args:
            lat: Latitude
            lon: Longitude
            fire_date: Fire date
            chirps_dir: Directory containing CHIRPS data
            
        Returns:
            Dictionary with validation results
        """
        t_start, t_end = self.calculate_dtw(fire_date, lat, lon, chirps_dir)
        
        # Get precipitation series for the window
        precip_series = self.get_precipitation_series(lat, lon, t_start, t_end, chirps_dir)
        
        # Calculate statistics
        total_precip = precip_series.sum()
        max_daily = precip_series.max()
        days_with_precip = (precip_series >= self.thdp).sum()
        window_length = (t_end - t_start).days
        
        validation = {
            'fire_date': fire_date,
            'location': (lat, lon),
            'dtw_start': t_start,
            'dtw_end': t_end,
            'window_length_days': window_length,
            'total_precipitation_mm': total_precip,
            'max_daily_precipitation_mm': max_daily,
            'days_with_significant_precip': days_with_precip,
            'thresholds': {
                'thcp': self.thcp,
                'thdp': self.thdp,
                'max_window': self.max_window_days
            },
            'precipitation_series': precip_series.to_dict()
        }
        
        return validation

def main():
    """Example usage of DTW calculator."""
    # Initialize DTW calculator
    dtw = DynamicTimeWindow(thcp=30.0, thdp=10.0, max_window_days=90)
    
    # Example fire point
    fire_date = datetime(2024, 4, 15)
    lat, lon = 30.0, 79.0  # Example location in Uttarakhand
    
    # Calculate DTW
    t_start, t_end = dtw.calculate_dtw(fire_date, lat, lon)
    
    print(f"Fire date: {fire_date.date()}")
    print(f"DTW window: {t_start.date()} to {t_end.date()}")
    print(f"Window length: {(t_end - t_start).days} days")
    
    # Validate calculation
    validation = dtw.validate_dtw(lat, lon, fire_date)
    print(f"Total precipitation in window: {validation['total_precipitation_mm']:.1f} mm")

if __name__ == "__main__":
    main()


