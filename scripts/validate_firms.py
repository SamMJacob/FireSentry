#!/usr/bin/env python3
"""
FIRMS Fire Data Validator

Validates FIRMS CSV file for proper format, date range, and geographic coverage.
Provides instructions for manual download if needed.

Usage:
    python scripts/validate_firms.py [path_to_firms_csv]

Requirements:
    - pandas
    - geopandas
"""

import os
import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FIRMSValidator:
    def __init__(self, config_path=".env"):
        """Initialize FIRMS validator with configuration."""
        self.load_config(config_path)
        self.bbox_geom = self.create_bbox_geometry()
        
    def load_config(self, config_path):
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
        self.year_start = int(config.get('YEAR_START', 2020))
        self.year_end = int(config.get('YEAR_END', 2024))
        
        logger.info(f"Loaded config: bbox=({self.bbox_w}, {self.bbox_s}, {self.bbox_e}, {self.bbox_n}), "
                   f"years={self.year_start}-{self.year_end}")
    
    def create_bbox_geometry(self):
        """Create bounding box geometry for validation."""
        from shapely.geometry import box
        return box(self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n)
    
    def print_download_instructions(self):
        """Print manual download instructions."""
        print("\n" + "="*60)
        print("FIRMS MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print("1. Go to: https://firms.modaps.eosdis.nasa.gov/download/")
        print("2. Select 'MODIS Collection 6.1' (or VIIRS if preferred)")
        print("3. For region selection:")
        print(f"   - Draw bounding box: {self.bbox_w}°W, {self.bbox_s}°S to {self.bbox_e}°E, {self.bbox_n}°N")
        print("   - OR upload a shapefile of Uttarakhand")
        print(f"4. Date range: {self.year_start}-01-01 to {self.year_end}-12-31")
        print("5. Format: CSV")
        print("6. Download and save as: data/raw/firms/modis_2020_2024.csv")
        print("="*60)
    
    def validate_csv_structure(self, df):
        """Validate CSV has required columns."""
        required_columns = [
            'latitude', 'longitude', 'acq_date', 'acq_time',
            'satellite', 'confidence', 'brightness', 'scan', 'track'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("✓ CSV structure validation passed")
        return True
    
    def validate_date_range(self, df):
        """Validate date range covers required years."""
        # Convert acq_date to datetime
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        
        min_date = df['acq_date'].min()
        max_date = df['acq_date'].max()
        
        expected_start = datetime(self.year_start, 1, 1)
        expected_end = datetime(self.year_end, 12, 31)
        
        logger.info(f"Date range in data: {min_date.date()} to {max_date.date()}")
        logger.info(f"Expected range: {expected_start.date()} to {expected_end.date()}")
        
        if min_date > expected_start or max_date < expected_end:
            logger.warning("⚠ Date range doesn't fully cover expected period")
            return False
        
        logger.info("✓ Date range validation passed")
        return True
    
    def validate_geographic_coverage(self, df):
        """Validate geographic coverage of Uttarakhand."""
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs='EPSG:4326'
        )
        
        # Check if points are within bounding box
        within_bbox = gdf.within(self.bbox_geom)
        coverage_pct = within_bbox.mean() * 100
        
        logger.info(f"Points within Uttarakhand bbox: {coverage_pct:.1f}%")
        
        if coverage_pct < 80:
            logger.warning(f"⚠ Low geographic coverage: {coverage_pct:.1f}% within bbox")
            return False
        
        logger.info("✓ Geographic coverage validation passed")
        return True
    
    def validate_data_quality(self, df):
        """Validate data quality metrics."""
        # Check for missing values
        missing_pct = df.isnull().mean() * 100
        high_missing = missing_pct[missing_pct > 10]
        
        if len(high_missing) > 0:
            logger.warning(f"⚠ High missing values in columns: {high_missing.to_dict()}")
        
        # Check confidence values
        if 'confidence' in df.columns:
            conf_dist = df['confidence'].value_counts().sort_index()
            logger.info(f"Confidence distribution: {conf_dist.to_dict()}")
        
        # Check satellite distribution
        if 'satellite' in df.columns:
            sat_dist = df['satellite'].value_counts()
            logger.info(f"Satellite distribution: {sat_dist.to_dict()}")
        
        logger.info("✓ Data quality validation completed")
        return True
    
    def generate_summary(self, df):
        """Generate summary statistics."""
        summary = {
            'total_fires': len(df),
            'date_range': {
                'start': str(df['acq_date'].min().date()),
                'end': str(df['acq_date'].max().date())
            },
            'geographic_bounds': {
                'min_lat': float(df['latitude'].min()),
                'max_lat': float(df['latitude'].max()),
                'min_lon': float(df['longitude'].min()),
                'max_lon': float(df['longitude'].max())
            },
            'satellites': df['satellite'].value_counts().to_dict() if 'satellite' in df.columns else {},
            'confidence_distribution': df['confidence'].value_counts().to_dict() if 'confidence' in df.columns else {}
        }
        
        return summary
    
    def validate_file(self, file_path):
        """Validate FIRMS CSV file."""
        logger.info(f"Validating FIRMS file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            self.print_download_instructions()
            return False
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} fire records")
            
            # Run validations
            validations = [
                self.validate_csv_structure(df),
                self.validate_date_range(df),
                self.validate_geographic_coverage(df),
                self.validate_data_quality(df)
            ]
            
            all_passed = all(validations)
            
            if all_passed:
                logger.info("✓ All validations passed!")
            else:
                logger.warning("⚠ Some validations failed")
            
            # Generate summary
            summary = self.generate_summary(df)
            
            # Save summary
            output_dir = Path("data/raw/firms")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            summary_path = output_dir / "firms_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Summary saved to: {summary_path}")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return False

def main():
    """Main entry point."""
    validator = FIRMSValidator()
    
    # Check for file path argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default path
        file_path = "data/raw/firms/modis_2020_2024.csv"
    
    if not os.path.exists(file_path):
        logger.info(f"FIRMS file not found at: {file_path}")
        validator.print_download_instructions()
        sys.exit(1)
    
    success = validator.validate_file(file_path)
    
    if not success:
        logger.error("Validation failed. Please check the file and try again.")
        sys.exit(1)
    else:
        logger.info("FIRMS validation completed successfully!")

if __name__ == "__main__":
    main()

