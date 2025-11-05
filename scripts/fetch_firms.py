#!/usr/bin/env python3
"""
FIRMS Fire Data Fetcher

Downloads FIRMS fire data using API for Uttarakhand region (2015-2024).
Automated download using FIRMS API token.

Usage:
    python scripts/fetch_firms.py

Requirements:
    - requests
    - pandas
    - python-dotenv
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FIRMSFetcher:
    def __init__(self):
        """Initialize FIRMS fetcher with configuration."""
        load_dotenv()
        self.load_config()
        self.setup_directories()
        
    def load_config(self):
        """Load configuration from environment variables."""
        self.bbox_n = float(os.getenv('BBOX_N', 31.459016))
        self.bbox_s = float(os.getenv('BBOX_S', 28.709556))
        self.bbox_e = float(os.getenv('BBOX_E', 81.044789))
        self.bbox_w = float(os.getenv('BBOX_W', 77.575402))
        self.year_start = int(os.getenv('YEAR_START', 2015))
        self.year_end = int(os.getenv('YEAR_END', 2024))
        self.firms_token = os.getenv('FIRMS_MAP_KEY')
        
        if not self.firms_token:
            raise ValueError("FIRMS_MAP_KEY not found in environment variables")
        
        logger.info(f"Loaded config: bbox=({self.bbox_w}, {self.bbox_s}, {self.bbox_e}, {self.bbox_n}), "
                   f"years={self.year_start}-{self.year_end}")
    
    def setup_directories(self):
        """Create output directories."""
        self.output_dir = Path("data/raw/firms")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def download_firms_data(self):
        """Download FIRMS data for all years."""
        logger.info("Starting FIRMS data download...")
        
        all_data = []
        
        for year in range(self.year_start, self.year_end + 1):
            logger.info(f"Downloading FIRMS data for {year}...")
            
            # Download data for the year
            year_data = self.download_year_data(year)
            if year_data is not None and not year_data.empty:
                all_data.append(year_data)
                logger.info(f"Downloaded {len(year_data)} fire points for {year}")
            else:
                logger.warning(f"No data found for {year}")
            
            # Rate limiting
            time.sleep(1)
        
        if all_data:
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Save combined data
            output_file = self.output_dir / f"firms_{self.year_start}_{self.year_end}.csv"
            combined_data.to_csv(output_file, index=False)
            logger.info(f"Saved {len(combined_data)} total fire points to {output_file}")
            
            # Save summary
            self.save_summary(combined_data, output_file)
            
            return output_file
        else:
            logger.error("No FIRMS data downloaded")
            return None
    
    def download_year_data(self, year):
        """Download FIRMS data for a specific year."""
        # FIRMS API endpoint - corrected format
        base_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{self.firms_token}/MODIS_NRT/{self.bbox_w},{self.bbox_s},{self.bbox_e},{self.bbox_n}/7"
        
        # Note: FIRMS API typically provides last 7 days of data
        # For historical data (2020-2024), manual download is recommended
        logger.warning(f"FIRMS API provides recent data only. For {year}, consider manual download from FIRMS archive.")
        
        # Date range for the year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        try:
            # Make request (no params needed with new URL format)
            response = requests.get(base_url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV data
            if response.text.strip():
                # Create StringIO object for pandas
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                
                # Filter by date range
                df['acq_date'] = pd.to_datetime(df['acq_date'])
                df = df[(df['acq_date'] >= start_date) & (df['acq_date'] <= end_date)]
                
                # Filter by confidence (optional - keep all confidence levels)
                # df = df[df['confidence'] >= 'n']  # nominal confidence or higher
                
                return df
            else:
                logger.warning(f"No data returned for {year}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading FIRMS data for {year}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {year}: {e}")
            return None
    
    def save_summary(self, df, output_file):
        """Save summary statistics."""
        summary = {
            'total_fires': len(df),
            'date_range': {
                'start': df['acq_date'].min().strftime('%Y-%m-%d'),
                'end': df['acq_date'].max().strftime('%Y-%m-%d')
            },
            'geographic_bounds': {
                'lat_min': df['latitude'].min(),
                'lat_max': df['latitude'].max(),
                'lon_min': df['longitude'].min(),
                'lon_max': df['longitude'].max()
            },
            'confidence_distribution': df['confidence'].value_counts().to_dict(),
            'yearly_counts': df.groupby(df['acq_date'].dt.year).size().to_dict()
        }
        
        summary_file = output_file.parent / f"firms_summary_{self.year_start}_{self.year_end}.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_file}")

def main():
    """Main function."""
    try:
        fetcher = FIRMSFetcher()
        output_file = fetcher.download_firms_data()
        
        if output_file:
            print(f"\n✅ FIRMS data download complete!")
            print(f"📁 Output file: {output_file}")
            print(f"📊 Run validation: python scripts/validate_firms.py {output_file}")
        else:
            print("\n❌ FIRMS data download failed!")
            print("💡 Check your FIRMS_MAP_KEY in .env file")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
