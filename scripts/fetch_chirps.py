#!/usr/bin/env python3
"""
CHIRPS Precipitation Data Fetcher

Downloads daily CHIRPS precipitation GeoTIFFs for Uttarakhand region (2020-2024),
clips to bounding box, and saves organized by year.

Usage:
    python scripts/fetch_chirps.py

Requirements:
    - rasterio
    - requests
    - tqdm (for progress bars)
"""

import os
import sys
import argparse
import requests
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CHIRPSFetcher:
    def __init__(self, config_path=".env", year=None, start_month=None, end_month=None):
        """Initialize CHIRPS fetcher with configuration."""
        self.load_config(config_path)
        
        # Override with command line arguments if provided
        if year is not None:
            self.year_start = year
            self.year_end = year
        if start_month is not None and end_month is not None:
            self.start_month = start_month
            self.end_month = end_month
        else:
            self.start_month = 1
            self.end_month = 12
            
        self.setup_directories()
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
    
    def setup_directories(self):
        """Create output directories."""
        self.output_dir = Path("data/raw/chirps")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create year subdirectories
        for year in range(self.year_start, self.year_end + 1):
            (self.output_dir / str(year)).mkdir(exist_ok=True)
    
    def create_bbox_geometry(self):
        """Create bounding box geometry for clipping."""
        return box(self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n)
    
    def get_date_range(self, year):
        """Get all dates for a given year within the specified month range."""
        start_date = datetime(year, self.start_month, 1)
        
        # Calculate end date for the end month
        if self.end_month == 12:
            end_date = datetime(year, 12, 31)
        else:
            # Get first day of next month, then subtract one day
            next_month = self.end_month + 1
            if next_month > 12:
                next_month = 1
                next_year = year + 1
            else:
                next_year = year
            end_date = datetime(next_year, next_month, 1) - timedelta(days=1)
        
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        
        return dates
    
    def get_chirps_url(self, date):
        """Generate CHIRPS download URL for a given date."""
        year = date.year
        month = date.month
        day = date.day
        filename = f"chirps-v2.0.{year}.{month:02d}.{day:02d}.tif.gz"
        
        # CHIRPS URL structure with year directories
        base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"
        return f"{base_url}/{year}/{filename}"
    
    def download_file(self, url, output_path, max_retries=3):
        """Download file with retry logic."""
        logger.debug(f"Starting download: {url}")
        for attempt in range(max_retries):
            try:
                logger.debug(f"Download attempt {attempt + 1} for {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                logger.debug(f"Download successful, writing to: {output_path}")
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.debug(f"File saved: {output_path} ({output_path.stat().st_size} bytes)")
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download {url} after {max_retries} attempts")
                    return False
                
        return False
    
    def clip_to_bbox(self, input_path, output_path):
        """Clip GeoTIFF to bounding box."""
        try:
            with rasterio.open(input_path) as src:
                # Create GeoDataFrame for clipping
                bbox_gdf = gpd.GeoDataFrame([1], geometry=[self.bbox_geom], crs='EPSG:4326')
                
                # Reproject bbox to match raster CRS if needed
                if bbox_gdf.crs != src.crs:
                    bbox_gdf = bbox_gdf.to_crs(src.crs)
                
                # Clip the raster
                clipped_data, clipped_transform = mask(src, bbox_gdf.geometry, crop=True)
                
                # Update metadata
                clipped_meta = src.meta.copy()
                clipped_meta.update({
                    'driver': 'GTiff',
                    'height': clipped_data.shape[1],
                    'width': clipped_data.shape[2],
                    'transform': clipped_transform
                })
                
                # Write clipped raster
                with rasterio.open(output_path, 'w', **clipped_meta) as dst:
                    dst.write(clipped_data)
                
                return True
                
        except Exception as e:
            logger.error(f"Error clipping {input_path}: {e}")
            return False
    
    def process_year(self, year):
        """Download and process all CHIRPS files for a year."""
        logger.info(f"Processing year {year}")
        
        dates = self.get_date_range(year)
        year_dir = self.output_dir / str(year)
        
        logger.info(f"Year {year}: {len(dates)} dates to process")
        logger.info(f"Output directory: {year_dir}")
        
        successful_downloads = 0
        failed_downloads = 0
        
        for i, date in enumerate(tqdm(dates, desc=f"Downloading {year}")):
            if i % 50 == 0:  # Log every 50 files
                logger.info(f"Year {year}: Processing date {date} ({i+1}/{len(dates)})")
            # Skip if file already exists
            output_filename = f"chirps-v2.0.{year}.{date.timetuple().tm_yday:03d}.tif"
            output_path = year_dir / output_filename
            
            if output_path.exists():
                successful_downloads += 1
                continue
            
            # Download compressed file
            url = self.get_chirps_url(date)
            temp_path = year_dir / f"{output_filename}.gz"
            
            logger.debug(f"Downloading: {url}")
            if self.download_file(url, temp_path):
                # Decompress and clip
                import gzip
                import shutil
                
                try:
                    # Decompress
                    with gzip.open(temp_path, 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove compressed file
                    temp_path.unlink()
                    
                    # Clip to bbox
                    if self.clip_to_bbox(output_path, output_path):
                        successful_downloads += 1
                    else:
                        output_path.unlink()  # Remove failed clip
                        failed_downloads += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {date}: {e}")
                    if temp_path.exists():
                        temp_path.unlink()
                    if output_path.exists():
                        output_path.unlink()
                    failed_downloads += 1
            else:
                failed_downloads += 1
        
        logger.info(f"Year {year} complete: {successful_downloads} successful, {failed_downloads} failed")
        return successful_downloads, failed_downloads
    
    def generate_manifest(self):
        """Generate manifest file with all downloaded files."""
        manifest = {
            'dataset': 'CHIRPS',
            'bbox': [self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n],
            'years': f"{self.year_start}-{self.year_end}",
            'files': []
        }
        
        for year in range(self.year_start, self.year_end + 1):
            year_dir = self.output_dir / str(year)
            if year_dir.exists():
                for file_path in year_dir.glob("*.tif"):
                    file_info = {
                        'filename': file_path.name,
                        'path': str(file_path.relative_to(Path.cwd())),
                        'year': year,
                        'size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
                    manifest['files'].append(file_info)
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Generated manifest: {manifest_path}")
        return manifest
    
    def run(self):
        """Main execution function."""
        logger.info("Starting CHIRPS data download")
        logger.info(f"Years: {self.year_start} to {self.year_end}")
        logger.info(f"Bounding box: ({self.bbox_w}, {self.bbox_s}, {self.bbox_e}, {self.bbox_n})")
        logger.info(f"Output directory: {self.output_dir}")
        
        total_successful = 0
        total_failed = 0
        
        # Test connectivity first
        test_url = self.get_chirps_url(datetime(2024, 1, 1))
        logger.info(f"Testing connectivity with: {test_url}")
        try:
            response = requests.head(test_url, timeout=10)
            logger.info(f"Connectivity test: {response.status_code}")
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
        
        for year in range(self.year_start, self.year_end + 1):
            logger.info(f"Starting year {year}...")
            successful, failed = self.process_year(year)
            total_successful += successful
            total_failed += failed
        
        # Generate manifest
        manifest = self.generate_manifest()
        
        logger.info(f"CHIRPS download complete: {total_successful} successful, {total_failed} failed")
        logger.info(f"Total files: {len(manifest['files'])}")
        logger.info(f"Total size: {sum(f['size_mb'] for f in manifest['files']):.1f} MB")
        
        return total_successful, total_failed

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Download CHIRPS precipitation data')
    parser.add_argument('--year', type=int, help='Specific year to download')
    parser.add_argument('--start-month', type=int, help='Start month (1-12)')
    parser.add_argument('--end-month', type=int, help='End month (1-12)')
    
    args = parser.parse_args()
    
    fetcher = CHIRPSFetcher(
        year=args.year,
        start_month=args.start_month,
        end_month=args.end_month
    )
    successful, failed = fetcher.run()
    
    if failed > 0:
        logger.warning(f"Some downloads failed. Check logs for details.")
        sys.exit(1)
    else:
        logger.info("All downloads completed successfully!")

if __name__ == "__main__":
    main()

