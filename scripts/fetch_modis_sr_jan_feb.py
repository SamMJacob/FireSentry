#!/usr/bin/env python3
"""
MODIS Surface Reflectance Data Fetcher - January & February 2020

Downloads MODIS Surface Reflectance (MYD09GA) data for Jan-Feb 2020
to fill data gaps for DTW window calculations.

Usage:
    python scripts/fetch_modis_sr_jan_feb.py

Requirements:
    - earthaccess
    - rasterio
    - .netrc file with NASA Earthdata credentials
"""

import os
import sys
import earthaccess
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from tqdm import tqdm
import json
import subprocess
import numpy as np
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MODISSRFetcherJanFeb:
    def __init__(self):
        """Initialize MODIS SR fetcher for Jan-Feb 2020."""
        # Uttarakhand bounding box
        self.bbox_n = 31.459016
        self.bbox_s = 28.709556
        self.bbox_e = 81.044789
        self.bbox_w = 77.575402
        
        # Output directory
        self.output_dir = Path("data/raw/modis_sr")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MODIS tiles for Uttarakhand (correct tiles - 100% coverage verified)
        self.tiles = ['h24v05', 'h24v06', 'h25v06']
        
        # Years to download
        self.years = [2020, 2021, 2022, 2023, 2024]
        
        logger.info(f"Fetching MODIS SR data for Jan-Feb 2020-2024")
        logger.info(f"Target tiles: {self.tiles}")
        logger.info(f"Years: {self.years}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def authenticate(self):
        """Authenticate with NASA Earthdata."""
        logger.info("Authenticating with NASA Earthdata...")
        auth = earthaccess.login()
        if auth:
            logger.info("✅ Authentication successful")
        else:
            logger.error("❌ Authentication failed")
            sys.exit(1)
        return auth
    
    def search_modis_sr(self, start_date, end_date):
        """
        Search for MODIS Surface Reflectance granules.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of granule results
        """
        logger.info(f"Searching for MODIS SR data: {start_date.date()} to {end_date.date()}")
        
        # Expand bbox to ensure we get all required tiles
        bbox_search = (
            self.bbox_w - 1.0,  # West
            self.bbox_s - 1.0,  # South
            self.bbox_e + 1.0,  # East
            self.bbox_n + 1.0   # North
        )
        
        try:
            results = earthaccess.search_data(
                short_name='MYD09GA',
                version='061',
                temporal=(start_date, end_date),
                bounding_box=bbox_search
            )
            
            logger.info(f"Found {len(results)} granules")
            
            # Filter to only the tiles we need
            filtered_results = []
            for result in results:
                granule_name = result['umm']['GranuleUR']
                if any(tile in granule_name for tile in self.tiles):
                    filtered_results.append(result)
            
            logger.info(f"Filtered to {len(filtered_results)} granules for tiles {self.tiles}")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching MODIS data: {e}")
            return []
    
    def download_granule(self, granule, year_dir):
        """
        Download and process a single MODIS granule.
        
        Args:
            granule: Granule metadata
            year_dir: Output directory for the year
        """
        try:
            granule_name = granule['umm']['GranuleUR']
            logger.info(f"Processing: {granule_name}")
            
            # Download granule
            files = earthaccess.download(granule, str(year_dir))
            
            if not files:
                logger.warning(f"No files downloaded for {granule_name}")
                return
            
            hdf_file = files[0]
            logger.info(f"Downloaded: {hdf_file}")
            
            # Convert HDF to GeoTIFF for each band
            self.convert_hdf_to_geotiff(hdf_file, year_dir)
            
            # Clean up HDF file
            if os.path.exists(hdf_file):
                os.remove(hdf_file)
                logger.info(f"Cleaned up HDF file: {hdf_file}")
            
        except Exception as e:
            logger.error(f"Error downloading granule: {e}")
    
    def convert_hdf_to_geotiff(self, hdf_file, output_dir):
        """
        Convert HDF bands to GeoTIFF format.
        
        Args:
            hdf_file: Path to HDF file
            output_dir: Output directory
        """
        # Extract granule info from filename
        filename = Path(hdf_file).name
        parts = filename.split('.')
        
        if len(parts) < 4:
            logger.warning(f"Unexpected filename format: {filename}")
            return
        
        product = parts[0]  # MYD09GA
        date_str = parts[1]  # AYYYYDDD
        tile = parts[2]     # hXXvYY
        version = parts[3]  # 061
        
        # Bands to extract
        bands = {
            'sur_refl_b01': 'B01',  # Red
            'sur_refl_b02': 'B02',  # NIR
            'sur_refl_b03': 'B03',  # Blue
            'sur_refl_b04': 'B04',  # Green
            'sur_refl_b05': 'B05',  # NIR
            'sur_refl_b06': 'B06',  # SWIR1
            'sur_refl_b07': 'B07',  # SWIR2
        }
        
        for band_name, band_code in bands.items():
            try:
                # Output filename
                output_file = output_dir / f"{product}.{date_str}.{tile}.{version}.{band_code}.tif"
                
                # Skip if already exists
                if output_file.exists():
                    logger.info(f"Already exists: {output_file.name}")
                    continue
                
                # HDF subdataset path
                subdataset = f'HDF4_EOS:EOS_GRID:"{hdf_file}":MODIS_Grid_500m_2D:{band_name}'
                
                # Use gdal_translate to convert
                cmd = [
                    'gdal_translate',
                    '-of', 'GTiff',
                    '-co', 'COMPRESS=LZW',
                    '-co', 'TILED=YES',
                    subdataset,
                    str(output_file)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✅ Converted: {output_file.name}")
                else:
                    logger.error(f"❌ Failed to convert {band_name}: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Error converting band {band_name}: {e}")
    
    def download_jan_feb_all_years(self):
        """Download MODIS SR data for January and February 2020-2024."""
        logger.info("="*80)
        logger.info("DOWNLOADING MODIS SR DATA FOR JAN-FEB 2020-2024")
        logger.info("="*80)
        
        # Authenticate
        self.authenticate()
        
        total_granules = 0
        
        for year in self.years:
            logger.info(f"\n{'='*80}")
            logger.info(f"YEAR {year}")
            logger.info(f"{'='*80}")
            
            # Create year directory
            year_dir = self.output_dir / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            
            # Date ranges for Jan-Feb
            # Check if leap year for February
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                feb_days = 29
            else:
                feb_days = 28
            
            date_ranges = [
                (datetime(year, 1, 1), datetime(year, 1, 31)),   # January
                (datetime(year, 2, 1), datetime(year, 2, feb_days)),   # February
            ]
            
            year_granules = 0
            
            for start_date, end_date in date_ranges:
                logger.info(f"\n--- {start_date.strftime('%B %Y')} ---")
                
                # Search for granules
                results = self.search_modis_sr(start_date, end_date)
                year_granules += len(results)
                
                if not results:
                    logger.warning(f"No granules found for {start_date.strftime('%B %Y')}")
                    continue
                
                # Download and process each granule
                for i, granule in enumerate(results, 1):
                    logger.info(f"Granule {i}/{len(results)}")
                    self.download_granule(granule, year_dir)
                    time.sleep(1)
            
            logger.info(f"\n✅ Year {year} complete: {year_granules} granules")
            total_granules += year_granules
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("="*80)
        logger.info(f"Total granules processed: {total_granules}")
        logger.info(f"Years: {self.years}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Save manifest
        self.save_manifest()
    
    def save_manifest(self):
        """Save download manifest."""
        manifest = {
            'download_date': datetime.now().isoformat(),
            'years': self.years,
            'months': ['January', 'February'],
            'tiles': self.tiles,
            'bbox': {
                'north': self.bbox_n,
                'south': self.bbox_s,
                'east': self.bbox_e,
                'west': self.bbox_w
            },
            'status': 'JAN_FEB_2020_2024_COMPLETE'
        }
        
        manifest_path = self.output_dir / 'download_manifest_jan_feb_all_years.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest saved: {manifest_path}")

def main():
    """Main execution function."""
    try:
        fetcher = MODISSRFetcherJanFeb()
        fetcher.download_jan_feb_all_years()
        
        logger.info("\n✅ SUCCESS: Jan-Feb 2020-2024 MODIS SR data downloaded")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

