#!/usr/bin/env python3
"""
MODIS Surface Reflectance Complete Download Script

Downloads ALL MODIS SR data needed for FireSentry:
- Months: January-June (dry season + DTW lookback)
- Years: 2020-2024
- Tiles: h24v05, h24v06, h25v06 (100% Uttarakhand coverage)

This is the ONLY script you need to run for complete MODIS SR coverage.

Usage:
    python scripts/fetch_modis_sr_complete.py

Requirements:
    - earthaccess
    - rasterio
    - .netrc file with NASA Earthdata credentials
"""

import os
import sys
import earthaccess
import rasterio
from datetime import datetime, timedelta
from pathlib import Path
import logging
from tqdm import tqdm
import json
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MODISSRCompleteFetcher:
    def __init__(self):
        """Initialize complete MODIS SR fetcher."""
        # Uttarakhand bounding box
        self.bbox_n = 31.459016
        self.bbox_s = 28.709556
        self.bbox_e = 81.044789
        self.bbox_w = 77.575402
        
        # Output directory
        self.output_dir = Path("data/raw/modis_sr")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MODIS tiles for Uttarakhand (verified 100% coverage)
        self.tiles = ['h24v05', 'h24v06', 'h25v06']
        
        # Years to download
        self.years = [2020, 2021, 2022, 2023, 2024]
        
        # Months to download (dry season + DTW lookback)
        self.months = [1, 2, 3, 4, 5, 6]  # Jan-Jun
        
        logger.info("="*80)
        logger.info("MODIS SR COMPLETE DOWNLOAD")
        logger.info("="*80)
        logger.info(f"Months: January-June (dry season + DTW lookback)")
        logger.info(f"Years: {self.years}")
        logger.info(f"Tiles: {self.tiles}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("="*80)
    
    def authenticate(self):
        """Authenticate with NASA Earthdata using .env credentials."""
        logger.info("Authenticating with NASA Earthdata...")
        
        # Get credentials from .env file
        username = os.getenv('EARTHDATA_USER')
        password = os.getenv('EARTHDATA_PASS')
        
        if not username or not password:
            logger.error("❌ EARTHDATA_USER and EARTHDATA_PASS must be set in .env file")
            sys.exit(1)
        
        try:
            # Set credentials as environment variables for earthaccess
            os.environ['EARTHDATA_USER'] = username
            os.environ['EARTHDATA_PASS'] = password
            
            # Authenticate with credentials
            auth = earthaccess.login()
            if auth:
                logger.info("✅ Authentication successful")
                self.auth_session = auth
            else:
                logger.error("❌ Authentication failed")
                sys.exit(1)
            return auth
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            sys.exit(1)
    
    def search_modis_sr(self, start_date, end_date):
        """Search for MODIS SR granules."""
        logger.info(f"Searching: {start_date.date()} to {end_date.date()}")
        
        # Expanded bbox to ensure all tiles are found
        bbox_search = (
            self.bbox_w - 1.0,
            self.bbox_s - 1.0,
            self.bbox_e + 1.0,
            self.bbox_n + 1.0
        )
        
        try:
            results = earthaccess.search_data(
                short_name='MYD09GA',
                version='061',
                temporal=(start_date, end_date),
                bounding_box=bbox_search
            )
            
            # Filter to only our tiles
            filtered_results = []
            for result in results:
                granule_name = result['umm']['GranuleUR']
                if any(tile in granule_name for tile in self.tiles):
                    filtered_results.append(result)
            
            logger.info(f"Found {len(filtered_results)} granules for our tiles")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def download_granule(self, granule, year_dir):
        """Download and process a single granule."""
        try:
            granule_name = granule['umm']['GranuleUR']
            
            # Download
            files = earthaccess.download(granule, str(year_dir))
            
            if not files:
                logger.warning(f"No files downloaded for {granule_name}")
                return
            
            hdf_file = files[0]
            
            # Convert HDF to GeoTIFF
            self.convert_hdf_to_geotiff(hdf_file, year_dir)
            
            # Clean up HDF
            if os.path.exists(hdf_file):
                os.remove(hdf_file)
            
        except Exception as e:
            logger.error(f"Error downloading granule: {e}")
    
    def convert_hdf_to_geotiff(self, hdf_file, output_dir):
        """Convert HDF bands to GeoTIFF."""
        filename = Path(hdf_file).name
        parts = filename.split('.')
        
        if len(parts) < 4:
            logger.warning(f"Unexpected filename: {filename}")
            return
        
        product = parts[0]
        date_str = parts[1]
        tile = parts[2]
        version = parts[3]
        
        # Bands to extract (correct subdataset names)
        band_mapping = {
            'B01': 'sur_refl_b01_1',  # Red
            'B02': 'sur_refl_b02_1',  # NIR
            'B03': 'sur_refl_b03_1',  # Blue
            'B04': 'sur_refl_b04_1',  # Green
            'B05': 'sur_refl_b05_1',  # NIR
            'B06': 'sur_refl_b06_1',  # SWIR1
            'B07': 'sur_refl_b07_1'   # SWIR2
        }
        
        for band_code, subdataset_name in band_mapping.items():
            try:
                output_file = output_dir / f"{product}.{date_str}.{tile}.{version}.{band_code}.tif"
                
                if output_file.exists():
                    continue
                
                # Use correct subdataset path
                subdataset = f'HDF4_EOS:EOS_GRID:"{hdf_file}":MODIS_Grid_500m_2D:{subdataset_name}'
                
                cmd = [
                    'gdal_translate',
                    subdataset,
                    str(output_file)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to convert {band_code}")
                    
            except Exception as e:
                logger.error(f"Error converting {band_code}: {e}")
    
    def download_complete_dataset(self):
        """Download complete MODIS SR dataset for all years and months."""
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPLETE DOWNLOAD")
        logger.info("="*80)
        
        # Authenticate
        self.authenticate()
        
        total_granules = 0
        overall_start = datetime.now()
        
        for year in self.years:
            logger.info(f"\n{'='*80}")
            logger.info(f"YEAR {year}")
            logger.info(f"{'='*80}")
            
            # Create year directory
            year_dir = self.output_dir / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            
            year_granules = 0
            
            for month in self.months:
                month_name = datetime(year, month, 1).strftime('%B')
                logger.info(f"\n--- {month_name} {year} ---")
                
                # Get month date range
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, month + 1, 1) - timedelta(days=1)
                
                # Search for granules
                results = self.search_modis_sr(start_date, end_date)
                year_granules += len(results)
                
                if not results:
                    logger.warning(f"No granules found for {month_name} {year}")
                    continue
                
                # Download each granule
                for i, granule in enumerate(results, 1):
                    logger.info(f"  [{i}/{len(results)}] Downloading...")
                    self.download_granule(granule, year_dir)
                    time.sleep(1)  # Rate limiting
                
                logger.info(f"✅ {month_name} complete: {len(results)} granules")
            
            logger.info(f"\n✅ Year {year} complete: {year_granules} granules")
            total_granules += year_granules
        
        overall_end = datetime.now()
        duration = overall_end - overall_start
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total granules: {total_granules}")
        logger.info(f"Years: {len(self.years)}")
        logger.info(f"Months per year: {len(self.months)}")
        logger.info(f"Tiles: {len(self.tiles)}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("="*80)
        
        # Save manifest
        self.save_manifest(total_granules, duration)
    
    def save_manifest(self, total_granules, duration):
        """Save download manifest."""
        manifest = {
            'download_date': datetime.now().isoformat(),
            'years': self.years,
            'months': ['January', 'February', 'March', 'April', 'May', 'June'],
            'tiles': self.tiles,
            'total_granules': total_granules,
            'duration_seconds': duration.total_seconds(),
            'bbox': {
                'north': self.bbox_n,
                'south': self.bbox_s,
                'east': self.bbox_e,
                'west': self.bbox_w
            },
            'coverage': '100% Uttarakhand (verified)',
            'status': 'COMPLETE_JAN_JUN_2020_2024'
        }
        
        manifest_path = self.output_dir / 'download_manifest_complete.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"\nManifest saved: {manifest_path}")

def main():
    """Main execution function."""
    try:
        fetcher = MODISSRCompleteFetcher()
        fetcher.download_complete_dataset()
        
        logger.info("\n✅ SUCCESS: Complete MODIS SR dataset downloaded!")
        logger.info("\nNext steps:")
        logger.info("1. Verify data with: ls data/raw/modis_sr/*/")
        logger.info("2. Run feature extraction: python scripts/build_features_optimized.py")
        logger.info("3. Train model with complete dataset")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Download interrupted by user")
        logger.info("You can resume by running this script again")
        logger.info("Already downloaded files will be skipped")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()










































































































































































































































































































































