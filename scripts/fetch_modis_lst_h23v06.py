#!/usr/bin/env python3
"""
MODIS Land Surface Temperature Data Fetcher - h23v06 Tile Only

Downloads MODIS LST (MOD11A1) data for h23v06 tile (western Uttarakhand)
for dry season months (Feb-Jun) across 2020-2024.

This tile covers 67.0°E - 78.3°E, which includes the western edge
of Uttarakhand (77.5°E - 78.3°E).

Usage:
    python scripts/fetch_modis_lst_h23v06.py

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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MODISLSTFetcherH23V06:
    def __init__(self):
        """Initialize MODIS LST fetcher for h23v06 tile."""
        # Uttarakhand bounding box (expanded to ensure h23v06 coverage)
        self.bbox_n = 31.459016
        self.bbox_s = 28.709556
        self.bbox_e = 81.044789
        self.bbox_w = 75.0  # Expanded west to capture h23v06
        
        # Output directory
        self.output_dir = Path("data/raw/modis_lst")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Only download h23v06 tile
        self.target_tile = 'h23v06'
        
        # Dry season months (Feb-Jun)
        self.dry_months = [2, 3, 4, 5, 6]
        
        # Years to download
        self.years = [2020, 2021, 2022, 2023, 2024]
        
        logger.info(f"Fetching MODIS LST data for tile {self.target_tile}")
        logger.info(f"This tile covers western Uttarakhand (77.5°E - 78.3°E)")
        logger.info(f"Dry season months: Feb-Jun")
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
    
    def search_modis_lst(self, start_date, end_date):
        """Search for MODIS LST granules for h23v06 tile."""
        logger.info(f"Searching for h23v06 LST data: {start_date.date()} to {end_date.date()}")
        
        # Expanded bbox to ensure h23v06 is included
        bbox_search = (
            self.bbox_w - 2.0,  # West
            self.bbox_s - 1.0,  # South
            self.bbox_e + 1.0,  # East
            self.bbox_n + 1.0   # North
        )
        
        try:
            results = earthaccess.search_data(
                short_name='MOD11A1',  # MODIS LST product
                version='061',
                temporal=(start_date, end_date),
                bounding_box=bbox_search
            )
            
            logger.info(f"Found {len(results)} total granules")
            
            # Filter to only h23v06
            filtered_results = []
            for result in results:
                granule_name = result['umm']['GranuleUR']
                if self.target_tile in granule_name:
                    filtered_results.append(result)
            
            logger.info(f"Filtered to {len(filtered_results)} granules for {self.target_tile}")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching MODIS LST data: {e}")
            return []
    
    def download_granule(self, granule, year_dir):
        """Download and process a single MODIS LST granule."""
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
            
            # Convert HDF to GeoTIFF
            self.convert_hdf_to_geotiff(hdf_file, year_dir)
            
            # Clean up HDF file
            if os.path.exists(hdf_file):
                os.remove(hdf_file)
                logger.info(f"Cleaned up HDF file")
            
        except Exception as e:
            logger.error(f"Error downloading granule: {e}")
    
    def convert_hdf_to_geotiff(self, hdf_file, output_dir):
        """Convert HDF LST band to GeoTIFF format."""
        filename = Path(hdf_file).name
        parts = filename.split('.')
        
        if len(parts) < 4:
            logger.warning(f"Unexpected filename format: {filename}")
            return
        
        product = parts[0]  # MOD11A1
        date_str = parts[1]  # AYYYYDDD
        tile = parts[2]     # hXXvYY
        version = parts[3]  # 061
        
        # Extract LST_Day_1km band
        try:
            output_file = output_dir / f"{product}.{date_str}.{tile}.{version}.LST_Day_1km.tif"
            
            if output_file.exists():
                logger.info(f"Already exists: {output_file.name}")
                return
            
            # HDF subdataset path for LST Day
            subdataset = f'HDF4_EOS:EOS_GRID:"{hdf_file}":MODIS_Grid_Daily_1km_LST:LST_Day_1km'
            
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
                logger.error(f"❌ Failed to convert LST band")
                
        except Exception as e:
            logger.error(f"Error converting LST band: {e}")
    
    def download_h23v06_dry_season(self):
        """Download h23v06 LST tile for dry season (Feb-Jun) 2020-2024."""
        logger.info("="*80)
        logger.info("DOWNLOADING h23v06 LST TILE FOR DRY SEASON 2020-2024")
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
            
            year_granules = 0
            
            # Download each dry season month
            for month in self.dry_months:
                month_name = datetime(year, month, 1).strftime('%B')
                logger.info(f"\n--- {month_name} {year} ---")
                
                # Get month date range
                start_date = datetime(year, month, 1)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(year, month + 1, 1) - timedelta(days=1)
                
                # Search for granules
                results = self.search_modis_lst(start_date, end_date)
                year_granules += len(results)
                
                if not results:
                    logger.warning(f"No granules found for {month_name} {year}")
                    continue
                
                # Download and process each granule
                for i, granule in enumerate(results, 1):
                    logger.info(f"Granule {i}/{len(results)}")
                    self.download_granule(granule, year_dir)
                    time.sleep(1)  # Small delay
            
            logger.info(f"\n✅ Year {year} complete: {year_granules} granules")
            total_granules += year_granules
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("="*80)
        logger.info(f"Total granules processed: {total_granules}")
        logger.info(f"Tile: {self.target_tile}")
        logger.info(f"Period: Dry season (Feb-Jun) 2020-2024")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Save manifest
        self.save_manifest()
    
    def save_manifest(self):
        """Save download manifest."""
        manifest = {
            'download_date': datetime.now().isoformat(),
            'tile': self.target_tile,
            'years': self.years,
            'months': ['February', 'March', 'April', 'May', 'June'],
            'purpose': 'Fill western Uttarakhand LST coverage gap (77.5-78.3°E)',
            'product': 'MOD11A1 (MODIS Land Surface Temperature)',
            'status': 'H23V06_LST_DRY_SEASON_COMPLETE'
        }
        
        manifest_path = self.output_dir / 'download_manifest_h23v06_lst.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest saved: {manifest_path}")

def main():
    """Main execution function."""
    try:
        fetcher = MODISLSTFetcherH23V06()
        fetcher.download_h23v06_dry_season()
        
        logger.info("\n✅ SUCCESS: h23v06 LST tile downloaded for dry season")
        logger.info("This fills the LST coverage gap for western Uttarakhand")
        logger.info("\nNext steps:")
        logger.info("1. Also download h23v06 MODIS SR: python scripts/fetch_modis_sr_h23v06.py")
        logger.info("2. Re-run feature extraction with complete data")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

