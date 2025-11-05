#!/usr/bin/env python3
"""
MODIS Surface Reflectance Data Fetcher - h23v06 Tile Only

Downloads MODIS SR data for h23v06 tile (western Uttarakhand)
for the entire year 2020 to fill the coverage gap.

This tile covers 67.0°E - 78.3°E, which includes the western edge
of Uttarakhand (77.5°E - 78.3°E).

Usage:
    python scripts/fetch_modis_sr_h23v06.py

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

class MODISSRFetcherH23V06:
    def __init__(self):
        """Initialize MODIS SR fetcher for h23v06 tile."""
        # Uttarakhand bounding box (expanded to ensure h23v06 coverage)
        self.bbox_n = 31.459016
        self.bbox_s = 28.709556
        self.bbox_e = 81.044789
        self.bbox_w = 75.0  # Expanded west to capture h23v06
        
        # Output directory
        self.output_dir = Path("data/raw/modis_sr")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Only download h23v06 tile
        self.target_tile = 'h23v06'
        
        logger.info(f"Fetching MODIS SR data for tile {self.target_tile}")
        logger.info(f"This tile covers western Uttarakhand (77.5°E - 78.3°E)")
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
        """Search for MODIS SR granules for h23v06 tile."""
        logger.info(f"Searching for h23v06 data: {start_date.date()} to {end_date.date()}")
        
        # Expanded bbox to ensure h23v06 is included
        bbox_search = (
            self.bbox_w - 2.0,  # West
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
            logger.error(f"Error searching MODIS data: {e}")
            return []
    
    def download_granule(self, granule, year_dir):
        """Download and process a single MODIS granule."""
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
                logger.info(f"Cleaned up HDF file")
            
        except Exception as e:
            logger.error(f"Error downloading granule: {e}")
    
    def convert_hdf_to_geotiff(self, hdf_file, output_dir):
        """Convert HDF bands to GeoTIFF format."""
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
                output_file = output_dir / f"{product}.{date_str}.{tile}.{version}.{band_code}.tif"
                
                if output_file.exists():
                    logger.info(f"Already exists: {output_file.name}")
                    continue
                
                subdataset = f'HDF4_EOS:EOS_GRID:"{hdf_file}":MODIS_Grid_500m_2D:{band_name}'
                
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
                    logger.error(f"❌ Failed to convert {band_name}")
                    
            except Exception as e:
                logger.error(f"Error converting band {band_name}: {e}")
    
    def download_h23v06_2020(self):
        """Download h23v06 tile for entire year 2020."""
        logger.info("="*80)
        logger.info("DOWNLOADING h23v06 TILE FOR 2020")
        logger.info("="*80)
        
        # Authenticate
        self.authenticate()
        
        # Create year directory
        year_dir = self.output_dir / "2020"
        year_dir.mkdir(parents=True, exist_ok=True)
        
        # Download full year 2020
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 12, 31)
        
        logger.info(f"Downloading {self.target_tile} for full year 2020")
        
        # Search for granules
        results = self.search_modis_sr(start_date, end_date)
        
        if not results:
            logger.warning(f"No granules found for {self.target_tile}")
            return
        
        # Download and process each granule
        for i, granule in enumerate(results, 1):
            logger.info(f"\nGranule {i}/{len(results)}")
            self.download_granule(granule, year_dir)
            time.sleep(1)  # Small delay
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("="*80)
        logger.info(f"Total granules processed: {len(results)}")
        logger.info(f"Tile: {self.target_tile}")
        logger.info(f"Output directory: {year_dir}")
        
        # Save manifest
        self.save_manifest(year_dir, start_date, end_date)
    
    def save_manifest(self, year_dir, start_date, end_date):
        """Save download manifest."""
        manifest = {
            'download_date': datetime.now().isoformat(),
            'date_range': (str(start_date.date()), str(end_date.date())),
            'tile': self.target_tile,
            'purpose': 'Fill western Uttarakhand coverage gap (77.5-78.3°E)',
            'status': 'H23V06_2020_COMPLETE'
        }
        
        manifest_path = year_dir / 'download_manifest_h23v06.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest saved: {manifest_path}")

def main():
    """Main execution function."""
    try:
        fetcher = MODISSRFetcherH23V06()
        fetcher.download_h23v06_2020()
        
        logger.info("\n✅ SUCCESS: h23v06 tile downloaded for 2020")
        logger.info("This fills the coverage gap for western Uttarakhand")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

