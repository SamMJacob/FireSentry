#!/usr/bin/env python3
"""
MODIS LST Data Fetcher

Downloads MODIS LST (MOD11A1) data for Uttarakhand region using earthaccess library.
Focuses on dry season months (Feb-Jun) to reduce download size.

Usage:
    python scripts/fetch_modis_lst.py

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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MODISLSTFetcher:
    def __init__(self, config_path=".env"):
        """Initialize MODIS LST fetcher with configuration."""
        self.load_config(config_path)
        self.setup_directories()
        self.bbox_geom = self.create_bbox_geometry()
        self.tiles = ['h24v05', 'h24v06', 'h25v05', 'h25v06']  # Tiles covering Uttarakhand
        
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
        
        # Dry season months (Feb-Jun) to reduce download size
        self.dry_season_months = [2, 3, 4, 5, 6]
        
        logger.info(f"Loaded config: bbox=({self.bbox_w}, {self.bbox_s}, {self.bbox_e}, {self.bbox_n}), "
                   f"years={self.year_start}-{self.year_end}, dry season months={self.dry_season_months}")
    
    def setup_directories(self):
        """Create output directories."""
        self.output_dir = Path("data/raw/modis_lst")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create year subdirectories
        for year in range(self.year_start, self.year_end + 1):
            (self.output_dir / str(year)).mkdir(exist_ok=True)
    
    def create_bbox_geometry(self):
        """Create bounding box geometry for clipping."""
        return box(self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n)
    
    def setup_earthdata_auth(self):
        """Setup Earthdata authentication."""
        try:
            # Check if .netrc exists
            netrc_path = Path.home() / '.netrc'
            if not netrc_path.exists():
                logger.error("No .netrc file found. Please create one with your NASA Earthdata credentials:")
                logger.error("echo 'machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASS' >> ~/.netrc")
                logger.error("chmod 600 ~/.netrc")
                return False
            
            # Login to Earthdata
            auth = earthaccess.login()
            if auth:
                logger.info("✓ Successfully authenticated with NASA Earthdata")
                return True
            else:
                logger.error("Failed to authenticate with NASA Earthdata")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def get_date_ranges(self):
        """Get date ranges for dry season months across all years."""
        date_ranges = []
        
        for year in range(self.year_start, self.year_end + 1):
            for month in self.dry_season_months:
                start_date = datetime(year, month, 1)
                if month == 6:  # June
                    end_date = datetime(year, month, 30)
                elif month == 2:  # February
                    if year % 4 == 0:  # Leap year
                        end_date = datetime(year, month, 29)
                    else:
                        end_date = datetime(year, month, 28)
                else:  # March, April, May
                    end_date = datetime(year, month + 1, 1) - timedelta(days=1)
                
                date_ranges.append((start_date, end_date))
        
        return date_ranges
    
    def search_modis_lst(self, start_date, end_date):
        """Search for MODIS LST products."""
        try:
            results = earthaccess.search_data(
                short_name='MOD11A1',
                version='061',
                bounding_box=(self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n),
                temporal=(start_date, end_date),
                count=1000
            )
            
            logger.info(f"Found {len(results)} MODIS LST files for {start_date.date()} to {end_date.date()}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching MODIS LST: {e}")
            return []
    
    def download_files(self, results, year):
        """Download MODIS LST files."""
        if not results:
            return 0, 0
        
        year_dir = self.output_dir / str(year)
        successful = 0
        failed = 0
        
        try:
            # Download all files at once
            logger.info(f"Downloading {len(results)} MODIS LST files for {year}...")
            earthaccess.download(results, local_path=str(year_dir))
            
            # Process each downloaded HDF file
            hdf_files = list(year_dir.glob("*.hdf"))
            logger.info(f"Found {len(hdf_files)} HDF files to process")
            
            for hdf_file in tqdm(hdf_files, desc=f"Processing MODIS LST {year}"):
                try:
                    # Extract date from filename
                    filename = hdf_file.name
                    date_str = filename.split('.')[1]  # YYYYDDD format
                    
                    # Create output filename
                    output_filename = f"MOD11A1.A{date_str}.h{filename.split('.h')[1].split('.')[0]}.061.LST_Day_1km.tif"
                    output_path = year_dir / output_filename
                    
                    # Skip if already exists
                    if output_path.exists():
                        hdf_file.unlink()  # Remove HDF file
                        successful += 1
                        continue
                    
                    # Convert HDF to GeoTIFF and clip
                    if self.convert_and_clip_hdf(hdf_file, output_path):
                        # Remove HDF file
                        hdf_file.unlink()
                        successful += 1
                    else:
                        failed += 1
                        if hdf_file.exists():
                            hdf_file.unlink()
                            
                except Exception as e:
                    logger.error(f"Error processing {hdf_file.name}: {e}")
                    failed += 1
                    if hdf_file.exists():
                        hdf_file.unlink()
                        
        except Exception as e:
            logger.error(f"Error downloading MODIS LST files for {year}: {e}")
            failed += len(results)
        
        return successful, failed
    
    def convert_and_clip_hdf(self, hdf_path, output_path):
        """Convert HDF to GeoTIFF and clip to bounding box."""
        try:
            # Use gdal_translate to extract LST_Day_1km band and convert to GeoTIFF
            temp_tif = output_path.parent / f"temp_{output_path.name}"
            
            # Extract LST_Day_1km band (band 1 in MOD11A1)
            cmd = [
                'gdal_translate',
                f'HDF4_EOS:EOS_GRID:"{hdf_path}":MODIS_Grid_Daily_1km_LST:LST_Day_1km',
                str(temp_tif)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"gdal_translate failed: {result.stderr}")
                return False
            
            # Clip to bounding box
            if self.clip_to_bbox(temp_tif, output_path):
                temp_tif.unlink()
                return True
            else:
                if temp_tif.exists():
                    temp_tif.unlink()
                return False
                
        except Exception as e:
            logger.error(f"Error converting HDF: {e}")
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
    
    def generate_manifest(self):
        """Generate manifest file with all downloaded files."""
        manifest = {
            'dataset': 'MODIS_LST',
            'product': 'MOD11A1.061',
            'bbox': [self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n],
            'years': f"{self.year_start}-{self.year_end}",
            'dry_season_months': self.dry_season_months,
            'tiles': self.tiles,
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
        logger.info("Starting MODIS LST data download")
        
        # Setup authentication
        if not self.setup_earthdata_auth():
            logger.error("Authentication failed. Cannot proceed.")
            return 0, 0
        
        total_successful = 0
        total_failed = 0
        
        # Get date ranges for dry season
        date_ranges = self.get_date_ranges()
        
        for start_date, end_date in date_ranges:
            year = start_date.year
            
            # Search for files
            results = self.search_modis_lst(start_date, end_date)
            
            # Download files
            successful, failed = self.download_files(results, year)
            total_successful += successful
            total_failed += failed
        
        # Generate manifest
        manifest = self.generate_manifest()
        
        logger.info(f"MODIS LST download complete: {total_successful} successful, {total_failed} failed")
        logger.info(f"Total files: {len(manifest['files'])}")
        logger.info(f"Total size: {sum(f['size_mb'] for f in manifest['files']):.1f} MB")
        
        return total_successful, total_failed

def main():
    """Main entry point."""
    fetcher = MODISLSTFetcher()
    successful, failed = fetcher.run()
    
    if failed > 0:
        logger.warning(f"Some downloads failed. Check logs for details.")
        sys.exit(1)
    else:
        logger.info("All downloads completed successfully!")

if __name__ == "__main__":
    main()

