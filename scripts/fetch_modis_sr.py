#!/usr/bin/env python3
"""
MODIS Surface Reflectance Data Fetcher

Downloads MODIS Surface Reflectance (MYD09GA) data for Uttarakhand region using earthaccess library.
Focuses on dry season months (Feb-Jun) to reduce download size.

Usage:
    python scripts/fetch_modis_sr.py

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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MODISSRFetcher:
    def __init__(self, config_path=".env"):
        """Initialize MODIS SR fetcher with configuration."""
        self.load_config(config_path)
        self.setup_directories()
        self.bbox_geom = self.create_bbox_geometry()
        # Force correct tiles for Uttarakhand based on actual MODIS georeference
        # (MODIS Sinusoidal projection causes tile boundaries to shift)
        # h24v05: 78.3°-80.8°E, 30°-40°N (contains Dehradun and western Uttarakhand)
        # h24v06: 69.3°-74.5°E, 20°-30°N
        # h25v05: 91.4°-92.4°E, 30°-40°N  
        # h25v06: 80.8°-85.1°E, 20°-30°N (contains eastern Uttarakhand)
        self.tiles = ['h24v05', 'h24v06', 'h25v06']
        
    def load_config(self, config_path):
        """Load configuration from environment file."""
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        config[key] = value
        
        # Use expanded bounding box to ensure all 4 tiles are downloaded
        # This covers the full extent of all 4 MODIS tiles needed for Uttarakhand
        self.bbox_n = 31.5  # Slightly north of Uttarakhand
        self.bbox_s = 28.5  # Slightly south of Uttarakhand  
        self.bbox_e = 83.5  # Extended east to cover h26v05, h26v06
        self.bbox_w = 77.0  # Extended west to cover h25v05, h25v06
        self.year_start = int(config.get('YEAR_START', 2020))
        self.year_end = int(config.get('YEAR_END', 2024))
        
        # Dry season months (Mar-May) to reduce download size for 200GB target
        self.dry_season_months = [3, 4, 5]
        
        logger.info(f"Loaded config: bbox=({self.bbox_w}, {self.bbox_s}, {self.bbox_e}, {self.bbox_n}), "
                   f"years={self.year_start}-{self.year_end}, dry season months={self.dry_season_months}")
    
    def calculate_modis_tiles(self):
        """Calculate MODIS sinusoidal tiles that cover the bounding box."""
        tiles = set()
        
        # Sample points across bbox to determine tiles
        lat_samples = np.linspace(self.bbox_s, self.bbox_n, 10)
        lon_samples = np.linspace(self.bbox_w, self.bbox_e, 10)
        
        for lat in lat_samples:
            for lon in lon_samples:
                tile = self.get_modis_tile(lat, lon)
                tiles.add(tile)
        
        tiles_list = sorted(list(tiles))
        logger.info(f"Calculated MODIS tiles for bbox: {tiles_list}")
        return tiles_list
    
    def get_modis_tile(self, lat, lon):
        """Get MODIS sinusoidal tile identifier for given coordinates."""
        # MODIS sinusoidal grid calculation
        h = int((lon + 180) / 10)
        v = int((90 - lat) / 10)
        return f"h{h:02d}v{v:02d}"
    
    def setup_directories(self):
        """Create output directories."""
        # Download to local project directory (C drive)
        self.output_dir = Path("data/raw/modis_sr")
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
    
    def search_modis_sr(self, start_date, end_date):
        """Search for MODIS Surface Reflectance products."""
        try:
            results = earthaccess.search_data(
                short_name='MYD09GA',
                version='061',
                bounding_box=(self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n),
                temporal=(start_date, end_date),
                count=1000
            )
            
            # Filter results to only include our calculated tiles
            filtered_results = []
            for result in results:
                # Extract tile from granule name
                granule_name = result['umm']['GranuleUR']
                for tile in self.tiles:
                    if tile in granule_name:
                        filtered_results.append(result)
                        break
            
            logger.info(f"Found {len(results)} total, filtered to {len(filtered_results)} MODIS SR files for tiles {self.tiles}")
            logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching MODIS SR: {e}")
            return []
    
    def download_files(self, results, year):
        """Download MODIS SR files."""
        if not results:
            return 0, 0
        
        year_dir = self.output_dir / str(year)
        successful = 0
        failed = 0
        
        # Download with retry logic
        max_retries = 3
        retry_delay = 30  # seconds
        
        for attempt in range(max_retries):
            try:
                # Download all files at once
                logger.info(f"Downloading {len(results)} MODIS SR files for {year} (attempt {attempt + 1}/{max_retries})...")
                earthaccess.download(results, local_path=str(year_dir))
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All download attempts failed for {year}")
                    return 0, len(results)
        
        # Process each downloaded HDF file
        hdf_files = list(year_dir.glob("*.hdf"))
        logger.info(f"Found {len(hdf_files)} HDF files to process")
        
        invalid_files = 0
        
        for hdf_file in tqdm(hdf_files, desc=f"Processing MODIS SR {year}"):
            try:
                # Validate HDF file before processing
                if not self.validate_hdf_file(hdf_file):
                    logger.warning(f"Skipping invalid HDF file: {hdf_file.name}")
                    invalid_files += 1
                    failed += 1
                    continue
                
                # Extract date from filename
                filename = hdf_file.name
                date_str = filename.split('.')[1]  # YYYYDDD format
                
                # Create output filenames for each band
                bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07']
                band_success = 0
                
                for band in bands:
                    output_filename = f"MYD09GA.A{date_str}.h{filename.split('.h')[1].split('.')[0]}.061.{band}.tif"
                    output_path = year_dir / output_filename
                    
                    # Skip if already exists
                    if output_path.exists():
                        band_success += 1
                        continue
                    
                    # Convert HDF to GeoTIFF for this band and clip
                    if self.convert_and_clip_hdf_band(hdf_file, output_path, band):
                        band_success += 1
                    else:
                        # Don't break - try other bands
                        continue
                
                # Remove HDF file after processing all bands
                hdf_file.unlink()
                
                if band_success >= 1:  # At least 1 band succeeded
                    successful += 1
                    logger.info(f"✓ {hdf_file.name}: {band_success}/7 bands processed successfully")
                elif band_success == 0:
                    failed += 1
                    logger.warning(f"✗ {hdf_file.name}: 0/7 bands processed successfully")
                    
            except Exception as e:
                logger.error(f"Error processing {hdf_file.name}: {e}")
                failed += 1
                if hdf_file.exists():
                    hdf_file.unlink()
        
        logger.info(f"Processing summary: {successful} successful, {failed} failed, {invalid_files} invalid HDF files skipped")
        return successful, failed
    
    def validate_hdf_file(self, hdf_path):
        """Validate HDF file has valid data before processing."""
        try:
            # Use gdalinfo to check if HDF has valid structure
            cmd = ['gdalinfo', str(hdf_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"HDF file invalid: {hdf_path.name}")
                return False
                
            # Check for valid dimensions in gdalinfo output
            if "Size is 0, 1" in result.stdout or "Size is 1, 0" in result.stdout:
                logger.warning(f"HDF file has invalid dimensions: {hdf_path.name}")
                return False
            
            # Check for required subdatasets (surface reflectance bands)
            required_subdatasets = [
                'sur_refl_b01_1', 'sur_refl_b02_1', 'sur_refl_b03_1', 
                'sur_refl_b04_1', 'sur_refl_b05_1', 'sur_refl_b06_1', 'sur_refl_b07_1'
            ]
            
            missing_bands = []
            for band in required_subdatasets:
                if f'MODIS_Grid_500m_2D:{band}' not in result.stdout:
                    missing_bands.append(band)
            
            if missing_bands:
                logger.warning(f"HDF file missing bands {missing_bands}: {hdf_path.name}")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Error validating HDF {hdf_path.name}: {e}")
            return False

    def convert_and_clip_hdf_band(self, hdf_path, output_path, band):
        """Convert HDF to GeoTIFF for specific band and clip to bounding box."""
        try:
            # Validate HDF file first
            if not self.validate_hdf_file(hdf_path):
                return False
                
            # Use gdal_translate to extract specific band and convert to GeoTIFF
            temp_tif = output_path.parent / f"temp_{output_path.name}"
            
            # Map band names to correct subdataset names
            band_mapping = {
                'B01': 'sur_refl_b01_1',
                'B02': 'sur_refl_b02_1', 
                'B03': 'sur_refl_b03_1',
                'B04': 'sur_refl_b04_1',
                'B05': 'sur_refl_b05_1',
                'B06': 'sur_refl_b06_1',
                'B07': 'sur_refl_b07_1'
            }
            
            subdataset_name = band_mapping.get(band)
            if not subdataset_name:
                logger.error(f"Unknown band: {band}")
                return False
            
            # Extract specific band from MYD09GA using correct subdataset path
            cmd = [
                'gdal_translate',
                f'HDF4_EOS:EOS_GRID:"{hdf_path}":MODIS_Grid_500m_2D:{subdataset_name}',
                str(temp_tif)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"GDAL translate failed for {band} in {hdf_path.name}")
                logger.error(f"STDERR: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout}")
                return False
            
            # Check if temp file was created and has valid dimensions
            if not temp_tif.exists():
                logger.warning(f"Temp file not created for {band} in {hdf_path.name}")
                return False
                
            if temp_tif.stat().st_size < 1000:
                logger.warning(f"Temp file too small ({temp_tif.stat().st_size} bytes) for {band} in {hdf_path.name}")
                temp_tif.unlink()
                return False
            
            # Skip clipping - use full MODIS tile to ensure complete coverage
            # (Clipping was removing data that includes fire points on tile boundaries)
            import shutil
            shutil.move(str(temp_tif), str(output_path))
            logger.debug(f"Successfully processed {band} for {hdf_path.name}")
            return True
                
        except Exception as e:
            logger.error(f"Exception processing {band} in {hdf_path.name}: {e}")
            return False
    
    def clip_to_bbox(self, input_path, output_path):
        """Clip GeoTIFF to bounding box."""
        try:
            with rasterio.open(input_path) as src:
                # Transform bbox to raster's coordinate system
                from rasterio.warp import transform_bounds
                
                # Get raster CRS
                raster_crs = src.crs
                
                # Transform Uttarakhand bbox from WGS84 to raster CRS (not the expanded search bbox)
                uttarakhand_bbox = (77.575402, 28.709556, 81.044789, 31.459016)
                bbox_transformed = transform_bounds('EPSG:4326', raster_crs, *uttarakhand_bbox)
                
                logger.debug(f"Coordinate transformation: Uttarakhand {uttarakhand_bbox} -> {raster_crs} {bbox_transformed}")
                
                # Check if raster overlaps with transformed bbox
                raster_bounds = src.bounds
                if (bbox_transformed[2] < raster_bounds.left or bbox_transformed[0] > raster_bounds.right or
                    bbox_transformed[3] < raster_bounds.bottom or bbox_transformed[1] > raster_bounds.top):
                    logger.debug(f"No overlap: raster bounds {raster_bounds}, transformed bbox {bbox_transformed}")
                    return False
                
                # Create GeoDataFrame for clipping using transformed bbox
                from shapely.geometry import box
                transformed_geom = box(bbox_transformed[0], bbox_transformed[1], 
                                     bbox_transformed[2], bbox_transformed[3])
                bbox_gdf = gpd.GeoDataFrame([1], geometry=[transformed_geom], crs=raster_crs)
                
                # Clip the raster
                clipped_data, clipped_transform = mask(src, bbox_gdf.geometry, crop=True)
                
                # Check if clipping resulted in valid data
                if clipped_data.size == 0 or clipped_data.shape[1] <= 0 or clipped_data.shape[2] <= 0:
                    logger.warning(f"Invalid clipping result: shape={clipped_data.shape}, size={clipped_data.size}")
                    return False
                
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
                
                logger.debug(f"Clipped and saved: {output_path.name}")
                return True
                
        except Exception as e:
            logger.error(f"Error clipping {input_path.name}: {e}")
            return False
    
    def generate_manifest(self):
        """Generate manifest file with all downloaded files."""
        manifest = {
            'dataset': 'MODIS_SR_COMPLETE',
            'product': 'MYD09GA.061',
            'uttarakhand_bbox': [77.575402, 28.709556, 81.044789, 31.459016],
            'search_bbox': [self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n],
            'years': f"{self.year_start}-{self.year_end}",
            'dry_season_months': self.dry_season_months,
            'tiles': self.tiles,
            'bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07'],
            'files': []
        }
        
        for year in range(self.year_start, self.year_end + 1):
            year_dir = self.output_dir / str(year)
            if year_dir.exists():
                for file_path in year_dir.glob("*.tif"):
                    file_info = {
                        'filename': file_path.name,
                        'path': str(file_path),
                        'year': year,
                        'band': file_path.name.split('.')[-2],  # Extract band name
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
        logger.info("Starting MODIS Surface Reflectance COMPLETE data download")
        logger.info(f"Target tiles: {self.tiles}")
        logger.info(f"Search bbox: ({self.bbox_w}, {self.bbox_s}, {self.bbox_e}, {self.bbox_n})")
        
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
            year_dir = self.output_dir / str(year)
            
            # Check if year already has data
            if year_dir.exists() and len(list(year_dir.glob("*.tif"))) > 0:
                logger.info(f"Year {year} already has data, skipping...")
                continue
            
            # Search for files
            results = self.search_modis_sr(start_date, end_date)
            
            # Download files
            successful, failed = self.download_files(results, year)
            total_successful += successful
            total_failed += failed
        
        # Generate manifest
        manifest = self.generate_manifest()
        
        logger.info(f"MODIS SR COMPLETE download finished: {total_successful} successful, {total_failed} failed")
        logger.info(f"Total files: {len(manifest['files'])}")
        logger.info(f"Total size: {sum(f['size_mb'] for f in manifest['files']):.1f} MB")
        
        return total_successful, total_failed

def main():
    """Main entry point."""
    fetcher = MODISSRFetcher()
    successful, failed = fetcher.run()
    
    if failed > 0:
        logger.warning(f"Some downloads failed. Check logs for details.")
        # Don't exit with error if we have some successful downloads
        if successful > 0:
            logger.info(f"Proceeding with {successful} successful downloads")
    else:
        logger.info("All downloads completed successfully!")

if __name__ == "__main__":
    main()
