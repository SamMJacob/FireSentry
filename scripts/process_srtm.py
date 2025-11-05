#!/usr/bin/env python3
"""
SRTM DEM Processing Script

Processes SRTM DEM data to compute slope, aspect, and clip to Uttarakhand region.
Handles .hgt.zip files and creates organized output structure.

Usage:
    python scripts/process_srtm.py

Requirements:
    - rasterio
    - geopandas
    - gdal (for gdaldem commands)
    - SRTM .hgt.zip files in data/raw/srtm/
"""

import os
import sys
import subprocess
import zipfile
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from pathlib import Path
import logging
from dotenv import load_dotenv
import json
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SRTMProcessor:
    def __init__(self, config_path=".env"):
        """Initialize SRTM processor with configuration."""
        self.load_config(config_path)
        self.setup_directories()
        self.bbox_geom = self.create_bbox_geometry()
        
    def load_config(self, config_path):
        """Load configuration from environment file."""
        load_dotenv(dotenv_path=config_path)
        
        # Default values
        self.bbox_n = float(os.getenv('BBOX_N', 31.5))
        self.bbox_s = float(os.getenv('BBOX_S', 28.7))
        self.bbox_e = float(os.getenv('BBOX_E', 81.0))
        self.bbox_w = float(os.getenv('BBOX_W', 77.5))
        
        logger.info(f"Loaded config: bbox=({self.bbox_w}, {self.bbox_s}, {self.bbox_e}, {self.bbox_n})")
    
    def setup_directories(self):
        """Create output directories."""
        self.raw_dir = Path("data/raw/srtm")
        self.derived_dir = Path("data/derived/terrain")
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Raw SRTM directory: {self.raw_dir}")
        logger.info(f"Derived terrain directory: {self.derived_dir}")
    
    def create_bbox_geometry(self):
        """Create bounding box geometry for clipping."""
        return box(self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n)
    
    def extract_srtm_files(self):
        """Extract .hgt.zip files to get .hgt files."""
        logger.info("Extracting SRTM .hgt.zip files...")
        
        extracted_files = []
        zip_files = list(self.raw_dir.glob("*.zip"))
        
        for zip_file in zip_files:
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Extract to same directory
                    zip_ref.extractall(self.raw_dir)
                    
                    # Find the extracted .hgt file
                    hgt_files = [f for f in zip_ref.namelist() if f.endswith('.hgt')]
                    for hgt_file in hgt_files:
                        extracted_path = self.raw_dir / hgt_file
                        if extracted_path.exists():
                            extracted_files.append(extracted_path)
                            logger.info(f"Extracted: {hgt_file}")
                
            except Exception as e:
                logger.error(f"Error extracting {zip_file}: {e}")
        
        logger.info(f"Extracted {len(extracted_files)} .hgt files")
        return extracted_files
    
    def find_srtm_files(self):
        """Find all SRTM files in the raw directory."""
        srtm_files = []
        
        # Look for .hgt files first
        hgt_files = list(self.raw_dir.glob("*.hgt"))
        if hgt_files:
            srtm_files.extend(hgt_files)
        else:
            # If no .hgt files, extract from .zip files
            srtm_files = self.extract_srtm_files()
        
        # Also look for .tif files
        tif_files = list(self.raw_dir.glob("*.tif"))
        srtm_files.extend(tif_files)
        
        logger.info(f"Found {len(srtm_files)} SRTM files")
        for file in srtm_files:
            logger.info(f"  - {file.name}")
        
        return srtm_files
    
    def validate_coverage(self, srtm_files):
        """Validate that SRTM files cover the bounding box."""
        logger.info("Validating SRTM coverage...")
        
        coverage_info = []
        total_bounds = None
        
        for srtm_file in srtm_files:
            try:
                with rasterio.open(srtm_file) as src:
                    # Get bounds
                    bounds = src.bounds
                    
                    # Check if bbox intersects with this file
                    file_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                    intersects = self.bbox_geom.intersects(file_geom)
                    
                    coverage_info.append({
                        'file': srtm_file.name,
                        'bounds': bounds,
                        'intersects': intersects,
                        'crs': src.crs,
                        'shape': src.shape
                    })
                    
                    if intersects:
                        logger.info(f"✓ {srtm_file.name} covers project area")
                        if total_bounds is None:
                            total_bounds = bounds
                        else:
                            # Expand total bounds
                            total_bounds = rasterio.coords.BoundingBox(
                                min(total_bounds.left, bounds.left),
                                min(total_bounds.bottom, bounds.bottom),
                                max(total_bounds.right, bounds.right),
                                max(total_bounds.top, bounds.top)
                            )
                    else:
                        logger.warning(f"✗ {srtm_file.name} does not cover project area")
                        
            except Exception as e:
                logger.error(f"Error reading {srtm_file}: {e}")
        
        if total_bounds:
            logger.info(f"Total coverage bounds: {total_bounds}")
            return coverage_info, total_bounds
        else:
            logger.error("No SRTM files cover the project area!")
            return coverage_info, None
    
    def merge_srtm_files(self, srtm_files, output_path):
        """Merge multiple SRTM files into a single mosaic."""
        logger.info("Merging SRTM files...")
        
        try:
            # Use gdal_merge to create mosaic
            cmd = ['gdal_merge.py', '-o', str(output_path)]
            
            # Add all SRTM files
            for srtm_file in srtm_files:
                cmd.append(str(srtm_file))
            
            # Run gdal_merge
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully merged SRTM files to {output_path}")
                return True
            else:
                logger.error(f"gdal_merge failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error merging SRTM files: {e}")
            return False
    
    def clip_to_bbox(self, input_path, output_path):
        """Clip raster to bounding box."""
        logger.info(f"Clipping {input_path.name} to project area...")
        
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
                
                logger.info(f"Successfully clipped to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error clipping {input_path}: {e}")
            return False
    
    def compute_slope(self, dem_path, slope_path):
        """Compute slope from DEM using gdaldem."""
        logger.info("Computing slope...")
        
        try:
            cmd = [
                'gdaldem', 'slope',
                str(dem_path),
                str(slope_path),
                '-of', 'GTiff',
                '-co', 'COMPRESS=LZW'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully computed slope: {slope_path}")
                return True
            else:
                logger.error(f"gdaldem slope failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error computing slope: {e}")
            return False
    
    def compute_aspect(self, dem_path, aspect_path):
        """Compute aspect from DEM using gdaldem."""
        logger.info("Computing aspect...")
        
        try:
            cmd = [
                'gdaldem', 'aspect',
                str(dem_path),
                str(aspect_path),
                '-of', 'GTiff',
                '-co', 'COMPRESS=LZW'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully computed aspect: {aspect_path}")
                return True
            else:
                logger.error(f"gdaldem aspect failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error computing aspect: {e}")
            return False
    
    def generate_manifest(self, dem_path, slope_path, aspect_path):
        """Generate manifest file with processing information."""
        manifest = {
            'dataset': 'SRTM_DEM',
            'product': 'SRTMGL1.003',
            'bbox': [self.bbox_w, self.bbox_s, self.bbox_e, self.bbox_n],
            'files': {
                'elevation': {
                    'filename': dem_path.name,
                    'path': str(dem_path.relative_to(Path.cwd())),
                    'description': 'Digital Elevation Model (meters)'
                },
                'slope': {
                    'filename': slope_path.name,
                    'path': str(slope_path.relative_to(Path.cwd())),
                    'description': 'Slope in degrees (0-90)'
                },
                'aspect': {
                    'filename': aspect_path.name,
                    'path': str(aspect_path.relative_to(Path.cwd())),
                    'description': 'Aspect in degrees (0-360, 0=North)'
                }
            }
        }
        
        # Add file sizes
        for file_type, file_info in manifest['files'].items():
            file_path = Path(file_info['path'])
            if file_path.exists():
                file_info['size_mb'] = file_path.stat().st_size / (1024 * 1024)
        
        manifest_path = self.derived_dir / "srtm_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Generated manifest: {manifest_path}")
        return manifest
    
    def run(self):
        """Main execution function."""
        logger.info("Starting SRTM DEM processing")
        
        # Find SRTM files
        srtm_files = self.find_srtm_files()
        if not srtm_files:
            logger.error("No SRTM files found!")
            return False
        
        # Validate coverage
        coverage_info, total_bounds = self.validate_coverage(srtm_files)
        if total_bounds is None:
            logger.error("No SRTM files cover the project area!")
            return False
        
        # Define output paths
        dem_path = self.derived_dir / "elevation.tif"
        slope_path = self.derived_dir / "slope.tif"
        aspect_path = self.derived_dir / "aspect.tif"
        
        # Process files
        if len(srtm_files) == 1:
            # Single file - just clip
            logger.info("Single SRTM file found, clipping directly...")
            if not self.clip_to_bbox(srtm_files[0], dem_path):
                return False
        else:
            # Multiple files - merge first, then clip
            temp_mosaic = self.derived_dir / "temp_mosaic.tif"
            if not self.merge_srtm_files(srtm_files, temp_mosaic):
                return False
            
            if not self.clip_to_bbox(temp_mosaic, dem_path):
                return False
            
            # Clean up temp file
            if temp_mosaic.exists():
                temp_mosaic.unlink()
        
        # Compute slope and aspect
        if not self.compute_slope(dem_path, slope_path):
            return False
        
        if not self.compute_aspect(dem_path, aspect_path):
            return False
        
        # Generate manifest
        manifest = self.generate_manifest(dem_path, slope_path, aspect_path)
        
        logger.info("SRTM processing complete!")
        logger.info(f"Output files:")
        logger.info(f"  - Elevation: {dem_path}")
        logger.info(f"  - Slope: {slope_path}")
        logger.info(f"  - Aspect: {aspect_path}")
        
        return True

def main():
    """Main entry point."""
    processor = SRTMProcessor()
    success = processor.run()
    
    if not success:
        logger.error("SRTM processing failed!")
        sys.exit(1)
    else:
        logger.info("SRTM processing completed successfully!")

if __name__ == "__main__":
    main()



