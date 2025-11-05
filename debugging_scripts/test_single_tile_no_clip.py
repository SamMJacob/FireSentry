#!/usr/bin/env python3
"""
Test script to verify that processing without clipping will fix the issue.
Downloads and processes a single MODIS tile to test.
"""

import earthaccess
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS

# Test parameters
test_date = datetime(2020, 3, 1)  # March 1, 2020 (DOY 61)
test_tile = "h25v05"
test_point = (30.3165, 78.0322)  # Dehradun

print("=" * 70)
print("MODIS SR Test - Processing WITHOUT Clipping")
print("=" * 70)
print(f"Test date: {test_date.date()}")
print(f"Test tile: {test_tile}")
print(f"Test point: {test_point} (Dehradun)")
print()

# Create test directory
test_dir = Path("test_modis_no_clip")
test_dir.mkdir(exist_ok=True)

print("Step 1: Authenticating with NASA Earthdata...")
try:
    auth = earthaccess.login()
    if not auth:
        print("❌ Authentication failed")
        exit(1)
    print("✓ Authenticated")
except Exception as e:
    print(f"❌ Authentication error: {e}")
    exit(1)

print()
print("Step 2: Searching for MODIS SR data...")
try:
    results = earthaccess.search_data(
        short_name='MYD09GA',
        version='061',
        temporal=(test_date, test_date),
        count=100
    )
    
    # Find our specific tile
    target_granule = None
    for result in results:
        links = result.data_links()
        if links:
            filename = links[0].split('/')[-1]
            if test_tile in filename and f"A{test_date.year}{test_date.timetuple().tm_yday:03d}" in filename:
                target_granule = result
                print(f"✓ Found: {filename}")
                break
    
    if not target_granule:
        print(f"❌ Could not find tile {test_tile} for {test_date.date()}")
        exit(1)
        
except Exception as e:
    print(f"❌ Search error: {e}")
    exit(1)

print()
print("Step 3: Downloading HDF file...")
try:
    downloaded = earthaccess.download([target_granule], local_path=str(test_dir))
    if not downloaded or len(downloaded) == 0:
        print("❌ Download failed")
        exit(1)
    
    hdf_file = Path(downloaded[0])
    print(f"✓ Downloaded: {hdf_file.name}")
    print(f"  Size: {hdf_file.stat().st_size / (1024*1024):.1f} MB")
except Exception as e:
    print(f"❌ Download error: {e}")
    exit(1)

print()
print("Step 4: Converting HDF to GeoTIFF (B01 - Red band) WITHOUT clipping...")
try:
    temp_tif = test_dir / "test_B01.tif"
    
    cmd = [
        'gdal_translate',
        f'HDF4_EOS:EOS_GRID:"{hdf_file}":MODIS_Grid_500m_2D:sur_refl_b01_1',
        str(temp_tif)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ gdal_translate failed")
        print(f"STDERR: {result.stderr}")
        exit(1)
    
    print(f"✓ Converted to GeoTIFF: {temp_tif.name}")
    print(f"  Size: {temp_tif.stat().st_size / (1024*1024):.1f} MB")
except Exception as e:
    print(f"❌ Conversion error: {e}")
    exit(1)

print()
print("Step 5: Checking if test point is within the FULL (unclipped) tile...")
try:
    with rasterio.open(temp_tif) as src:
        print(f"  Raster CRS: {src.crs}")
        print(f"  Raster shape: {src.shape}")
        print(f"  Raster bounds: {src.bounds}")
        
        # Transform test point to raster CRS
        x, y = transform(
            CRS.from_epsg(4326),
            src.crs,
            [test_point[1]], [test_point[0]]
        )
        
        print(f"\n  Test point in WGS84: ({test_point[1]}, {test_point[0]})")
        print(f"  Test point in raster CRS: ({x[0]:.2f}, {y[0]:.2f})")
        
        # Check bounds
        within = (src.bounds.left <= x[0] <= src.bounds.right and 
                  src.bounds.bottom <= y[0] <= src.bounds.top)
        
        print(f"\n  Left bound: {src.bounds.left:.2f}, Point X: {x[0]:.2f}, Right bound: {src.bounds.right:.2f}")
        print(f"  Bottom bound: {src.bounds.bottom:.2f}, Point Y: {y[0]:.2f}, Top bound: {src.bounds.top:.2f}")
        
        if within:
            print("\n✓✓✓ SUCCESS! Test point IS within the unclipped raster bounds ✓✓✓")
            
            # Try to extract value
            row, col = src.index(x[0], y[0])
            if 0 <= row < src.height and 0 <= col < src.width:
                value = src.read(1)[row, col]
                print(f"\n  Pixel value at test point: {value}")
                print(f"  Row: {row}, Col: {col}")
            else:
                print(f"\n  ⚠ Row/Col outside raster: row={row}, col={col}")
        else:
            print("\n✗✗✗ FAILED! Test point is STILL outside raster bounds ✗✗✗")
            print("  This means the issue is not with clipping.")
            
except Exception as e:
    print(f"❌ Check error: {e}")
    exit(1)

print()
print("=" * 70)
if within:
    print("CONCLUSION: Downloading without clipping WILL FIX the issue!")
    print("  Proceed with full download using the updated script.")
else:
    print("CONCLUSION: The issue is NOT with clipping alone.")
    print("  Further investigation needed.")
print("=" * 70)

# Cleanup
print(f"\nCleaning up test directory: {test_dir}")
shutil.rmtree(test_dir)
print("✓ Cleanup complete")


