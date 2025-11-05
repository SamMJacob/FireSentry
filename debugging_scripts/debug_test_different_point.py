#!/usr/bin/env python3
"""
Test with a different point that should have data available
"""

from datetime import datetime
from pathlib import Path
import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS

# Test with a point that should be within h25v05 bounds
# Let's try a point in the center of the h25v05 tile
# h25v05 covers roughly 78-88°E, 30-40°N
lat, lon = 35.0, 83.0  # Center of h25v05 tile

print(f"Testing with different point: ({lat}, {lon})")
print("="*60)

# Calculate tile
h = int((lon + 180) / 10)
v = int((90 - lat) / 10)
tile = f"h{h:02d}v{v:02d}"
print(f"Calculated tile: {tile}")

# Test with a known date that has data
date = datetime(2020, 3, 1)  # DOY 61
doy = date.timetuple().tm_yday
print(f"Test date: {date.date()} (DOY: {doy})")

# Check if files exist
year_dir = Path("data/raw/modis_sr/2020")
band_mapping = {
    'red': 'B01',
    'nir': 'B02', 
    'blue': 'B03',
    'swir1': 'B06',
    'swir2': 'B07'
}

print(f"\nChecking files for tile {tile}:")
all_files_exist = True
for band_name, band_code in band_mapping.items():
    band_file = f"MYD09GA.AA{date.year}{doy:03d}.{tile}.061.{band_code}.tif"
    filepath = year_dir / band_file
    exists = filepath.exists()
    print(f"  {band_name} ({band_code}): {exists}")
    if not exists:
        all_files_exist = False

if all_files_exist:
    print(f"\n✅ All band files exist for this point and date!")
    
    # Test coordinate transformation
    test_file = year_dir / f"MYD09GA.AA{date.year}{doy:03d}.{tile}.061.B01.tif"
    try:
        with rasterio.open(test_file) as src:
            print(f"\nTesting coordinate transformation:")
            print(f"  File: {test_file.name}")
            print(f"  Bounds: {src.bounds}")
            
            # Transform coordinates
            x, y = transform(
                CRS.from_epsg(4326),  # WGS84
                src.crs,              # MODIS Sinusoidal
                [lon], [lat]
            )
            
            print(f"  Transformed coordinates: ({x[0]:.2f}, {y[0]:.2f})")
            
            # Check if point is within bounds
            bounds = src.bounds
            in_bounds = (bounds.left <= x[0] <= bounds.right and 
                        bounds.bottom <= y[0] <= bounds.top)
            print(f"  Within bounds: {in_bounds}")
            
            if in_bounds:
                row, col = src.index(x[0], y[0])
                print(f"  Row, Col: ({row}, {col})")
                
                if 0 <= row < src.height and 0 <= col < src.width:
                    value = src.read(1)[row, col]
                    print(f"  Raw value: {value}")
                    if value != src.nodata:
                        scaled_value = value / 10000.0
                        print(f"  Scaled value: {scaled_value}")
                        print(f"  ✅ This point should work for vegetation indices!")
                    else:
                        print(f"  NoData value")
                else:
                    print(f"  Row/Col out of bounds")
            else:
                print(f"  Point outside raster bounds")
                
    except Exception as e:
        print(f"  Error: {e}")
else:
    print(f"\n❌ Some band files are missing for this point and date")


