#!/usr/bin/env python3
"""
Find dates with available MODIS SR data for the test point
"""

from datetime import datetime, timedelta
from pathlib import Path
import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS

# Test parameters
lat, lon = 30.3165, 78.0322

print(f"Searching for MODIS SR data for point ({lat}, {lon})")
print("="*60)

# Calculate tile
h = int((lon + 180) / 10)
v = int((90 - lat) / 10)
tile = f"h{h:02d}v{v:02d}"
print(f"Calculated tile: {tile}")

# Check if h24 tiles exist (the correct tile for this point)
year_dir = Path("data/raw/modis_sr/2020")
if year_dir.exists():
    # Look for any h24 tiles
    h24_files = list(year_dir.glob("MYD09GA.AA2020*.h24v05.061.B01.tif"))
    print(f"\nFound {len(h24_files)} h24v05 files in 2020")
    
    if h24_files:
        print("Sample h24v05 files:")
        for file in h24_files[:5]:
            print(f"  {file.name}")
        
        # Test one of the h24v05 files
        test_file = h24_files[0]
        print(f"\nTesting file: {test_file.name}")
        
        try:
            with rasterio.open(test_file) as src:
                print(f"  CRS: {src.crs}")
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
                            print(f"  ✅ Data available for this point!")
                        else:
                            print(f"  NoData value")
                    else:
                        print(f"  Row/Col out of bounds")
                else:
                    print(f"  Point outside raster bounds")
                    
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("No h24v05 files found - checking h25v05 files...")
        
        # Check h25v05 files
        h25_files = list(year_dir.glob("MYD09GA.AA2020*.h25v05.061.B01.tif"))
        print(f"Found {len(h25_files)} h25v05 files in 2020")
        
        if h25_files:
            print("Sample h25v05 files:")
            for file in h25_files[:5]:
                print(f"  {file.name}")
            
            # Test one of the h25v05 files
            test_file = h25_files[0]
            print(f"\nTesting file: {test_file.name}")
            
            try:
                with rasterio.open(test_file) as src:
                    print(f"  CRS: {src.crs}")
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
                                print(f"  ✅ Data available for this point!")
                            else:
                                print(f"  NoData value")
                        else:
                            print(f"  Row/Col out of bounds")
                    else:
                        print(f"  Point outside raster bounds")
                        
            except Exception as e:
                print(f"  Error: {e}")
else:
    print("MODIS SR directory not found!")


