#!/usr/bin/env python3
"""
Debug vegetation indices file finding
"""

from datetime import datetime
from pathlib import Path
from features.indices import VegetationIndices

# Test parameters
lat, lon = 30.3165, 78.0322
date = datetime(2020, 3, 15)
doy = date.timetuple().tm_yday

print(f"Testing vegetation indices file finding:")
print(f"  Point: ({lat}, {lon})")
print(f"  Date: {date.date()}")
print(f"  DOY: {doy}")

# Calculate tile
h = int((lon + 180) / 10)
v = int((90 - lat) / 10)
tile = f"h{h:02d}v{v:02d}"
print(f"  Calculated tile: {tile}")

# Initialize vegetation indices
vi = VegetationIndices()

# Test file finding
print(f"\n1. Testing band file finding:")
band_files = vi.find_modis_sr_band_files(lat, lon, date)
print(f"   Found band files: {len(band_files)}")
for band_name, file_path in band_files.items():
    print(f"     {band_name}: {file_path}")

if band_files:
    print(f"\n2. Testing band extraction:")
    bands = vi.extract_modis_bands(band_files, lat, lon)
    print(f"   Extracted bands:")
    for band_name, value in bands.items():
        print(f"     {band_name}: {value}")
    
    print(f"\n3. Testing index calculation:")
    indices = vi.calculate_indices_for_date(lat, lon, date)
    print(f"   Calculated indices:")
    for index_name, value in indices.items():
        if index_name != 'bands':  # Skip debug bands
            print(f"     {index_name}: {value}")
else:
    print("   No band files found!")
    
    # Let's check what files actually exist
    print(f"\n4. Checking what files exist for this date:")
    year_dir = Path("data/raw/modis_sr/2020")
    if year_dir.exists():
        # Look for files with this DOY
        pattern = f"MYD09GA.AA{date.year}{doy:03d}.*.tif"
        matching_files = list(year_dir.glob(pattern))
        print(f"   Files matching pattern {pattern}: {len(matching_files)}")
        for file in matching_files[:10]:  # Show first 10
            print(f"     {file.name}")
        if len(matching_files) > 10:
            print(f"     ... and {len(matching_files) - 10} more")
    else:
        print(f"   Year directory {year_dir} does not exist!")


