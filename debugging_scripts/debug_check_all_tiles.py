#!/usr/bin/env python3
"""
Download and check the georeference of ALL available tiles to find which one
actually contains Dehradun (78°E, 30°N).
"""

import earthaccess
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

test_dir = Path("test_all_tiles")
test_dir.mkdir(exist_ok=True)

print("=" * 70)
print("Checking georeference of all available MODIS tiles")
print("=" * 70)
print()

# Authenticate
auth = earthaccess.login()

# Search for all tiles on this date
results = earthaccess.search_data(
    short_name='MYD09GA',
    version='061',
    temporal=(datetime(2020, 3, 1), datetime(2020, 3, 1)),
    bounding_box=(77.0, 28.5, 83.5, 31.5),  # Expanded search area
    count=1000
)

print(f"Found {len(results)} results")
print()

# Extract tile names
tiles_found = {}
for result in results:
    links = result.data_links()
    if links:
        filename = links[0].split('/')[-1]
        # Extract tile from filename (e.g., h25v05)
        parts = filename.split('.')
        for part in parts:
            if part.startswith('h') and 'v' in part and len(part) == 6:
                if 'A2020061' in filename:  # DOY 61 = March 1
                    tiles_found[part] = result
                break

print(f"Tiles available for March 1, 2020:")
for tile in sorted(tiles_found.keys()):
    print(f"  {tile}")
print()

# Download and check a few tiles
tiles_to_check = [t for t in sorted(tiles_found.keys()) if t.startswith(('h24', 'h25', 'h26', 'h27'))][:6]

print(f"Checking georeference of {len(tiles_to_check)} tiles...")
print()

target_lon, target_lat = 78.0322, 30.3165
print(f"Target point (Dehradun): {target_lon}°E, {target_lat}°N")
print("=" * 70)
print()

for tile in tiles_to_check:
    if tile not in tiles_found:
        continue
        
    print(f"Tile {tile}:")
    
    try:
        # Download
        downloaded = earthaccess.download([tiles_found[tile]], local_path=str(test_dir))
        hdf_file = Path(downloaded[0])
        
        # Check georeference
        subdataset = f'HDF4_EOS:EOS_GRID:"{hdf_file}":MODIS_Grid_500m_2D:sur_refl_b01_1'
        cmd = ['gdalinfo', subdataset]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extract bounds
            for line in result.stdout.split('\n'):
                if 'Upper Left' in line:
                    # Extract lon, lat from: Upper Left  ( 7783653.638, 4447802.079) ( 91d22'42.64"E, 40d 0' 0.00"N)
                    parts = line.split('(')
                    if len(parts) >= 3:
                        coords = parts[2].split(')')[0].strip()
                        print(f"  Upper Left (geographic): {coords}")
                elif 'Lower Right' in line:
                    parts = line.split('(')
                    if len(parts) >= 3:
                        coords = parts[2].split(')')[0].strip()
                        print(f"  Lower Right (geographic): {coords}")
                        
                        # Parse longitude to check if target is in range
                        try:
                            lon_str = coords.split(',')[0].strip()
                            # Convert from DMS to decimal (rough approximation)
                            if 'd' in lon_str:
                                deg = float(lon_str.split('d')[0])
                                ul_line = [l for l in result.stdout.split('\n') if 'Upper Left' in l][0]
                                ul_coords = ul_line.split('(')[2].split(')')[0].strip()
                                ul_lon = float(ul_coords.split(',')[0].split('d')[0])
                                
                                if ul_lon <= target_lon <= deg or deg <= target_lon <= ul_lon:
                                    print(f"  ✓✓✓ THIS TILE MIGHT CONTAIN DEHRADUN! ✓✓✓")
                        except:
                            pass
        
        # Cleanup
        hdf_file.unlink()
        print()
        
    except Exception as e:
        print(f"  Error: {e}")
        print()

# Cleanup
shutil.rmtree(test_dir)
print("✓ Cleanup complete")

