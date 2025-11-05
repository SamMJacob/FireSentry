#!/usr/bin/env python3
"""
Check what subdatasets are in the MODIS HDF file and verify we're extracting the right one.
"""

import subprocess
from pathlib import Path
import earthaccess
from datetime import datetime

# Download one HDF file for inspection
test_dir = Path("test_hdf_inspect")
test_dir.mkdir(exist_ok=True)

print("Downloading test HDF file...")
auth = earthaccess.login()

results = earthaccess.search_data(
    short_name='MYD09GA',
    version='061',
    temporal=(datetime(2020, 3, 1), datetime(2020, 3, 1)),
    count=100
)

# Find h25v05
target = None
for result in results:
    links = result.data_links()
    if links:
        filename = links[0].split('/')[-1]
        if 'h25v05' in filename and 'A2020061' in filename:
            target = result
            break

if target:
    downloaded = earthaccess.download([target], local_path=str(test_dir))
    hdf_file = Path(downloaded[0])
    print(f"✓ Downloaded: {hdf_file.name}\n")
    
    # List all subdatasets
    print("=" * 70)
    print("Listing all subdatasets in HDF file:")
    print("=" * 70)
    
    cmd = ['gdalinfo', str(hdf_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Extract subdataset lines
        lines = result.stdout.split('\n')
        in_subdatasets = False
        subdatasets = []
        
        for line in lines:
            if 'Subdatasets:' in line:
                in_subdatasets = True
                continue
            if in_subdatasets:
                if line.strip().startswith('SUBDATASET_'):
                    if '_NAME=' in line:
                        subdataset = line.split('=', 1)[1]
                        subdatasets.append(subdataset)
                elif not line.strip():
                    break
        
        for i, sd in enumerate(subdatasets, 1):
            print(f"{i}. {sd}")
        
        print()
        print("=" * 70)
        print("Checking georeference of the subdataset we're using:")
        print("=" * 70)
        
        # Check the specific subdataset we extract
        subdataset = f'HDF4_EOS:EOS_GRID:"{hdf_file}":MODIS_Grid_500m_2D:sur_refl_b01_1'
        print(f"\nSubdataset: {subdataset}\n")
        
        cmd = ['gdalinfo', subdataset]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extract key information
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['Size is', 'Origin =', 'Pixel Size =', 'Upper Left', 'Lower Right', 'Center']):
                    print(line)
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("\n✓ Cleanup complete")


