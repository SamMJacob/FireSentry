#!/usr/bin/env python3
"""
Quick CHIRPS Data Test

Tests that CHIRPS data can be read and point extraction works.
"""

import sys
from pathlib import Path
from datetime import datetime
import rasterio

sys.path.append(str(Path(__file__).parent.parent))

def test_chirps_file():
    """Test reading a CHIRPS file."""
    # Test with day 1 of 2020 (Jan 1, 2020)
    test_file = Path("data/raw/chirps/2020/chirps-v2.0.2020.001.tif")
    
    print("="*60)
    print("CHIRPS FILE TEST")
    print("="*60)
    print(f"Testing file: {test_file}")
    
    if not test_file.exists():
        print(f"❌ File not found: {test_file}")
        return False
    
    print(f"✅ File exists")
    
    try:
        with rasterio.open(test_file) as src:
            print(f"\nFile Information:")
            print(f"  CRS: {src.crs}")
            print(f"  Bounds: {src.bounds}")
            print(f"  Shape: {src.shape} (height x width)")
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  NoData value: {src.nodata}")
            
            # Test point in Uttarakhand (Dehradun)
            test_lat = 30.3165
            test_lon = 78.0322
            
            print(f"\nTest Point Extraction:")
            print(f"  Location: ({test_lat}, {test_lon}) - Dehradun area")
            
            # Get row, col for the point
            row, col = src.index(test_lon, test_lat)
            
            print(f"  Raster indices: row={row}, col={col}")
            print(f"  Raster dimensions: height={src.height}, width={src.width}")
            
            # Check bounds
            if 0 <= row < src.height and 0 <= col < src.width:
                # Read the value
                value = src.read(1)[row, col]
                print(f"  ✅ Point is within raster bounds")
                print(f"  Precipitation value: {value:.2f} mm")
                return True
            else:
                print(f"  ❌ Point is OUTSIDE raster bounds")
                print(f"     This means CHIRPS files might not cover Uttarakhand properly")
                return False
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def test_dtw_filename_format():
    """Test that DTW filename format matches actual files."""
    from datetime import datetime
    
    print("\n" + "="*60)
    print("DTW FILENAME FORMAT TEST")
    print("="*60)
    
    test_date = datetime(2020, 1, 1)
    doy = test_date.timetuple().tm_yday
    filename = f"chirps-v2.0.{test_date.year}.{doy:03d}.tif"
    filepath = Path("data/raw/chirps") / str(test_date.year) / filename
    
    print(f"Test date: {test_date.date()}")
    print(f"Day of year: {doy}")
    print(f"Generated filename: {filename}")
    print(f"Full path: {filepath}")
    
    if filepath.exists():
        print(f"✅ File exists - DTW format is correct!")
        return True
    else:
        print(f"❌ File not found - DTW format is incorrect!")
        
        # Check what files actually exist
        year_dir = Path("data/raw/chirps") / str(test_date.year)
        if year_dir.exists():
            files = list(year_dir.glob("*.tif"))[:5]
            print(f"\nActual files in directory (first 5):")
            for f in files:
                print(f"  {f.name}")
        return False

def main():
    """Run all quick tests."""
    print("\n" + "="*80)
    print("FIRESENTRY CHIRPS QUICK TEST")
    print("="*80)
    
    success = True
    success = test_dtw_filename_format() and success
    success = test_chirps_file() and success
    
    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED - CHIRPS data is ready!")
    else:
        print("❌ TESTS FAILED - Check issues above")
    print("="*80 + "\n")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



