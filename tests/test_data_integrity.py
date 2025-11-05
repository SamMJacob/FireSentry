#!/usr/bin/env python3
"""
Test Data Integrity

Verifies that all required data exists and is accessible before running the pipeline.
Tests each data source individually to catch issues early.

Usage:
    python tests/test_data_integrity.py
"""

import sys
from pathlib import Path
from datetime import datetime
import rasterio
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class DataIntegrityTester:
    """Test suite for data integrity checks."""
    
    def __init__(self):
        self.results = {
            'firms': False,
            'chirps': False,
            'srtm': False,
            'terrain': False,
            'modis_sr': False,
            'modis_lst': False
        }
        self.errors = []
        
    def test_firms_data(self):
        """Test FIRMS fire data."""
        print("\n" + "="*60)
        print("TESTING FIRMS DATA")
        print("="*60)
        
        firms_file = Path("data/raw/firms/firms_uttarakhand_2020_2025.csv")
        
        if not firms_file.exists():
            self.errors.append(f"❌ FIRMS file not found: {firms_file}")
            print(f"❌ FIRMS file not found: {firms_file}")
            return False
        
        try:
            df = pd.read_csv(firms_file)
            print(f"✅ FIRMS file loaded: {len(df)} records")
            
            # Check required columns
            required_cols = ['latitude', 'longitude', 'acq_date', 'confidence']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                self.errors.append(f"❌ Missing columns: {missing}")
                print(f"❌ Missing columns: {missing}")
                return False
            
            print(f"✅ All required columns present")
            
            # Check date range
            df['date'] = pd.to_datetime(df['acq_date'])
            print(f"✅ Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Check bbox
            bbox = {
                'lat_min': df['latitude'].min(),
                'lat_max': df['latitude'].max(),
                'lon_min': df['longitude'].min(),
                'lon_max': df['longitude'].max()
            }
            print(f"✅ Bounding box: ({bbox['lon_min']:.2f}, {bbox['lat_min']:.2f}) to ({bbox['lon_max']:.2f}, {bbox['lat_max']:.2f})")
            
            self.results['firms'] = True
            return True
            
        except Exception as e:
            self.errors.append(f"❌ Error reading FIRMS: {e}")
            print(f"❌ Error reading FIRMS: {e}")
            return False
    
    def test_chirps_data(self):
        """Test CHIRPS precipitation data."""
        print("\n" + "="*60)
        print("TESTING CHIRPS DATA")
        print("="*60)
        
        chirps_dir = Path("data/raw/chirps")
        
        if not chirps_dir.exists():
            self.errors.append(f"❌ CHIRPS directory not found: {chirps_dir}")
            print(f"❌ CHIRPS directory not found: {chirps_dir}")
            return False
        
        # Check years
        years = [2019, 2020, 2021, 2022, 2023, 2024]
        missing_years = []
        
        for year in years:
            year_dir = chirps_dir / str(year)
            if not year_dir.exists():
                missing_years.append(year)
        
        if missing_years:
            print(f"⚠️  Missing years: {missing_years}")
        else:
            print(f"✅ All years present: {years}")
        
        # Test a sample file
        test_files = list(chirps_dir.glob("2020/*.tif"))
        if not test_files:
            self.errors.append("❌ No CHIRPS files found in 2020")
            print("❌ No CHIRPS files found in 2020")
            return False
        
        test_file = test_files[0]
        print(f"✅ Found {len(test_files)} files in 2020")
        print(f"   Testing file: {test_file.name}")
        
        try:
            with rasterio.open(test_file) as src:
                print(f"   CRS: {src.crs}")
                print(f"   Bounds: {src.bounds}")
                print(f"   Shape: {src.shape}")
                print(f"   Data type: {src.dtypes[0]}")
                
                # Read a small sample
                data = src.read(1, window=((0, 10), (0, 10)))
                print(f"   Sample values: min={data.min():.2f}, max={data.max():.2f}")
                
            print(f"✅ CHIRPS files are readable")
            self.results['chirps'] = True
            return True
            
        except Exception as e:
            self.errors.append(f"❌ Error reading CHIRPS: {e}")
            print(f"❌ Error reading CHIRPS: {e}")
            return False
    
    def test_chirps_point_extraction(self):
        """Test extracting precipitation value from CHIRPS for a known point."""
        print("\n" + "="*60)
        print("TESTING CHIRPS POINT EXTRACTION")
        print("="*60)
        
        # Test point in Uttarakhand (Dehradun area)
        test_lat = 30.3165
        test_lon = 78.0322
        test_date = datetime(2020, 1, 15)
        
        print(f"Test point: ({test_lat}, {test_lon}) on {test_date.date()}")
        
        # Find the file
        doy = test_date.timetuple().tm_yday
        filename = f"chirps-v2.0.{test_date.year}.{doy:03d}.tif"
        filepath = Path("data/raw/chirps") / str(test_date.year) / filename
        
        print(f"Looking for file: {filepath}")
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return False
        
        print(f"✅ File found")
        
        try:
            with rasterio.open(filepath) as src:
                # Get row, col for the point
                row, col = src.index(test_lon, test_lat)
                
                print(f"   Row: {row}, Col: {col}")
                print(f"   Raster shape: {src.shape}")
                
                # Check bounds
                if 0 <= row < src.height and 0 <= col < src.width:
                    # Read the value
                    value = src.read(1)[row, col]
                    print(f"✅ Point is within raster bounds")
                    print(f"✅ Precipitation value: {value:.2f} mm")
                    return True
                else:
                    print(f"❌ Point ({test_lat}, {test_lon}) outside raster bounds")
                    print(f"   Raster height: {src.height}, width: {src.width}")
                    return False
                    
        except Exception as e:
            self.errors.append(f"❌ Error extracting from CHIRPS: {e}")
            print(f"❌ Error extracting from CHIRPS: {e}")
            return False
    
    def test_srtm_terrain(self):
        """Test SRTM terrain data."""
        print("\n" + "="*60)
        print("TESTING SRTM TERRAIN DATA")
        print("="*60)
        
        terrain_dir = Path("data/derived/terrain")
        required_files = ["elevation.tif", "slope.tif", "aspect.tif"]
        
        all_exist = True
        for filename in required_files:
            filepath = terrain_dir / filename
            if filepath.exists():
                print(f"✅ {filename} exists")
                
                # Test file
                try:
                    with rasterio.open(filepath) as src:
                        print(f"   Shape: {src.shape}, CRS: {src.crs}")
                except Exception as e:
                    print(f"❌ Error reading {filename}: {e}")
                    all_exist = False
            else:
                print(f"❌ {filename} not found")
                self.errors.append(f"Missing terrain file: {filename}")
                all_exist = False
        
        self.results['terrain'] = all_exist
        return all_exist
    
    def test_modis_sr(self):
        """Test MODIS Surface Reflectance data."""
        print("\n" + "="*60)
        print("TESTING MODIS SR DATA (OPTIONAL)")
        print("="*60)
        
        modis_sr_dir = Path("data/raw/modis_sr")
        
        if not modis_sr_dir.exists():
            print("⚠️  MODIS SR directory not found (optional for MVP)")
            return False
        
        # Check for files
        tif_files = list(modis_sr_dir.glob("**/*.tif"))
        if tif_files:
            print(f"✅ Found {len(tif_files)} MODIS SR TIF files")
            self.results['modis_sr'] = True
            return True
        else:
            print("⚠️  No MODIS SR files found (optional for MVP)")
            return False
    
    def test_modis_lst(self):
        """Test MODIS LST data."""
        print("\n" + "="*60)
        print("TESTING MODIS LST DATA (OPTIONAL)")
        print("="*60)
        
        modis_lst_dir = Path("data/raw/modis_lst")
        
        if not modis_lst_dir.exists():
            print("⚠️  MODIS LST directory not found (optional for MVP)")
            return False
        
        # Check for files
        tif_files = list(modis_lst_dir.glob("**/*.tif"))
        if tif_files:
            print(f"✅ Found {len(tif_files)} MODIS LST TIF files")
            self.results['modis_lst'] = True
            return True
        else:
            print("⚠️  No MODIS LST files found (optional for MVP)")
            return False
    
    def run_all_tests(self):
        """Run all data integrity tests."""
        print("\n" + "="*80)
        print("FIRESENTRY DATA INTEGRITY TEST SUITE")
        print("="*80)
        
        # Critical tests
        self.test_firms_data()
        self.test_chirps_data()
        self.test_chirps_point_extraction()
        self.test_srtm_terrain()
        
        # Optional tests
        self.test_modis_sr()
        self.test_modis_lst()
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        critical_passed = (
            self.results['firms'] and 
            self.results['chirps'] and 
            self.results['terrain']
        )
        
        print("\nCritical Components:")
        print(f"  FIRMS:   {'✅ PASS' if self.results['firms'] else '❌ FAIL'}")
        print(f"  CHIRPS:  {'✅ PASS' if self.results['chirps'] else '❌ FAIL'}")
        print(f"  Terrain: {'✅ PASS' if self.results['terrain'] else '❌ FAIL'}")
        
        print("\nOptional Components:")
        print(f"  MODIS SR:  {'✅ PASS' if self.results['modis_sr'] else '⚠️  SKIP'}")
        print(f"  MODIS LST: {'✅ PASS' if self.results['modis_lst'] else '⚠️  SKIP'}")
        
        if self.errors:
            print("\n❌ ERRORS FOUND:")
            for error in self.errors:
                print(f"  {error}")
        
        if critical_passed:
            print("\n✅ ALL CRITICAL TESTS PASSED - Ready for feature pipeline!")
            return True
        else:
            print("\n❌ CRITICAL TESTS FAILED - Fix issues before running pipeline")
            return False

def main():
    """Main entry point."""
    tester = DataIntegrityTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

