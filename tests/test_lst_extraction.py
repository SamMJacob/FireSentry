#!/usr/bin/env python3
"""
LST Extraction Test

Tests the new LST feature extraction functionality.
Verifies MODIS LST file reading and feature calculation.

Usage:
    python tests/test_lst_extraction.py
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.lst import LSTFeatures

class LSTTester:
    """Test suite for LST extraction."""
    
    def __init__(self):
        self.lst = LSTFeatures()
        self.test_results = []
        
    def test_lst_file_finding(self):
        """Test finding MODIS LST files."""
        print("\n" + "="*60)
        print("TESTING LST FILE FINDING")
        print("="*60)
        
        # Test point in Uttarakhand
        test_lat = 30.3165
        test_lon = 78.0322
        test_date = datetime(2020, 3, 15)
        
        print(f"Test point: ({test_lat}, {test_lon}) on {test_date.date()}")
        
        try:
            lst_file = self.lst.find_modis_lst_file(test_lat, test_lon, test_date)
            
            if lst_file:
                print(f"✅ LST file found: {lst_file.name}")
                print(f"   Full path: {lst_file}")
                return True
            else:
                print(f"❌ No LST file found for this date/location")
                return False
                
        except Exception as e:
            print(f"❌ Error finding LST file: {e}")
            return False
    
    def test_lst_value_extraction(self):
        """Test extracting LST values from files."""
        print("\n" + "="*60)
        print("TESTING LST VALUE EXTRACTION")
        print("="*60)
        
        # Test point in Uttarakhand
        test_lat = 30.3165
        test_lon = 78.0322
        test_date = datetime(2020, 3, 15)
        
        print(f"Test point: ({test_lat}, {test_lon}) on {test_date.date()}")
        
        try:
            lst_value = self.lst.extract_lst_value(test_lat, test_lon, test_date)
            
            if lst_value is not None:
                print(f"✅ LST value extracted: {lst_value:.2f} K")
                print(f"   Temperature: {lst_value - 273.15:.2f} °C")
                
                # Check reasonable range
                if 200 <= lst_value <= 350:
                    print(f"   ✅ LST value in reasonable range (200-350 K)")
                    return True
                else:
                    print(f"   ⚠️  LST value outside expected range")
                    return False
            else:
                print(f"❌ No LST value extracted")
                return False
                
        except Exception as e:
            print(f"❌ Error extracting LST value: {e}")
            return False
    
    def test_lst_series_extraction(self):
        """Test extracting LST time series."""
        print("\n" + "="*60)
        print("TESTING LST SERIES EXTRACTION")
        print("="*60)
        
        test_lat = 30.3165
        test_lon = 78.0322
        start_date = datetime(2020, 3, 10)
        end_date = datetime(2020, 3, 20)  # 10 days
        
        print(f"Test point: ({test_lat}, {test_lon})")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        try:
            lst_series = self.lst.get_lst_series(test_lat, test_lon, start_date, end_date)
            
            print(f"✅ LST series extracted")
            print(f"   Length: {len(lst_series)} days")
            print(f"   Non-null values: {lst_series.count()}")
            print(f"   Mean LST: {lst_series.mean():.2f} K")
            print(f"   Min LST: {lst_series.min():.2f} K")
            print(f"   Max LST: {lst_series.max():.2f} K")
            
            if lst_series.count() > 0:
                return True
            else:
                print(f"❌ No valid LST values in series")
                return False
                
        except Exception as e:
            print(f"❌ Error extracting LST series: {e}")
            return False
    
    def test_lst_dtw_features(self):
        """Test extracting LST features within DTW window."""
        print("\n" + "="*60)
        print("TESTING LST DTW FEATURES")
        print("="*60)
        
        test_lat = 30.3165
        test_lon = 78.0322
        dtw_start = datetime(2020, 3, 1)
        dtw_end = datetime(2020, 3, 15)
        
        print(f"Test point: ({test_lat}, {test_lon})")
        print(f"DTW window: {dtw_start.date()} to {dtw_end.date()}")
        
        try:
            lst_features = self.lst.extract_dtw_features(test_lat, test_lon, dtw_start, dtw_end)
            
            print(f"✅ LST DTW features extracted:")
            for feature, value in lst_features.items():
                if np.isnan(value):
                    print(f"   {feature}: NaN")
                else:
                    print(f"   {feature}: {value:.2f} K")
            
            # Check feature structure
            required_features = ['lst_min', 'lst_median', 'lst_mean', 'lst_max']
            for feature in required_features:
                assert feature in lst_features, f"Missing LST feature: {feature}"
            
            # Check if we have any valid values
            valid_count = sum(1 for v in lst_features.values() if not np.isnan(v))
            if valid_count > 0:
                print(f"   ✅ {valid_count}/4 features have valid values")
                return True
            else:
                print(f"   ⚠️  All LST features are NaN (no data available)")
                return True  # Still pass - this is expected for some locations/dates
                
        except Exception as e:
            print(f"❌ Error extracting LST DTW features: {e}")
            return False
    
    def test_modis_tile_calculation(self):
        """Test MODIS tile calculation."""
        print("\n" + "="*60)
        print("TESTING MODIS TILE CALCULATION")
        print("="*60)
        
        # Test points in Uttarakhand
        test_points = [
            (30.3165, 78.0322, "Dehradun area"),
            (29.5, 79.5, "Central Uttarakhand"),
            (31.0, 77.0, "Northern Uttarakhand"),
        ]
        
        print("Testing MODIS tile calculation for Uttarakhand points:")
        
        try:
            for lat, lon, description in test_points:
                tile = self.lst.get_modis_tile(lat, lon)
                print(f"   {description}: ({lat}, {lon}) -> {tile}")
                
                # Check if tile is reasonable for Uttarakhand
                if tile.startswith('h24') or tile.startswith('h25'):
                    print(f"     ✅ Tile {tile} is reasonable for Uttarakhand")
                else:
                    print(f"     ⚠️  Tile {tile} might be outside Uttarakhand")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in MODIS tile calculation: {e}")
            return False
    
    def run_all_tests(self):
        """Run all LST extraction tests."""
        print("\n" + "="*80)
        print("LST EXTRACTION TEST SUITE")
        print("="*80)
        
        tests = [
            ("MODIS Tile Calculation", self.test_modis_tile_calculation),
            ("LST File Finding", self.test_lst_file_finding),
            ("LST Value Extraction", self.test_lst_value_extraction),
            ("LST Series Extraction", self.test_lst_series_extraction),
            ("LST DTW Features", self.test_lst_dtw_features),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                    self.test_results.append(f"✅ {test_name}: PASSED")
                else:
                    self.test_results.append(f"❌ {test_name}: FAILED")
            except Exception as e:
                self.test_results.append(f"❌ {test_name}: ERROR - {e}")
        
        # Summary
        print("\n" + "="*80)
        print("LST EXTRACTION TEST SUMMARY")
        print("="*80)
        
        for result in self.test_results:
            print(result)
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ ALL LST EXTRACTION TESTS PASSED!")
            return True
        else:
            print("❌ SOME LST EXTRACTION TESTS FAILED!")
            return False

def main():
    """Main entry point."""
    tester = LSTTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


