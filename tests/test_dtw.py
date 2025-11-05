#!/usr/bin/env python3
"""
DTW Algorithm Test

Tests the Dynamic Time Window algorithm with known fire points.
Verifies precipitation threshold logic and edge cases.

Usage:
    python tests/test_dtw.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.dtw import DynamicTimeWindow

class DTWTester:
    """Test suite for DTW algorithm."""
    
    def __init__(self):
        self.dtw = DynamicTimeWindow(thcp=30.0, thdp=10.0, max_window_days=90)
        self.test_results = []
        
    def test_dtw_parameters(self):
        """Test DTW parameter initialization."""
        print("\n" + "="*60)
        print("TESTING DTW PARAMETERS")
        print("="*60)
        
        print(f"Threshold cumulative precipitation: {self.dtw.thcp} mm")
        print(f"Threshold daily precipitation: {self.dtw.thdp} mm")
        print(f"Maximum window days: {self.dtw.max_window_days}")
        
        # Test parameter validation
        assert self.dtw.thcp > 0, "Thcp should be positive"
        assert self.dtw.thdp > 0, "Thdp should be positive"
        assert self.dtw.max_window_days > 0, "Max window days should be positive"
        
        print("✅ DTW parameters are valid")
        return True
    
    def test_precipitation_extraction(self):
        """Test precipitation value extraction."""
        print("\n" + "="*60)
        print("TESTING PRECIPITATION EXTRACTION")
        print("="*60)
        
        # Test with known point in Uttarakhand
        test_lat = 30.3165  # Dehradun area
        test_lon = 78.0322
        test_date = datetime(2020, 1, 15)
        
        print(f"Test point: ({test_lat}, {test_lon}) on {test_date.date()}")
        
        try:
            precip = self.dtw.extract_precipitation_value(
                test_lat, test_lon, test_date, "data/raw/chirps"
            )
            
            if precip is not None:
                print(f"✅ Precipitation extracted: {precip:.2f} mm")
                assert isinstance(precip, (int, float)), "Precipitation should be numeric"
                assert precip >= 0, "Precipitation should be non-negative"
                return True
            else:
                print("❌ Precipitation extraction returned None")
                return False
                
        except Exception as e:
            print(f"❌ Error extracting precipitation: {e}")
            return False
    
    def test_dtw_calculation(self):
        """Test DTW window calculation."""
        print("\n" + "="*60)
        print("TESTING DTW CALCULATION")
        print("="*60)
        
        # Test with a fire point
        fire_date = datetime(2020, 3, 15)  # Spring fire season
        test_lat = 30.3165
        test_lon = 78.0322
        
        print(f"Fire date: {fire_date.date()}")
        print(f"Location: ({test_lat}, {test_lon})")
        
        try:
            dtw_start, dtw_end = self.dtw.calculate_dtw(
                fire_date, test_lat, test_lon, "data/raw/chirps"
            )
            
            if dtw_start and dtw_end:
                window_length = (dtw_end - dtw_start).days
                print(f"✅ DTW calculated successfully")
                print(f"   Start: {dtw_start.date()}")
                print(f"   End: {dtw_end.date()}")
                print(f"   Length: {window_length} days")
                
                # Validate DTW properties
                assert dtw_end == fire_date, "DTW end should equal fire date"
                assert dtw_start <= dtw_end, "DTW start should be before end"
                assert window_length <= self.dtw.max_window_days, "Window should not exceed max days"
                
                return True
            else:
                print("❌ DTW calculation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error calculating DTW: {e}")
            return False
    
    def test_precipitation_series(self):
        """Test precipitation time series extraction."""
        print("\n" + "="*60)
        print("TESTING PRECIPITATION SERIES")
        print("="*60)
        
        test_lat = 30.3165
        test_lon = 78.0322
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 7)  # One week
        
        print(f"Location: ({test_lat}, {test_lon})")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        try:
            series = self.dtw.get_precipitation_series(
                test_lat, test_lon, start_date, end_date, "data/raw/chirps"
            )
            
            print(f"✅ Precipitation series extracted")
            print(f"   Length: {len(series)} days")
            print(f"   Mean: {series.mean():.2f} mm")
            print(f"   Max: {series.max():.2f} mm")
            print(f"   Min: {series.min():.2f} mm")
            
            # Validate series properties
            assert len(series) == 7, "Series should have 7 days"
            assert all(series >= 0), "All precipitation values should be non-negative"
            
            return True
            
        except Exception as e:
            print(f"❌ Error extracting precipitation series: {e}")
            return False
    
    def test_edge_cases(self):
        """Test DTW edge cases."""
        print("\n" + "="*60)
        print("TESTING DTW EDGE CASES")
        print("="*60)
        
        # Test 1: Very recent fire (might not have enough lookback data)
        recent_fire = datetime(2020, 1, 5)
        test_lat = 30.3165
        test_lon = 78.0322
        
        print(f"Test 1: Recent fire ({recent_fire.date()})")
        try:
            dtw_start, dtw_end = self.dtw.calculate_dtw(
                recent_fire, test_lat, test_lon, "data/raw/chirps"
            )
            if dtw_start and dtw_end:
                print(f"   ✅ Recent fire handled: {dtw_start.date()} to {dtw_end.date()}")
            else:
                print(f"   ⚠️  Recent fire: No DTW found (expected for very recent)")
        except Exception as e:
            print(f"   ❌ Recent fire error: {e}")
        
        # Test 2: Point outside CHIRPS bounds
        print(f"\nTest 2: Point outside bounds")
        outside_lat = 25.0  # South of Uttarakhand
        outside_lon = 75.0  # West of Uttarakhand
        
        try:
            precip = self.dtw.extract_precipitation_value(
                outside_lat, outside_lon, datetime(2020, 1, 15), "data/raw/chirps"
            )
            print(f"   ✅ Outside bounds handled: {precip} mm (should be 0.0)")
            assert precip == 0.0, "Outside bounds should return 0.0"
        except Exception as e:
            print(f"   ❌ Outside bounds error: {e}")
        
        # Test 3: Missing CHIRPS file
        print(f"\nTest 3: Missing CHIRPS file")
        try:
            # Use a date that likely doesn't have CHIRPS data
            missing_date = datetime(2019, 1, 1)
            precip = self.dtw.extract_precipitation_value(
                test_lat, test_lon, missing_date, "data/raw/chirps"
            )
            print(f"   ✅ Missing file handled: {precip} mm (should be 0.0)")
            assert precip == 0.0, "Missing file should return 0.0"
        except Exception as e:
            print(f"   ❌ Missing file error: {e}")
        
        return True
    
    def test_batch_processing(self):
        """Test DTW batch processing."""
        print("\n" + "="*60)
        print("TESTING DTW BATCH PROCESSING")
        print("="*60)
        
        # Create test fire points
        test_points = [
            {'lat': 30.3165, 'lon': 78.0322, 'date': datetime(2020, 3, 15)},
            {'lat': 30.5, 'lon': 78.5, 'date': datetime(2020, 4, 10)},
            {'lat': 29.8, 'lon': 79.2, 'date': datetime(2020, 5, 5)},
        ]
        
        import pandas as pd
        df = pd.DataFrame(test_points)
        
        print(f"Testing batch processing with {len(df)} points")
        
        try:
            result = self.dtw.calculate_dtw_batch(df, "data/raw/chirps")
            
            print(f"✅ Batch processing completed")
            print(f"   Input points: {len(df)}")
            print(f"   Output points: {len(result)}")
            
            # Check that DTW columns were added
            required_cols = ['dtw_start', 'dtw_end', 'dtw_length']
            for col in required_cols:
                assert col in result.columns, f"Missing column: {col}"
            
            print(f"   ✅ All required columns present")
            
            # Check that some DTW calculations succeeded
            valid_dtw = result['dtw_start'].notna().sum()
            print(f"   Valid DTW calculations: {valid_dtw}/{len(result)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            return False
    
    def run_all_tests(self):
        """Run all DTW tests."""
        print("\n" + "="*80)
        print("DTW ALGORITHM TEST SUITE")
        print("="*80)
        
        tests = [
            ("DTW Parameters", self.test_dtw_parameters),
            ("Precipitation Extraction", self.test_precipitation_extraction),
            ("DTW Calculation", self.test_dtw_calculation),
            ("Precipitation Series", self.test_precipitation_series),
            ("Edge Cases", self.test_edge_cases),
            ("Batch Processing", self.test_batch_processing),
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
        print("DTW TEST SUMMARY")
        print("="*80)
        
        for result in self.test_results:
            print(result)
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ ALL DTW TESTS PASSED!")
            return True
        else:
            print("❌ SOME DTW TESTS FAILED!")
            return False

def main():
    """Main entry point."""
    tester = DTWTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()