#!/usr/bin/env python3
"""
Feature Extraction Test

Tests vegetation indices, terrain features, and precipitation feature extraction.
Verifies the 24-dimensional feature matrix generation.

Usage:
    python tests/test_features.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.dtw import DynamicTimeWindow
from features.indices import VegetationIndices
from features.terrain import TerrainFeatures
from features.lst import LSTFeatures

class FeatureTester:
    """Test suite for feature extraction."""
    
    def __init__(self):
        self.dtw = DynamicTimeWindow(thcp=30.0, thdp=10.0, max_window_days=90)
        self.vi = VegetationIndices()
        self.tf = TerrainFeatures()
        self.lst = LSTFeatures()
        self.test_results = []
        
    def test_terrain_features(self):
        """Test terrain feature extraction."""
        print("\n" + "="*60)
        print("TESTING TERRAIN FEATURES")
        print("="*60)
        
        # Test point in Uttarakhand
        test_lat = 30.3165
        test_lon = 78.0322
        
        print(f"Test point: ({test_lat}, {test_lon})")
        
        try:
            terrain_features = self.tf.extract_terrain_features(test_lat, test_lon)
            
            print(f"✅ Terrain features extracted:")
            for feature, value in terrain_features.items():
                if np.isnan(value):
                    print(f"   {feature}: NaN (file may not exist)")
                else:
                    print(f"   {feature}: {value:.2f}")
            
            # Validate feature structure
            required_features = ['elevation', 'slope', 'aspect']
            for feature in required_features:
                assert feature in terrain_features, f"Missing terrain feature: {feature}"
            
            return True
            
        except Exception as e:
            print(f"❌ Error extracting terrain features: {e}")
            return False
    
    def test_vegetation_indices(self):
        """Test vegetation indices calculation."""
        print("\n" + "="*60)
        print("TESTING VEGETATION INDICES")
        print("="*60)
        
        test_lat = 30.3165
        test_lon = 78.0322
        test_date = datetime(2020, 3, 15)
        
        print(f"Test point: ({test_lat}, {test_lon}) on {test_date.date()}")
        
        try:
            indices = self.vi.calculate_indices_for_date(test_lat, test_lon, test_date)
            
            print(f"✅ Vegetation indices calculated:")
            for index, value in indices.items():
                if index != 'bands':  # Skip raw band values
                    if np.isnan(value):
                        print(f"   {index}: NaN (MODIS file may not exist)")
                    else:
                        print(f"   {index}: {value:.4f}")
            
            # Validate index structure
            required_indices = ['ndvi', 'evi', 'ndwi']
            for index in required_indices:
                assert index in indices, f"Missing vegetation index: {index}"
            
            return True
            
        except Exception as e:
            print(f"❌ Error calculating vegetation indices: {e}")
            return False
    
    def test_precipitation_features(self):
        """Test precipitation feature extraction within DTW window."""
        print("\n" + "="*60)
        print("TESTING PRECIPITATION FEATURES")
        print("="*60)
        
        test_lat = 30.3165
        test_lon = 78.0322
        fire_date = datetime(2020, 3, 15)
        
        print(f"Test point: ({test_lat}, {test_lon})")
        print(f"Fire date: {fire_date.date()}")
        
        try:
            # Calculate DTW window
            dtw_start, dtw_end = self.dtw.calculate_dtw(
                fire_date, test_lat, test_lon, "data/raw/chirps"
            )
            
            if not dtw_start or not dtw_end:
                print("❌ Could not calculate DTW window")
                return False
            
            print(f"DTW window: {dtw_start.date()} to {dtw_end.date()}")
            
            # Extract precipitation series
            precip_series = self.dtw.get_precipitation_series(
                test_lat, test_lon, dtw_start, dtw_end, "data/raw/chirps"
            )
            
            if len(precip_series) == 0:
                print("❌ No precipitation data in DTW window")
                return False
            
            # Calculate precipitation features
            precip_features = {
                'prec_min': float(precip_series.min()),
                'prec_median': float(precip_series.median()),
                'prec_mean': float(precip_series.mean()),
                'prec_max': float(precip_series.max()),
                'prec_sum': float(precip_series.sum())
            }
            
            print(f"✅ Precipitation features calculated:")
            for feature, value in precip_features.items():
                print(f"   {feature}: {value:.2f} mm")
            
            # Validate feature structure
            required_features = ['prec_min', 'prec_median', 'prec_mean', 'prec_max', 'prec_sum']
            for feature in required_features:
                assert feature in precip_features, f"Missing precipitation feature: {feature}"
            
            return True
            
        except Exception as e:
            print(f"❌ Error extracting precipitation features: {e}")
            return False
    
    def test_vegetation_dtw_features(self):
        """Test vegetation indices within DTW window."""
        print("\n" + "="*60)
        print("TESTING VEGETATION DTW FEATURES")
        print("="*60)
        
        test_lat = 30.3165
        test_lon = 78.0322
        fire_date = datetime(2020, 3, 15)
        
        print(f"Test point: ({test_lat}, {test_lon})")
        print(f"Fire date: {fire_date.date()}")
        
        try:
            # Calculate DTW window
            dtw_start, dtw_end = self.dtw.calculate_dtw(
                fire_date, test_lat, test_lon, "data/raw/chirps"
            )
            
            if not dtw_start or not dtw_end:
                print("❌ Could not calculate DTW window")
                return False
            
            print(f"DTW window: {dtw_start.date()} to {dtw_end.date()}")
            
            # Extract vegetation indices within DTW window
            vi_features = self.vi.extract_dtw_features(test_lat, test_lon, dtw_start, dtw_end)
            
            print(f"✅ Vegetation DTW features calculated:")
            for feature, value in vi_features.items():
                if np.isnan(value):
                    print(f"   {feature}: NaN (MODIS files may not exist)")
                else:
                    print(f"   {feature}: {value:.4f}")
            
            # Validate feature structure
            required_features = []
            for index in ['ndvi', 'evi', 'ndwi']:
                for stat in ['min', 'median', 'mean', 'max']:
                    required_features.append(f"{index}_{stat}")
            
            for feature in required_features:
                assert feature in vi_features, f"Missing vegetation DTW feature: {feature}"
            
            return True
            
        except Exception as e:
            print(f"❌ Error extracting vegetation DTW features: {e}")
            return False
    
    def test_lst_features(self):
        """Test LST feature extraction."""
        print("\n" + "="*60)
        print("TESTING LST FEATURES")
        print("="*60)
        
        test_lat = 30.3165
        test_lon = 78.0322
        fire_date = datetime(2020, 3, 15)
        
        print(f"Test point: ({test_lat}, {test_lon})")
        print(f"Fire date: {fire_date.date()}")
        
        try:
            # Calculate DTW window
            dtw_start, dtw_end = self.dtw.calculate_dtw(
                fire_date, test_lat, test_lon, "data/raw/chirps"
            )
            
            if not dtw_start or not dtw_end:
                print("❌ Could not calculate DTW window")
                return False
            
            print(f"DTW window: {dtw_start.date()} to {dtw_end.date()}")
            
            # Extract LST features
            lst_features = self.lst.extract_dtw_features(test_lat, test_lon, dtw_start, dtw_end)
            
            print(f"✅ LST features calculated:")
            for feature, value in lst_features.items():
                if np.isnan(value):
                    print(f"   {feature}: NaN (LST file may not exist)")
                else:
                    print(f"   {feature}: {value:.2f} K")
            
            # Check required features
            required_features = ['lst_min', 'lst_median', 'lst_mean', 'lst_max']
            for feature in required_features:
                assert feature in lst_features, f"Missing LST feature: {feature}"
            
            return True
            
        except Exception as e:
            print(f"❌ Error extracting LST features: {e}")
            return False
    
    def test_complete_feature_extraction(self):
        """Test complete 24-feature extraction for a single point."""
        print("\n" + "="*60)
        print("TESTING COMPLETE FEATURE EXTRACTION")
        print("="*60)
        
        test_lat = 30.3165
        test_lon = 78.0322
        fire_date = datetime(2020, 3, 15)
        
        print(f"Test point: ({test_lat}, {test_lon})")
        print(f"Fire date: {fire_date.date()}")
        
        try:
            # Calculate DTW window
            dtw_start, dtw_end = self.dtw.calculate_dtw(
                fire_date, test_lat, test_lon, "data/raw/chirps"
            )
            
            if not dtw_start or not dtw_end:
                print("❌ Could not calculate DTW window")
                return False
            
            print(f"DTW window: {dtw_start.date()} to {dtw_end.date()}")
            
            # Use pipeline's extract_all_features method to ensure test matches pipeline logic
            from features.pipeline import FeaturePipeline
            pipeline = FeaturePipeline()
            all_features = pipeline.extract_all_features(test_lat, test_lon, dtw_start, dtw_end)
            
            print(f"✅ Complete feature extraction successful")
            print(f"   Total features: {len(all_features)}")
            
            # Validate 24-feature structure
            expected_features = [
                # Precipitation (5)
                'prec_min', 'prec_median', 'prec_mean', 'prec_max', 'prec_sum',
                # LST (4)
                'lst_min', 'lst_median', 'lst_mean', 'lst_max',
                # NDVI (4)
                'ndvi_min', 'ndvi_median', 'ndvi_mean', 'ndvi_max',
                # EVI (4)
                'evi_min', 'evi_median', 'evi_mean', 'evi_max',
                # NDWI (4)
                'ndwi_min', 'ndwi_median', 'ndwi_mean', 'ndwi_max',
                # Terrain (3)
                'elevation', 'slope', 'aspect'
            ]
            
            missing_features = [f for f in expected_features if f not in all_features]
            if missing_features:
                print(f"❌ Missing features: {missing_features}")
                return False
            
            print(f"✅ All 24 features present")
            
            # Count non-NaN features
            non_nan_count = sum(1 for v in all_features.values() if not np.isnan(v))
            print(f"   Non-NaN features: {non_nan_count}/24")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in complete feature extraction: {e}")
            return False
    
    def test_feature_validation(self):
        """Test feature value validation."""
        print("\n" + "="*60)
        print("TESTING FEATURE VALIDATION")
        print("="*60)
        
        # Test with multiple points
        test_points = [
            {'lat': 30.3165, 'lon': 78.0322, 'date': datetime(2020, 3, 15)},
            {'lat': 30.5, 'lon': 78.5, 'date': datetime(2020, 4, 10)},
            {'lat': 29.8, 'lon': 79.2, 'date': datetime(2020, 5, 5)},
        ]
        
        print(f"Testing feature validation with {len(test_points)} points")
        
        valid_count = 0
        for i, point in enumerate(test_points):
            print(f"\nPoint {i+1}: ({point['lat']}, {point['lon']}) on {point['date'].date()}")
            
            try:
                # Calculate DTW
                dtw_start, dtw_end = self.dtw.calculate_dtw(
                    point['date'], point['lat'], point['lon'], "data/raw/chirps"
                )
                
                if dtw_start and dtw_end:
                    # Extract terrain features
                    terrain = self.tf.extract_terrain_features(point['lat'], point['lon'])
                    
                    # Check for reasonable values
                    if 'elevation' in terrain and not np.isnan(terrain['elevation']):
                        if 0 <= terrain['elevation'] <= 8000:  # Reasonable elevation range
                            valid_count += 1
                            print(f"   ✅ Valid elevation: {terrain['elevation']:.0f}m")
                        else:
                            print(f"   ⚠️  Unusual elevation: {terrain['elevation']:.0f}m")
                    else:
                        print(f"   ⚠️  No elevation data")
                else:
                    print(f"   ⚠️  No DTW window")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n✅ Valid points: {valid_count}/{len(test_points)}")
        return valid_count > 0
    
    def run_all_tests(self):
        """Run all feature extraction tests."""
        print("\n" + "="*80)
        print("FEATURE EXTRACTION TEST SUITE")
        print("="*80)
        
        tests = [
            ("Terrain Features", self.test_terrain_features),
            ("Vegetation Indices", self.test_vegetation_indices),
            ("Precipitation Features", self.test_precipitation_features),
            ("Vegetation DTW Features", self.test_vegetation_dtw_features),
            ("LST Features", self.test_lst_features),
            ("Complete Feature Extraction", self.test_complete_feature_extraction),
            ("Feature Validation", self.test_feature_validation),
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
        print("FEATURE EXTRACTION TEST SUMMARY")
        print("="*80)
        
        for result in self.test_results:
            print(result)
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ ALL FEATURE EXTRACTION TESTS PASSED!")
            return True
        else:
            print("❌ SOME FEATURE EXTRACTION TESTS FAILED!")
            return False

def main():
    """Main entry point."""
    tester = FeatureTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
