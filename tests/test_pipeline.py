#!/usr/bin/env python3
"""
Pipeline Integration Test

Tests the complete feature engineering pipeline with a small sample.
Verifies end-to-end processing from fire points to feature matrix.

Usage:
    python tests/test_pipeline.py
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.pipeline import FeaturePipeline

class PipelineTester:
    """Test suite for pipeline integration."""
    
    def __init__(self):
        self.pipeline = FeaturePipeline()
        self.test_results = []
        
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        print("\n" + "="*60)
        print("TESTING PIPELINE INITIALIZATION")
        print("="*60)
        
        try:
            # Check components
            assert hasattr(self.pipeline, 'dtw'), "DTW component missing"
            assert hasattr(self.pipeline, 'vi'), "Vegetation indices component missing"
            assert hasattr(self.pipeline, 'tf'), "Terrain features component missing"
            
            # Check feature columns
            assert hasattr(self.pipeline, 'feature_columns'), "Feature columns missing"
            assert len(self.pipeline.feature_columns) == 24, f"Expected 24 features, got {len(self.pipeline.feature_columns)}"
            
            print(f"✅ Pipeline initialized successfully")
            print(f"   DTW component: {type(self.pipeline.dtw).__name__}")
            print(f"   VI component: {type(self.pipeline.vi).__name__}")
            print(f"   TF component: {type(self.pipeline.tf).__name__}")
            print(f"   Feature columns: {len(self.pipeline.feature_columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing pipeline: {e}")
            return False
    
    def test_fire_data_loading(self):
        """Test fire data loading."""
        print("\n" + "="*60)
        print("TESTING FIRE DATA LOADING")
        print("="*60)
        
        try:
            fire_points = self.pipeline.load_fire_data()
            
            if len(fire_points) == 0:
                print("❌ No fire data loaded")
                return False
            
            print(f"✅ Fire data loaded successfully")
            print(f"   Number of fire points: {len(fire_points)}")
            print(f"   Columns: {list(fire_points.columns)}")
            print(f"   Date range: {fire_points['date'].min()} to {fire_points['date'].max()}")
            print(f"   Lat range: {fire_points['lat'].min():.2f} to {fire_points['lat'].max():.2f}")
            print(f"   Lon range: {fire_points['lon'].min():.2f} to {fire_points['lon'].max():.2f}")
            
            # Validate structure
            required_cols = ['lat', 'lon', 'date']
            for col in required_cols:
                assert col in fire_points.columns, f"Missing column: {col}"
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading fire data: {e}")
            return False
    
    def test_small_sample_processing(self):
        """Test processing a small sample of fire points."""
        print("\n" + "="*60)
        print("TESTING SMALL SAMPLE PROCESSING")
        print("="*60)
        
        try:
            # Load fire data
            fire_points = self.pipeline.load_fire_data()
            
            if len(fire_points) == 0:
                print("❌ No fire data available")
                return False
            
            # Take a small sample (10 points)
            sample_size = min(10, len(fire_points))
            sample_points = fire_points.sample(n=sample_size, random_state=42)
            
            print(f"Processing sample of {sample_size} fire points")
            
            # Generate pseudo fire points (small ratio)
            pseudo_points = self.pipeline.generate_pseudo_fire_points(sample_points, ratio=0.5)
            
            print(f"Generated {len(pseudo_points)} pseudo fire points")
            
            # Build feature matrix
            feature_matrix = self.pipeline.build_feature_matrix(sample_points, pseudo_points)
            
            print(f"✅ Feature matrix built successfully")
            print(f"   Total samples: {len(feature_matrix)}")
            print(f"   Features: {len(self.pipeline.feature_columns)}")
            print(f"   Target distribution: {feature_matrix['target'].value_counts().to_dict()}")
            
            # Validate structure
            assert len(feature_matrix) > 0, "Feature matrix is empty"
            assert 'target' in feature_matrix.columns, "Target column missing"
            
            # Check feature columns
            missing_features = [f for f in self.pipeline.feature_columns if f not in feature_matrix.columns]
            if missing_features:
                print(f"❌ Missing feature columns: {missing_features}")
                return False
            
            # Check for reasonable feature values
            feature_stats = feature_matrix[self.pipeline.feature_columns].describe()
            print(f"   Feature statistics:")
            print(f"     Non-null values: {feature_matrix[self.pipeline.feature_columns].count().sum()}")
            print(f"     NaN values: {feature_matrix[self.pipeline.feature_columns].isnull().sum().sum()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in small sample processing: {e}")
            return False
    
    def test_dtw_calculation_batch(self):
        """Test DTW calculation for multiple points."""
        print("\n" + "="*60)
        print("TESTING DTW CALCULATION BATCH")
        print("="*60)
        
        try:
            # Load fire data
            fire_points = self.pipeline.load_fire_data()
            
            if len(fire_points) == 0:
                print("❌ No fire data available")
                return False
            
            # Take a small sample
            sample_size = min(5, len(fire_points))
            sample_points = fire_points.sample(n=sample_size, random_state=42)
            
            print(f"Testing DTW calculation for {sample_size} points")
            
            # Calculate DTW for all points
            dtw_results = self.pipeline.dtw.calculate_dtw_batch(sample_points, "data/raw/chirps")
            
            print(f"✅ DTW batch calculation completed")
            print(f"   Input points: {len(sample_points)}")
            print(f"   Output points: {len(dtw_results)}")
            
            # Check DTW columns
            required_cols = ['dtw_start', 'dtw_end', 'dtw_length']
            for col in required_cols:
                assert col in dtw_results.columns, f"Missing DTW column: {col}"
            
            # Check DTW validity
            valid_dtw = dtw_results['dtw_start'].notna().sum()
            print(f"   Valid DTW calculations: {valid_dtw}/{len(dtw_results)}")
            
            if valid_dtw > 0:
                avg_length = dtw_results['dtw_length'].mean()
                print(f"   Average DTW length: {avg_length:.1f} days")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in DTW batch calculation: {e}")
            return False
    
    def test_pseudo_fire_points_generation(self):
        """Test pseudo fire points generation."""
        print("\n" + "="*60)
        print("TESTING PSEUDO FIRE POINTS GENERATION")
        print("="*60)
        
        try:
            # Load fire data
            fire_points = self.pipeline.load_fire_data()
            
            if len(fire_points) == 0:
                print("❌ No fire data available")
                return False
            
            # Take a small sample
            sample_size = min(5, len(fire_points))
            sample_points = fire_points.sample(n=sample_size, random_state=42)
            
            print(f"Generating pseudo fire points for {sample_size} real fire points")
            
            # Generate pseudo points
            pseudo_points = self.pipeline.generate_pseudo_fire_points(sample_points, ratio=1.0)
            
            print(f"✅ Pseudo fire points generated")
            print(f"   Real fire points: {len(sample_points)}")
            print(f"   Pseudo fire points: {len(pseudo_points)}")
            
            # Validate structure
            required_cols = ['lat', 'lon', 'date', 'is_fire']
            for col in required_cols:
                assert col in pseudo_points.columns, f"Missing pseudo point column: {col}"
            
            # Check that all pseudo points are marked as non-fire
            assert all(pseudo_points['is_fire'] == False), "Pseudo points should be marked as non-fire"
            
            # Check spatial distribution
            print(f"   Lat range: {pseudo_points['lat'].min():.2f} to {pseudo_points['lat'].max():.2f}")
            print(f"   Lon range: {pseudo_points['lon'].min():.2f} to {pseudo_points['lon'].max():.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating pseudo fire points: {e}")
            return False
    
    def test_feature_matrix_validation(self):
        """Test feature matrix validation and quality."""
        print("\n" + "="*60)
        print("TESTING FEATURE MATRIX VALIDATION")
        print("="*60)
        
        try:
            # Load fire data
            fire_points = self.pipeline.load_fire_data()
            
            if len(fire_points) == 0:
                print("❌ No fire data available")
                return False
            
            # Take a small sample
            sample_size = min(10, len(fire_points))
            sample_points = fire_points.sample(n=sample_size, random_state=42)
            
            # Generate pseudo points
            pseudo_points = self.pipeline.generate_pseudo_fire_points(sample_points, ratio=0.5)
            
            # Build feature matrix
            feature_matrix = self.pipeline.build_feature_matrix(sample_points, pseudo_points)
            
            print(f"✅ Feature matrix validation")
            print(f"   Total samples: {len(feature_matrix)}")
            print(f"   Features: {len(self.pipeline.feature_columns)}")
            
            # Check data quality
            feature_cols = self.pipeline.feature_columns
            
            # Count non-null values
            non_null_counts = feature_matrix[feature_cols].count()
            null_counts = feature_matrix[feature_cols].isnull().sum()
            
            print(f"   Data quality:")
            for i, col in enumerate(feature_cols):
                non_null = non_null_counts.iloc[i]
                null = null_counts.iloc[i]
                total = len(feature_matrix)
                print(f"     {col}: {non_null}/{total} non-null ({non_null/total*100:.1f}%)")
            
            # Check for reasonable value ranges
            print(f"   Value ranges:")
            for col in feature_cols[:5]:  # Show first 5 features
                if col in feature_matrix.columns:
                    values = feature_matrix[col].dropna()
                    if len(values) > 0:
                        print(f"     {col}: {values.min():.2f} to {values.max():.2f}")
            
            # Check target distribution
            target_dist = feature_matrix['target'].value_counts()
            print(f"   Target distribution: {target_dist.to_dict()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in feature matrix validation: {e}")
            return False
    
    def test_pipeline_performance(self):
        """Test pipeline performance with timing."""
        print("\n" + "="*60)
        print("TESTING PIPELINE PERFORMANCE")
        print("="*60)
        
        import time
        
        try:
            # Load fire data
            fire_points = self.pipeline.load_fire_data()
            
            if len(fire_points) == 0:
                print("❌ No fire data available")
                return False
            
            # Take a small sample for performance testing
            sample_size = min(20, len(fire_points))
            sample_points = fire_points.sample(n=sample_size, random_state=42)
            
            print(f"Performance test with {sample_size} fire points")
            
            # Time pseudo point generation
            start_time = time.time()
            pseudo_points = self.pipeline.generate_pseudo_fire_points(sample_points, ratio=0.5)
            pseudo_time = time.time() - start_time
            
            print(f"   Pseudo point generation: {pseudo_time:.2f}s")
            
            # Time feature matrix building
            start_time = time.time()
            feature_matrix = self.pipeline.build_feature_matrix(sample_points, pseudo_points)
            feature_time = time.time() - start_time
            
            print(f"   Feature matrix building: {feature_time:.2f}s")
            
            total_time = pseudo_time + feature_time
            total_points = len(sample_points) + len(pseudo_points)
            time_per_point = total_time / total_points
            
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Time per point: {time_per_point:.2f}s")
            print(f"   Points processed: {total_points}")
            
            # Performance targets (from plan.md)
            target_time_per_point = 5.0  # seconds
            if time_per_point <= target_time_per_point:
                print(f"✅ Performance target met: {time_per_point:.2f}s <= {target_time_per_point}s per point")
            else:
                print(f"⚠️  Performance target missed: {time_per_point:.2f}s > {target_time_per_point}s per point")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in performance test: {e}")
            return False
    
    def run_all_tests(self):
        """Run all pipeline integration tests."""
        print("\n" + "="*80)
        print("PIPELINE INTEGRATION TEST SUITE")
        print("="*80)
        
        tests = [
            ("Pipeline Initialization", self.test_pipeline_initialization),
            ("Fire Data Loading", self.test_fire_data_loading),
            ("Small Sample Processing", self.test_small_sample_processing),
            ("DTW Calculation Batch", self.test_dtw_calculation_batch),
            ("Pseudo Fire Points Generation", self.test_pseudo_fire_points_generation),
            ("Feature Matrix Validation", self.test_feature_matrix_validation),
            ("Pipeline Performance", self.test_pipeline_performance),
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
        print("PIPELINE INTEGRATION TEST SUMMARY")
        print("="*80)
        
        for result in self.test_results:
            print(result)
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ ALL PIPELINE INTEGRATION TESTS PASSED!")
            return True
        else:
            print("❌ SOME PIPELINE INTEGRATION TESTS FAILED!")
            return False

def main():
    """Main entry point."""
    tester = PipelineTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


