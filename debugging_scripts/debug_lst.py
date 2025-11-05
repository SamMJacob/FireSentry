#!/usr/bin/env python3
"""
Debug LST extraction for test point
"""

from datetime import datetime
from features.lst import LSTFeatures

# Test parameters
lat, lon = 30.3165, 78.0322
date = datetime(2020, 3, 15)

print(f"Testing LST extraction for:")
print(f"  Point: ({lat}, {lon})")
print(f"  Date: {date.date()}")
print(f"  DOY: {date.timetuple().tm_yday}")

# Initialize LST features
lst = LSTFeatures()

# Test file finding
print(f"\n1. Testing file finding:")
lst_file = lst.find_modis_lst_file(lat, lon, date)
print(f"   Found file: {lst_file}")

if lst_file:
    print(f"   File exists: {lst_file.exists()}")
    
    # Test value extraction
    print(f"\n2. Testing value extraction:")
    lst_value = lst.extract_lst_value(lat, lon, date)
    print(f"   LST value: {lst_value} K")
    
    # Test DTW features
    print(f"\n3. Testing DTW features:")
    dtw_start = datetime(2020, 3, 14)
    dtw_end = datetime(2020, 3, 15)
    dtw_features = lst.extract_dtw_features(lat, lon, dtw_start, dtw_end)
    print(f"   DTW features: {dtw_features}")
else:
    print("   No file found!")


