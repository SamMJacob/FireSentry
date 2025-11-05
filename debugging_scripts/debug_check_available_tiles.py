#!/usr/bin/env python3
"""
Debug script to check what MODIS tiles are available in NASA Earthdata
for our date range and bounding box.
"""

import earthaccess
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_available_tiles():
    """Check what MODIS tiles are available for our search criteria."""
    
    # Same parameters as the main script
    bbox_w, bbox_s, bbox_e, bbox_n = 77.0, 28.5, 83.5, 31.5
    start_date = datetime(2020, 3, 1)
    end_date = datetime(2020, 3, 31)
    
    print(f"Searching for MODIS SR tiles:")
    print(f"  Bounding box: ({bbox_w}, {bbox_s}, {bbox_e}, {bbox_n})")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print()
    
    try:
        # Search for MODIS Surface Reflectance products
        results = earthaccess.search_data(
            short_name='MYD09GA',
            version='061',
            bounding_box=(bbox_w, bbox_s, bbox_e, bbox_n),
            temporal=(start_date, end_date),
            count=1000
        )
        
        print(f"Found {len(results)} results")
        print()
        
        # Debug: Print first result structure
        if len(results) > 0:
            print("First result structure:")
            print(f"  Type: {type(results[0])}")
            if hasattr(results[0], '__dict__'):
                print(f"  Attributes: {dir(results[0])}")
            print()
        
        # Extract unique tiles from results
        tiles_found = set()
        for i, result in enumerate(results):
            try:
                # Use the data_links() method to get download URLs
                links = result.data_links()
                
                if links and len(links) > 0:
                    # Get filename from the first data link
                    url = links[0]
                    filename = url.split('/')[-1]
                    
                    # Debug: Print first few filenames
                    if i < 5:
                        print(f"  Result {i}: {filename}")
                    
                    if filename and 'h' in filename and 'v' in filename:
                        # Find tile pattern like h25v05
                        parts = filename.split('.')
                        for part in parts:
                            if part.startswith('h') and 'v' in part and len(part) == 6:
                                tiles_found.add(part)
                                break
                            
            except Exception as e:
                # Print error for debugging
                if i < 5:
                    print(f"  Result {i}: Error - {e}")
                continue
        
        print()
        
        print("Tiles found in NASA Earthdata:")
        for tile in sorted(tiles_found):
            print(f"  {tile}")
        
        print()
        print("Required tiles for Uttarakhand:")
        required_tiles = ['h25v05', 'h25v06', 'h26v05', 'h26v06']
        for tile in required_tiles:
            status = "✓" if tile in tiles_found else "✗"
            print(f"  {status} {tile}")
        
        missing_tiles = set(required_tiles) - tiles_found
        if missing_tiles:
            print(f"\nMissing tiles: {missing_tiles}")
            print("\nPossible reasons:")
            print("1. Tiles don't exist for this date range")
            print("2. Tiles are outside the bounding box")
            print("3. Data is not available in NASA Earthdata")
        else:
            print("\n✓ All required tiles are available!")
            
    except Exception as e:
        print(f"Error searching for tiles: {e}")

if __name__ == "__main__":
    check_available_tiles()
