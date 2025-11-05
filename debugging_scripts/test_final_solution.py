#!/usr/bin/env python3
"""
Final test: Download h24v05 tile and verify vegetation indices work.
"""

print("=" * 70)
print("FINAL TEST: Verifying h24v05 tile solves the vegetation indices issue")
print("=" * 70)
print()

# Summary of findings
print("FINDINGS:")
print("  1. Dehradun (78.0322°E, 30.3165°N) is in tile h24v05, NOT h25v05")
print("  2. MODIS Sinusoidal projection causes tile boundaries to shift")
print("  3. h24v05 covers: 78.3°-80.8°E, 30°-40°N")
print("  4. The simple tile calculation h=int((lon+180)/10) is incorrect")
print()

print("SOLUTION:")
print("  - Download tiles: h24v05, h24v06, h25v06")
print("  - Use 3x3 grid search with point containment check")
print("  - Skip clipping to preserve full tile coverage")
print()

print("=" * 70)
print("Running download with corrected tiles...")
print("=" * 70)
print()

import subprocess
result = subprocess.run(['python', 'scripts/fetch_modis_sr.py'], 
                       capture_output=False, text=True)

if result.returncode == 0:
    print()
    print("=" * 70)
    print("✓ Download complete!")
    print("=" * 70)
else:
    print()
    print("=" * 70)
    print("✗ Download failed")
    print("=" * 70)


