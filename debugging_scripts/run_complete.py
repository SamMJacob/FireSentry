#!/usr/bin/env python3
"""
Complete FireSentry Pipeline Runner
Downloads MODIS SR data and runs feature extraction + model training
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import time

def main():
    # Load environment variables
    load_dotenv()
    
    print("🚀 FireSentry Complete Pipeline")
    print("========================================")
    print("1. Downloading MODIS SR data (Jan-Jun 2020-2024)")
    print("2. Building features when download completes")
    print("3. Training model")
    print("========================================")
    
    # Check if .env has credentials
    if not os.getenv('EARTHDATA_USER') or not os.getenv('EARTHDATA_PASS'):
        print("❌ EARTHDATA_USER and EARTHDATA_PASS must be set in .env file")
        sys.exit(1)
    
    # Start download
    print("📥 Starting MODIS SR download...")
    try:
        result = subprocess.run([sys.executable, "scripts/fetch_modis_sr_complete.py"], 
                              check=True, capture_output=True, text=True)
        print("✅ Download completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    
    # Run feature extraction
    print("🚀 Starting feature extraction and model training...")
    try:
        result = subprocess.run([sys.executable, "scripts/build_features.py"], 
                              check=True, capture_output=True, text=True)
        print("✅ Pipeline completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
