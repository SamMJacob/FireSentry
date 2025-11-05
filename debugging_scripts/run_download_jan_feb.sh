#!/bin/bash
# Download MODIS SR data for Jan-Feb 2020
# This fills the data gap for DTW windows that extend into early 2020

echo "🚀 Starting MODIS SR download for Jan-Feb 2020..."
echo "This will run in the background"
echo ""
echo "Output: data/raw/modis_sr/2020/"
echo "Logs: Check terminal for progress"
echo ""

# Run the download script
python scripts/fetch_modis_sr_jan_feb.py

echo ""
echo "✅ Download complete!"
echo "Check data/raw/modis_sr/2020/ for new files"

