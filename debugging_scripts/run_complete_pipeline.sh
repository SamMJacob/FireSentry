#!/bin/bash
# FireSentry Complete Pipeline Runner
# Downloads MODIS SR data and runs feature extraction + model training
# Usage: ./run_complete_pipeline.sh

set -e  # Exit on any error

echo "🚀 FireSentry Complete Pipeline"
echo "========================================"
echo "1. Downloading MODIS SR data (Jan-Jun 2020-2024)"
echo "2. Building features when download completes"
echo "3. Training model"
echo "========================================"

# Function to check if download is complete
check_download_complete() {
    local years=(2020 2021 2022 2023 2024)
    local all_complete=true
    
    for year in "${years[@]}"; do
        if [ ! -d "data/raw/modis_sr/$year" ]; then
            all_complete=false
            break
        fi
        
        # Check if we have files for June (DOY 150-180)
        local june_files=$(find "data/raw/modis_sr/$year" -name "*.B01.tif" | grep -E "AA${year}1[5-8][0-9]" | wc -l)
        if [ "$june_files" -lt 10 ]; then
            all_complete=false
            break
        fi
    done
    
    echo $all_complete
}

# Function to show progress
show_progress() {
    local year=$1
    local total_days=180  # Jan-Jun = 180 days
    local current_files=$(find "data/raw/modis_sr/$year" -name "*.B01.tif" 2>/dev/null | wc -l)
    local expected_files=$((total_days * 21))  # 21 files per day (7 bands × 3 tiles)
    local progress=$((current_files * 100 / expected_files))
    
    echo "📊 $year Progress: $current_files/$expected_files files ($progress%)"
}

# Start download in background
echo "📥 Starting MODIS SR download..."
python scripts/fetch_modis_sr_complete.py &
DOWNLOAD_PID=$!

# Monitor download progress
echo "⏳ Monitoring download progress..."
while kill -0 $DOWNLOAD_PID 2>/dev/null; do
    echo "📊 Download still running... (PID: $DOWNLOAD_PID)"
    
    # Show progress for each year
    for year in 2020 2021 2022 2023 2024; do
        if [ -d "data/raw/modis_sr/$year" ]; then
            show_progress $year
        fi
    done
    
    sleep 300  # Check every 5 minutes
done

# Wait for download to fully complete
echo "⏳ Download process finished, checking data completeness..."
sleep 60  # Give it a moment to finalize

# Check if download was successful
if [ $(check_download_complete) = "true" ]; then
    echo "✅ MODIS SR download completed successfully!"
    echo "🚀 Starting feature extraction and model training..."
    
    # Run the complete pipeline
    python scripts/build_features.py
    
    if [ $? -eq 0 ]; then
        echo "🎉 FireSentry pipeline completed successfully!"
        echo "📊 Results saved to:"
        echo "  - Feature matrix: data/processed/features.parquet"
        echo "  - Model: models/fire_prediction_model.pkl"
        echo "  - Evaluation: docs/fig_*.png"
        echo "  - Presentation: docs/Mini_Project_review_F1[1].pptx"
    else
        echo "❌ Pipeline failed. Check build.log for details."
        exit 1
    fi
else
    echo "❌ Download incomplete. Check for errors and retry."
    exit 1
fi

echo "🏁 Complete pipeline finished!"
