#!/usr/bin/env python3
"""
Rename MODIS Files from A to AA Format

This script renames all MODIS files from the old A format to the new AA format
that the pipeline expects.

Example:
- MYD09GA.A2021037.h24v05.061.B01.tif → MYD09GA.AA2021037.h24v05.061.B01.tif
- MOD11A1.A2021037.h24v05.061.LST_Day_1km.tif → MOD11A1.AA2021037.h24v05.061.LST_Day_1km.tif

Usage:
    python scripts/rename_modis_files.py
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def rename_modis_files(base_dir="data/raw"):
    """
    Rename all MODIS files from A format to AA format.
    
    Args:
        base_dir: Base directory containing MODIS data
    """
    logger.info("🔄 Starting MODIS file renaming from A to AA format...")
    
    # Directories to process
    modis_dirs = [
        Path(base_dir) / "modis_sr",
        Path(base_dir) / "modis_lst"
    ]
    
    total_renamed = 0
    
    for modis_dir in modis_dirs:
        if not modis_dir.exists():
            logger.warning(f"Directory not found: {modis_dir}")
            continue
            
        logger.info(f"Processing directory: {modis_dir}")
        
        # Find all files with A format
        pattern = "*.A[0-9][0-9][0-9][0-9][0-9][0-9][0-9].*"
        files_to_rename = list(modis_dir.rglob(pattern))
        
        logger.info(f"Found {len(files_to_rename)} files to rename in {modis_dir}")
        
        for file_path in files_to_rename:
            try:
                # Get the filename
                filename = file_path.name
                
                # Check if it's already in AA format
                if ".AA" in filename:
                    logger.debug(f"Already in AA format: {filename}")
                    continue
                
                # Replace .A with .AA
                if ".A" in filename:
                    new_filename = filename.replace(".A", ".AA")
                    new_path = file_path.parent / new_filename
                    
                    # Check if target file already exists
                    if new_path.exists():
                        logger.warning(f"Target file already exists: {new_path}")
                        continue
                    
                    # Rename the file
                    file_path.rename(new_path)
                    total_renamed += 1
                    
                    if total_renamed % 100 == 0:
                        logger.info(f"Renamed {total_renamed} files...")
                        
            except Exception as e:
                logger.error(f"Error renaming {file_path}: {e}")
    
    logger.info(f"✅ Renaming complete! Total files renamed: {total_renamed}")
    return total_renamed

def verify_renaming(base_dir="data/raw"):
    """
    Verify that files were renamed correctly.
    
    Args:
        base_dir: Base directory containing MODIS data
    """
    logger.info("🔍 Verifying file renaming...")
    
    modis_dirs = [
        Path(base_dir) / "modis_sr",
        Path(base_dir) / "modis_lst"
    ]
    
    total_aa_files = 0
    total_a_files = 0
    
    for modis_dir in modis_dirs:
        if not modis_dir.exists():
            continue
            
        # Count AA format files
        aa_pattern = "*.AA[0-9][0-9][0-9][0-9][0-9][0-9][0-9].*"
        aa_files = list(modis_dir.rglob(aa_pattern))
        total_aa_files += len(aa_files)
        
        # Count remaining A format files
        a_pattern = "*.A[0-9][0-9][0-9][0-9][0-9][0-9][0-9].*"
        a_files = list(modis_dir.rglob(a_pattern))
        total_a_files += len(a_files)
        
        logger.info(f"{modis_dir.name}: {len(aa_files)} AA files, {len(a_files)} A files")
    
    logger.info(f"📊 Summary: {total_aa_files} AA format files, {total_a_files} A format files")
    
    if total_a_files == 0:
        logger.info("✅ All files successfully converted to AA format!")
    else:
        logger.warning(f"⚠️  {total_a_files} files still in A format")

def main():
    """Main function."""
    logger.info("🚀 MODIS File Renaming Script")
    logger.info("=" * 50)
    
    start_time = datetime.now()
    
    try:
        # Rename files
        total_renamed = rename_modis_files()
        
        # Verify renaming
        verify_renaming()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 50)
        logger.info("🎉 RENAMING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Files renamed: {total_renamed}")
        logger.info(f"Duration: {duration}")
        logger.info("You can now run the pipeline with AA format files!")
        
    except Exception as e:
        logger.error(f"❌ Renaming failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
