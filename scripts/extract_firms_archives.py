#!/usr/bin/env python3
"""
FIRMS Annual Archives Extractor and Filter

Extracts MODIS annual fire data archives (modis_YYYY_all_countries.zip),
filters the 'modis_YYYY_India.csv' files within them to the Uttarakhand bounding box,
and combines the filtered data into a single CSV file.

Usage:
    python scripts/extract_firms_archives.py

Requirements:
    - pandas
    - python-dotenv
    - zipfile (built-in)
"""

import os
import sys
import zipfile
import pandas as pd
from pathlib import Path
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FIRMSArchiveExtractor:
    def __init__(self, config_path=".env"):
        """Initialize FIRMS archive extractor with configuration."""
        load_dotenv(dotenv_path=config_path)
        self.bbox_n = float(os.getenv('BBOX_N', 31.459016))
        self.bbox_s = float(os.getenv('BBOX_S', 28.709556))
        self.bbox_e = float(os.getenv('BBOX_E', 81.044789))
        self.bbox_w = float(os.getenv('BBOX_W', 77.575402))
        
        # User specified 2020-2024 for available archives (no 2025)
        self.year_start = int(os.getenv('YEAR_START', 2020))
        self.year_end = int(os.getenv('YEAR_END', 2024))  # Adjusted to 2024 based on user input

        self.archive_dir = Path("data/raw/firms/archives")
        self.output_dir = Path("data/raw/firms")
        self.temp_extract_dir = Path("data/temp/firms_extract")

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FIRMS Archive Extractor initialized for years {self.year_start}-{self.year_end}")
        logger.info(f"Bounding Box (Uttarakhand): ({self.bbox_w}, {self.bbox_s}, {self.bbox_e}, {self.bbox_n})")
        logger.info(f"Archives expected in: {self.archive_dir}")
        logger.info(f"Output will be saved to: {self.output_dir}")

    def extract_and_filter_year(self, year: int) -> pd.DataFrame:
        """
        Extracts the India CSV from the annual zip, filters it to Uttarakhand,
        and returns the filtered DataFrame.
        """
        archive_filename = f"modis_{year}_all_countries.zip"
        archive_path = self.archive_dir / archive_filename
        
        if not archive_path.exists():
            logger.warning(f"Archive not found for year {year}: {archive_path}. Skipping.")
            return pd.DataFrame()

        logger.info(f"Processing archive: {archive_path}")
        
        # Create a temporary directory for this year's extraction
        year_temp_dir = self.temp_extract_dir / str(year)
        year_temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # The path inside the zip is "modis/YYYY/modis_YYYY_India.csv"
                csv_in_zip_path = f"modis/{year}/modis_{year}_India.csv"
                
                if csv_in_zip_path not in zip_ref.namelist():
                    logger.error(f"CSV file '{csv_in_zip_path}' not found inside {archive_filename}. Skipping.")
                    return pd.DataFrame()
                
                # Extract only the required CSV file
                extracted_csv_path = zip_ref.extract(csv_in_zip_path, year_temp_dir)
                
                logger.info(f"Loading extracted CSV: {extracted_csv_path}")
                df = pd.read_csv(extracted_csv_path)
                
                # Filter to Uttarakhand bounding box
                df_filtered = df[
                    (df['latitude'] >= self.bbox_s) & (df['latitude'] <= self.bbox_n) &
                    (df['longitude'] >= self.bbox_w) & (df['longitude'] <= self.bbox_e)
                ].copy()  # Use .copy() to avoid SettingWithCopyWarning
                
                logger.info(f"Year {year}: {len(df)} total records, {len(df_filtered)} records in Uttarakhand.")
                return df_filtered

        except zipfile.BadZipFile:
            logger.error(f"Bad zip file: {archive_path}. Please check the file integrity.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing {archive_path}: {e}")
            return pd.DataFrame()
        finally:
            # Clean up temporary extracted files for this year
            if year_temp_dir.exists():
                shutil.rmtree(year_temp_dir)

    def run(self):
        """Main execution function to extract and combine FIRMS data."""
        logger.info("Starting FIRMS archive extraction and filtering process.")
        
        all_firms_data = []
        
        for year in tqdm(range(self.year_start, self.year_end + 1), desc="Processing FIRMS archives"):
            df_year = self.extract_and_filter_year(year)
            if not df_year.empty:
                all_firms_data.append(df_year)
        
        if not all_firms_data:
            logger.warning("No FIRMS data extracted for any year. Output CSV will be empty.")
            # Create an empty CSV to avoid errors in later steps
            output_filename = f"firms_uttarakhand_{self.year_start}_{self.year_end}.csv"
            output_path = self.output_dir / output_filename
            pd.DataFrame(columns=['latitude', 'longitude', 'brightness', 'scan', 'track',
                                  'acq_date', 'acq_time', 'satellite', 'instrument confidence',
                                  'version', 'bright_t31', 'frp', 'daynight', 'type']).to_csv(output_path, index=False)
            logger.info(f"Empty FIRMS CSV created at {output_path} as a placeholder.")
            return

        combined_df = pd.concat(all_firms_data, ignore_index=True)
        
        # Ensure 'acq_date' is in datetime format for sorting
        combined_df['acq_date'] = pd.to_datetime(combined_df['acq_date'])
        combined_df = combined_df.sort_values(by='acq_date').reset_index(drop=True)

        output_filename = f"firms_uttarakhand_{self.year_start}_{self.year_end}.csv"
        output_path = self.output_dir / output_filename
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully combined and saved {len(combined_df)} FIRMS records to {output_path}")
        logger.info("FIRMS archive extraction and filtering process complete.")

        # Clean up the main temporary extraction directory
        if self.temp_extract_dir.exists():
            shutil.rmtree(self.temp_extract_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_extract_dir}")

if __name__ == "__main__":
    extractor = FIRMSArchiveExtractor()
    extractor.run()



