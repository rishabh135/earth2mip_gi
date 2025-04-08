import os
import sys
import time
import argparse
from datetime import datetime, timedelta
import importlib.util
import json
import logging
import os, time
import sys
import logging
from datetime import timedelta
import cdsapi
from collections import defaultdict
import dotenv
import xarray as xr
import numpy as np
import torch
import cdsapi
import pandas as pd
import zarr
import argparse

import importlib.util
import json
import logging
import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
import matplotlib
matplotlib.use('Agg')  # Needed for headless environments
import matplotlib.pyplot as plt
import dotenv
import xarray as xr
import numpy as np
import torch
import cdsapi
import pandas as pd



import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.px.sfno import VARIABLES
import asyncio
import logging
import os
from typing import List
from tqdm import tqdm

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from earth2studio.data import ARCO
from earth2studio.models.px.sfno import VARIABLES
import logging
import os
from typing import List
from tqdm import tqdm
import traceback

# --- Configuration ---
# Script paths and environment (adjust as needed)
USERNAME = "gupt1075"
BASE_SCRATCH_PATH = f"/scratch/gilbreth/{USERNAME}/fcnv2/ARCO_data_73_channels"
EARTH2MIP_PATH = "/scratch/gilbreth/gupt1075/fcnv2/earth2mip"

# --- Enhanced Logging Setup ---
def setup_logging(output_dir):
    """Configures logging to file and console with Pacific Time timestamps"""
    class PSTFormatter(logging.Formatter):
        def converter(self, timestamp):
            return datetime.fromtimestamp(timestamp, tz=timezone(timedelta(hours=-8)))

        def formatTime(self, record, datefmt=None):
            dt = self.converter(record.created)
            if datefmt:
                return dt.strftime(datefmt)
            else:
                return dt.isoformat()

    # Corrected variable name from timestmap to timestamp
    timestamp = datetime.now(timezone(timedelta(hours=-7))).strftime("%d_%B_%H_%M")
    
    log_file = os.path.join(output_dir, f"download_ensemble_pipeline_{str(timestamp)}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler with detailed logging
    file_handler = logging.FileHandler(log_file)
    file_formatter = PSTFormatter(
        "%(asctime)s [%(levelname)-8s] %(name)-25s - %(message)s",
        datefmt="%d_%B_%H:%M:"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler with basic info
    console_handler = logging.StreamHandler()
    console_formatter = PSTFormatter(
        "%(asctime)s [%(levelname)-8s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

def download_arco_data(
    start_date: datetime,
    end_date: datetime,
    times_of_day: List[str],
    variables: List[str],
    download_dir: str,
    logger: logging.Logger,
    max_retries: int = 3
):
    """
    Optimized synchronous ARCO data download with enhanced error handling
    and Zarr version compatibility checks.
    """
    # Validate input dates first
    try:
        ARCO._validate_time([start_date, end_date])
    except ValueError as e:
        logger.error(f"Invalid date range: {str(e)}")
        raise

    logger.info(f"Starting ARCO data download from {start_date} to {end_date}")
    logger.info(f"Downloading times of day: {times_of_day}")
    logger.info(f"Downloading variables: {len(variables)} variables")
    logger.info(f"Saving data to directory: {download_dir}")

    # Initialize ARCO with extended timeout
    try:
        
        arco = ARCO(cache=True, verbose=False, async_timeout=3600, saving_dir=download_dir)
        logger.info("ARCO client initialized successfully with custom saving directory")
        
        
    except Exception as e:
        logger.error(f"Failed to initialize ARCO client: {str(e)}")
        raise

    # Create download directory with existence check
    try:
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"Ensured download directory exists: {download_dir}")
    except OSError as e:
        logger.error(f"Failed to create download directory: {str(e)}")
        raise

    # Generate all datetimes to download with validation
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)

    datetimes = []
    for date in date_list:
        for time_str in times_of_day:
            try:
                hour, minute = map(int, time_str.split(":"))
                dt = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                ARCO._validate_time([dt])
                datetimes.append(dt)
            except ValueError as e:
                logger.error(f"Invalid time {time_str} for date {date}: {str(e)}")
                continue

    if not datetimes:
        logger.error("No valid datetimes to download after validation")
        raise ValueError("No valid datetimes specified")

    logger.info(f"Total datetimes to process: {len(datetimes)}")
    
    downloaded_data = []
    failed_dates = []
    success_count = 0

    # Use tqdm for progress bar with timeout handling
    with tqdm(total=len(datetimes), desc="Downloading ARCO Data") as pbar:
        for dt in datetimes:  # Changed variable name from time to dt
            retries = 0
            success = False
            start_time = time.time()  # Use qualified module reference
            
            while retries < max_retries and not success:
                try:
                    logger.debug(f"Attempt {retries+1}/{max_retries} for {dt}")
                    da = arco(time=dt, variable=variables)
                    downloaded_data.append(da)
                    success_count += 1
                    success = True
                    logger.info(f"Successfully downloaded {dt}")
                except Exception as e:
                    logger.error(f"Attempt {retries+1} failed for {dt}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    retries += 1
                    if retries >= max_retries:
                        failed_dates.append(dt)
                        logger.error(f"Permanent failure for {dt} after {max_retries} attempts")
            
            # Update progress bar with timing info
            pbar.update(1)
            elapsed = time.time() - start_time  # Use qualified module reference
            pbar.set_postfix_str(
                f"Last: {elapsed:.2f}s | Success: {success_count}, Failed: {len(failed_dates)}"
            )

    if len(downloaded_data) == 0:
        logger.error("No data was successfully downloaded. Check previous errors.")
        raise RuntimeError("No data downloaded")

    logger.info(f"Successfully downloaded {len(downloaded_data)}/{len(datetimes)} time steps")
    
    if failed_dates:
        logger.warning(f"Failed to download {len(failed_dates)} time steps:")
        for date in failed_dates:
            logger.warning(f"- {date}")

    try:
        logger.info("Combining downloaded data...")
        combined_da = xr.concat(downloaded_data, dim="time")
        
        logger.info("Converting to numpy array...")
        np_array = combined_da.to_numpy().astype('float32')
        
        filename = f"START_{start_date.strftime('%d%B%Y')}__END_{end_date.strftime('%d%B%Y')}_ics_frames_{len(datetimes)}.npy"
        filepath = os.path.join(download_dir, filename)
        
        logger.info(f"Saving data to {filepath}...")
        np.save(filepath, np_array)
        
        logger.info(f"Successfully saved data to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to process/save data: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError("Data processing failed") from e





if __name__ == "__main__":
    # Example usage with proper date validation
    # 6 and 11 February 2020
    start_date = datetime(2020, 2, 8)
    end_date = datetime(2020, 2, 9)
    times_of_day = ["00:00", "06:00", "12:00", "18:00"]
    variables = VARIABLES
    download_dir = os.path.join(BASE_SCRATCH_PATH, "data")
    
    try:
        # Setup logging first
        logging_dir = os.path.join(BASE_SCRATCH_PATH, "logging")
        os.makedirs(logging_dir, exist_ok=True)
        logger = setup_logging(logging_dir)

        # Validate Zarr version compatibility
        zarr_version = tuple(map(int, zarr.__version__.split('.')))
        if zarr_version < (3,):
            logger.warning(f"Using Zarr v{zarr_version} - consider upgrading to v3+ for better performance")

        result_path = download_arco_data(
            start_date=start_date,
            end_date=end_date,
            times_of_day=times_of_day,
            variables=variables,
            download_dir=download_dir,
            logger=logger
        )
        logger.info(f"Download completed successfully. Data saved to: {result_path}")
        
    except Exception as e:
        logger.critical(f"Critical error occurred: {str(e)}", exc_info=True)
        exit(1)

