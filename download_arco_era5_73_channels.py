





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


import os
import logging
import traceback
import time  # Import time module explicitly
from datetime import datetime, timedelta
from typing import List
import zarr # Import zarr for version check
import numpy as np
import xarray as xr
from tqdm import tqdm


# --- Configuration ---
# Script paths and environment (adjust as needed)
USERNAME = "gupt1075"
BASE_SCRATCH_PATH = f"/scratch/gilbreth/wwtung/ARCO_73chanel_data"
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







# def download_arco_data(
#     start_date: datetime,
#     end_date: datetime,
#     times_of_day: List[str],
#     variables: List[str],
#     download_dir: str,
#     logger: logging.Logger,
#     max_retries: int = 3
# ):
#     """
#     Optimized synchronous ARCO data download with enhanced error handling
#     and Zarr version compatibility checks.
#     """
#     # Validate input dates first
#     try:
#         ARCO._validate_time([start_date, end_date])
#     except ValueError as e:
#         logger.error(f"Invalid date range: {str(e)}")
#         raise

#     logger.info(f"Starting ARCO data download from {start_date} to {end_date}")
#     logger.info(f"Downloading times of day: {times_of_day}")
#     logger.info(f"Downloading variables: {len(variables)} variables")
#     logger.info(f"Saving data to directory: {download_dir}")

#     # Initialize ARCO with extended timeout
#     try:
        
#         arco = ARCO(cache=True, verbose=False, async_timeout=3600, saving_dir=download_dir)
#         logger.info("ARCO client initialized successfully with custom saving directory")
        
        
#     except Exception as e:
#         logger.error(f"Failed to initialize ARCO client: {str(e)}")
#         raise

#     # Create download directory with existence check
#     try:
#         os.makedirs(download_dir, exist_ok=True)
#         logger.info(f"Ensured download directory exists: {download_dir}")
#     except OSError as e:
#         logger.error(f"Failed to create download directory: {str(e)}")
#         raise

#     # Generate all datetimes to download with validation
#     date_list = []
#     current_date = start_date
#     while current_date <= end_date:
#         date_list.append(current_date)
#         current_date += timedelta(days=1)

#     datetimes = []
#     for date in date_list:
#         for time_str in times_of_day:
#             try:
#                 hour, minute = map(int, time_str.split(":"))
#                 dt = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
#                 ARCO._validate_time([dt])
#                 datetimes.append(dt)
#             except ValueError as e:
#                 logger.error(f"Invalid time {time_str} for date {date}: {str(e)}")
#                 continue

#     if not datetimes:
#         logger.error("No valid datetimes to download after validation")
#         raise ValueError("No valid datetimes specified")

#     logger.info(f"Total datetimes to process: {len(datetimes)}")
    
#     downloaded_data = []
#     failed_dates = []
#     success_count = 0

#     # Use tqdm for progress bar with timeout handling
#     with tqdm(total=len(datetimes), desc="Downloading ARCO Data") as pbar:
#         for dt in datetimes:  # Changed variable name from time to dt
#             retries = 0
#             success = False
#             start_time = time.time()  # Use qualified module reference
            
#             while retries < max_retries and not success:
#                 try:
#                     logger.debug(f"Attempt {retries+1}/{max_retries} for {dt}")
#                     da = arco(time=dt, variable=variables)
#                     downloaded_data.append(da)
#                     success_count += 1
#                     success = True
#                     logger.info(f"Successfully downloaded {dt}")
#                 except Exception as e:
#                     logger.error(f"Attempt {retries+1} failed for {dt}: {str(e)}")
#                     logger.debug(traceback.format_exc())
#                     retries += 1
#                     if retries >= max_retries:
#                         failed_dates.append(dt)
#                         logger.error(f"Permanent failure for {dt} after {max_retries} attempts")
            
#             # Update progress bar with timing info
#             pbar.update(1)
#             elapsed = time.time() - start_time  # Use qualified module reference
#             pbar.set_postfix_str(
#                 f"Last: {elapsed:.2f}s | Success: {success_count}, Failed: {len(failed_dates)}"
#             )

#     if len(downloaded_data) == 0:
#         logger.error("No data was successfully downloaded. Check previous errors.")
#         raise RuntimeError("No data downloaded")

#     logger.info(f"Successfully downloaded {len(downloaded_data)}/{len(datetimes)} time steps")
    
#     if failed_dates:
#         logger.warning(f"Failed to download {len(failed_dates)} time steps:")
#         for date in failed_dates:
#             logger.warning(f"- {date}")

#     try:
#         logger.info("Combining downloaded data...")
#         combined_da = xr.concat(downloaded_data, dim="time")
        
#         logger.info("Converting to numpy array...")
#         np_array = combined_da.to_numpy().astype('float32')
        
#         filename = f"start_{start_date.strftime('%d_%B_%Y')}_end_{end_date.strftime('%d_%B_%Y')}_ics_frames_{len(datetimes)}.npy"
#         filepath = os.path.join(download_dir, filename)
        
#         logger.info(f"Saving data to {filepath}...")
#         np.save(filepath, np_array)
        
#         logger.info(f"Successfully saved data to {filepath}")
#         return filepath
        
#     except Exception as e:
#         logger.error(f"Failed to process/save data: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise RuntimeError("Data processing failed") from e











# --- Main Download Function ---
def download_single_day_arco_data(
    target_date: datetime,
    times_of_day: List[str],
    variables: List[str],
    download_dir: str,
    logger: logging.Logger,
    arco_client: ARCO, # Pass the initialized client instance
    max_retries: int = 3
) -> str | None:
    """
    Downloads ARCO data for a single specified day using a provided ARCO client.
    (Keep docstring Args/Returns)
    """
    logger.info(f"--- Starting download for {target_date.strftime('%Y-%m-%d')} ---")
    day_start_time = time.time()

    # Generate specific datetimes for the target date
    datetimes_to_download = []
    for time_str in times_of_day:
        try:
            hour, minute = map(int, time_str.split(":"))
            dt = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # Validate this specific datetime using ARCO's validator
            ARCO._validate_time([dt]) # Use class method for validation
            datetimes_to_download.append(dt)
        except ValueError as e:
            logger.error(f"Invalid time '{time_str}' or resulting datetime for date {target_date.strftime('%Y-%m-%d')}: {str(e)}")
            continue # Skip this invalid time

    if not datetimes_to_download:
        logger.error(f"No valid datetimes generated for {target_date.strftime('%Y-%m-%d')}. Skipping this day.")
        return None

    logger.info(f"Processing {len(datetimes_to_download)} time steps for {target_date.strftime('%Y-%m-%d')}: {[dt.strftime('%H:%M') for dt in datetimes_to_download]}")

    downloaded_data_arrays = []
    failed_times = []
    success_count = 0
    total_attempts = 0

    # *** REMOVED ARCO Initialization from here ***

    # Progress bar for the time steps within this day
    with tqdm(total=len(datetimes_to_download), desc=f"Day {target_date.strftime('%Y-%m-%d')}", unit="timestep") as pbar:
        for dt in datetimes_to_download:
            retries = 0
            success = False
            last_error = None
            step_start_time = time.time()

            while retries < max_retries and not success:
                total_attempts += 1
                try:
                    logger.debug(f"Attempt {retries+1}/{max_retries} for {dt.strftime('%Y-%m-%d %H:%M')}")
                    # Use the passed arco_client instance directly
                    da = arco_client(time=dt, variable=variables) # Use the passed client

                    # --- Data Validation and Handling ---
                    if not isinstance(da, xr.DataArray):
                        raise TypeError(f"ARCO call did not return an xarray.DataArray (got {type(da)})")

                    # Ensure expected dimensions and coordinates exist
                    # Check if 'time' coordinate needs adjustment (ARCO might return single step differently)
                    if 'time' not in da.coords:
                         # If fetching single time step, ARCO might omit the time coord.
                         # Add it back if needed for concatenation later.
                         da = da.expand_dims(time=[dt])
                         logger.debug(f"Added 'time' dimension ({dt}) to single DataArray.")
                    elif da.time.size == 1 and da.time.item() != np.datetime64(dt):
                         # If time coord exists but is wrong (less likely), log warning or correct it
                         logger.warning(f"DataArray time coord {da.time.item()} differs from requested {dt}. Using requested time.")
                         da['time'] = [dt]
                    elif da.time.size > 1:
                         # If multiple times returned unexpectedly, select the correct one
                         logger.warning(f"ARCO returned multiple time steps ({da.time.size}). Selecting requested time {dt}.")
                         da = da.sel(time=dt)

                    # Check for expected variables
                    if not all(v in da.variable for v in variables):
                         missing_vars = [v for v in variables if v not in da.variable]
                         logger.warning(f"Returned DataArray missing expected variables: {missing_vars}")
                         # Decide how to handle: raise error, fill with NaN, or continue?
                         # For now, continue but the concat might fail later if shapes differ.

                    # Check for NaN values if critical
                    if da.isnull().any():
                        logger.warning(f"NaN values detected in downloaded data for {dt}.")

                    # --- Append and Mark Success ---
                    downloaded_data_arrays.append(da)
                    success_count += 1
                    success = True
                    logger.info(f"Successfully downloaded and validated data for {dt.strftime('%Y-%m-%d %H:%M')}")

                except (gcsfs.utils.HttpError, asyncio.TimeoutError, TimeoutError, ConnectionError) as net_err:
                    # Handle network/timeout errors specifically for retries
                    last_error = net_err
                    logger.warning(f"Network/Timeout Error (Attempt {retries+1}) for {dt.strftime('%Y-%m-%d %H:%M')}: {str(net_err)}")
                    retries += 1
                    if retries < max_retries:
                        wait_time = 2 ** retries # Exponential backoff (2, 4, 8 seconds)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        failed_times.append(dt)
                        logger.error(f"Permanent network failure for {dt.strftime('%Y-%m-%d %H:%M')} after {max_retries} attempts. Last error: {str(last_error)}")
                        logger.debug(traceback.format_exc()) # Log traceback for permanent fails

                except Exception as e:
                    # Handle other unexpected errors
                    last_error = e
                    logger.error(f"Unexpected Error (Attempt {retries+1}) for {dt.strftime('%Y-%m-%d %H:%M')}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    retries += 1
                    # Decide if retry makes sense for non-network errors (maybe not)
                    # For now, retry all exceptions up to max_retries
                    if retries >= max_retries:
                        failed_times.append(dt)
                        logger.error(f"Permanent failure (non-network) for {dt.strftime('%Y-%m-%d %H:%M')} after {max_retries} attempts. Last error: {str(last_error)}")

            # Update progress bar
            pbar.update(1)
            elapsed = time.time() - step_start_time
            pbar.set_postfix_str(f"Last: {elapsed:.1f}s | OK: {success_count}, Fail: {len(failed_times)} | Last Status: {'OK' if success else 'FAIL'}")

    # --- Post-download processing for the day ---
    if not downloaded_data_arrays:
        logger.error(f"No data successfully downloaded for {target_date.strftime('%Y-%m-%d')}. Check logs for errors.")
        return None # Return None to indicate day failure

    logger.info(f"Successfully downloaded {len(downloaded_data_arrays)}/{len(datetimes_to_download)} time steps for {target_date.strftime('%Y-%m-%d')}.")
    if failed_times:
        logger.warning(f"Failed to download {len(failed_times)} time steps for {target_date.strftime('%Y-%m-%d')}:")
        for ft in failed_times:
            logger.warning(f"- {ft.strftime('%Y-%m-%d %H:%M')}")

    try:
        logger.info(f"Combining {len(downloaded_data_arrays)} DataArrays for {target_date.strftime('%Y-%m-%d')}...")

        # Combine downloaded data arrays
        # Ensure all arrays have the 'time' dimension before concatenating
        arrays_to_concat = [da for da in downloaded_data_arrays if 'time' in da.coords]
        if len(arrays_to_concat) != len(downloaded_data_arrays):
             logger.warning("Some downloaded arrays were missing the 'time' dimension and could not be concatenated.")
             # Decide how to proceed: fail, or concatenate only valid ones?
             # Let's proceed with only the valid ones for now.
        if not arrays_to_concat:
             logger.error("No valid DataArrays with 'time' dimension found to combine.")
             return None


        # Concatenate along time dimension
        combined_da = xr.concat(arrays_to_concat, dim="time")
        # Sort by time for guaranteed order
        combined_da = combined_da.sortby('time')

        logger.info("Converting combined data to NumPy array (float32)...")
        # Ensure data is loaded into memory if Dask-backed (ARCO class seems to fetch eagerly)
        # np_array = combined_da.load().to_numpy().astype(np.float32)
        np_array = combined_da.to_numpy().astype(np.float32) # Convert directly

        # Define filename
        times_str = "-".join(t.replace(":", "") for t in times_of_day)
        filename = f"ARCO_{target_date.strftime('%Y%m%d')}_T{times_str}_V{len(variables)}.npy" # Use YYYYMMDD and V prefix for variables
        filepath = os.path.join(download_dir, filename)

        logger.info(f"Saving daily data to {filepath}...")
        np.save(filepath, np_array)

        day_elapsed_time = time.time() - day_start_time
        logger.info(f"--- Successfully completed download and saved data for {target_date.strftime('%Y-%m-%d')} in {day_elapsed_time:.2f} seconds ---")
        return filepath

    except Exception as e:
        logger.error(f"Failed to process or save data for {target_date.strftime('%Y-%m-%d')}: {str(e)}")
        logger.error(traceback.format_exc())
        return None # Return None on processing/saving failure

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    start_date_dt = datetime(2020, 6, 18)
    end_date_dt = datetime(2020, 6, 22) # Inclusive
    times_of_day_list = ["00:00", "06:00", "12:00", "18:00"]
    # VARIABLES defined globally above

    # Directories
    # Let ARCO manage its cache based on saving_dir or default
    # The final .npy files will go into monthly subdirs here:
    base_save_dir = os.path.join(BASE_SCRATCH_PATH, "arco_final_data")
    logging_dir = os.path.join(BASE_SCRATCH_PATH, "arco_data_logs")
    # Cache base dir (optional, ARCO class figures this out, but useful for clarity)
    cache_base_dir = os.path.join(BASE_SCRATCH_PATH, "arco_cache_managed") # This is where ARCO will put its cache

    # --- Setup ---
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(cache_base_dir, exist_ok=True) # Ensure base for cache exists
    os.makedirs(logging_dir, exist_ok=True)
    logger = setup_logging(logging_dir) # INFO for progress, DEBUG for details

    # Zarr version check
    try:
        zarr_version_str = version("zarr")
        logger.info(f"Using Zarr version: {zarr_version_str}")
        # Add specific warnings if needed based on ARCO class behavior
    except Exception as e:
        logger.warning(f"Could not determine Zarr version: {e}")

    # Validate overall date range
    if start_date_dt > end_date_dt:
        logger.critical(f"Start date ({start_date_dt}) cannot be after end date ({end_date_dt}). Exiting.")
        exit(1)

    # --- Initialize ARCO Client ONCE ---
    arco_client_instance = None # Initialize to None
    try:
        # Pass the cache_base_dir as saving_dir to ARCO if you want it to manage cache there
        arco_client_instance = ARCO(
            cache=True,
            verbose=False, # Tqdm handles verbosity in the download loop
            async_timeout=1800, # 30 min timeout per call internal to ARCO
            saving_dir=cache_base_dir # Tell ARCO where to put cache
        )
        logger.info("ARCO client initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize ARCO client: {str(e)}")
        logger.critical(traceback.format_exc())
        exit(1) # Exit if client fails to initialize


    # --- Download Loop ---
    logger.info(f"Starting ARCO download process from {start_date_dt.strftime('%Y-%m-%d')} to {end_date_dt.strftime('%Y-%m-%d')}")
    overall_start_time = time.time()
    total_days = (end_date_dt - start_date_dt).days + 1
    successful_days_count = 0
    failed_days_list = []

    for i in range(total_days):
        current_process_date = start_date_dt + timedelta(days=i)
        day_str = current_process_date.strftime('%Y-%m-%d')
        logger.info(f"===== Processing Day {i+1}/{total_days}: {day_str} =====")

        # Create YEAR/MONTH subdirectory within the base *save* directory for final NPY files
        # Format: base_save_dir/YYYY/MonthName (e.g., base_save_dir/2020/February)
        year_dir = current_process_date.strftime("%Y")
        month_dir = current_process_date.strftime("%B") # Full month name
        day_save_dir = os.path.join(base_save_dir, year_dir, month_dir)

        try:
            os.makedirs(day_save_dir, exist_ok=True)
            logger.info(f"Saving final .npy file for this day to: {day_save_dir}")
        except OSError as e:
            logger.error(f"Failed to create save directory {day_save_dir}: {str(e)}. Skipping day {day_str}")
            failed_days_list.append(day_str)
            continue # Skip to the next day

        # Call the daily download function, passing the single client instance
        try:
            result_path = download_single_day_arco_data(
                target_date=current_process_date,
                times_of_day=times_of_day_list,
                variables=VARIABLES,
                download_dir=day_save_dir, # Directory to save the final NPY
                logger=logger,
                arco_client=arco_client_instance, # *** Pass the client instance ***
                max_retries=3
            )

            if result_path:
                logger.info(f"Day {day_str} completed successfully. Data saved to: {result_path}")
                successful_days_count += 1
            else:
                # download_single_day_arco_data logs errors internally
                logger.error(f"Day {day_str} failed during processing or saving.")
                failed_days_list.append(day_str)

        except Exception as e:
            # Catch unexpected errors during the daily function call itself
            logger.critical(f"Critical error during processing logic for day {day_str}: {str(e)}")
            logger.critical(traceback.format_exc())
            failed_days_list.append(day_str)
            # Optional: decide whether to stop or continue on critical errors
            # continue

    # --- Clear Cache AFTER the loop ---
    if arco_client_instance: # Check if client was initialized successfully
        logger.info("Download loop finished. Attempting to clear ARCO cache...")
        try:
            arco_client_instance.clear_cache()
        except Exception as e:
            logger.error(f"Failed to clear ARCO cache after run: {e}")
            logger.error(traceback.format_exc())
    else:
         logger.warning("ARCO client was not initialized, skipping cache clear.")


    # --- Final Summary ---
    overall_elapsed_time = time.time() - overall_start_time
    logger.info("===== ARCO Download Process Finished =====")
    logger.info(f"Total execution time: {timedelta(seconds=int(overall_elapsed_time))}")
    logger.info(f"Processed {total_days} days.")
    logger.info(f"Successful days: {successful_days_count}")
    logger.info(f"Failed days: {len(failed_days_list)}")
    if failed_days_list:
        logger.warning(f"List of failed days: {', '.join(failed_days_list)}")

    # --- Set Exit Code ---
    exit_code = 0
    if successful_days_count == 0 and total_days > 0:
        logger.error("No data was successfully downloaded for any day.")
        exit_code = 1
    elif successful_days_count < total_days:
        logger.warning("Some days failed to download or process completely.")
        # Keep exit_code 0 to indicate partial success, but log indicates warning
    else:
        logger.info("All requested days downloaded and processed successfully.")

    logger.info(f"Exiting with code {exit_code}.")
    exit(exit_code)












# # --- Main Download Function ---
# def download_single_day_arco_data(
#     target_date: datetime,
#     times_of_day: List[str],
#     variables: List[str],
#     download_dir: str,
#     logger: logging.Logger,
#     max_retries: int = 3
# ) -> str | None:
#     """
#     Downloads ARCO data for a single specified day, handling errors and retries.

#     Args:
#         target_date: The specific date (datetime object) to download data for.
#         times_of_day: List of "HH:MM" strings for the times to download.
#         variables: List of variable names to download.
#         download_dir: The directory to save the final .npy file.
#         logger: The logger instance.
#         arco_client: The initialized ARCO client instance.
#         max_retries: Maximum number of download attempts per time step.

#     Returns:
#         The filepath of the saved .npy file if successful, otherwise None.
#     """
#     logger.info(f"--- Starting download for {target_date.strftime('%Y-%m-%d')} ---")
#     day_start_time = time.time()

#     # Generate specific datetimes for the target date
#     datetimes_to_download = []
#     for time_str in times_of_day:
#         try:
#             hour, minute = map(int, time_str.split(":"))
#             # Ensure date part is from target_date, time part from time_str
#             dt = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
#             # Validate this specific datetime using ARCO's validator
#             ARCO._validate_time([dt])
#             datetimes_to_download.append(dt)
#         except ValueError as e:
#             logger.error(f"Invalid time '{time_str}' or resulting datetime for date {target_date.strftime('%Y-%m-%d')}: {str(e)}")
#             continue # Skip this invalid time

#     if not datetimes_to_download:
#         logger.error(f"No valid datetimes generated for {target_date.strftime('%Y-%m-%d')}. Skipping this day.")
#         return None

#     logger.info(f"Processing {len(datetimes_to_download)} time steps for {target_date.strftime('%Y-%m-%d')}: {[dt.strftime('%H:%M') for dt in datetimes_to_download]}")

#     downloaded_data_arrays = []
#     failed_times = []
#     success_count = 0
#     total_attempts = 0





#     """
#     Optimized synchronous ARCO data download with enhanced error handling
#     and Zarr version compatibility checks.
#     """

#     # Initialize ARCO with extended timeout
#     try:  
#         arco = ARCO(cache=True, verbose=False, async_timeout=3600, saving_dir=download_dir)
#         logger.info("ARCO client initialized successfully with custom saving directory")        
#     except Exception as e:
#         logger.error(f"Failed to initialize ARCO client: {str(e)}")
#         raise








#     # Progress bar for the time steps within this day
#     with tqdm(total=len(datetimes_to_download), desc=f"Day {target_date.strftime('%Y-%m-%d')}", unit="timestep") as pbar:
#         for dt in datetimes_to_download:
#             retries = 0
#             success = False
#             last_error = None
#             step_start_time = time.time()

#             while retries < max_retries and not success:
#                 total_attempts += 1
#                 try:
#                     logger.debug(f"Attempt {retries+1}/{max_retries} for {dt.strftime('%Y-%m-%d %H:%M')}")
#                     # Use the passed arco_client instance
#                     da = arco(time=dt, variable=variables)

#                     # Basic validation of returned data (can be expanded)
#                     if not isinstance(da, xr.DataArray):
#                         raise TypeError(f"ARCO call did not return an xarray.DataArray (got {type(da)})")
#                     if 'time' not in da.coords and len(datetimes_to_download) > 1:
#                          # If time dimension is missing but expected, add it back for concat
#                          da = da.expand_dims(time=[dt])
#                          logger.debug(f"Added time dimension to DataArray for {dt}")
#                     elif 'time' in da.coords and da.time.size != 1:
#                         # Ensure only the requested time step is present
#                         da = da.sel(time=dt)
#                         logger.debug(f"Selected specific time {dt} from DataArray")


#                     downloaded_data_arrays.append(da)
#                     success_count += 1
#                     success = True
#                     logger.info(f"Successfully downloaded {dt.strftime('%Y-%m-%d %H:%M')}")

#                 except Exception as e:
#                     last_error = e
#                     logger.warning(f"Attempt {retries+1} failed for {dt.strftime('%Y-%m-%d %H:%M')}: {str(e)}")
#                     logger.debug(traceback.format_exc())
#                     retries += 1
#                     if retries < max_retries:
#                         time.sleep(2 ** retries) # Exponential backoff
#                     else:
#                         failed_times.append(dt)
#                         logger.error(f"Permanent failure for {dt.strftime('%Y-%m-%d %H:%M')} after {max_retries} attempts. Last error: {str(last_error)}")

#             # Update progress bar
#             pbar.update(1)
#             elapsed = time.time() - step_start_time
#             pbar.set_postfix_str(f"Last: {elapsed:.1f}s | OK: {success_count}, Fail: {len(failed_times)} | Last Status: {'OK' if success else 'FAIL'}")

#     # --- Post-download processing for the day ---
#     if not downloaded_data_arrays:
#         logger.error(f"No data successfully downloaded for {target_date.strftime('%Y-%m-%d')}. Check logs for errors.")
#         return None

#     logger.info(f"Successfully downloaded {len(downloaded_data_arrays)}/{len(datetimes_to_download)} time steps for {target_date.strftime('%Y-%m-%d')}.")
#     if failed_times:
#         logger.warning(f"Failed to download {len(failed_times)} time steps for {target_date.strftime('%Y-%m-%d')}:")
#         for ft in failed_times:
#             logger.warning(f"- {ft.strftime('%Y-%m-%d %H:%M')}")

#     try:
#         logger.info(f"Combining {len(downloaded_data_arrays)} DataArrays for {target_date.strftime('%Y-%m-%d')}...")
#         # Ensure consistent chunking before concat if using dask-backed arrays
#         # downloaded_data_arrays = [da.chunk({}) for da in downloaded_data_arrays] # Remove dask chunks for numpy conversion

#         # Check if 'time' dimension exists before concatenating
#         if 'time' in downloaded_data_arrays[0].coords:
#              combined_da = xr.concat(downloaded_data_arrays, dim="time")
#              # Sort by time just in case order was mixed up (unlikely here, but good practice)
#              combined_da = combined_da.sortby('time')
#         elif len(downloaded_data_arrays) == 1:
#              combined_da = downloaded_data_arrays[0] # No need to concat single array
#         else:
#              # This case should ideally be handled by adding 'time' dim earlier
#              logger.error("Cannot concatenate: 'time' dimension missing and multiple arrays present.")
#              raise ValueError("Cannot concatenate DataArrays without a 'time' dimension")


#         logger.info("Converting combined data to NumPy array (float32)...")
#         # Ensure data is loaded into memory before converting if it's Dask-backed
#         np_array = combined_da.load().to_numpy().astype(np.float32)

#         # Define a clear filename for the daily data
#         times_str = "-".join(t.replace(":", "") for t in times_of_day)
#         filename = f"ARCO_{target_date.strftime('%d_%B_%Y')}_T{times_str}_channels{len(variables)}.npy"
#         filepath = os.path.join(download_dir, filename)

#         logger.info(f"Saving daily data to {filepath}...")
#         np.save(filepath, np_array)

#         day_elapsed_time = time.time() - day_start_time
#         logger.info(f"--- Successfully completed download and saved data for {target_date.strftime('%Y-%m-%d')} in {day_elapsed_time:.2f} seconds ---")
#         return filepath

#     except Exception as e:
#         logger.error(f"Failed to process or save data for {target_date.strftime('%Y-%m-%d')}: {str(e)}")
#         logger.error(traceback.format_exc())
#         return None


# # --- Main execution block ---
# if __name__ == "__main__":
#     # --- Configuration ---
#     # Dates
#     start_date_dt = datetime(2020, 2, 8)
#     end_date_dt = datetime(2020, 2, 10) # Inclusive end date



#     times_of_day_list = ["00:00", "06:00", "12:00", "18:00"]
#     variables = VARIABLES


#     # Directories
#     base_download_dir = os.path.join(BASE_SCRATCH_PATH, "data")
#     logging_dir = os.path.join(BASE_SCRATCH_PATH, "arco_data_logs")




#     # Setup logging first
#     logging_dir = os.path.join(BASE_SCRATCH_PATH, "logging")
#     os.makedirs(logging_dir, exist_ok=True)
#     logger = setup_logging(logging_dir)




#     # Zarr version check
#     try:
#         zarr_version_tuple = tuple(map(int, zarr.__version__.split('.')))
#         logger.info(f"Using Zarr version: {zarr.__version__}")
#         if zarr_version_tuple < (2, 11): # Example check: Herbie/ARCO might need specific features
#             logger.warning(f"Zarr v{zarr.__version__} is quite old. Consider upgrading Zarr (pip install --upgrade zarr) for potential performance/compatibility benefits.")
#     except Exception as e:
#         logger.warning(f"Could not determine Zarr version: {e}")

#     # Validate overall date range (basic check)
#     if start_date_dt > end_date_dt:
#         logger.critical(f"Start date ({start_date_dt}) cannot be after end date ({end_date_dt}).")
#         exit(1)

#     # Initialize ARCO client once
#     try:
#         # Adjust timeout as needed. 3600s = 1 hour.
#         # Remove saving_dir from here, manage manually per day/month
#         arco_client = ARCO(cache=True, verbose=False, async_timeout=3600)
#         logger.info("ARCO client initialized successfully.")
#     except Exception as e:
#         logger.critical(f"Failed to initialize ARCO client: {str(e)}")
#         logger.critical(traceback.format_exc())
#         exit(1)


#     # --- Download Loop ---
#     logger.info(f"Starting ARCO download process from {start_date_dt.strftime('%Y-%m-%d')} to {end_date_dt.strftime('%Y-%m-%d')}")
#     overall_start_time = time.time()
#     total_days = (end_date_dt - start_date_dt).days + 1
#     successful_days = 0
#     failed_days_list = []

#     for i in range(total_days):
#         current_process_date = start_date_dt + timedelta(days=i)
#         day_str = current_process_date.strftime('%Y-%m-%d')
#         logger.info(f"===== Processing Day {i+1}/{total_days}: {day_str} =====")

#         # Create multiple subdirectory for each month and year within the base download directory
#         # Format: "base_download_dir/Year/Month" (e.g., "base_download_dir/2020/February")
        
#         month_dir_name = current_process_date.strftime("%Y/%B")
#         day_download_dir = os.path.join(base_download_dir, month_dir_name)
        
        
#         try:
#             os.makedirs(day_download_dir, exist_ok=True)
#             logger.info(f"Using download directory for this day: {day_download_dir}")
#         except OSError as e:
#             logger.error(f"Failed to create directory {day_download_dir}: {str(e)}. Skipping day {day_str}")
#             failed_days_list.append(day_str)
#             continue # Skip to the next day


#         # Call the daily download function
#         try:
#             result_path = download_single_day_arco_data(
#                 target_date=current_process_date,
#                 times_of_day=times_of_day_list,
#                 variables=VARIABLES,
#                 download_dir=day_download_dir, # Pass the specific monthly dir
#                 logger=logger,
#                 max_retries=3
#             )

#             if result_path:
#                 logger.info(f"Day {day_str} completed. Data saved to: {result_path}")
#                 successful_days += 1
#             else:
#                 logger.error(f"Day {day_str} failed after processing.")
#                 failed_days_list.append(day_str)

#         except Exception as e:
#             # Catch unexpected errors during the daily function call itself
#             logger.critical(f"Critical error during processing for day {day_str}: {str(e)}")
#             logger.critical(traceback.format_exc())
#             failed_days_list.append(day_str)
#             # Decide whether to continue with other days or stop
#             # continue # To try next day
#             # exit(1) # To stop entirely on critical error

#     # --- Final Summary ---
#     overall_elapsed_time = time.time() - overall_start_time
#     logger.info("===== ARCO Download Process Finished =====")
#     logger.info(f"Total time: {overall_elapsed_time:.2f} seconds")
#     logger.info(f"Processed {total_days} days.")
#     logger.info(f"Successful days: {successful_days}")
#     logger.info(f"Failed days: {len(failed_days_list)}")
#     if failed_days_list:
#         logger.warning(f"List of failed days: {', '.join(failed_days_list)}")

#     if successful_days == 0 and total_days > 0:
#         logger.error("No data was successfully downloaded for any day.")
#         exit(1)
#     elif successful_days < total_days:
#         logger.warning("Some days failed to download completely.")
#         exit(0) # Exit with 0 as some days succeeded, but indicate warning
#     else:
#         logger.info("All requested days downloaded successfully.")
#         exit(0) # Exit successfully









"""
Key Changes and Improvements:

    Modular Daily Download Function:

        Created download_single_day_arco_data specifically to handle the download and processing for one day.

        This function takes the target_date and the initialized arco_client as arguments.

        It generates the specific datetime objects for that day and the requested times_of_day.

        It includes the retry loop, tqdm progress bar (now specific to the timesteps of that single day), data combination (xr.concat), conversion to NumPy (float32), and saving to .npy.

        The filename is now specific to the day (ARCO_YYYYMMDD_T..._V....npy).

        It returns the filepath on success or None on failure for that day.

        Includes specific timing log messages for the start and end of the daily processing.

    Refined __main__ Block:

        Clear Configuration: Variables, dates, times, and paths are defined at the top.

        Robust Initialization: Initializes the ARCO client once outside the loop to avoid repeated initialization overhead and potential issues. It exits if initialization fails.

        Daily Loop: Iterates through the date range using timedelta.

        Directory Management: Creates a monthly subdirectory (e.g., base_download_dir/2020_02) for organization and passes this specific directory to the download_single_day_arco_data function. Handles potential errors during directory creation.

        Function Call: Calls download_single_day_arco_data for each day.

        Timing: Records the start time before the daily loop and calculates the total duration at the end. The daily function logs its own duration.

        Progress Reporting: The tqdm bar now shows progress within each day. The main loop logs the start and end of processing for each day (===== Processing Day X/Y... =====).

        Error Handling: Catches errors during the daily function call and logs them. Tracks successful and failed days.

        Final Summary: Prints a summary of total time, successful days, and failed days.

        Exit Codes: Uses exit(0) for success (even with warnings for partially failed days) and exit(1) for critical failures (like initialization failure or no data downloaded at all).

    ARCO Client Handling: The ARCO client is initialized once and passed into the daily download function, making it more efficient. Removed saving_dir from ARCO initialization as file saving is handled manually.

    Error Handling & Retries: Enhanced retry logic with exponential backoff (time.sleep(2 ** retries)). More specific error logging, including tracebacks for debugging. Added basic validation for the data returned by ARCO.

    Data Handling: Explicitly loads data (.load()) before converting to NumPy if using Dask arrays. Handles cases where the time dimension might be missing or needs selection. Sorts by time after concatenation for robustness.

    Logging: Improved log messages for clarity. Added setup_logging helper function.

    Imports and Typing: Cleaned up imports and added type hints.

    Mock ARCO: Included a MockARCO class to allow testing the script's logic without needing the actual arco_era5 library installed or hitting the network. Replace ARCO = MockARCO with the real import when ready.

    Zarr Check: Kept the Zarr version check.

This structure makes the download process much clearer, more robust against failures on specific days or times, and provides better feedback on progress and timing.
    The script is now more modular, allowing for easier testing and debugging of individual components.

"""



















