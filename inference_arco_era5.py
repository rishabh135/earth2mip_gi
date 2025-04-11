import numpy as np
import datetime
from datetime import timezone, timedelta
import os
import logging
import argparse
import time
import sys
import _pickle
import torch
import xarray as xr
from dateutil.relativedelta import relativedelta
import dotenv
import pytz

# Need these imports if not already present at the top
import collections
import pickle
import pytz
from typing import List, Dict, Optional, Callable, Any, Tuple # For type hints

import torch
import re
import datetime
from datetime import timedelta
import datetime
import time
import logging
from typing import Optional, Any, Iterator, Tuple, List, Dict
import collections # For deque



import os
import re
from datetime import datetime
from typing import Optional





import torch
import numpy as np
import datetime
import time
import logging
import collections
from typing import Optional, Callable, Dict, Any, Tuple, List # Added Tuple, List




# --- Configuration ---
USERNAME = "gupt1075"
MODEL_REGISTRY_BASE = f"/scratch/gilbreth/gupt1075/fcnv2/"
EARTH2MIP_PATH = f"/scratch/gilbreth/gupt1075/fcnv2/earth2mip"

# --- Add earth2mip to Python path ---
if EARTH2MIP_PATH not in sys.path:
    sys.path.insert(0, EARTH2MIP_PATH)
    print(f"Added {EARTH2MIP_PATH} to Python path.")

# --- Environment Setup ---
os.environ["WORLD_SIZE"] = "1"
os.environ["MODEL_REGISTRY"] = MODEL_REGISTRY_BASE
print(f"Set MODEL_REGISTRY environment variable to: {MODEL_REGISTRY_BASE}")

# --- Logging Setup ---
def setup_logging(log_dir, log_level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    pacific_tz = pytz.timezone("America/Los_Angeles")
    timestamp_str = datetime.datetime.now(pacific_tz).strftime("%d_%B_%H_%M_")
    log_filename = os.path.join(log_dir, f"inference_pipeline_{timestamp_str}.log")

    class PytzFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None, tz=None):
            super().__init__(fmt, datefmt)
            self.tz = tz if tz else pytz.utc

        def formatTime(self, record, datefmt=None):
            dt = datetime.datetime.fromtimestamp(record.created, self.tz)
            if datefmt:
                return dt.strftime(datefmt)
            else:
                return dt.strftime("%Y-%m-%d %H:%M:%S.%f %Z%z")

    logger = logging.getLogger("FCNv2Inference")
    logger.setLevel(log_level)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename, mode="w")
    file_formatter = PytzFormatter("%(asctime)s [%(levelname)-8s] [%(name)s:%(lineno)d] %(message)s", tz=pacific_tz)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = PytzFormatter("%(asctime)s [%(levelname)-8s] %(message)s", tz=pacific_tz)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)

    logger.info(f"Logging configured. Level: {logging.getLevelName(logger.level)}")
    logger.info(f"Log file: {log_filename}")
    return logger











# --- Determine Output Directory and Setup Logging ---
pacific_tz = pytz.timezone("America/Los_Angeles")
timestamp = datetime.datetime.now(pacific_tz).strftime("%d_%B_%H_%M")
OUTPUT_DIR = f"/scratch/gilbreth/wwtung/FourCastNetV2_RESULTS_2025/{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
logger = setup_logging(LOG_DIR)

logger.info(f"Using Output Directory: {OUTPUT_DIR}")

# --- Load Environment Variables (optional) ---
dotenv.load_dotenv()
logger.info("Checked for .env file.")

# --- Earth-2 MIP Imports (AFTER setting env vars and sys.path) ---
try:
    from earth2mip import registry
    from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
    print("Successfully imported earth2mip components.")
except ImportError as e:
    print(f"Error importing earth2mip components: {e}")
    print("Please ensure earth2mip is installed correctly and EARTH2MIP_PATH is correct.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during earth2mip import: {e}")
    sys.exit(1)




import earth2mip.grid
from earth2mip import (
    ModelRegistry,
    loaders,
    model_registry,
    registry,
    schema,
    time_loop,
)

# --- Core Inference Function (adapted from main_inference) ---
# [ ... run_inference function remains the same as in your previous code ... ]
# Ensure logger calls within run_inference use the passed 'logger' object.















"""

Key Changes in run_inference:

    initial_time_dt Argument: Added initial_time_dt (a datetime object) as a required argument.

    Get n_history: Retrieves n_history from the model_inference object (defaulting to 0).

    Initial State Preparation:

        Creates the 4D ensemble batch (E, C, H, W).

        Unsqueezes it to the 5D shape (E, n_history+1, C, H, W) required by the time_loop.

    Normalization/Perturbation:

        Normalizes the initial 5D state using model_inference.normalize().

        Applies perturbation (if configured) to this initial normalized state.

    time_loop Initialization: Calls iterator = model_inference(time=initial_time_dt, x=initial_state_perturbed_norm_5d). This passes the required time and the prepared initial (perturbed, normalized) state x.

    Iteration:

        Replaces the manual for step in range(simulation_length): loop.

        Iterates over the iterator obtained from model_inference().

        The loop runs simulation_length + 1 times to get the initial state plus the forecast steps.

        It extracts the data_denorm yielded by the iterator at each step.

    Output Collection: Appends data_denorm.cpu() to output_tensors_denorm based on output_freq.

    Stacking: Stacks the collected 4D tensors along dim=1 to create the final 5D output (E, T_out, C, H, W).

    Logging: Updated logs to reflect the use of the iterator and the steps involved.

How to Update Your Main Script (inference_arco_73_numpy.py):

You need to modify the loop in your main function where run_inference is called to pass the initial_time datetime object.

      


"""











"""

xplanation and Usage Notes:

    Modular Design:

        The core logic is now centered around iterating the time_loop.

        Normalization and perturbation happen before the loop starts.

        Output handling (intermediate saving or full collection) is clearly separated.

        A dedicated save_output_steps function handles the specialized saving logic.

    Ensemble Creation:

        Ensembles are created for each IC at the beginning of run_inference using initial_state_tensor.repeat(n_ensemble, ...).

        Perturbation (noise) is added to this ensemble batch (excluding member 0).

        The time_loop processes this entire ensemble batch in parallel on the GPU.

    Simulation Length:

        Controlled by config['simulation_length']. This is the number of steps after the initial time (t=0).

        The time_loop iterator is run simulation_length + 1 times.

        How deep can you go? This depends on:

            Model Stability: Numerical errors accumulate. Models might become unstable after a certain number of steps (days/weeks). FCNv2 is generally stable for medium-range forecasts (e.g., 10-15 days, which is 40-60 steps). Longer S2S forecasts might show drift.

            Compute Time: Each step takes time.

            Memory (if collecting full history): Storing the entire history uses E * T_out * C * H * W * 4 bytes.

            Memory (with intermediate saving): Memory usage is dominated by the current state tensor (E, T, C, H, W) on the GPU during the loop, plus the small CPU buffer (buffer_size steps). This is much lower.

    Performance & Memory:

        Using time_loop: This is generally efficient as it avoids manual state copying between CPU/GPU within the loop.

        GPU Usage: The model runs on the GPU. The main memory bottleneck during the loop is the current state tensor x inside _iterate, which has shape (E, n_history+1, C, H, W).

        CPU Usage:

            If collecting full history: The list output_tensors_full_history grows, consuming CPU RAM.

            If saving intermediates: The output_history_buffer (deque) holds only buffer_size tensors on the CPU RAM, which is minimal. The main CPU load comes from the save_output_steps function (xarray creation, NetCDF writing).

        Intermediate Saving: Calling save_output_steps inside the loop is the key to minimizing peak CPU RAM usage if the full history is too large. It trades RAM for I/O overhead. collections.deque is efficient for the buffer. Moving data to CPU immediately (data_denorm.cpu()) frees GPU memory faster.

    Variable Subsetting (Lowest Memory):

        Challenge: FCNv2 (and the Inference wrapper) are built assuming all 73 input channels are needed to predict all 73 output channels at each step. You cannot simply "turn off" calculations for certain variables during the autoregressive loop without fundamentally changing the model or wrapper.

        Post-Processing (Standard): The standard approach is to run the full model and then select/save only the variables you need from the output files using tools like xarray. This doesn't save memory during the run.

        Modifying Inference (Advanced): You could modify the Inference class. In _iterate, after next_state_norm = self.model(...), you could select only the channels needed for the next step's input from next_state_norm before assigning it back to x. However, this assumes the model can actually run with fewer input channels, which is unlikely for FCNv2 without retraining or significant architectural changes. It would likely break the physics/dynamics learned by the model.

        Modifying save_output_steps (Practical): If you only need final output files with fewer variables, you can modify save_output_steps to select specific channels before creating the xarray.DataArray.

              
        # Inside save_output_steps, after stacking output_tensor
        if 'variables_to_save' in config:
            try:
                var_indices = [channels.index(v) for v in config['variables_to_save']]
                output_tensor = output_tensor[:, :, var_indices, :, :] # Select channels
                channels_coord = config['variables_to_save'] # Update coords
                logger.info(f"Selected {len(channels_coord)} variables for saving.")
            except ValueError as e:
                logger.error(f"Invalid variable name in 'variables_to_save': {e}. Saving all variables.")
                channels_coord = channels # Fallback
        else:
            channels_coord = channels # Save all if not specified
        # ... proceed to create DataArray with potentially subsetted tensor and channels_coord ...

            

        IGNORE_WHEN_COPYING_START

        Use code with caution.Python
        IGNORE_WHEN_COPYING_END

        This saves disk space and makes subsequent loading faster, but doesn't reduce memory usage during the simulation.

        Conclusion: True memory reduction by only processing needed variables during the loop is generally not feasible with pre-trained monolithic models like FCNv2 without model modification/retraining. Intermediate saving is the best strategy for managing output memory.

    save_output_steps Implementation:

        Takes a dictionary mapping step indices to data tensors.

        Handles assembling these potentially non-contiguous steps into a single xarray object for saving.

        Uses step as the primary coordinate, with time associated.

        Generates a filename including the current_model_step.

How to Use the New Structure:

    Replace run_inference: Update your inference_arco_73_numpy.py with the new run_inference function definition and the save_output_steps helper function.

    Modify main Loop: Ensure the call to run_inference passes the initial_time_dt.

    Configure Saving:

        To save intermediates (e.g., t and t-2):

            Make sure output_dir is correctly set and passed.

            Keep save_func=save_output_steps (the default).

            Set save_steps_config={'steps_to_save': [-2, 0]} (or your desired offsets).

            run_inference will return None. The output files appear in output_dir as the loop progresses.

        To collect full history in memory:

            Pass save_func=None when calling run_inference.

            run_inference will return the full 5D tensor (E, T_out, C, H, W).

            You will need a separate call to a save_output function (like your original one) after run_inference finishes to save this large tensor. Make sure your system has enough RAM.

            Use config['output_frequency'] to control how many steps are stored in the full history tensor (e.g., output_frequency=4 saves every 4th step).

This revised structure provides a robust way to run the inference using the intended time_loop pattern and offers flexibility in how you handle the output for memory efficiency.


"""





# Assuming earth2mip imports are handled in the main script
# from earth2mip import time_loop

# --- Function to Save Specific Time Steps ---



def save_output_steps(
    data_dict: Dict[int, torch.Tensor], # Dict mapping step_index to tensor (E, C, H, W)
    time_dict: Dict[int, datetime.datetime], # Dict mapping step_index to datetime
    channels: List[str],
    lat: np.ndarray,
    lon: np.ndarray,
    config: dict,
    output_dir: str,
    current_model_step: int, # The latest step index included in data_dict
    logger: logging.Logger
):
    """
    Saves specified forecast steps (e.g., current and t-2) to a NetCDF file.

    Args:
        data_dict: Dictionary where keys are step indices (e.g., 0, 8, 10)
                   and values are corresponding denormalized data tensors
                   (on CPU, shape: E, C, H, W).
        time_dict: Dictionary mapping step indices to their datetime objects.
        channels: List of channel names.
        lat: Latitude coordinates (1D numpy array).
        lon: Longitude coordinates (1D numpy array).
        config: The main inference configuration dictionary.
        output_dir: Directory to save the NetCDF file.
        current_model_step: The primary model step this save file corresponds to
                             (used for naming).
        logger: Logger object.
    """
    if not data_dict or not time_dict:
        logger.warning(f"Save request for step {current_model_step} received no data/time to save.")
        return

    n_ensemble = config["ensemble_members"]
    n_channels = len(channels)
    model_name = config.get("weather_model", "unknown_model")
    initial_time = time_dict.get(0, list(time_dict.values())[0]) # Get t=0 time if available

    logger.info(f"Preparing to save specific steps {list(data_dict.keys())} for forecast step {current_model_step}.")

    # --- Data Validation ---
    expected_n_channels = len(channels)
    first_tensor = list(data_dict.values())[0]
    if first_tensor.shape[1] != expected_n_channels:
         logger.error(f"Channel mismatch in save_output_steps! Expected {expected_n_channels}, got {first_tensor.shape[1]}.")
         # Decide how to handle: return, raise error, or use generic index
         return # Safer to not save incorrect data

    # --- Create Coordinates ---
    sorted_steps = sorted(data_dict.keys())
    time_coords = [time_dict[step] for step in sorted_steps]
    step_coords = np.array(sorted_steps) # Coordinate for the 'time' dimension

    # --- Assemble Data ---
    # Stack the tensors along a new 'time' dimension
    try:
        # Ensure all tensors have the same shape (except potentially batch if E=1)
        ref_shape = first_tensor.shape
        for step, tensor in data_dict.items():
             if tensor.shape[1:] != ref_shape[1:]: # Check C, H, W dims
                  logger.error(f"Tensor shape mismatch in data_dict for step {step}. Expected {ref_shape}, got {tensor.shape}")
                  return
        # Stack along dim=1 (new time dimension)
        output_tensor = torch.stack([data_dict[step] for step in sorted_steps], dim=1)
        # Shape is now (E, T_out, C, H, W) where T_out = len(sorted_steps)
        logger.debug(f"Stacked tensor for saving: {output_tensor.shape}")
    except Exception as e:
         logger.error(f"Failed to stack tensors for saving step {current_model_step}: {e}", exc_info=True)
         return

    n_time_out = output_tensor.shape[1]



    # Convert lat/lon to numpy arrays if they're lists
    # This is a safety measure to ensure lat/lon are numpy arrays
    if isinstance(lat, list):
        lat = np.array(lat)
    if isinstance(lon, list):
        lon = np.array(lon)
    


    # --- Create xarray Dataset ---
    try:
        # Ensure lat/lon are numpy arrays
        lat_np = lat if isinstance(lat, np.ndarray) else lat.cpu().numpy()
        lon_np = lon if isinstance(lon, np.ndarray) else lon.cpu().numpy()

        # Convert time_coords to datetime and log it to logger
        time_coords = [np.datetime64(t) for t in time_coords]
        logger.debug(f"Time coordinates for saving step {current_model_step}: {time_coords}")



        forecast_da = xr.DataArray(
            output_tensor.numpy(), # Convert final tensor
            coords={
                'ensemble': np.arange(n_ensemble),
                'step': step_coords, # Use 'step' as the time coordinate name
                'channel': channels,
                'lat': lat_np,
                'lon': lon_np,
                'time': ('step', time_coords) # Add actual datetime as a coordinate associated with 'step'
            },
            dims=['ensemble', 'step', 'channel', 'lat', 'lon'],
            name='forecast',
            attrs={
                'description': f"{model_name} ensemble forecast output for specific steps",
                'forecast_step': current_model_step, # Indicate which step this file primarily represents
                'saved_steps': str(sorted_steps), # List the steps included
                # Add other relevant metadata from config
                'model': model_name,
                'simulation_length_steps': config.get("simulation_length", "N/A"), # Total intended length
                'ensemble_members': n_ensemble,
                'initial_condition_time': initial_time.isoformat() if initial_time else "N/A",
                'noise_amplitude': config.get("noise_amplitude", 0.0),
                'perturbation_strategy': config.get("perturbation_strategy", "N/A"),
                'creation_date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                # Add versions if needed
            }
        )
        
        
        
        
        
        logger.debug("Created xarray DataArray for specific steps.")

        # Convert to Dataset
        forecast_ds = forecast_da.to_dataset(dim='channel')
        logger.info(f"Converted DataArray to Dataset for saving step {current_model_step}.")

    except Exception as e:
        logger.error(f"Failed to create xarray Dataset for step {current_model_step}: {e}", exc_info=True)
        return



    # --- Define Filename and Save ---
    ic_time_str = initial_time.strftime('%d_%B_%H_%M') if initial_time else "unknownIC"
    start_frame = sorted_steps[0]  # First saved frame (e.g., 0)
    end_frame = sorted_steps[-1]   # Last saved frame (e.g., 2)

    output_filename = os.path.join(
        output_dir,
        f"{model_name}_ensemble{n_ensemble}_IC{ic_time_str}_"
        f"startFrame{start_frame:04d}_endFrame{end_frame:04d}_"
        f"currentStep{current_model_step:04d}.nc"
    )




    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving forecast steps {list(data_dict.keys())} to: {output_filename}")

    try:
        encoding = {var: {'zlib': True, 'complevel': 5, '_FillValue': -9999.0} for var in forecast_ds.data_vars}
        start_save = time.time()
        forecast_ds.to_netcdf(output_filename, encoding=encoding, engine='netcdf4')
        end_save = time.time()
        file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
        logger.info(f"Save complete for step {current_model_step}. Time: {end_save - start_save:.2f}s. Size: {file_size_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to save NetCDF file for step {current_model_step}: {e}", exc_info=True)
        # Attempt removal
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                logger.warning(f"Removed potentially corrupted file: {output_filename}")
            except OSError as oe:
                logger.error(f"Failed to remove corrupted file {output_filename}: {oe}")

# --- Main Inference Function using time_loop ---




def save_output(output_tensor, initial_time, time_step, channels, lat, lon, config, output_dir, logger):
    """Saves the forecast output tensor to a NetCDF file."""

    if output_tensor is None:
        logger.error("Cannot save output, tensor is None.")
        return

    try:
        # output_tensor shape: (E, T_out, C, H, W)
        n_ensemble, n_time_out, n_channels, n_lat, n_lon = output_tensor.shape
        output_freq = config.get("output_frequency", 1)

        logger.info("Preparing output for saving...")
        logger.debug(f"Output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
        logger.debug(f"Number of channels: {n_channels}, Expected channels from model: {len(channels)}")
        logger.debug(f"Grid Lat shape: {lat.shape}, Lon shape: {lon.shape}")

        if n_channels != len(channels):
            logger.error(f"Mismatch between channels in output tensor ({n_channels}) and provided channel names ({len(channels)}). Saving with generic channel indices.")
            channels_coord = np.arange(n_channels)  # Use generic index if names mismatch
            channel_dim_name = "channel_idx"  # Use a different name to indicate mismatch
        else:
            channels_coord = channels
            channel_dim_name = "channel"

        # Create time coordinates
        time_coords = []
        current_time = initial_time
        # Add initial time (t=0)
        time_coords.append(current_time)
        # Generate forecast times
        for i in range(1, n_time_out):
            # Calculate time delta: i * output_freq * base_time_step
            try:
                # Ensure time_step is timedelta
                if not isinstance(time_step, datetime.timedelta):
                    logger.warning(f"Model time_step is not a timedelta ({type(time_step)}), assuming hours.")
                    actual_time_step = datetime.timedelta(hours=time_step)
                else:
                    actual_time_step = time_step

                # Time for the i-th saved output step (corresponds to model step i * output_freq)
                forecast_step_number = i * output_freq
                current_time = initial_time + forecast_step_number * actual_time_step
                time_coords.append(current_time)

            except TypeError as te:
                logger.error(f"TypeError calculating time coordinates at step {i}: {te}. Check time_step type.")
                # Fallback to index if calculation fails
                time_coords = np.arange(n_time_out)
                break  # Stop trying to calculate time coords
            except Exception as e:
                logger.error(f"Error calculating time coordinates at step {i}: {e}", exc_info=True)
                time_coords = np.arange(n_time_out)
                break

        logger.debug(f"Generated {len(time_coords)} time coordinates. First: {time_coords[0].isoformat() if time_coords else 'N/A'}, Last: {time_coords[-1].isoformat() if len(time_coords)>0 else 'N/A'}")
        if len(time_coords) != n_time_out:
            logger.warning(f"Generated {len(time_coords)} time coordinates, but expected {n_time_out}. Using indices instead.")
            time_coords = np.arange(n_time_out)

        # Ensure lat/lon are numpy arrays on CPU
        lat_np = lat.cpu().numpy() if isinstance(lat, torch.Tensor) else np.asarray(lat)
        lon_np = lon.cpu().numpy() if isinstance(lon, torch.Tensor) else np.asarray(lon)

        # Create DataArray
        logger.debug("Creating xarray DataArray...")
        # Check for NaNs before creating DataArray
        if np.isnan(output_tensor.numpy()).any():
            logger.warning("NaNs present in the output tensor before saving to NetCDF!")

        forecast_da = xr.DataArray(
            output_tensor.numpy(),  # Convert tensor to numpy array
            coords={
                "ensemble": np.arange(n_ensemble),
                "time": time_coords,
                channel_dim_name: channels_coord,  # Use dynamic channel dimension name
                "lat": lat_np,
                "lon": lon_np,
            },
            dims=["ensemble", "time", channel_dim_name, "lat", "lon"],
            name="forecast",
            attrs={
                "description": f"{config['weather_model']} ensemble forecast output",
                "model": config["weather_model"],
                "simulation_length_steps": config["simulation_length"],
                "output_frequency_steps": output_freq,
                "ensemble_members": n_ensemble,
                "initial_condition_time": initial_time.isoformat(),
                "time_step_seconds": actual_time_step.total_seconds() if isinstance(actual_time_step, datetime.timedelta) else "unknown",
                "noise_amplitude": config["noise_amplitude"],
                "perturbation_strategy": config["perturbation_strategy"],
                "creation_date": datetime.datetime.now(pytz.utc).isoformat(),
                "pytorch_version": torch.__version__,
                "numpy_version": np.__version__,
                "xarray_version": xr.__version__,
            },
        )
        logger.info("Created xarray DataArray.")

        # Convert to Dataset with each channel as a variable for better compatibility
        # Handle potential channel name issues (e.g., invalid characters for variable names)
        logger.debug(f"Converting DataArray to Dataset using dimension '{channel_dim_name}'...")
        try:
            forecast_ds = forecast_da.to_dataset(dim=channel_dim_name)
            logger.info("Converted DataArray to Dataset (channels as variables).")
        except Exception as e:
            logger.error(f"Failed to convert DataArray to Dataset (dim='{channel_dim_name}'): {e}. Saving as DataArray instead.", exc_info=True)
            # Fallback: save the DataArray directly if conversion fails
            forecast_ds = forecast_da

        # Define output filename
        ic_time_str = initial_time.strftime("%Y%m%d_%H%M%S")
        # Add ensemble size and sim length to filename for clarity
        output_filename = os.path.join(output_dir, f"{config['weather_model']}_ensemble{n_ensemble}_sim{config['simulation_length']}_{ic_time_str}.nc")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving forecast output to: {output_filename}")

        # Define encoding for compression (optional but recommended)
        # Apply encoding only if saving as Dataset
        encoding = {}
        if isinstance(forecast_ds, xr.Dataset):
            encoding = {var: {"zlib": True, "complevel": 5, "_FillValue": -9999.0} for var in forecast_ds.data_vars}  # Add FillValue
            logger.debug(f"Applying encoding to variables: {list(forecast_ds.data_vars.keys())}")
        elif isinstance(forecast_ds, xr.DataArray):
            encoding = {forecast_ds.name: {"zlib": True, "complevel": 5, "_FillValue": -9999.0}}
            logger.debug(f"Applying encoding to DataArray: {forecast_ds.name}")

        start_save = time.time()
        forecast_ds.to_netcdf(output_filename, encoding=encoding, engine="netcdf4")  # Specify engine
        end_save = time.time()
        # Check file size
        file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
        logger.info(f"Save complete. Time taken: {end_save - start_save:.2f} seconds. File size: {file_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Failed during the save_output process: {e}", exc_info=True)
        # Attempt to remove potentially corrupted file
        if "output_filename" in locals() and os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                logger.warning(f"Removed potentially corrupted file: {output_filename}")
            except OSError as oe:
                logger.error(f"Failed to remove corrupted file {output_filename}: {oe}")















# def run_inference(
#     model_inference: time_loop.time_loop, # Use time_loop type hint
#     initial_state_tensor: torch.Tensor, # Shape (1, C, H, W) - Single IC
#     initial_time_dt: datetime.datetime, # Starting datetime object for the IC
#     config: dict,                       # Configuration dictionary
#     logger: logging.Logger,             # Logger instance
#     save_func: Optional[Callable] = save_output_steps, # Function to call for saving steps
#     save_steps_config: Dict[str, Any] = {'steps_to_save': [-2, 0]}, # Config for saving steps: offset relative to current step
#     output_dir: Optional[str] = None, # Required if save_func is provided
# ):
#     """
#     Runs the autoregressive ensemble forecast using the time_loop interface provided
#     by the earth2mip.networks.Inference object.

#     Handles ensemble creation, normalization, perturbation, time stepping via
#     the iterator, and memory-efficient saving of specified output steps.

#     Args:
#         model_inference: An instance of earth2mip.networks.Inference (or compatible time_loop).
#         initial_state_tensor: The initial condition for ONE time step.
#                                Shape: (1, C, H, W), on CPU or GPU.
#         initial_time_dt: The datetime object corresponding to the initial_state_tensor.
#                          Must be timezone-aware (e.g., UTC).
#         config: Dictionary containing inference parameters:
#                 'ensemble_members' (int): Number of ensemble members (E).
#                 'simulation_length' (int): Number of forecast steps *after* t=0.
#                 'output_frequency' (int, Optional): How often to log/potentially save. Defaults to 1.
#                 'noise_amplitude' (float, Optional): Amplitude for Gaussian noise perturbation. Defaults to 0.0.
#                 'perturbation_strategy' (str, Optional): Currently only 'gaussian' placeholder supported.
#                 'weather_model' (str, Optional): Name of the model for metadata.
#         logger: A configured Python logger instance.
#         save_func: A callable function to save output during the loop.
#                    Expected signature: save_func(data_dict, time_dict, channels, lat, lon, config, output_dir, current_model_step, logger)
#                    If None, no intermediate saving is performed.
#         save_steps_config: Dictionary configuring which steps to save relative to the current step.
#                            Example: {'steps_to_save': [0]} saves only the current step.
#                            Example: {'steps_to_save': [-2, 0]} saves current step and step t-2.
#                            Offsets are relative to the *current* model step being processed.
#         output_dir: The directory where save_func should save files. Required if save_func is not None.

#     Returns:
#         Optional[torch.Tensor]: If save_func is None, returns the full forecast history tensor
#                                 (denormalized, on CPU) with shape (E, T_out, C, H, W).
#                                 T_out depends on simulation_length and output_frequency.
#                                 Returns None if save_func is provided OR if an error occurs.

#     Raises:
#         ValueError: If input tensor shape is invalid or configuration is missing.
#         AttributeError: If model_inference object lacks expected methods/attributes.
#         TypeError: If datetime object is naive or issues with callable checks.
#     """

#     # --- Configuration Extraction ---
#     n_ensemble = config.get("ensemble_members", 1)
#     simulation_length = config.get("simulation_length", 0) # Num steps AFTER t=0
#     output_freq = config.get("output_frequency", 1) # Frequency for collecting output in memory if not saving intermediates
#     noise_amp = config.get("noise_amplitude", 0.0)
#     pert_strategy = config.get("perturbation_strategy", "gaussian") # Placeholder

#     logger.info(f"Starting inference run for IC: {initial_time_dt.isoformat()}")
#     logger.info(f"Ensemble members: {n_ensemble}, Simulation steps: {simulation_length}")
#     logger.info(f"Output collection/save frequency: {output_freq}")
#     logger.info(f"Perturbation: Amp={noise_amp}, Strategy='{pert_strategy}' (placeholder)")

#     # --- Validation ---
#     if not isinstance(initial_time_dt, datetime.datetime):
#         raise TypeError("initial_time_dt must be a datetime.datetime object.")
#     if initial_time_dt.tzinfo is None or initial_time_dt.tzinfo.utcoffset(initial_time_dt) is None:
#         logger.warning(f"Initial time {initial_time_dt.isoformat()} is timezone naive. Assuming UTC for consistency.")
#         initial_time_dt = initial_time_dt.replace(tzinfo=datetime.timezone.utc)

#     if initial_state_tensor.dim() != 4 or initial_state_tensor.shape[0] != 1:
#         logger.error(f"Initial state tensor has unexpected shape: {initial_state_tensor.shape}. Expected (1, C, H, W).")
#         raise ValueError("Invalid initial state tensor shape for run_inference input")

#     if simulation_length <= 0:
#         logger.warning("Simulation length is 0 or negative. Only initial state will be processed.")
#         # Allow proceeding to just get t=0 output if needed

#     perform_intermediate_saving = callable(save_func)
#     if perform_intermediate_saving and not output_dir:
#         logger.error("output_dir must be provided when save_func is specified.")
#         raise ValueError("output_dir is required for intermediate saving.")

#     # --- Get Model Properties ---
#     try:
#         device = model_inference.device
#         n_history = getattr(model_inference, 'n_history', 0)
#         time_step_delta = model_inference.time_step
#         all_channels = model_inference.in_channel_names # Assuming in==out
#         lat = model_inference.grid.lat
#         lon = model_inference.grid.lon
#         logger.info(f"Model properties: Device={device}, n_history={n_history}, time_step={time_step_delta}")
#     except AttributeError as e:
#         logger.error(f"Failed to get required attributes from model_inference object: {e}")
#         raise AttributeError(f"model_inference object missing required attributes (device, n_history, time_step, etc.): {e}")

#     # --- 1. Prepare Initial State for time_loop ---
#     # Shape (1, C, H, W) -> (E, C, H, W) -> (E, T, C, H, W) where T = n_history + 1

#     logger.debug("Preparing initial state...")
#     # a. Create ensemble batch
#     batch_tensor_4d = initial_state_tensor.repeat(n_ensemble, 1, 1, 1).to(device)
#     logger.debug(f"  Created ensemble batch (4D, device={device}): {batch_tensor_4d.shape}")

#     # b. Add time dimension T = n_history + 1
#     initial_state_5d = batch_tensor_4d.unsqueeze(1)
#     if n_history > 0:
#          # If history is needed, the input initial_state_tensor should ideally represent
#          # the *latest* time step, and previous steps need loading or simulation.
#          # This example assumes n_history=0, so we just repeat the initial state
#          # if T > 1 was required. For n_history=0, T=1 is correct.
#          if initial_state_5d.shape[1] != n_history + 1:
#               logger.warning(f"Shape mismatch for history. Expected T={n_history+1}, got {initial_state_5d.shape[1]}. Assuming n_history=0.")
#               # This might indicate an issue if the model truly expects history.
#               # For simplicity now, we assume n_history=0 if input is 4D.
#               if initial_state_5d.shape[1] == 1 and n_history > 0:
#                    logger.warning(f"Repeating initial state to match n_history={n_history}. This might be incorrect if true history is needed.")
#                    initial_state_5d = initial_state_5d.repeat(1, n_history + 1, 1, 1, 1)

#     logger.info(f"  Prepared initial state for time_loop (5D): {initial_state_5d.shape}")


#     # --- 2. Normalize Initial State ---
#     logger.debug("Normalizing initial 5D state...")
#     try:
#         # Use the normalize method added to the Inference class
#         if not hasattr(model_inference, 'normalize') or not callable(model_inference.normalize):
#              logger.error("model_inference object does not have a callable 'normalize' method!")
#              raise AttributeError("model_inference missing required 'normalize' method.")
#         initial_state_norm_5d = model_inference.normalize(initial_state_5d)
#         logger.info("  Normalized initial 5D state.")
#         logger.debug(f"  Normalized initial state shape: {initial_state_norm_5d.shape}, dtype: {initial_state_norm_5d.dtype}")
#         if torch.isnan(initial_state_norm_5d).any():
#             logger.warning("NaNs detected in initial normalized state!")
#     except Exception as e:
#         logger.error(f"Error during initial state normalization: {e}", exc_info=True)
#         raise # Propagate error


#     # --- 3. Apply Perturbation ---
#     initial_state_perturbed_norm_5d = initial_state_norm_5d.clone()
#     if noise_amp > 0 and n_ensemble > 1:
#         logger.info(f"Applying perturbation noise to initial normalized state (Amp={noise_amp:.4f}, Strategy='{pert_strategy}')")
#         if pert_strategy != "gaussian": # Only Gaussian implemented here
#              logger.warning(f"Perturbation strategy '{pert_strategy}' requested, but using simple Gaussian noise placeholder.")

#         noise = torch.randn_like(initial_state_perturbed_norm_5d) * noise_amp
#         logger.debug(f"  Generated noise tensor: shape={noise.shape}, std={torch.std(noise):.4f}")

#         if n_ensemble > 1:
#            noise[0, ...] = 0 # Ensure member 0 is deterministic (index entire first dimension)
#            logger.debug("  Set noise for ensemble member 0 to zero.")

#         initial_state_perturbed_norm_5d += noise
#         logger.info("  Applied Gaussian noise to initial state.")
#         if torch.isnan(initial_state_perturbed_norm_5d).any():
#             logger.warning("NaNs detected after adding noise to initial state!")
#     else:
#         logger.info("No perturbation noise applied to initial state.")


#     # --- 4. Execute time_loop Iterator ---
#     # Prepare storage for results
#     output_history_buffer = collections.deque() # Holds recent (step_index, time, data_denorm_cpu) tuples
#     output_tensors_full_history = [] # Only used if perform_intermediate_saving is False
#     steps_relative_to_save = sorted(save_steps_config.get('steps_to_save', [0])) # e.g., [-2, 0]
#     max_offset = abs(min(steps_relative_to_save)) if steps_relative_to_save else 0 # Max lookback needed, e.g., 2 for [-2, 0]
#     buffer_size = max_offset + 1 # Need to store current + lookback steps

#     inference_times = []
#     logger.info(f"Initializing time_loop iterator starting from {initial_time_dt.isoformat()}")
#     logger.info(f"Target simulation steps: {simulation_length}. Iterator will run {simulation_length + 1} times.")
#     if perform_intermediate_saving:
#          logger.info(f"Intermediate saving enabled. Steps relative to current to save: {steps_relative_to_save}. Buffer size: {buffer_size}")
#          logger.info(f"Output files will be saved to: {output_dir}")
#     else:
#          logger.info(f"Intermediate saving disabled. Collecting full history in memory (output_freq={output_freq}).")

#     overall_start_time = time.time()
#     model_step_counter = 0 # Tracks the forecast step number (0, 1, 2...)

#     try:
#         # Initialize the iterator. Pass the perturbed NORMALIZED state.
#         iterator = model_inference(time=initial_time_dt, x=initial_state_perturbed_norm_5d)

#         # Iterate simulation_length + 1 times (for t=0 up to t=simulation_length*time_step)
#         for i in range(simulation_length + 1):
#             iter_start_time = time.time()
#             logger.debug(f"--- Iterator Step {i} (Model Step {model_step_counter}) ---")

#             # Get next state from the iterator
#             try:
#                 time_out, data_denorm, _ = next(iterator) # We don't need restart_state here
#                 logger.debug(f"  Iterator yielded: Time={time_out.isoformat()}, Output shape={data_denorm.shape}")
#             except StopIteration:
#                 logger.warning(f"Iterator stopped unexpectedly after {i} iterations (model step {model_step_counter}). Expected {simulation_length + 1} iterations.")
#                 break # Exit loop if iterator finishes early

#             # --- Output Handling ---
#             data_denorm_cpu = data_denorm.cpu() # Move to CPU immediately to free GPU memory

#             if perform_intermediate_saving:
#                 # Add current step data to the buffer
#                 output_history_buffer.append((model_step_counter, time_out, data_denorm_cpu))
#                 # Maintain buffer size
#                 while len(output_history_buffer) > buffer_size:
#                     output_history_buffer.popleft() # Remove oldest entry

#                 # Check if we can save based on the *current* step and required offsets
#                 can_save_all_steps = True
#                 steps_to_save_indices = {} # step_index -> tensor
#                 times_to_save = {}       # step_index -> datetime
#                 for offset in steps_relative_to_save:
#                     target_step_index = model_step_counter + offset
#                     found = False
#                     # Search buffer for the target step index
#                     for step_idx, step_time, step_data in output_history_buffer:
#                         if step_idx == target_step_index:
#                             steps_to_save_indices[target_step_index] = step_data
#                             times_to_save[target_step_index] = step_time
#                             found = True
#                             break
#                     if not found:
#                         # We don't have the required history yet (e.g., asking for t-2 at step 0 or 1)
#                         can_save_all_steps = False
#                         logger.debug(f"  Cannot save yet: required step {target_step_index} (offset {offset}) not in buffer.")
#                         break # No need to check other offsets for this iteration

#                 # If all required steps are found in the buffer, trigger the save
#                 if can_save_all_steps:
#                     logger.debug(f"  All required steps {list(steps_to_save_indices.keys())} found in buffer. Triggering save...")
#                     try:
#                         save_func(
#                             data_dict=steps_to_save_indices,
#                             time_dict=times_to_save,
#                             channels=all_channels,
#                             lat=lat,
#                             lon=lon,
#                             config=config,
#                             output_dir=output_dir,
#                             current_model_step=model_step_counter, # Pass the current step index
#                             logger=logger
#                         )
#                     except Exception as save_e:
#                          logger.error(f"Error occurred during intermediate save call for step {model_step_counter}: {save_e}", exc_info=True)
#                          # Decide whether to continue or stop the loop on save error

#             else:
#                 # Collect full history in memory based on output_freq
#                 if model_step_counter % output_freq == 0:
#                     logger.debug(f"  Collecting output for model step {model_step_counter} in memory.")
#                     output_tensors_full_history.append(data_denorm_cpu)

#             # --- Timing and Increment ---
#             iter_end_time = time.time()
#             step_duration = iter_end_time - iter_start_time
#             inference_times.append(step_duration)
#             logger.debug(f"  Iterator Step {i} finished in {step_duration:.3f} seconds.")

#             # Increment model step counter for the next iteration
#             model_step_counter += 1

#         logger.info(f"Finished {i+1} iterations over time_loop.")

#     except Exception as e:
#         logger.error(f"Error occurred during time_loop iteration: {e}", exc_info=True)
#         return None # Indicate failure

#     finally:
#          # Clean up iterator if needed (though exiting the loop should be sufficient)
#          # Explicitly delete large tensors? Maybe not necessary if loop finishes.
#          # logger.debug("Clearing intermediate tensors...")
#          # del initial_state_5d, initial_state_norm_5d, initial_state_perturbed_norm_5d
#          # del output_history_buffer # If saving intermediates
#          # if 'iterator' in locals(): del iterator # Might help GC
#          if torch.cuda.is_available(): torch.cuda.empty_cache()


#     overall_end_time = time.time()
#     total_duration = overall_end_time - overall_start_time
#     avg_inference_time = np.mean(inference_times) if inference_times else 0
#     logger.info(f"time_loop execution finished. Total time: {total_duration:.2f}s. Average step time: {avg_inference_time:.3f}s.")

#     # --- 5. Combine and Return Full History (if not saving intermediates) ---
#     if not perform_intermediate_saving:
#         if not output_tensors_full_history:
#             logger.warning("No output tensors were collected for full history!")
#             return None

#         logger.info(f"Stacking {len(output_tensors_full_history)} collected output tensors for final result...")
#         try:
#             # Each tensor is (E, C, H, W)
#             final_output_tensor = torch.stack(output_tensors_full_history, dim=1) # Shape: (E, T_out, C, H, W)
#             logger.info(f"Final aggregated output tensor shape: {final_output_tensor.shape}")
#             if torch.isnan(final_output_tensor).any():
#                 logger.warning("NaNs detected in the final aggregated output tensor!")
#             return final_output_tensor # Return the full history tensor on CPU
#         except Exception as e:
#             logger.error(f"Failed to stack collected output tensors: {e}", exc_info=True)
#             return None
#     else:
#         logger.info("Intermediate saving was performed. Returning None as full history was not stored.")
#         return None # Indicate success but no tensor returned















def get_gpu_memory_usage(device: torch.device, logger: logging.Logger) -> str:
    """Helper to get current GPU memory usage string."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        return f"Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB"
    return "Device is CPU"




"""
Key Optimizations and Robustness Improvements:

    CPU Preprocessing:

        Ensures initial_state_tensor is on the CPU.

        repeat (for ensemble) and unsqueeze/repeat (for history) are done on CPU tensors.

        Normalization (center, scale lookup and arithmetic) is performed manually using CPU tensors, completely avoiding the large pre-inference allocation on the GPU that caused the OOM.

        Perturbation (noise generation and addition) is also done on the CPU.

    Just-in-Time GPU Transfer: The final prepared tensor (initial_state_perturbed_norm_5d_cpu) is moved to the GPU (initial_state_perturbed_norm_gpu) only when needed by the model_inference iterator initialization.

    Aggressive Memory Management:

        del is used frequently to explicitly remove references to large intermediate tensors (like the CPU versions after GPU transfer, normalization stats, noise tensor, and the GPU output tensor after moving it to CPU).

        torch.cuda.empty_cache() is called after moving the model output to the CPU within the loop to potentially release fragmented memory sooner (note: this can add overhead, use judiciously or control with log_gpu_mem or another flag). It's also called after iterator initialization and in the final finally block.

        The input initial_state_perturbed_norm_gpu is deleted after the iterator is initialized, as the iterator now holds the necessary reference internally.

    Detailed Logging:

        Logs tensor shapes and devices (CPU/GPU) at each major step.

        Includes optional GPU memory logging (log_gpu_mem=True) via get_gpu_memory_usage helper at critical points (before/after transfers, yields, cleanup) to help diagnose memory issues.

        Logs cleanup actions.

        Uses logger.exception or exc_info=True for errors.

    Error Handling:

        More specific try...except blocks around CPU preprocessing steps (state prep, normalization, perturbation) and GPU transfer.

        Catches torch.cuda.OutOfMemoryError specifically during the loop.

        Checks for NaNs after normalization, perturbation, and in the model output, raising errors if found.

        Ensures cleanup happens in finally blocks even if errors occur mid-loop.

    Validation:

        Checks if center and scale attributes exist on model_inference.

        Handles timezone-naive datetime objects.

        Creates output_dir if it doesn't exist when saving.

This revised structure significantly reduces the peak GPU memory requirement before the inference loop starts, addressing the specific OOM error encountered, while adding layers of logging and error handling to make the process more robust.

"""


def run_inference(
    model_inference: time_loop, # Use time_loop type hint
    initial_state_tensor: torch.Tensor, # Shape (1, C, H, W) - Single IC, EXPECTED ON CPU INITIALLY FOR OPTIMIZATION
    initial_time_dt: datetime.datetime, # Starting datetime object for the IC
    config: dict,                       # Configuration dictionary
    logger: logging.Logger,             # Logger instance
    save_func: Optional[Callable] = save_output_steps, # Function to call for saving steps
    save_steps_config: Dict[str, Any] = {'steps_to_save': [-2, 0]}, # Config for saving steps: offset relative to current step
    output_dir: Optional[str] = None, # Required if save_func is provided
    log_gpu_mem: bool = False, # Option to log GPU memory usage frequently
):
    """
    Optimized and robust function to run autoregressive ensemble forecasts using time_loop.

    Performs initial state preparation, normalization, and perturbation on CPU
    to avoid GPU OOM errors with large ensemble/history states. Transfers data
    to GPU just before the inference loop. Includes aggressive memory management
    and detailed logging.

    Args:
        model_inference: An instance of earth2mip.networks.Inference (or compatible time_loop).
        initial_state_tensor: Initial condition tensor (1, C, H, W). Expected on CPU.
        initial_time_dt: Timezone-aware datetime for the initial state.
        config: Dictionary with keys 'ensemble_members', 'simulation_length', etc.
        logger: Configured logger instance.
        save_func: Callable for saving intermediate steps (see original docstring).
        save_steps_config: Configuration for save_func (see original docstring).
        output_dir: Directory for saving outputs. Required if save_func is used.
        log_gpu_mem: If True, logs GPU memory usage at key points (can add overhead).

    Returns:
        Optional[torch.Tensor]: Full forecast history (E, T_out, C, H, W) on CPU if
                                save_func is None. None otherwise or on error.

    Raises:
        ValueError: Invalid input shape, configuration, or missing output_dir.
        AttributeError: Missing required methods/attributes on model_inference.
        TypeError: Invalid types for inputs.
        RuntimeError: Errors during critical operations (e.g., GPU transfer, iteration).
    """
    overall_start_time = time.time()
    logger.info("="*50)
    logger.info(f"Starting run_inference for IC: {initial_time_dt.isoformat()}")
    logger.info("="*50)

    # --- Configuration Extraction ---
    n_ensemble = config.get("ensemble_members", 1)
    simulation_length = config.get("simulation_length", 0)
    output_freq = config.get("output_frequency", 1)
    noise_amp = config.get("noise_amplitude", 0.0)
    pert_strategy = config.get("perturbation_strategy", "gaussian")

    logger.info(f"Config - Ensemble: {n_ensemble}, Sim Length: {simulation_length}, Output Freq: {output_freq}")
    logger.info(f"Config - Perturbation: Amp={noise_amp:.4e}, Strategy='{pert_strategy}'")

    # --- Validation ---
    if not isinstance(initial_time_dt, datetime.datetime):
        raise TypeError("initial_time_dt must be a datetime.datetime object.")
    if initial_time_dt.tzinfo is None or initial_time_dt.tzinfo.utcoffset(initial_time_dt) is None:
        logger.warning(f"Initial time {initial_time_dt.isoformat()} is timezone naive. Assuming UTC.")
        initial_time_dt = initial_time_dt.replace(tzinfo=datetime.timezone.utc)

    if initial_state_tensor.dim() != 4 or initial_state_tensor.shape[0] != 1:
        logger.error(f"Input IC tensor shape invalid: {initial_state_tensor.shape}. Expected (1, C, H, W).")
        raise ValueError("Invalid initial state tensor shape.")










    # Ensure IC tensor is on CPU for initial prep steps
    if initial_state_tensor.is_cuda:
        logger.warning("Initial state tensor was on GPU. Moving to CPU for preprocessing.")
        initial_state_tensor = initial_state_tensor.cpu()

    if simulation_length <= 0:
        logger.warning("Simulation length <= 0. Only initial state (t=0) will be processed.")

    perform_intermediate_saving = callable(save_func)
    if perform_intermediate_saving:
        if not output_dir:
            logger.error("output_dir must be provided when save_func is specified.")
            raise ValueError("output_dir is required for intermediate saving.")
        if not os.path.exists(output_dir):
             try:
                  os.makedirs(output_dir)
                  logger.info(f"Created output directory: {output_dir}")
             except OSError as e:
                  logger.error(f"Failed to create output directory {output_dir}: {e}")
                  raise



    # --- Get Model Properties ---
    try:
        device = model_inference.device
        n_history = getattr(model_inference, 'n_history', 0)
        time_step_delta = model_inference.time_step
        all_channels = model_inference.in_channel_names # Check if out_channel_names exists if different
        lat = model_inference.grid.lat
        lon = model_inference.grid.lon
        center = getattr(model_inference, 'center', None) # For manual normalization
        scale = getattr(model_inference, 'scale', None)   # For manual normalization
        if center is None or scale is None:
             raise AttributeError("model_inference object must have 'center' and 'scale' attributes for normalization.")

        logger.info(f"Model Properties - Device: {device}, n_history: {n_history}, Time Step: {time_step_delta}")
        logger.debug(f"Model Channels: {len(all_channels)} channels") # Log channel names if not too long: {all_channels}")
        logger.debug(f"Model Grid: Lat {lat.shape}, Lon {lon.shape}")
    except AttributeError as e:
        logger.error(f"Failed to get required attributes from model_inference object: {e}", exc_info=True)
        raise AttributeError(f"model_inference object missing required attributes (device, n_history, time_step, center, scale, etc.): {e}")

    # Move center and scale to CPU for CPU-based normalization
    try:
        center_cpu = center.cpu()
        scale_cpu = scale.cpu()
        logger.info("Moved model normalization 'center' and 'scale' tensors to CPU.")
    except Exception as e:
        logger.error(f"Failed to move center/scale to CPU: {e}", exc_info=True)
        raise RuntimeError("Could not prepare normalization tensors on CPU.") from e



    # --- 1. Prepare Initial State on CPU ---
    initial_state_5d_cpu = None # Define variable outside try block
    try:
        logger.info("Preparing initial state on CPU...")
        # a. Create ensemble batch on CPU
        batch_tensor_4d_cpu = initial_state_tensor.repeat(n_ensemble, 1, 1, 1)
        logger.debug(f"  Created ensemble batch (4D, CPU): {batch_tensor_4d_cpu.shape}, dtype: {batch_tensor_4d_cpu.dtype}")

        # b. Add time dimension T = n_history + 1 on CPU
        initial_state_5d_cpu = batch_tensor_4d_cpu.unsqueeze(1)
        del batch_tensor_4d_cpu # Free memory
        if n_history > 0:
            logger.info(f"  Model requires history (n_history={n_history}). Repeating initial state on CPU.")
            # This assumes the provided IC is the *latest* time step and repeats it backward.
            # A more sophisticated approach would load actual history if available.
            initial_state_5d_cpu = initial_state_5d_cpu.repeat(1, n_history + 1, 1, 1, 1)

        logger.info(f"  Prepared initial state for time_loop (5D, CPU): {initial_state_5d_cpu.shape}")

    except Exception as e:
        logger.error(f"Error during initial state preparation on CPU: {e}", exc_info=True)
        if initial_state_5d_cpu is not None: del initial_state_5d_cpu
        raise RuntimeError("Failed to prepare initial state on CPU.") from e



    # --- 2. Normalize Initial State on CPU ---
    initial_state_norm_5d_cpu = None
    try:
        logger.info("Normalizing initial state on CPU...")
        # Perform normalization manually using CPU tensors
        initial_state_norm_5d_cpu = (initial_state_5d_cpu - center_cpu.unsqueeze(1)) / scale_cpu.unsqueeze(1)
        logger.info(f"  Normalized initial state (5D, CPU): {initial_state_norm_5d_cpu.shape}, dtype: {initial_state_norm_5d_cpu.dtype}")

        # Check for NaNs after normalization
        if torch.isnan(initial_state_norm_5d_cpu).any():
            logger.error("NaNs detected in initial state AFTER normalization!")
            # Optionally raise an error or try to handle NaNs
            raise ValueError("Normalization resulted in NaN values.")

        # Clean up original 5D state and normalization tensors from CPU memory
        del initial_state_5d_cpu
        del center_cpu
        del scale_cpu
        logger.debug("  Cleaned up original 5D state and normalization tensors from CPU memory.")

    except Exception as e:
        logger.error(f"Error during initial state normalization on CPU: {e}", exc_info=True)
        if initial_state_norm_5d_cpu is not None: del initial_state_norm_5d_cpu
        if 'initial_state_5d_cpu' in locals() and initial_state_5d_cpu is not None: del initial_state_5d_cpu
        if 'center_cpu' in locals() and center_cpu is not None: del center_cpu
        if 'scale_cpu' in locals() and scale_cpu is not None: del scale_cpu
        raise RuntimeError("Failed to normalize initial state on CPU.") from e





    # --- 3. Apply Perturbation on CPU ---
    initial_state_perturbed_norm_5d_cpu = None
    try:
        initial_state_perturbed_norm_5d_cpu = initial_state_norm_5d_cpu.clone() # Clone first
        if noise_amp > 0 and n_ensemble > 1:
            logger.info(f"Applying perturbation noise on CPU (Amp={noise_amp:.4e}, Strategy='{pert_strategy}')")
            if pert_strategy != "gaussian":
                 logger.warning(f"Perturbation strategy '{pert_strategy}' not implemented, using Gaussian.")

            noise_cpu = torch.randn_like(initial_state_perturbed_norm_5d_cpu) * noise_amp
            logger.debug(f"  Generated noise tensor (CPU): shape={noise_cpu.shape}, std={torch.std(noise_cpu):.4f}")

            noise_cpu[0, ...] = 0 # Ensure member 0 is deterministic
            logger.debug("  Set noise for ensemble member 0 to zero.")

            initial_state_perturbed_norm_5d_cpu += noise_cpu
            logger.info("  Applied noise to initial state on CPU.")
            del noise_cpu # Clean up noise tensor

            if torch.isnan(initial_state_perturbed_norm_5d_cpu).any():
                logger.error("NaNs detected AFTER adding noise to initial state!")
                raise ValueError("Perturbation resulted in NaN values.")
        else:
            logger.info("No perturbation noise applied.")

        # Clean up the unperturbed normalized state from CPU memory
        del initial_state_norm_5d_cpu
        logger.debug("  Cleaned up unperturbed normalized state from CPU memory.")

    except Exception as e:
        logger.error(f"Error during perturbation on CPU: {e}", exc_info=True)
        if initial_state_perturbed_norm_5d_cpu is not None: del initial_state_perturbed_norm_5d_cpu
        if 'initial_state_norm_5d_cpu' in locals() and initial_state_norm_5d_cpu is not None: del initial_state_norm_5d_cpu
        if 'noise_cpu' in locals() and noise_cpu is not None: del noise_cpu
        raise RuntimeError("Failed to apply perturbation on CPU.") from e




    # --- 4. Transfer to GPU and Execute time_loop Iterator ---
    initial_state_perturbed_norm_gpu = None # Define outside try
    iterator = None # Define outside try
    output_history_buffer = collections.deque()
    output_tensors_full_history = []
    steps_relative_to_save = sorted(save_steps_config.get('steps_to_save', [0]))
    max_offset = abs(min(steps_relative_to_save)) if steps_relative_to_save else 0
    buffer_size = max_offset + 1

    inference_times = []
    model_step_counter = 0
    final_return_tensor = None # Initialize return tensor

    try:
        # Move the final prepared state to GPU just before the loop
        logger.info(f"Moving prepared initial state to target device: {device}...")
        transfer_start_time = time.time()
        initial_state_perturbed_norm_gpu = initial_state_perturbed_norm_5d_cpu.to(device)
        transfer_duration = time.time() - transfer_start_time
        logger.info(f"  Moved state to {device} in {transfer_duration:.3f}s. Shape: {initial_state_perturbed_norm_gpu.shape}")
        if log_gpu_mem: logger.info(f"  GPU Memory after transfer: {get_gpu_memory_usage(device, logger)}")

        # Clean up the CPU version immediately
        del initial_state_perturbed_norm_5d_cpu
        logger.debug("  Cleaned up final prepared state from CPU memory.")

        # Initialize the iterator
        logger.info(f"Initializing time_loop iterator starting from {initial_time_dt.isoformat()}")
        if log_gpu_mem: logger.info(f"  GPU Memory before iterator init: {get_gpu_memory_usage(device, logger)}")
        iterator = model_inference(time=initial_time_dt, x=initial_state_perturbed_norm_gpu)
        if log_gpu_mem: logger.info(f"  GPU Memory after iterator init: {get_gpu_memory_usage(device, logger)}")

        # No longer need the initial GPU state after iterator is initialized
        del initial_state_perturbed_norm_gpu
        logger.debug("  Cleaned up initial state from GPU memory (iterator holds reference).")
        if torch.cuda.is_available() and device.type == 'cuda':
             torch.cuda.empty_cache()
             logger.debug("  Called torch.cuda.empty_cache() after iterator init.")

        logger.info(f"Starting time_loop iteration for {simulation_length + 1} steps (0 to {simulation_length}).")
        if perform_intermediate_saving:
             logger.info(f"Intermediate saving: Steps relative={steps_relative_to_save}, Buffer={buffer_size}, Dir={output_dir}")
        else:
             logger.info(f"Full history collection: Output Freq={output_freq}")

        # --- Main Loop ---
        for i in range(simulation_length + 1): # t=0 included
            iter_start_time = time.time()
            logger.debug(f"--- Iteration {i} (Model Step {model_step_counter}) ---")
            if log_gpu_mem: logger.debug(f"  GPU Memory start of step {model_step_counter}: {get_gpu_memory_usage(device, logger)}")

            # Get next state from the iterator
            time_out, data_denorm_gpu, _ = next(iterator) # Assume yields (time, data_denorm_gpu, restart_state)
            logger.debug(f"  Iterator yielded: Time={time_out.isoformat()}, GPU Output shape={data_denorm_gpu.shape}")
            if log_gpu_mem: logger.debug(f"  GPU Memory after yield: {get_gpu_memory_usage(device, logger)}")

            # --- Output Handling (Prioritize moving off GPU) ---
            data_denorm_cpu = data_denorm_gpu.cpu()
            logger.debug(f"  Moved output to CPU: shape={data_denorm_cpu.shape}")

            # Explicitly delete the GPU tensor
            del data_denorm_gpu
            logger.debug("  Deleted denormalized tensor from GPU memory.")

            # Optional: Aggressive cache clearing (can add overhead)
            if torch.cuda.is_available() and device.type == 'cuda':
                 torch.cuda.empty_cache()
                 logger.debug("  Called torch.cuda.empty_cache() after moving output to CPU.")
                 if log_gpu_mem: logger.debug(f"  GPU Memory after cleanup: {get_gpu_memory_usage(device, logger)}")


            # Check for NaNs in output
            if torch.isnan(data_denorm_cpu).any():
                 logger.error(f"NaNs detected in output at model step {model_step_counter} (Time: {time_out.isoformat()})!")
                 # Decide how to handle: break, continue, raise?
                 raise ValueError(f"NaNs detected in model output at step {model_step_counter}")

            if perform_intermediate_saving:
                output_history_buffer.append((model_step_counter, time_out, data_denorm_cpu))
                while len(output_history_buffer) > buffer_size:
                    removed_step, _, _ = output_history_buffer.popleft()
                    logger.debug(f"  Removed step {removed_step} from buffer.")

                can_save_all_steps = True
                steps_to_save_indices: Dict[int, torch.Tensor] = {}
                times_to_save: Dict[int, datetime.datetime] = {}
                for offset in steps_relative_to_save:
                    target_step_index = model_step_counter + offset
                    found = False
                    for step_idx, step_time, step_data in output_history_buffer:
                        if step_idx == target_step_index:
                            steps_to_save_indices[target_step_index] = step_data
                            times_to_save[target_step_index] = step_time
                            found = True
                            break
                    if not found:
                        can_save_all_steps = False
                        break

                if can_save_all_steps:
                    logger.debug(f"  Triggering save for steps: {list(steps_to_save_indices.keys())}")
                    try:
                        save_func(
                            data_dict=steps_to_save_indices, time_dict=times_to_save,
                            channels=all_channels, lat=lat, lon=lon, config=config,
                            output_dir=output_dir, current_model_step=model_step_counter, logger=logger
                        )
                        logger.debug(f"  Save function completed for model step {model_step_counter}.")
                    except Exception as save_e:
                         logger.error(f"Error during intermediate save call for step {model_step_counter}: {save_e}", exc_info=True)
                         # Continue loop even if saving fails? Or raise? Let's continue but log error.

            else: # Collect full history
                if model_step_counter % output_freq == 0:
                    logger.debug(f"  Collecting output for model step {model_step_counter} in memory.")
                    output_tensors_full_history.append(data_denorm_cpu)
                else:
                     # Need to delete the CPU tensor if not saved or collected
                     del data_denorm_cpu
                     logger.debug("  Deleted unused CPU output tensor.")


            # --- Timing and Increment ---
            iter_end_time = time.time()
            step_duration = iter_end_time - iter_start_time
            inference_times.append(step_duration)
            logger.debug(f"  Iteration {i} (Step {model_step_counter}) finished in {step_duration:.3f}s.")

            model_step_counter += 1
        # --- End of Loop ---

        logger.info(f"Finished {i + 1} iterations over time_loop.") # Use final 'i' value

    except StopIteration:
         logger.warning(f"Iterator stopped prematurely after {i} iterations (model step {model_step_counter-1}). Expected {simulation_length + 1} iterations.")
         # Allow proceeding with collected data
    except torch.cuda.OutOfMemoryError as oom_err:
        logger.error(f"CUDA Out of Memory error occurred during time_loop iteration step {model_step_counter}: {oom_err}", exc_info=True)
        logger.error(f"GPU Memory state at OOM: {get_gpu_memory_usage(device, logger)}")
        return None # Indicate failure clearly
    except Exception as e:
        logger.error(f"Unhandled exception occurred during time_loop iteration step {model_step_counter}: {e}", exc_info=True)
        return None # Indicate failure

    finally:
         # --- Cleanup ---
         logger.debug("Performing final cleanup...")
         # Explicitly delete iterator and buffer to help GC
         if iterator is not None: del iterator
         if output_history_buffer: output_history_buffer.clear()
         # Delete potentially large initial state tensors if they somehow persisted
         if 'initial_state_perturbed_norm_gpu' in locals() and initial_state_perturbed_norm_gpu is not None: del initial_state_perturbed_norm_gpu
         if 'initial_state_perturbed_norm_5d_cpu' in locals() and initial_state_perturbed_norm_5d_cpu is not None: del initial_state_perturbed_norm_5d_cpu
         # Final cache empty
         if torch.cuda.is_available() and device.type == 'cuda':
              torch.cuda.empty_cache()
              logger.debug("  Called final torch.cuda.empty_cache().")
              if log_gpu_mem: logger.info(f"  GPU Memory after final cleanup: {get_gpu_memory_usage(device, logger)}")


    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    logger.info(f"time_loop execution finished. Total time: {total_duration:.2f}s.")
    if inference_times:
        logger.info(f"Avg step time: {avg_inference_time:.3f}s. Min: {np.min(inference_times):.3f}s, Max: {np.max(inference_times):.3f}s.")




    # --- 5. Combine and Return Full History (if applicable) ---
    if not perform_intermediate_saving:
        if not output_tensors_full_history:
            logger.warning("No output tensors were collected for full history! Returning None.")
            return None

        logger.info(f"Stacking {len(output_tensors_full_history)} collected output tensors (CPU) for final result...")
        try:
            final_return_tensor = torch.stack(output_tensors_full_history, dim=1) # Stack on CPU
            logger.info(f"Final aggregated output tensor shape: {final_return_tensor.shape}")
            if torch.isnan(final_return_tensor).any():
                logger.warning("NaNs detected in the final aggregated output tensor!")
            # Clean up the list of tensors
            del output_tensors_full_history
            return final_return_tensor
        except Exception as e:
            logger.error(f"Failed to stack collected output tensors: {e}", exc_info=True)
            del output_tensors_full_history # Still try to cleanup
            return None
    else:
        logger.info("Intermediate saving was performed. Returning None as full history was not stored.")
        return None # Indicate success but no tensor returned



























# Keep this function definition in your inference_arco_73_numpy.py script
# Renamed from the original save_output
def save_full_output(
    output_tensor: torch.Tensor, # Expects (E, T_out, C, H, W)
    initial_time: datetime.datetime,
    time_step: datetime.timedelta, # Needs time_step to build time coordinate
    channels: List[str],
    lat: np.ndarray,
    lon: np.ndarray,
    config: dict,
    output_dir: str,
    logger: logging.Logger
):
    """Saves the full forecast history tensor to a single NetCDF file."""

    if output_tensor is None:
        logger.error("Cannot save full output, tensor is None.")
        return

    try:
        # output_tensor shape: (E, T_out, C, H, W)
        n_ensemble, n_time_out, n_channels, n_lat, n_lon = output_tensor.shape
        # output_frequency used ONLY to calculate time coords if collecting full history
        output_freq_for_coords = config.get("output_frequency", 1)

        logger.info("Preparing full forecast history for saving...")
        logger.debug(f"Full output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
        logger.debug(f"Number of channels: {n_channels}, Expected channels: {len(channels)}")
        logger.debug(f"Grid Lat shape: {lat.shape}, Lon shape: {lon.shape}")

        # --- Channel Handling ---
        if n_channels != len(channels):
            logger.error(f"Mismatch channels in full tensor ({n_channels}) vs names ({len(channels)}). Saving with generic indices.")
            channels_coord = np.arange(n_channels)
            channel_dim_name = "channel_idx"
        else:
            channels_coord = channels
            channel_dim_name = "channel"

        # --- Create Time Coordinates ---
        time_coords = []
        try:
             # Time coordinate generation based on output frequency during collection
             # t=0 is the first element in output_tensors_full_history
             time_coords = [initial_time + i * output_freq_for_coords * time_step for i in range(n_time_out)]
             logger.debug(f"Generated {len(time_coords)} time coordinates for full history.")
        except Exception as e:
            logger.error(f"Failed to create time coordinates for full history: {e}", exc_info=True)
            time_coords = np.arange(n_time_out) # Fallback

        if len(time_coords) != n_time_out:
            logger.warning(f"Generated {len(time_coords)} time coords, expected {n_time_out}. Using indices.")
            time_coords = np.arange(n_time_out)





        # --- Ensure Lat/Lon are Numpy ---
        lat_np = lat.cpu().numpy() if isinstance(lat, torch.Tensor) else np.asarray(lat)
        lon_np = lon.cpu().numpy() if isinstance(lon, torch.Tensor) else np.asarray(lon)




        # --- Create DataArray & Dataset ---
        logger.debug("Creating xarray DataArray for full history...")
        if np.isnan(output_tensor.numpy()).any():
            logger.warning("NaNs present in the full output tensor before saving!")

        forecast_da = xr.DataArray(
            output_tensor.numpy(), # Use the full tensor
            coords={
                'ensemble': np.arange(n_ensemble),
                'time': time_coords, # Use the generated time coordinates
                channel_dim_name: channels_coord,
                'lat': lat_np,
                'lon': lon_np,
            },
            dims=['ensemble', 'time', channel_dim_name, 'lat', 'lon'],
            name='forecast',
            attrs={
                'description': f"{config['weather_model']} full ensemble forecast output",
                'model': config['weather_model'],
                'simulation_length_steps': config['simulation_length'],
                # output_frequency here refers to how often steps were stored in the tensor
                'output_frequency_stored': output_freq_for_coords,
                'ensemble_members': n_ensemble,
                'initial_condition_time': initial_time.isoformat(),
                'time_step_seconds': time_step.total_seconds() if isinstance(time_step, datetime.timedelta) else 'unknown',
                'noise_amplitude': config['noise_amplitude'],
                'perturbation_strategy': config['perturbation_strategy'],
                'creation_date': datetime.datetime.now(pytz.utc).isoformat(),
                # Add relevant versions
            }
        )
        logger.info("Created xarray DataArray for full history.")
        forecast_ds = forecast_da.to_dataset(dim=channel_dim_name)
        logger.info("Converted DataArray to Dataset for full history.")

        # --- Define Filename & Save ---
        ic_time_str = initial_time.strftime('%Y%m%d_%H%M%S')
        output_filename = os.path.join(
            output_dir,
            f"{config['weather_model']}_ensemble{n_ensemble}_simulated_steps{config['simulation_length']}_IC_{ic_time_str}_FULL.nc" # Indicate full history
        )

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving full forecast history to: {output_filename}")

        encoding = {var: {'zlib': True, 'complevel': 5, '_FillValue': -9999.0} for var in forecast_ds.data_vars}
        start_save = time.time()
        forecast_ds.to_netcdf(output_filename, encoding=encoding, engine='netcdf4')
        end_save = time.time()
        file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
        logger.info(f"Save complete (full history). Time: {end_save - start_save:.2f}s. Size: {file_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Failed during the save_full_output process: {e}", exc_info=True)
        # Optional: Attempt to remove potentially corrupted file
        if 'output_filename' in locals() and os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                logger.warning(f"Removed potentially corrupted file: {output_filename}")
            except OSError as oe:
                logger.error(f"Failed to remove corrupted file {output_filename}: {oe}")
























def parse_date_from_filename(fname: str) -> Optional[datetime]:
    """
    Robustly extracts the start date from a filename with the pattern:
    START_<day>_<month>_<year>_END_...npy
    
    Args:
        fname: Full file path containing the date pattern
        
    Returns:
        datetime object representing the parsed date
        
    Raises:
        ValueError: If date cannot be parsed from the filename
    """
    # Predefined month mapping for case-insensitive lookup
    MONTH_MAP = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    # Extract basename and handle empty/malformed paths
    basename = os.path.basename(fname)
    if not basename:
        raise ValueError("Invalid filename path structure")

    # Robust regex pattern with multiple validation checks
    pattern = re.compile(
        r"""
        START_                     # Start marker
        (?P<day>\d{1,2})           # Day (1-31)
        _                          
        (?P<month>[A-Za-z]+)       # Month name
        _                          
        (?P<year>\d{4})            # Year (4 digits)
        (?=_END)                   # Positive lookahead for _END
        """, 
        re.VERBOSE | re.IGNORECASE
    )

    match = pattern.search(basename)
    if not match:
        raise ValueError(f"Date pattern not found in filename: {basename}")

    try:
        # Extract and normalize components
        day = int(match.group('day'))
        month_str = match.group('month').lower()
        year = int(match.group('year'))
        
        # Validate month
        month = MONTH_MAP.get(month_str)
        if not month:
            raise ValueError(f"Invalid month name: {match.group('month')}")

        # Validate date components
        if not (1 <= day <= 31):
            raise ValueError(f"Day out of range: {day}")
        if not (1900 <= year <= 2100):
            raise ValueError(f"Year out of reasonable range: {year}")

        # Construct and validate actual date
        date_obj = datetime(year, month, day)
        return date_obj

    except ValueError as e:
        raise ValueError(f"Invalid date components in filename: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error parsing date: {str(e)}") from e










# --- Function Definitions (setup_logging, save_output_steps, save_full_output, run_inference) ---
# Make sure save_output_steps and run_inference from the previous answer are defined here
# Also include save_full_output defined above




























# # --- Updated Main Pipeline Function ---
# def main(args, save_steps_config, netcdf_output_dir): # Add new args
#     """Main pipeline execution function."""

#     logger.info("========================================================")
#     logger.info(" Starting FCNv2-SM Inference Pipeline from NumPy ICs")
#     logger.info("========================================================")
#     logger.info(f"Full command line arguments: {sys.argv}")
#     logger.info(f"Parsed arguments: {vars(args)}")
#     logger.info(f"Save mode: {args.save_mode}")
#     if args.save_mode == 'intermediate':
#         logger.info(f"Intermediate save steps config: {save_steps_config}")
#     logger.info(f"Effective MODEL_REGISTRY: {os.environ.get('MODEL_REGISTRY', 'Not Set')}")
#     logger.info(f"NetCDF output directory: {netcdf_output_dir}")


#     # --- Environment and Setup ---
#     if args.gpu >= 0 and torch.cuda.is_available():
#         try:
#             device = torch.device(f"cuda:{args.gpu}")
#             torch.cuda.set_device(device)
#             logger.info(f"Attempting to use GPU: {args.gpu} ({torch.cuda.get_device_name(device)})")
#             logger.info(f"CUDA version: {torch.version.cuda}")
#             logger.info(f"PyTorch version: {torch.__version__}")
#         except Exception as e:
#             logger.error(f"Failed to set CUDA device {args.gpu}: {e}. Falling back to CPU.", exc_info=True)
#             device = torch.device("cpu")
#             logger.info("Using CPU.")
#     else:
#         device = torch.device("cpu")
#         if args.gpu >= 0:
#             logger.warning(f"GPU {args.gpu} requested, but CUDA not available. Using CPU.")
#         else:
#             logger.info("Using CPU.")
            
            
#     log_gpu_memory(logger, "Start of main")



#     # --- Load Model ---
#     model_id = "fcnv2_sm"  # Use the small version
#     logger.info(f"Loading {model_id} model...")
#     try:
#         logger.info(f"Fetching model package for '{model_id}' from registry: {os.environ.get('MODEL_REGISTRY')}")
#         package = registry.get_model(model_id)
#         if package is None:
#             logger.error(f"Failed to get model package for '{model_id}'. Check registry path and model name.")
#             sys.exit(1)
#         logger.info(f"Found model package: {package}. Root path: {package.root}")

#         # Add logging inside the load function if possible (by editing earth2mip source)
#         # or log parameters being passed here
#         logger.info(f"Calling {model_id}_load with package root: {package.root}, device: {device}, pretrained: True")
#         model_inference = fcnv2_sm_load(package, device=device, pretrained=True)
#         model_inference.eval()  # Set model to evaluation mode
#         log_gpu_memory(logger, "After model load")

#         # Verification after loading
#         logger.info(f"{model_id} model loaded successfully to device: {next(model_inference.parameters()).device}.")
#         logger.info(f"Model expects {len(model_inference.in_channel_names)} input channels.")
#         logger.debug(f"Model input channels: {model_inference.in_channel_names}")
#         logger.info(f"Model output channels: {model_inference.out_channel_names}") # Usually same as input
#         # logger.info(f"Model grid: {model_inference.grid}")
#         logger.info(f"Model time step: {model_inference.time_step}")

#     except FileNotFoundError as e:
#         logger.error(f"Model loading failed: Required file not found - {e}", exc_info=True)
#         logger.error(f"Please check that weights.tar, global_means.npy, global_stds.npy exist within {os.path.join(os.environ.get('MODEL_REGISTRY'), model_id)}")
#         sys.exit(1)
#     except _pickle.UnpicklingError as e:
#         logger.error(f"Model loading failed due to UnpicklingError: {e}", exc_info=False)  # Don't need full traceback again
#         logger.error("This usually means torch.load failed with weights_only=True (default in PyTorch >= 2.6).")
#         logger.error(f"Ensure you have modified '{EARTH2MIP_PATH}/earth2mip/networks/fcnv2_sm.py' to use 'torch.load(..., weights_only=False)'.")
#         sys.exit(1)
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)
#         sys.exit(1)

#     # --- Load Initial Conditions from NumPy file ---
#     logger.info(f"Loading initial conditions from: {args.ic_file_path}")
#     if not os.path.exists(args.ic_file_path):
#         logger.error(f"Initial condition file not found: {args.ic_file_path}")
#         sys.exit(1)
#     try:
#         initial_conditions_np = np.load(args.ic_file_path)
#         logger.info(f"Loaded NumPy data with shape: {initial_conditions_np.shape}, dtype: {initial_conditions_np.dtype}")
#         # Expected shape: (num_times, num_channels, height, width)
#         if initial_conditions_np.ndim != 4:
#             raise ValueError(f"Expected 4 dimensions (time, channel, lat, lon), but got {initial_conditions_np.ndim}")

#         num_ics, num_channels, height, width = initial_conditions_np.shape
#         logger.info(f"Found {num_ics} initial conditions in the file. Grid size: {height}x{width}")

#         # Validate channel count
#         model_channels = model_inference.in_channel_names
#         if num_channels != len(model_channels):
#             logger.error(f"Channel mismatch! Model expects {len(model_channels)} channels, but NumPy file has {num_channels} channels.")
#             logger.error(f"Model channels: {model_channels}")
#             logger.error("Please ensure the NumPy file was created with the correct channels in the expected order.")
#             sys.exit(1)
#         else:
#             logger.info("Channel count matches model requirements.")

#         # Validate grid size (optional but good practice)
#         model_lat, model_lon = model_inference.grid.lat, model_inference.grid.lon
#         if height != len(model_lat) or width != len(model_lon):
#             logger.warning(f"Grid mismatch! Model grid is {len(model_lat)}x{len(model_lon)}, but NumPy file grid is {height}x{width}.")
#             logger.warning("Ensure the NumPy file represents data on the model's native grid.")
#             # Decide if this is critical - for now, just warn.
#             # sys.exit(1)

#     except Exception as e:
#         logger.error(f"Failed to load or validate NumPy file: {e}", exc_info=True)
#         sys.exit(1)

#     # --- Define Timestamps for the loaded ICs ---
#     # Ensure this matches the actual content and order of your .npy file
#     try:

#         try:
#             fname = os.path.basename(args.ic_file_path)
#             logger.info(f"Extracting date from filename: {fname}")
#             parsed_date = parse_date_from_filename(fname)
#             logger.info(f"Successfully parsed date: {parsed_date.strftime('%Y-%m-%d')}")
#             base_date = parsed_date
#             # year, month, day = extract_date_from_fname(fname)
#             # base_date = datetime.datetime(year, month, day, tzinfo=pytz.utc)  # Assume UTC if not specified
#             # logger.info(f"Inferred base date from filename: {base_date.strftime('%Y-%m-%d')}")

#         except:
#             # Fallback to hardcoded date if filename parsing fails
#             base_date = datetime.datetime(2020, 6, 22, tzinfo=pytz.utc)  # Make timezone aware (UTC is standard for IFS/ERA)
#             logger.warning(f"Could not infer date from filename, using default: {base_date.strftime('%Y-%m-%d')}")

#         # Generate timestamps assuming 6-hourly intervals starting at 00Z
#         ic_timestamps = [base_date + datetime.timedelta(hours=i * 6) for i in range(num_ics)]

#     except Exception as e:
#         logger.error(f"Error determining timestamps for ICs: {e}. Using generic indices.", exc_info=True)
#         ic_timestamps = list(range(num_ics))  # Fallback

#     logger.info(f"Using the following timestamps/indices for the {num_ics} loaded ICs:")
#     for i, ts in enumerate(ic_timestamps):
#         if isinstance(ts, datetime.datetime):
#             logger.info(f"- IC {i}: {ts.isoformat()}")
#         else:
#             logger.info(f"- IC {i}: Index {ts}")

#     # --- Prepare Inference Configuration (passed to run_inference and save functions) ---
#     inference_config = {
#         "ensemble_members": args.ensemble_members,
#         "noise_amplitude": args.noise_amplitude,
#         "simulation_length": args.simulation_length,
#         "output_frequency": args.output_frequency, # Used for full history collection frequency
#         "weather_model": model_id,
#         "perturbation_strategy": args.perturbation_strategy,
#         # Add any other relevant config needed by save functions
#         'variables_to_save': None # Example: Set this to ['t2m', 'u10m'] to only save specific vars
#     }
#     logger.info(f"Inference Configuration: {inference_config}")

#     # --- Run Inference for each Initial Condition ---
#     num_ics_processed = 0
#     num_ics_failed = 0
#     total_start_time = time.time()

#     for i, initial_time in enumerate(ic_timestamps):
#         # --- Prepare IC Time ---
#         if not isinstance(initial_time, datetime.datetime):
#              logger.error(f"IC timestamp for index {i} is not datetime ({type(initial_time)}). Skipping.")
#              num_ics_failed += 1
#              continue
#         if initial_time.tzinfo is None or initial_time.tzinfo.utcoffset(initial_time) is None:
#              initial_time = initial_time.replace(tzinfo=pytz.utc) # Ensure timezone aware

#         time_label = initial_time.isoformat()
#         logger.info(f"--- Processing Initial Condition {i+1}/{num_ics}: {time_label} ---")

        
        
#         log_gpu_memory(logger, f"Start of IC {i+1} Loop")

#         # --- Prepare IC Tensor ---
#         ic_data_np = initial_conditions_np[i]
#         try:
#             initial_state_tensor = torch.from_numpy(ic_data_np).unsqueeze(0).float() # Shape (1, C, H, W)
#             logger.debug(f"Prepared IC tensor (1, C, H, W): {initial_state_tensor.shape}")
#         except Exception as e:
#             logger.error(f"Failed to convert NumPy slice {i} to tensor: {e}", exc_info=True)
#             num_ics_failed += 1
#             continue

#         # --- Execute Inference based on Save Mode ---
#         start_run = time.time()
#         output_tensor_full = None # Variable to store full history if needed

#         try:
#             if args.save_mode == 'intermediate':
#                 logger.info("Running inference in 'intermediate' save mode.")
#                 # run_inference returns None in this mode
#                 run_inference(
#                     model_inference=model_inference,
#                     initial_state_tensor=initial_state_tensor,
#                     initial_time_dt=initial_time,
#                     config=inference_config,
#                     logger=logger,
#                     save_func=save_output_steps, # Pass the intermediate save function
#                     save_steps_config=save_steps_config, # Pass the step config
#                     output_dir=netcdf_output_dir       # Pass the output directory
#                 )
#                 # No tensor is returned, success is assumed if no exception occurred
#                 run_successful = True # Assume success if no exception

#             elif args.save_mode == 'full':
#                 logger.info("Running inference in 'full' save mode (collecting history in RAM).")
#                 # run_inference returns the full tensor (or None on error)
#                 output_tensor_full = run_inference(
#                     model_inference=model_inference,
#                     initial_state_tensor=initial_state_tensor,
#                     initial_time_dt=initial_time,
#                     config=inference_config,
#                     logger=logger,
#                     save_func=None, # Disable intermediate saving
#                     save_steps_config={}, # Not used in full mode
#                     output_dir=None      # Not used in full mode
#                 )
#                 run_successful = output_tensor_full is not None
#             else:
#                 # Should not happen due to argparse choices, but handle defensively
#                 logger.error(f"Invalid save_mode: {args.save_mode}")
#                 run_successful = False

#             end_run = time.time()
#             log_gpu_memory(logger, f"After run_inference IC {i+1}")
            
            
            
            
#             # --- Post-Inference Processing ---
#             if run_successful:
#                 logger.info(f"Inference run for IC {time_label} completed in {end_run - start_run:.2f} seconds.")
#                 num_ics_processed += 1

#                 # Save the full history if collected
#                 if args.save_mode == 'full' and output_tensor_full is not None:
#                     logger.info("Saving collected full history tensor...")
#                     try:
#                         save_full_output( # Use the dedicated function
#                             output_tensor=output_tensor_full,
#                             initial_time=initial_time,
#                             time_step=model_inference.time_step, # Get from model
#                             channels=model_inference.in_channel_names,
#                             lat=model_inference.grid.lat,
#                             lon=model_inference.grid.lon,
#                             config=inference_config,
#                             output_dir=netcdf_output_dir, # Save to the main output dir
#                             logger=logger
#                         )
#                     finally:
#                         del output_tensor_full # Delete the large tensor from CPU RAM
#                         log_gpu_memory(logger, f"After saving full history IC {i+1}")
#             else:
#                 logger.error(f"Inference run failed for IC {time_label}.")
#                 num_ics_failed += 1

#         except Exception as run_err:
#              logger.error(f"Unhandled exception during run_inference or saving for IC {time_label}: {run_err}", exc_info=True)
#              num_ics_failed += 1
#              end_run = time.time() # Still record end time if possible
#              log_gpu_memory(logger, f"After run_inference IC {i+1}")
#              logger.info(f"Inference run attempt for IC {time_label} took {end_run - start_run:.2f} seconds before failing.")

#         # --- Cleanup GPU Cache ---
#         # More aggressive cache clearing after each IC
#         logger.debug("Attempting to clear CUDA cache after processing IC...")
#         del initial_state_tensor # Delete IC tensor explicitly
#         # Any other large tensors created outside run_inference for this IC? Delete them.
#         if torch.cuda.is_available():
#              torch.cuda.empty_cache()
#              logger.debug("Cleared CUDA cache.")
#         log_gpu_memory(logger, f"End of IC {i+1} Loop")

#     # --- Final Summary ---
#     total_end_time = time.time()
#     logger.info(f"--- Total processing time for {num_ics} ICs: {total_end_time - total_start_time:.2f} seconds ---")
#     logger.info(f"Successfully processed {num_ics_processed} initial conditions.")
#     if num_ics_failed > 0:
#         logger.warning(f"Failed to process {num_ics_failed} initial conditions.")
#     logger.info(f"Output NetCDF files saved in: {netcdf_output_dir}")
#     logger.info(f"Log file saved in: {LOG_DIR}")
#     logger.info("========================================================")
#     logger.info(" FCNv2-SM Inference Pipeline Finished ")
#     logger.info("========================================================")
#     log_gpu_memory(logger, "End of main")












# --- Add Memory Logging Utility ---
def log_gpu_memory(logger, point="Point"):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        logger.info(f"GPU Memory @ {point}: Allocated={allocated:.2f} GiB, Reserved={reserved:.2f} GiB")
        logger.debug(f"GPU Memory Peak @ {point}: Max Allocated={max_allocated:.2f} GiB, Max Reserved={max_reserved:.2f} GiB")
    else:
        logger.debug(f"GPU Memory @ {point}: CUDA not available.")


















# --- Helper Function: Setup Device ---
def setup_device(args: argparse.Namespace, logger: logging.Logger) -> torch.device:
    """Sets up the computation device (GPU or CPU)."""
    if args.gpu >= 0 and torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(device)
            logger.info(f"Successfully set device to GPU: {args.gpu} ({torch.cuda.get_device_name(device)})")
            logger.info(f"CUDA version: {torch.version.cuda}, PyTorch version: {torch.__version__}")
            return device
        except Exception as e:
            logger.error(f"Failed to set CUDA device {args.gpu}: {e}. Falling back to CPU.", exc_info=True)
            return torch.device("cpu")
    else:
        if args.gpu >= 0:
            logger.warning(f"GPU {args.gpu} requested, but CUDA not available. Using CPU.")
        else:
            logger.info("Using CPU.")
        return torch.device("cpu")








# --- Helper Function: Load Model ---
def load_model(model_id: str, registry_path: str, device: torch.device, logger: logging.Logger):
    """Loads the specified model, handling potential errors."""
    logger.info(f"Loading model '{model_id}'...")
    try:
        logger.info(f"Fetching model package from registry: {registry_path}")
        package = registry.get_model(model_id) # Ensure registry points to correct path via MODEL_REGISTRY env var
        if package is None:
            raise FileNotFoundError(f"Model package '{model_id}' not found in registry '{registry_path}'.")
        logger.info(f"Found model package. Root path: {package.root}")

        # Explicitly log parameters for the load function
        logger.info(f"Calling load function for '{model_id}' with root='{package.root}', device='{device}', pretrained=True")
        # Assuming a generic load function mapping or specific imports like fcnv2_sm_load
        if model_id == "fcnv2_sm":
            model_inference = fcnv2_sm_load(package, device=device, pretrained=True)
        else:
            # Add logic for other models if needed
            raise ValueError(f"Unsupported model_id for loading: {model_id}")

        model_inference.eval() # Set to evaluation mode
        logger.info(f"Model '{model_id}' loaded successfully to device: {next(model_inference.parameters()).device}.")

        # --- Post-load Verification ---
        logger.info(f"Model expects {len(model_inference.in_channel_names)} input channels.")
        logger.debug(f"Input channels: {model_inference.in_channel_names}")
        logger.info(f"Output channels: {model_inference.out_channel_names}")
        logger.info(f"Model time step: {model_inference.time_step}")
        # Safe grid logging (assuming lat/lon are attributes, might need np.asarray if list)
        try:
             lat = np.asarray(model_inference.grid.lat)
             lon = np.asarray(model_inference.grid.lon)
             logger.info(f"Model grid shape: Lat {lat.shape}, Lon {lon.shape}")
        except Exception as grid_e:
             logger.warning(f"Could not log model grid details: {grid_e}")

        return model_inference

    except FileNotFoundError as e:
        logger.error(f"Model loading failed: Required file not found - {e}", exc_info=True)
        logger.error("Check registry path and ensure model files (weights.tar, etc.) exist.")
        sys.exit(1)
    except _pickle.UnpicklingError as e:
        logger.error(f"Model loading failed (UnpicklingError): {e}", exc_info=False)
        logger.error("This often indicates torch.load failure (weights_only=True/False issue?).")
        logger.error("Ensure model loading function handles PyTorch version compatibility.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
        sys.exit(1)








# --- Helper Function: Load Initial Conditions ---
def load_initial_conditions(ic_file_path: str, model_channels: list, logger: logging.Logger) -> tuple[np.ndarray, list]:
    """Loads ICs from NumPy file, validates, and determines timestamps."""
    logger.info(f"Loading initial conditions from: {ic_file_path}")
    if not os.path.exists(ic_file_path):
        logger.error(f"Initial condition file not found: {ic_file_path}")
        sys.exit(1)

    try:
        initial_conditions_np = np.load(ic_file_path)
        logger.info(f"Loaded NumPy data: Shape={initial_conditions_np.shape}, Dtype={initial_conditions_np.dtype}")

        if initial_conditions_np.ndim != 4:
            raise ValueError(f"Expected 4D array (time, channel, lat, lon), got {initial_conditions_np.ndim}D.")

        num_ics, num_channels, height, width = initial_conditions_np.shape
        logger.info(f"Found {num_ics} ICs. Grid: {height}x{width}, Channels: {num_channels}")

        # Validate channel count
        if num_channels != len(model_channels):
            raise ValueError(f"Channel mismatch! Model={len(model_channels)}, File={num_channels}.")
        logger.info("Channel count matches model.")

        # Determine Timestamps
        try:
            fname = os.path.basename(ic_file_path)
            base_date = parse_date_from_filename(fname)
            logger.info(f"Inferred base date from filename '{fname}': {base_date.strftime('%Y-%m-%d')}")
            # Ensure timezone awareness (default to UTC)
            if base_date.tzinfo is None:
                 base_date = base_date.replace(tzinfo=pytz.utc)
            # Generate timestamps assuming 6-hourly intervals starting from base_date 00Z (adjust logic if needed)
            start_time_for_seq = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
            ic_timestamps = [start_time_for_seq + datetime.timedelta(hours=i * 6) for i in range(num_ics)]
            logger.info("Generated 6-hourly timestamps starting from 00Z of inferred date.")

        except Exception as e:
            logger.warning(f"Could not determine timestamps from filename ({e}). Using indices.", exc_info=True)
            ic_timestamps = list(range(num_ics))

        logger.info(f"Using Timestamps/Indices for {num_ics} ICs:")
        for i, ts in enumerate(ic_timestamps):
            if isinstance(ts, datetime.datetime): logger.info(f" - IC {i}: {ts.isoformat()}")
            else: logger.info(f" - IC {i}: Index {ts}")

        return initial_conditions_np, ic_timestamps

    except ValueError as ve:
        logger.error(f"Validation error loading ICs: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load or process NumPy file: {e}", exc_info=True)
        sys.exit(1)







# --- Helper Function: Process Single Initial Condition ---
def process_single_ic(
    ic_index: int,
    num_total_ics: int,
    ic_data_np: np.ndarray, # Single IC data (C, H, W)
    initial_time: datetime.datetime | int, # Timestamp or index
    model_inference, # The loaded model
    inference_config: dict,
    save_mode: str,
    save_steps_config: dict,
    output_dir: str,
    logger: logging.Logger,
    run_inference_func: callable, # Pass the run_inference function
    save_full_output_func: callable, # Pass the save_full_output function
    save_intermediate_func: callable # Pass the save_output_steps function
) -> bool:
    """Processes a single initial condition through the inference pipeline."""

    # --- Prepare Time Label and Validate ---
    if isinstance(initial_time, datetime.datetime):
        # Ensure timezone aware for logging/processing
        if initial_time.tzinfo is None or initial_time.tzinfo.utcoffset(initial_time) is None:
             initial_time = initial_time.replace(tzinfo=pytz.utc) # Default to UTC
        time_label = initial_time.isoformat()
    else:
        time_label = f"Index {initial_time}"
        logger.error(f"IC timestamp for index {ic_index} is not datetime ({type(initial_time)}). Cannot proceed.")
        return False # Cannot run inference without proper datetime

    logger.info(f"--- Processing IC {ic_index + 1}/{num_total_ics}: {time_label} ---")
    log_gpu_memory(logger, f"Start of IC {ic_index + 1}")
    start_run_time = time.time()

    # --- Prepare IC Tensor (on CPU) ---
    initial_state_tensor_cpu = None # Define before try block
    try:
        # Ensure numpy slice is C, H, W
        if ic_data_np.ndim != 3:
             raise ValueError(f"Expected 3D numpy array (C,H,W), got {ic_data_np.ndim}D")
        # Convert to tensor, add batch dim (1), ensure float32, KEEP ON CPU
        initial_state_tensor_cpu = torch.from_numpy(ic_data_np).unsqueeze(0).float().cpu()
        logger.debug(f"Prepared IC tensor on CPU (1, C, H, W): {initial_state_tensor_cpu.shape}, dtype: {initial_state_tensor_cpu.dtype}")
    except Exception as e:
        logger.error(f"Failed to prepare tensor from NumPy slice {ic_index}: {e}", exc_info=True)
        return False # Cannot proceed

    # --- Execute Inference ---
    output_tensor_full = None
    run_successful = False
    try:
        if save_mode == 'intermediate':
            logger.info("Running inference in 'intermediate' save mode.")
            run_inference_func(
                model_inference=model_inference,
                initial_state_tensor=initial_state_tensor_cpu, # Pass CPU tensor
                initial_time_dt=initial_time,
                config=inference_config,
                logger=logger,
                save_func=save_intermediate_func,
                save_steps_config=save_steps_config,
                output_dir=output_dir
            )
            # run_inference returns None, success inferred if no exception
            run_successful = True

        elif save_mode == 'full':
            logger.info("Running inference in 'full' save mode (collecting history).")
            output_tensor_full = run_inference_func(
                model_inference=model_inference,
                initial_state_tensor=initial_state_tensor_cpu, # Pass CPU tensor
                initial_time_dt=initial_time,
                config=inference_config,
                logger=logger,
                save_func=None, # Disable intermediate saving
                save_steps_config={}
                # output_dir not needed here
            )
            run_successful = output_tensor_full is not None
        else:
            logger.error(f"Invalid save_mode: {save_mode}")
            run_successful = False

    except Exception as run_err:
         logger.error(f"Unhandled exception during run_inference for IC {time_label}: {run_err}", exc_info=True)
         run_successful = False # Mark as failed
    finally:
        # --- Cleanup and Post-processing ---
        end_run_time = time.time()
        log_gpu_memory(logger, f"After run_inference IC {ic_index + 1}")

        if run_successful:
            logger.info(f"Inference run for IC {time_label} completed in {end_run_time - start_run_time:.2f} seconds.")
            # Save full history if applicable
            if save_mode == 'full' and output_tensor_full is not None:
                logger.info("Saving collected full history tensor...")
                try:
                    save_full_output_func(
                        output_tensor=output_tensor_full,
                        initial_time=initial_time,
                        time_step=model_inference.time_step,
                        channels=model_inference.in_channel_names, # Assuming in==out
                        lat=np.asarray(model_inference.grid.lat), # Ensure numpy
                        lon=np.asarray(model_inference.grid.lon), # Ensure numpy
                        config=inference_config,
                        output_dir=output_dir,
                        logger=logger
                    )
                except Exception as save_err:
                     logger.error(f"Failed to save full output for IC {time_label}: {save_err}", exc_info=True)
                     # Decide if this constitutes a full failure for the IC
                     run_successful = False # Treat save failure as IC failure
                finally:
                    del output_tensor_full # Delete from CPU RAM
                    log_gpu_memory(logger, f"After saving full history IC {ic_index + 1}")
        else:
            # Error logged inside try/except block or by run_inference
            logger.error(f"Inference run FAILED for IC {time_label}. Attempt took {end_run_time - start_run_time:.2f} seconds.")


        # --- Explicit Cleanup for this IC ---
        logger.debug(f"Cleaning up resources for IC {ic_index + 1}...")
        del initial_state_tensor_cpu # Delete CPU IC tensor
        # output_tensor_full should be deleted above if it existed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache.")
        log_gpu_memory(logger, f"End of IC {ic_index + 1} Loop")

        return run_successful










# --- Updated Main Pipeline Function ---
def main(args, save_steps_config, netcdf_output_dir): # Keep args signature
    """Main pipeline execution function (Refactored)."""
    global logger # Assuming logger is configured globally or passed differently
    # If logger is not global, it needs to be initialized here or passed in.
    # logger = setup_logging(...) # Example

    start_time_main = time.time()
    logger.info("="*50)
    logger.info(" Starting FCNv2 Inference Pipeline (Refactored)")
    logger.info("="*50)
    logger.info(f"Run args: {vars(args)}")
    logger.info(f"Save mode: {args.save_mode}")
    if args.save_mode == 'intermediate': logger.info(f"Intermediate save config: {save_steps_config}")
    logger.info(f"NetCDF output dir: {netcdf_output_dir}")
    logger.info(f"MODEL_REGISTRY env var: {os.environ.get('MODEL_REGISTRY', 'Not Set')}")

    # --- Setup ---
    device = setup_device(args, logger)
    log_gpu_memory(logger, "Start of main")
    model_id = "fcnv2_sm" # Or get from args

    # --- Load Model ---
    model_inference = load_model(model_id, os.environ.get('MODEL_REGISTRY'), device, logger)
    log_gpu_memory(logger, "After model load")

    # --- Load Initial Conditions ---
    # Pass expected model channels for validation
    initial_conditions_np, ic_timestamps = load_initial_conditions(
        args.ic_file_path,
        model_inference.in_channel_names,
        logger
    )

    # --- Prepare Inference Configuration ---
    inference_config = {
        "ensemble_members": args.ensemble_members,
        "noise_amplitude": args.noise_amplitude,
        "simulation_length": args.simulation_length,
        "output_frequency": args.output_frequency, # For full history collection
        "weather_model": model_id,
        "perturbation_strategy": args.perturbation_strategy,
        'variables_to_save': None # Customize if needed
    }
    logger.info(f"Inference Configuration: {inference_config}")

    # --- Run Inference Loop ---
    num_ics = len(ic_timestamps)
    num_processed_successfully = 0
    num_failed = 0

    for i, initial_time in enumerate(ic_timestamps):
        success = process_single_ic(
            ic_index=i,
            num_total_ics=num_ics,
            ic_data_np=initial_conditions_np[i], # Pass the single IC numpy slice (C,H,W)
            initial_time=initial_time,
            model_inference=model_inference,
            inference_config=inference_config,
            save_mode=args.save_mode,
            save_steps_config=save_steps_config,
            output_dir=netcdf_output_dir,
            logger=logger,
            run_inference_func=run_inference, # Pass the actual function
            save_full_output_func=save_full_output, # Pass the actual function
            save_intermediate_func=save_output_steps # Pass the actual function
        )
        if success:
            num_processed_successfully += 1
        else:
            num_failed += 1

    # --- Final Summary ---
    end_time_main = time.time()
    logger.info("="*50)
    logger.info(" FCNv2 Inference Pipeline Finished ")
    logger.info("="*50)
    logger.info(f"Total execution time: {end_time_main - start_time_main:.2f} seconds")
    logger.info(f"Processed {num_processed_successfully}/{num_ics} initial conditions successfully.")
    if num_failed > 0:
        logger.warning(f"Failed to process {num_failed} initial conditions.")
    logger.info(f"Output NetCDF files saved in: {netcdf_output_dir}")
    # logger.info(f"Log file saved in: {LOG_DIR}") # Assuming LOG_DIR is defined globally or known
    log_gpu_memory(logger, "End of main")

    if num_failed > 0:
        sys.exit(1) # Exit with error code if any ICs failed
    else:
        sys.exit(0) # Exit successfully



































# In the if __name__ == "__main__": block

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCNv2-SM Inference Pipeline using initial conditions from a NumPy file.")

    # Input/Output paths
    parser.add_argument("--ic-file-path", type=str, default=f"/scratch/gilbreth/wwtung/ARCO_73chanel_data/data/2020/June/START_22_June_2020_END_22_June_2020.npy", help="Path to the NumPy file containing initial conditions (shape: T, C, H, W).")
    parser.add_argument("-o", "--output-path", type=str, default=os.path.join(OUTPUT_DIR, "inference_output"), help="Directory to save output NetCDF files.") # Changed default subfolder name

    # Inference parameters
    parser.add_argument("-sim", "--simulation-length", type=int, default=2, help="Number of autoregressive steps (forecast lead time in model steps).")
    parser.add_argument("-ef", "--output-frequency", type=int, default=1, help="Frequency (in steps) to store outputs when collecting full history (save_mode='full').")
    parser.add_argument("-ens", "--ensemble-members", type=int, default=1, help="Number of ensemble members (>=1).")
    parser.add_argument("-na", "--noise-amplitude", type=float, default=0.0, help="Amplitude for perturbation noise (if ensemble_members > 1). Set to 0 for no noise.") # Default 0 noise
    parser.add_argument("-ps", "--perturbation-strategy", type=str, default="gaussian", choices=["gaussian", "correlated", "none"], help="Perturbation strategy (currently uses Gaussian placeholder).")

    # ****** ADDED SAVING ARGUMENTS ******
    parser.add_argument(
        "--save-mode", type=str, default="intermediate", choices=["intermediate", "full"],
        help="Saving mode: 'intermediate' saves recent steps during the loop (memory efficient), 'full' collects entire history in RAM and saves at the end."
    )
    parser.add_argument(
        "--save-steps", type=str, default="-2,0",
        help="Comma-separated list of step offsets relative to the current step to save in 'intermediate' mode (e.g., '0' for current only, '-2,0' for current and t-2)."
    )
    # ****** END ADDED SAVING ARGUMENTS ******

    # System parameters
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (-1 for CPU).")
    parser.add_argument("--debug", type=bool, default=True, help="Enable debug logging.")

    args = parser.parse_args()

    # --- Process save_steps argument ---
    try:
        save_steps_list = [int(step.strip()) for step in args.save_steps.split(',')]
    except ValueError:
        logger.error(f"Invalid format for --save-steps argument: '{args.save_steps}'. Please use comma-separated integers (e.g., '-2,0').")
        sys.exit(1)
    save_steps_config = {'steps_to_save': save_steps_list}
    # ------------------------------------

    # Ensure the main output directory exists before potentially setting logger level
    # Note: args.output_path is now the *base* directory for netcdf files
    netcdf_output_dir = args.output_path
    os.makedirs(netcdf_output_dir, exist_ok=True) # Create the specific netcdf output dir

    # Adjust logger level if debug flag is set (Logger setup happens before arg parsing, need to adjust after)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG) # Ensure handlers also get debug level
        logger.info("Debug logging enabled.")

    # Call main function
    try:
        main(args, save_steps_config, netcdf_output_dir) # Pass processed args
    except Exception as e:
        try:
            logger.critical(f"Critical pipeline failure: {str(e)}", exc_info=True)
        except NameError:
            logging.critical(f"Critical pipeline failure before logger setup: {str(e)}", exc_info=True)
        sys.exit(1)
        
        
        
        
        
"""


Reasoning for the Fixes:

    Explicit Shape Logging: By adding logger.debug(f" Shape AFTER ...") in run_inference after unsqueeze, normalization, and perturbation, and detailed logging inside _iterate before the check, we can pinpoint exactly where the shape deviates from the expected (E, 1, 73, H, W).

    Catching Errors Around next(iterator): Specifically catching the ValueError around the next(iterator) call allows us to log the shape of the tensor just before it was passed into the __call__ method, providing crucial context if the error happens on the very first iteration.

    Clarity in Error Message: The modified ValueError message inside _iterate now explicitly mentions which dimension index (index 1, the Time dimension) failed the check and what the full shape was.

How to Proceed:

    Apply the logging additions to both earth2mip/networks/__init__.py and inference_arco_73_numpy.py.

    Run the script again.

    Carefully examine the logs leading up to the ValueError:

        What shape is reported after unsqueeze(1) in run_inference?

        What shape is reported after normalization in run_inference?

        What shape is reported after perturbation in run_inference?

        What shape is logged as being passed to model_inference.__call__?

        What full shape is reported by the _iterate PRE-CHECK state: log message inside networks/__init__.py?

The mismatch should become evident from these logs, likely revealing an unexpected dimension permutation or an incorrect assumption about n_history. If the logs show the shape is correctly (E, 1, 73, H, W) all the way until the check inside _iterate, then something very strange is happening with the indexing or the self.n_history value within the Inference object instance.

"""
