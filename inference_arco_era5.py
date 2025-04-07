import numpy as np
import datetime
import os
import logging
import argparse
import time
import sys
import torch
import xarray as xr

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
import datetime
from datetime import datetime, timedelta
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
from dateutil.relativedelta import relativedelta # More robust time stepping



# --- Configuration ---
# Script paths and environment (adjust as needed)
USERNAME = "gupt1075"
BASE_SCRATCH_PATH = f"/scratch/gilbreth/{USERNAME}/fcnv2/ARCO_data_73_channels"
EARTH2MIP_PATH = "/scratch/gilbreth/gupt1075/fcnv2/earth2mip"


# --- 1. Environment Setup ---
# Set number of GPUs to use (adjust if using multi-GPU inference later)
# For this script focusing on single GPU logic from run_inference, WORLD_SIZE=1 is appropriate.
os.environ["WORLD_SIZE"] = "1"


# Set model registry as a local folder (modify path if needed)
# script_dir = os.path.dirname(os.path.realpath(__file__)) if "__file__" in locals() else os.getcwd()
# model_registry = os.path.join(script_dir, "models")
# os.makedirs(model_registry, exist_ok=True)
# os.environ["MODEL_REGISTRY"] = model_registry
# logger.info(f"MODEL_REGISTRY set to: {model_registry}")


# package = registry.get_model("fcnv2")












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


#create timstamp for current time using pytz in pacific timezone
timestamp = datetime.now(timezone(timedelta(hours=-7))).strftime("%d_%B_%H_%M")
# create os.makedirs for output_dir if not exists
OUTPUT_DIR = f"/scratch/gilbreth/gupt1075/fcnv2/ARCO_inference_output_{timestamp}"
os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
logger = setup_logging(OUTPUT_DIR)




# Add earth2mip to Python path
if EARTH2MIP_PATH not in sys.path:
    sys.path.append(EARTH2MIP_PATH)
    logger.info(f"Added {EARTH2MIP_PATH} to Python path.")






# --- Earth-2 MIP Imports (after setting env vars) ---
# try:
#     from earth2mip import registry
#     from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
#     # We are not using cds or inference_ensemble.run_basic_inference anymore
# except ImportError as e:
#     logger.info(f"Error importing earth2mip components: {e}")
#     logger.info("Please ensure earth2mip is installed correctly.")
#     sys.exit(1)



from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
# Add inference_ensemble specific imports
from typing import Any, Optional
from earth2mip import initial_conditions, regrid, time_loop
from earth2mip.ensemble_utils import (
    generate_bred_vector,
    generate_noise_correlated,
    generate_noise_grf
)
from earth2mip.schema import EnsembleRun, PerturbationStrategy
from earth2mip.time_loop import TimeLoop
from modulus.distributed.manager import DistributedManager






dotenv.load_dotenv()

# With the enviroment variables set now we import Earth-2 MIP
from earth2mip import inference_ensemble, registry
from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load

logging.warning("Fetching model package...")


package = registry.get_model("fcnv2")




# --- Core Inference Function (adapted from main_inference) ---
def run_inference(model_inference, initial_state_tensor, config, logger):
    """Runs the autoregressive ensemble forecast for a single initial condition."""
    
    n_ensemble = config['ensemble_members']
    simulation_length = config['simulation_length']
    device = next(model_inference.parameters()).device # Get device from model
    output_freq = config.get('output_frequency', 1)
    
    logger.info(f"Starting inference: {n_ensemble} members, {simulation_length} steps.")
    logger.info(f"Output frequency: Every {output_freq} steps.")
    logger.info(f"Running on device: {device}")

    # Ensure initial_state_tensor is on the correct device and has batch dim 1
    # Shape expected: (1, C, H, W)
    if initial_state_tensor.dim() != 4 or initial_state_tensor.shape[0] != 1:
        logger.error(f"Initial state tensor has unexpected shape: {initial_state_tensor.shape}. Expected (1, C, H, W).")
        raise ValueError("Invalid initial state tensor shape")
    initial_state_tensor = initial_state_tensor.to(device)
    logger.debug(f"Initial state tensor shape (on device): {initial_state_tensor.shape}")

    # Create ensemble batch by repeating initial state
    # Shape: (E, C, H, W)
    batch_tensor = initial_state_tensor.repeat(n_ensemble, 1, 1, 1)
    logger.info(f"Created ensemble batch shape: {batch_tensor.shape}")

    # --- Normalization and Perturbation ---
    # Normalize the entire batch first
    try:
        batch_tensor_normalized = model_inference.normalize(batch_tensor)
        logger.info("Normalized initial ensemble batch.")
        logger.debug(f"Normalized batch tensor shape: {batch_tensor_normalized.shape}")
    except Exception as e:
        logger.error(f"Error during normalization: {e}", exc_info=True)
        raise

    batch_tensor_perturbed_normalized = batch_tensor_normalized.clone() # Start with normalized state

    if config['noise_amplitude'] > 0 and n_ensemble > 1:
         pert_strategy = config['perturbation_strategy']
         noise_amp = config['noise_amplitude']
         logger.info(f"Applying perturbation noise (amplitude: {noise_amp:.4f}). Strategy: {pert_strategy}")
         
         # Placeholder: Gaussian noise applied to normalized data
         # Note: 'correlated' strategy requires more sophisticated methods, potentially
         # using earth2mip.perturbation if available and suitable for fcnv2_sm.
         # This implementation uses simple Gaussian noise as in the example.
         if pert_strategy != "correlated":
             logger.warning(f"Perturbation strategy '{pert_strategy}' requested, but using simple Gaussian noise placeholder.")
         else:
             logger.warning("Using simplified Gaussian noise placeholder for 'correlated' strategy.")

         # Generate noise (ensure std deviation matches amplitude definition if needed)
         # Assuming noise_amplitude directly scales std dev of standard normal noise
         noise = torch.randn_like(batch_tensor_normalized) * noise_amp
         
         # Ensure first member is deterministic (no noise)
         noise[0, :, :, :] = 0
         
         batch_tensor_perturbed_normalized += noise
         logger.info("Applied placeholder Gaussian noise to normalized state (excluding member 0).")
         
         # Optional: Clamp values if normalization created bounds, though often not needed for Gaussian noise
         # batch_tensor_perturbed_normalized = torch.clamp(batch_tensor_perturbed_normalized, min_val, max_val)

    else:
        logger.info("No perturbation noise applied (amplitude is 0 or ensemble size is 1).")

    # --- Autoregressive Loop ---
    output_tensors_denorm = [] # Store denormalized outputs on CPU
    current_state_normalized = batch_tensor_perturbed_normalized # Shape (E, C, H, W)

    # Store initial state (t=0) - denormalized
    try:
        initial_state_denormalized = model_inference.denormalize(current_state_normalized)
        if 0 % output_freq == 0:
            logger.debug("Saving initial state (t=0).")
            output_tensors_denorm.append(initial_state_denormalized.cpu()) # Store on CPU
    except Exception as e:
        logger.error(f"Error during initial state denormalization: {e}", exc_info=True)
        raise

    logger.info(f"Model time step: {model_inference.time_step}")
    logger.info("Autoregressive loop starting...")

    inference_times = []
    with torch.no_grad(): # Essential for inference
        for step in range(simulation_length):
            step_num = step + 1
            start_time = time.time()
            logger.debug(f"Step {step_num}/{simulation_length} - Input shape: {current_state_normalized.shape}")

            # Model prediction (expects normalized input, outputs normalized prediction)
            try:
                next_state_normalized = model_inference(current_state_normalized)
                logger.debug(f"Step {step_num}/{simulation_length} - Raw model output shape: {next_state_normalized.shape}")
                
                # Simple check for NaN/Inf in model output
                if torch.isnan(next_state_normalized).any() or torch.isinf(next_state_normalized).any():
                    logger.error(f"NaN or Inf detected in model output at step {step_num}. Aborting.")
                    return None # Indicate failure

            except Exception as e:
                logger.error(f"Error during model forward pass at step {step_num}: {e}", exc_info=True)
                return None # Indicate failure

            # Denormalize for saving
            try:
                output_denormalized = model_inference.denormalize(next_state_normalized)
            except Exception as e:
                logger.error(f"Error during denormalization at step {step_num}: {e}", exc_info=True)
                # Decide whether to continue or abort. Let's try to continue but log the error.
                output_denormalized = next_state_normalized.clone() # Save normalized if denorm fails
                logger.warning(f"Saving normalized output for step {step_num} due to denormalization error.")


            # Store output based on frequency
            if step_num % output_freq == 0:
                logger.debug(f"Saving output for step {step_num}")
                output_tensors_denorm.append(output_denormalized.cpu()) # Store on CPU to save GPU memory

            # Update state for next iteration
            current_state_normalized = next_state_normalized

            end_time = time.time()
            step_time = end_time - start_time
            inference_times.append(step_time)
            logger.debug(f"Step {step_num} completed in {step_time:.3f} seconds.")

    avg_inference_time = np.mean(inference_times) if inference_times else 0
    logger.info(f"Autoregressive loop finished. Average step time: {avg_inference_time:.3f} seconds.")

    # Combine outputs
    if not output_tensors_denorm:
        logger.warning("No output tensors were saved!")
        return None

    # Concatenate along a new 'time' dimension
    # Each tensor in output_tensors_denorm has shape (E, C, H, W)
    try:
        final_output_tensor = torch.stack(output_tensors_denorm, dim=1) # Shape: (E, T_out, C, H, W)
        logger.info(f"Final aggregated output tensor shape: {final_output_tensor.shape}")
    except Exception as e:
        logger.error(f"Failed to stack output tensors: {e}", exc_info=True)
        return None

    return final_output_tensor


# --- Save Output Function (adapted from main_inference) ---
def save_output(output_tensor, initial_time, time_step, channels, lat, lon, config, output_dir, logger):
    """Saves the forecast output tensor to a NetCDF file."""

    if output_tensor is None:
        logger.error("Cannot save output, tensor is None.")
        return

    # output_tensor shape: (E, T_out, C, H, W)
    n_ensemble, n_time_out, n_channels, n_lat, n_lon = output_tensor.shape
    output_freq = config.get('output_frequency', 1)
    
    logger.info("Preparing output for saving...")
    logger.debug(f"Output tensor shape: {output_tensor.shape}")
    logger.debug(f"Number of channels: {n_channels}, Expected channels: {len(channels)}")
    logger.debug(f"Grid Lat shape: {lat.shape}, Lon shape: {lon.shape}")
    
    if n_channels != len(channels):
        logger.error(f"Mismatch between channels in output tensor ({n_channels}) and provided channel names ({len(channels)}).")
        # Attempt to save anyway, but channels dimension might be incorrect
        channels_coord = np.arange(n_channels) # Use generic index if names mismatch
    else:
         channels_coord = channels

    # Create time coordinates
    # Use relativedelta for robustness with time steps (e.g., hours)
    try:
        # Convert model time_step (timedelta) to relativedelta if possible, or use seconds
        if isinstance(time_step, datetime.timedelta):
            dt_step = relativedelta(seconds=time_step.total_seconds())
        else:
            logger.warning(f"Unexpected time_step type: {type(time_step)}. Assuming it represents hours.")
            dt_step = relativedelta(hours=time_step) # Adapt if time_step is different

        time_coords = [initial_time + i * dt_step * output_freq for i in range(n_time_out)]
        logger.debug(f"Generated {len(time_coords)} time coordinates starting from {initial_time.isoformat()}.")
    except Exception as e:
        logger.error(f"Failed to create time coordinates: {e}", exc_info=True)
        # Fallback to simple integer index for time
        time_coords = np.arange(n_time_out)


    # Create DataArray
    try:
        # Ensure lat/lon are numpy arrays
        lat_np = lat if isinstance(lat, np.ndarray) else lat.cpu().numpy()
        lon_np = lon if isinstance(lon, np.ndarray) else lon.cpu().numpy()

        forecast_da = xr.DataArray(
            output_tensor.numpy(), # Convert tensor to numpy array
            coords={
                'ensemble': np.arange(n_ensemble),
                'time': time_coords,
                'channel': channels_coord,
                'lat': lat_np,
                'lon': lon_np,
            },
            dims=['ensemble', 'time', 'channel', 'lat', 'lon'],
            name='forecast',
            attrs={
                'description': 'FCNv2-SM ensemble forecast output',
                'model': config['weather_model'],
                'simulation_length_steps': config['simulation_length'],
                'output_frequency_steps': output_freq,
                'ensemble_members': n_ensemble,
                'initial_condition_time': initial_time.isoformat(),
                'noise_amplitude': config['noise_amplitude'],
                'perturbation_strategy': config['perturbation_strategy'],
                'creation_date': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        )
        logger.info("Created xarray DataArray.")

        # Convert to Dataset with each channel as a variable for better compatibility
        forecast_ds = forecast_da.to_dataset(dim='channel')
        logger.info("Converted DataArray to Dataset (channels as variables).")

    except Exception as e:
        logger.error(f"Failed to create xarray Dataset: {e}", exc_info=True)
        return

    # Define output filename
    ic_time_str = initial_time.strftime('%Y%m%d_%H%M%S')
    output_filename = os.path.join(output_dir, f"fcnv2_sm_ensemble_{ic_time_str}.nc")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving forecast output to: {output_filename}")
    try:
        # Define encoding for compression (optional but recommended)
        encoding = {var: {'zlib': True, 'complevel': 5} for var in forecast_ds.data_vars}
        
        # Specify unlimited dimension for time if needed (good practice)
        # forecast_ds.encoding['unlimited_dims'] = {'time'}

        start_save = time.time()
        forecast_ds.to_netcdf(output_filename, encoding=encoding)
        end_save = time.time()
        logger.info(f"Save complete. Time taken: {end_save - start_save:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed to save output NetCDF file: {e}", exc_info=True)


# --- Main Pipeline Function ---
def main(args):
    """Main pipeline execution function."""
    
    # Setup logging (use output dir for logs)
    # log_dir = os.path.join(args.output_path, "logs")
    # logger = setup_logging(log_dir, log_level=logging.INFO if not args.debug else logging.DEBUG)

    logger.info("========================================================")
    logger.info(" Starting FCNv2-SM Inference Pipeline from NumPy ICs")
    logger.info("========================================================")
    logger.info(f"Run arguments: {vars(args)}")

    # --- Environment and Setup ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")

    # --- Load Model ---
    logger.info("Loading FCNv2-SM model...")
    try:
        package = registry.get_model("fcnv2_sm")
        logger.info(f"Found model package: {package}")
        # Load model onto the specified device
        model_inference = fcnv2_sm_load(package, device=device, pretrained=True)
        model_inference.eval() # Set model to evaluation mode
        logger.info("FCNv2-SM model loaded successfully.")
        logger.info(f"Model expects {len(model_inference.in_channel_names)} input channels.")
        logger.debug(f"Model input channels: {model_inference.in_channel_names}")
        logger.info(f"Model output channels: {model_inference.out_channel_names}") # Usually same as input for weather models
        logger.info(f"Model grid: {model_inference.grid}")
        logger.info(f"Model time step: {model_inference.time_step}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        logger.error("Ensure the model 'fcnv2_sm' is correctly registered or downloadable.")
        logger.error(f"Checked registry path: {os.environ.get('MODEL_REGISTRY', 'Not set')}")
        sys.exit(1)

    # --- Load Initial Conditions from NumPy file ---
    logger.info(f"Loading initial conditions from: {args.ic_file_path}")
    try:
        initial_conditions_np = np.load(args.ic_file_path)
        logger.info(f"Loaded NumPy data with shape: {initial_conditions_np.shape}")
        # Expected shape: (num_times, num_channels, height, width)
        if initial_conditions_np.ndim != 4:
            raise ValueError(f"Expected 4 dimensions (time, channel, lat, lon), but got {initial_conditions_np.ndim}")
        
        num_ics, num_channels, _, _ = initial_conditions_np.shape
        logger.info(f"Found {num_ics} initial conditions in the file.")

        # Validate channel count
        if num_channels != len(model_inference.in_channel_names):
            logger.error(f"Channel mismatch! Model expects {len(model_inference.in_channel_names)} channels, "
                         f"but NumPy file has {num_channels} channels.")
            logger.error("Please ensure the NumPy file was created with the correct channels in the expected order.")
            # Decide whether to exit or try to continue (exiting is safer)
            sys.exit(1)
            # logger.warning("Attempting to proceed despite channel mismatch...") # Alternative if you want to risk it
        else:
            logger.info("Channel count matches model requirements.")

    except FileNotFoundError:
        logger.error(f"Initial condition file not found: {args.ic_file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load or validate NumPy file: {e}", exc_info=True)
        sys.exit(1)

    # --- Define Timestamps for the loaded ICs ---
    # Based on the prompt: 14 Jan 2018, 00Z, 06Z, 12Z, 18Z
    # Ensure this matches the actual content of your .npy file order
    base_date = datetime.datetime(2018, 1, 14)
    ic_timestamps = [
        base_date.replace(hour=0, minute=0, second=0, microsecond=0),
        base_date.replace(hour=6, minute=0, second=0, microsecond=0),
        base_date.replace(hour=12, minute=0, second=0, microsecond=0),
        base_date.replace(hour=18, minute=0, second=0, microsecond=0),
    ]
    if len(ic_timestamps) != num_ics:
        logger.error(f"Mismatch between expected number of ICs (4 based on prompt) and loaded ICs ({num_ics}).")
        logger.error("Adjust the 'ic_timestamps' list in the script to match your NumPy file content.")
        # Fallback: Generate generic timestamps if needed, but specific times are better.
        # ic_timestamps = [base_date + datetime.timedelta(hours=i*6) for i in range(num_ics)] # Example fallback
        sys.exit(1)
    logger.info(f"Using the following timestamps for the {num_ics} loaded ICs:")
    for ts in ic_timestamps:
        logger.info(f"- {ts.isoformat()}")

    # --- Prepare Inference Configuration ---
    inference_config = {
        "ensemble_members": args.ensemble_members,
        "noise_amplitude": args.noise_amplitude,
        "simulation_length": args.simulation_length,
        "output_frequency": args.output_frequency,
        "weather_model": "fcnv2_sm", # Model name for metadata
        "perturbation_strategy": args.perturbation_strategy,
    }
    logger.info(f"Inference Configuration: {inference_config}")

    # --- Run Inference for each Initial Condition ---
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir}")

    num_ics_processed = 0
    num_ics_failed = 0

    for i, initial_time in enumerate(ic_timestamps):
        logger.info(f"--- Processing Initial Condition {i+1}/{num_ics}: {initial_time.isoformat()} ---")
        
        # Select the i-th initial condition from the loaded NumPy array
        ic_data_np = initial_conditions_np[i] # Shape: (C, H, W)
        
        # Convert to PyTorch Tensor, add batch dimension, and move to device
        try:
            # Add batch dimension: (1, C, H, W)
            initial_state_tensor = torch.from_numpy(ic_data_np).unsqueeze(0).float()
            logger.debug(f"Prepared initial state tensor from NumPy slice, initial shape: {initial_state_tensor.shape}")
        except Exception as e:
            logger.error(f"Failed to convert NumPy slice to tensor for IC {initial_time}: {e}", exc_info=True)
            num_ics_failed += 1
            continue # Skip to the next IC

        # Run the forecast
        start_run = time.time()
        output_tensor = run_inference(
            model_inference=model_inference,
            initial_state_tensor=initial_state_tensor,
            config=inference_config,
            logger=logger
        )
        end_run = time.time()

        if output_tensor is not None:
            logger.info(f"Inference run for IC {initial_time} completed in {end_run - start_run:.2f} seconds.")
            # Save the output
            save_output(
                output_tensor=output_tensor,
                initial_time=initial_time,
                time_step=model_inference.time_step, # Get timedelta from model
                channels=model_inference.in_channel_names, # Use model's channel names
                lat=model_inference.grid.lat, # Get lat/lon from model grid
                lon=model_inference.grid.lon,
                config=inference_config,
                output_dir=output_dir,
                logger=logger
            )
            num_ics_processed += 1
        else:
            logger.error(f"Inference failed for IC {initial_time}. No output generated.")
            num_ics_failed += 1

        # Optional: Clear CUDA cache if memory is an issue between runs
        if device.type == 'cuda':
             torch.cuda.empty_cache()
             logger.debug("Cleared CUDA cache.")

    # --- Final Summary ---
    logger.info("--- Inference Loop Finished ---")
    logger.info(f"Successfully processed {num_ics_processed} initial conditions.")
    if num_ics_failed > 0:
        logger.warning(f"Failed to process {num_ics_failed} initial conditions.")
    logger.info("========================================================")
    logger.info(" FCNv2-SM Inference Pipeline Finished ")
    logger.info("========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCNv2-SM Inference Pipeline using initial conditions from a NumPy file.")

    # Input/Output paths
    parser.add_argument("--ic-file-path", type=str,
                        default="/scratch/gilbreth/gupt1075/fcnv2/ARCO_data_73_channels/arco_data/arco_data_start_14_January_2018_end_14_January_2018_14_January_2018.npy",
                        help="Path to the NumPy file containing initial conditions (shape: T, C, H, W).")
    parser.add_argument("-o", "--output-path", type=str,
                        default=f"{OUTPUT_DIR}/saved_netcdf/",
                        help="Directory to save output NetCDF files.")

    # Inference parameters
    parser.add_argument("-sim", "--simulation-length", type=int, default=10,
                        help="Number of autoregressive steps.")
    parser.add_argument("-ef", "--output-frequency", type=int, default=1,
                        help="Frequency (in steps) to save output states (e.g., 1 = save every step).")
    parser.add_argument("-ens", "--ensemble-members", type=int, default=4,
                        help="Number of ensemble members (>=1).")
    parser.add_argument("-na", "--noise-amplitude", type=float, default=0.05,
                        help="Amplitude for perturbation noise (if ensemble_members > 1). Set to 0 for no noise.")
    parser.add_argument("-ps", "--perturbation-strategy", type=str, default="gaussian",
                        choices=["gaussian", "correlated", "none"], # Note: 'correlated' uses gaussian placeholder here
                        help="Perturbation strategy (currently uses Gaussian placeholder).")
    
    # System parameters
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (-1 for CPU).")
    parser.add_argument("--debug", action='store_true',
                        help="Enable debug logging.")

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        # Use root logger if setup failed, otherwise use configured logger
        try:
            logger = logging.getLogger("FCNv2Inference")
            if not logger.hasHandlers(): # Check if handlers were added
                 raise RuntimeError("Logging not configured")
            logger.critical(f"Critical pipeline failure: {str(e)}", exc_info=True)
        except Exception:
             logging.critical(f"Critical pipeline failure before logging setup or after failure: {str(e)}", exc_info=True) # Fallback basic logging
        sys.exit(1)
