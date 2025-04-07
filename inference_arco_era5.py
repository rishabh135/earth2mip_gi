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
OUTPUT_DIR = f"/scratch/gilbreth/{USERNAME}/fcnv2/ARCO_inference_output_{timestamp}"
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

        Unsqueezes it to the 5D shape (E, n_history+1, C, H, W) required by the TimeLoop.

    Normalization/Perturbation:

        Normalizes the initial 5D state using model_inference.normalize().

        Applies perturbation (if configured) to this initial normalized state.

    TimeLoop Initialization: Calls iterator = model_inference(time=initial_time_dt, x=initial_state_perturbed_norm_5d). This passes the required time and the prepared initial (perturbed, normalized) state x.

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


def run_inference(
    model_inference: time_loop.TimeLoop, # Type hint helps clarity
    initial_state_tensor: torch.Tensor, # Shape (1, C, H, W)
    initial_time_dt: datetime.datetime, # Add initial time argument
    config: dict,
    logger: logging.Logger,
):
    """Runs the autoregressive ensemble forecast using the TimeLoop interface."""

    n_ensemble = config["ensemble_members"]
    simulation_length = config["simulation_length"] # Number of forecast steps *after* t=0
    output_freq = config.get("output_frequency", 1)
    noise_amp = config.get("noise_amplitude", 0.0)
    pert_strategy = config.get("perturbation_strategy", "gaussian")
    n_history = getattr(model_inference, 'n_history', 0) # Get history length from model

    # Ensure device is fetched correctly
    try:
        device = model_inference.device
    except AttributeError:
        # Fallback if .device attribute isn't directly on the object
        logger.warning("Could not get device directly from model_inference.device, using parameter device.")
        if hasattr(model_inference, 'module'):
            device = next(model_inference.module.parameters()).device
        else:
            device = next(model_inference.parameters()).device

    logger.info(f"Starting inference using TimeLoop: {n_ensemble} members, {simulation_length} steps.")
    logger.info(f"Output frequency: Every {output_freq} steps.")
    logger.info(f"Running on device: {device}")
    logger.info(f"Model history steps (n_history): {n_history}")

    # --- Prepare Initial State for TimeLoop ---
    # TimeLoop expects 5D input: (B, T, C, H, W) where T = n_history + 1
    # Input initial_state_tensor is (1, C, H, W)

    if initial_state_tensor.dim() != 4 or initial_state_tensor.shape[0] != 1:
        logger.error(f"Initial state tensor has unexpected shape: {initial_state_tensor.shape}. Expected (1, C, H, W).")
        raise ValueError("Invalid initial state tensor shape for run_inference input")

    # Create ensemble batch (E, C, H, W)
    batch_tensor_4d = initial_state_tensor.repeat(n_ensemble, 1, 1, 1).to(device)
    logger.info(f"Created ensemble batch (4D): {batch_tensor_4d.shape}")

    # Add the time dimension T = n_history + 1
    # For n_history=0, T=1. Shape becomes (E, 1, C, H, W)
    initial_state_5d = batch_tensor_4d.unsqueeze(1)
    logger.info(f"Prepared initial state for TimeLoop (5D): {initial_state_5d.shape}")

    # --- Apply Normalization and Perturbation to the INITIAL state ---
    # Note: The TimeLoop interface assumes it receives the *initial* state
    # and manages normalization/denormalization internally based on its flags/methods.
    # However, perturbation needs to be applied *before* starting the loop.
    # Let's normalize, perturb, and then *denormalize* before passing to the loop,
    # OR modify the TimeLoop interface if it can handle perturbed normalized input.
    # Simpler approach: Normalize, perturb, pass the perturbed normalized state.
    # This requires that the TimeLoop's internal _iterate method correctly handles
    # starting from an already normalized state. (Our previous edit to _iterate assumes this).

    logger.debug("Normalizing initial 5D state...")
    try:
        initial_state_norm_5d = model_inference.normalize(initial_state_5d) # Use the added method
        logger.info("Normalized initial 5D state.")
        logger.debug(f"Normalized initial state shape: {initial_state_norm_5d.shape}, dtype: {initial_state_norm_5d.dtype}")
    except Exception as e:
        logger.error(f"Error during initial state normalization: {e}", exc_info=True)
        raise

    initial_state_perturbed_norm_5d = initial_state_norm_5d.clone()

    if noise_amp > 0 and n_ensemble > 1:
        logger.info(f"Applying perturbation noise to initial normalized state (amplitude: {noise_amp:.4f}, strategy: {pert_strategy})")
        if pert_strategy != "correlated":
             logger.warning(f"Perturbation strategy '{pert_strategy}' requested, but using simple Gaussian noise placeholder.")

        noise = torch.randn_like(initial_state_perturbed_norm_5d) * noise_amp
        logger.debug(f"Generated noise tensor, shape: {noise.shape}, std: {torch.std(noise):.4f}")

        if n_ensemble > 1:
           noise[0, :, :, :, :] = 0 # Ensure member 0 is deterministic
           logger.debug("Set noise for ensemble member 0 to zero.")

        initial_state_perturbed_norm_5d += noise
        logger.info("Applied placeholder Gaussian noise to normalized initial state.")
        if torch.isnan(initial_state_perturbed_norm_5d).any():
            logger.warning("NaNs detected after adding noise to initial state!")
    else:
        logger.info("No perturbation noise applied to initial state.")


    # --- Run the TimeLoop Iterator ---
    output_tensors_denorm = []
    inference_times = []
    logger.info(f"Initializing TimeLoop iterator starting from {initial_time_dt.isoformat()}")

    try:
        # Initialize the iterator. Pass the perturbed NORMALIZED state.
        # The `_iterate` method we modified assumes normalized input.
        iterator = model_inference(time=initial_time_dt, x=initial_state_perturbed_norm_5d)

        # The iterator yields the state *at* the given time step (including t=0)
        # We want `simulation_length` steps *after* t=0.
        # So, we need to iterate `simulation_length + 1` times.
        num_steps_to_iterate = simulation_length + 1
        logger.info(f"Iterating {num_steps_to_iterate} times over the TimeLoop iterator...")

        step_counter = 0 # Steps taken by the model (0, 1, 2, ...)
        saved_outputs = 0

        start_time = time.time()
        for i, (time_out, data_denorm, restart_state) in enumerate(iterator):
            current_time = time.time()
            step_time = current_time - start_time
            start_time = current_time # Reset timer for next step

            logger.debug(f"Iterator step {i}: Time = {time_out.isoformat()}, Output shape = {data_denorm.shape}, Step time = {step_time:.3f}s")

            # Store output based on frequency (relative to model steps taken)
            # Step 0 = initial state
            if step_counter % output_freq == 0:
                logger.debug(f"Saving output for model step {step_counter} (iterator step {i})")
                output_tensors_denorm.append(data_denorm.cpu()) # Store denormalized output on CPU
                saved_outputs += 1

            # Check if we have enough steps
            if i >= simulation_length: # Iterated simulation_length+1 times (0 to simulation_length)
                logger.info(f"Reached target simulation length ({simulation_length} steps after t=0).")
                break

            step_counter += 1 # Increment model step counter *after* processing the output for that step
            inference_times.append(step_time)

        logger.info(f"Finished iterating. Collected {len(output_tensors_denorm)} output tensors.")
        if step_counter < simulation_length:
             logger.warning(f"Iterator stopped early after {step_counter} model steps, expected {simulation_length}.")

    except Exception as e:
        logger.error(f"Error occurred while iterating TimeLoop: {e}", exc_info=True)
        return None # Indicate failure

    avg_inference_time = np.mean(inference_times) if inference_times else 0
    logger.info(f"TimeLoop iteration finished. Average step time: {avg_inference_time:.3f} seconds.")

    # --- Combine outputs ---
    if not output_tensors_denorm:
        logger.warning("No output tensors were collected!")
        return None

    try:
        logger.debug(f"Stacking {len(output_tensors_denorm)} collected output tensors...")
        # Each tensor in output_tensors_denorm should be (E, C, H, W)
        # We stack along dim=1 to create the time dimension T_out
        final_output_tensor = torch.stack(output_tensors_denorm, dim=1) # Shape: (E, T_out, C, H, W)
        logger.info(f"Final aggregated output tensor shape: {final_output_tensor.shape}")
        if torch.isnan(final_output_tensor).any():
             logger.warning("NaNs detected in the final aggregated output tensor!")
    except Exception as e:
        logger.error(f"Failed to stack output tensors: {e}", exc_info=True)
        return None

    return final_output_tensor









"""

xplanation and Usage Notes:

    Modular Design:

        The core logic is now centered around iterating the TimeLoop.

        Normalization and perturbation happen before the loop starts.

        Output handling (intermediate saving or full collection) is clearly separated.

        A dedicated save_output_steps function handles the specialized saving logic.

    Ensemble Creation:

        Ensembles are created for each IC at the beginning of run_inference using initial_state_tensor.repeat(n_ensemble, ...).

        Perturbation (noise) is added to this ensemble batch (excluding member 0).

        The TimeLoop processes this entire ensemble batch in parallel on the GPU.

    Simulation Length:

        Controlled by config['simulation_length']. This is the number of steps after the initial time (t=0).

        The TimeLoop iterator is run simulation_length + 1 times.

        How deep can you go? This depends on:

            Model Stability: Numerical errors accumulate. Models might become unstable after a certain number of steps (days/weeks). FCNv2 is generally stable for medium-range forecasts (e.g., 10-15 days, which is 40-60 steps). Longer S2S forecasts might show drift.

            Compute Time: Each step takes time.

            Memory (if collecting full history): Storing the entire history uses E * T_out * C * H * W * 4 bytes.

            Memory (with intermediate saving): Memory usage is dominated by the current state tensor (E, T, C, H, W) on the GPU during the loop, plus the small CPU buffer (buffer_size steps). This is much lower.

    Performance & Memory:

        Using TimeLoop: This is generally efficient as it avoids manual state copying between CPU/GPU within the loop.

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

This revised structure provides a robust way to run the inference using the intended TimeLoop pattern and offers flexibility in how you handle the output for memory efficiency.


"""










# --- Save Output Function (adapted from main_inference) ---
# [ ... save_output function remains the same as in your previous code ... ]
# Ensure logger calls within save_output use the passed 'logger' object.
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











# --- Main Pipeline Function ---
def main(args):
    """Main pipeline execution function."""

    logger.info("========================================================")
    logger.info(" Starting FCNv2-SM Inference Pipeline from NumPy ICs")
    logger.info("========================================================")
    logger.info(f"Full command line arguments: {sys.argv}")
    logger.info(f"Parsed arguments: {vars(args)}")
    logger.info(f"Effective MODEL_REGISTRY: {os.environ.get('MODEL_REGISTRY', 'Not Set')}")

    # --- Environment and Setup ---
    if args.gpu >= 0 and torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(device)
            logger.info(f"Attempting to use GPU: {args.gpu} ({torch.cuda.get_device_name(device)})")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch version: {torch.__version__}")
        except Exception as e:
            logger.error(f"Failed to set CUDA device {args.gpu}: {e}. Falling back to CPU.", exc_info=True)
            device = torch.device("cpu")
            logger.info("Using CPU.")
    else:
        device = torch.device("cpu")
        if args.gpu >= 0:
            logger.warning(f"GPU {args.gpu} requested, but CUDA not available. Using CPU.")
        else:
            logger.info("Using CPU.")

    # --- Load Model ---
    model_id = "fcnv2_sm"  # Use the small version
    logger.info(f"Loading {model_id} model...")
    try:
        logger.info(f"Fetching model package for '{model_id}' from registry: {os.environ.get('MODEL_REGISTRY')}")
        package = registry.get_model(model_id)
        if package is None:
            logger.error(f"Failed to get model package for '{model_id}'. Check registry path and model name.")
            sys.exit(1)
        logger.info(f"Found model package: {package}. Root path: {package.root}")

        # Add logging inside the load function if possible (by editing earth2mip source)
        # or log parameters being passed here
        logger.info(f"Calling {model_id}_load with package root: {package.root}, device: {device}, pretrained: True")
        model_inference = fcnv2_sm_load(package, device=device, pretrained=True)
        model_inference.eval()  # Set model to evaluation mode

        # Verification after loading
        logger.info(f"{model_id} model loaded successfully to device: {next(model_inference.parameters()).device}.")
        logger.info(f"Model expects {len(model_inference.in_channel_names)} input channels.")
        logger.debug(f"Model input channels: {model_inference.in_channel_names}")
        # logger.info(f"Model output channels: {model_inference.out_channel_names}") # Usually same as input
        logger.info(f"Model grid: {model_inference.grid}")
        logger.info(f"Model time step: {model_inference.time_step}")

    except FileNotFoundError as e:
        logger.error(f"Model loading failed: Required file not found - {e}", exc_info=True)
        logger.error(f"Please check that weights.tar, global_means.npy, global_stds.npy exist within {os.path.join(os.environ.get('MODEL_REGISTRY'), model_id)}")
        sys.exit(1)
    except _pickle.UnpicklingError as e:
        logger.error(f"Model loading failed due to UnpicklingError: {e}", exc_info=False)  # Don't need full traceback again
        logger.error("This usually means torch.load failed with weights_only=True (default in PyTorch >= 2.6).")
        logger.error(f"Ensure you have modified '{EARTH2MIP_PATH}/earth2mip/networks/fcnv2_sm.py' to use 'torch.load(..., weights_only=False)'.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Initial Conditions from NumPy file ---
    logger.info(f"Loading initial conditions from: {args.ic_file_path}")
    if not os.path.exists(args.ic_file_path):
        logger.error(f"Initial condition file not found: {args.ic_file_path}")
        sys.exit(1)
    try:
        initial_conditions_np = np.load(args.ic_file_path)
        logger.info(f"Loaded NumPy data with shape: {initial_conditions_np.shape}, dtype: {initial_conditions_np.dtype}")
        # Expected shape: (num_times, num_channels, height, width)
        if initial_conditions_np.ndim != 4:
            raise ValueError(f"Expected 4 dimensions (time, channel, lat, lon), but got {initial_conditions_np.ndim}")

        num_ics, num_channels, height, width = initial_conditions_np.shape
        logger.info(f"Found {num_ics} initial conditions in the file. Grid size: {height}x{width}")

        # Validate channel count
        model_channels = model_inference.in_channel_names
        if num_channels != len(model_channels):
            logger.error(f"Channel mismatch! Model expects {len(model_channels)} channels, but NumPy file has {num_channels} channels.")
            logger.error(f"Model channels: {model_channels}")
            logger.error("Please ensure the NumPy file was created with the correct channels in the expected order.")
            sys.exit(1)
        else:
            logger.info("Channel count matches model requirements.")

        # Validate grid size (optional but good practice)
        model_lat, model_lon = model_inference.grid.lat, model_inference.grid.lon
        if height != len(model_lat) or width != len(model_lon):
            logger.warning(f"Grid mismatch! Model grid is {len(model_lat)}x{len(model_lon)}, but NumPy file grid is {height}x{width}.")
            logger.warning("Ensure the NumPy file represents data on the model's native grid.")
            # Decide if this is critical - for now, just warn.
            # sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to load or validate NumPy file: {e}", exc_info=True)
        sys.exit(1)

    # --- Define Timestamps for the loaded ICs ---
    # Ensure this matches the actual content and order of your .npy file
    try:
        # Example: Infer base date from filename if possible (adjust logic as needed)
        fname = os.path.basename(args.ic_file_path)
        # Simple parsing assuming format like '..._YYYY_MM_DD...' or '..._YYYYMMDD...'
        import re

        match = re.search(r"(\d{4})_(\d{2})_(\d{2})", fname) or re.search(r"(\d{8})", fname)
        if match:
            if len(match.groups()) == 3:
                year, month, day = map(int, match.groups())
            else:  # Match YYYYMMDD
                date_str = match.group(1)
                year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
            base_date = datetime.datetime(year, month, day, tzinfo=pytz.utc)  # Assume UTC if not specified
            logger.info(f"Inferred base date from filename: {base_date.strftime('%Y-%m-%d')}")
        else:
            # Fallback to hardcoded date if filename parsing fails
            base_date = datetime.datetime(2018, 1, 14, tzinfo=pytz.utc)  # Make timezone aware (UTC is standard for IFS/ERA)
            logger.warning(f"Could not infer date from filename, using default: {base_date.strftime('%Y-%m-%d')}")

        # Generate timestamps assuming 6-hourly intervals starting at 00Z
        ic_timestamps = [base_date + datetime.timedelta(hours=i * 6) for i in range(num_ics)]

    except Exception as e:
        logger.error(f"Error determining timestamps for ICs: {e}. Using generic indices.", exc_info=True)
        ic_timestamps = list(range(num_ics))  # Fallback

    logger.info(f"Using the following timestamps/indices for the {num_ics} loaded ICs:")
    for i, ts in enumerate(ic_timestamps):
        if isinstance(ts, datetime.datetime):
            logger.info(f"- IC {i}: {ts.isoformat()}")
        else:
            logger.info(f"- IC {i}: Index {ts}")

    # --- Prepare Inference Configuration ---
    inference_config = {
        "ensemble_members": args.ensemble_members,
        "noise_amplitude": args.noise_amplitude,
        "simulation_length": args.simulation_length,
        "output_frequency": args.output_frequency,
        "weather_model": model_id,  # Use the actual model ID
        "perturbation_strategy": args.perturbation_strategy,
    }
    logger.info(f"Inference Configuration: {inference_config}")










    # --- Run Inference for each Initial Condition ---
    # ... (setup output_dir etc) ...
    num_ics_processed = 0
    num_ics_failed = 0
    total_start_time = time.time()

    for i, initial_time in enumerate(ic_timestamps): # initial_time is now datetime or index
        # Ensure initial_time is a datetime object
        if not isinstance(initial_time, datetime.datetime):
             logger.error(f"IC timestamp for index {i} is not a datetime object ({type(initial_time)}). Cannot proceed with this IC.")
             num_ics_failed += 1
             continue # Skip to the next IC
        # Ensure timezone-aware (UTC is standard for weather data)
        if initial_time.tzinfo is None or initial_time.tzinfo.utcoffset(initial_time) is None:
             logger.warning(f"Initial time {initial_time.isoformat()} is timezone naive. Assuming UTC.")
             initial_time = initial_time.replace(tzinfo=pytz.utc) # Or use datetime.timezone.utc if pytz isn't used

        time_label = initial_time.isoformat()
        logger.info(f"--- Processing Initial Condition {i+1}/{num_ics}: {time_label} ---")

        # Select the i-th initial condition (1, C, H, W) - already done before loop
        ic_data_np = initial_conditions_np[i]
        try:
            initial_state_tensor = torch.from_numpy(ic_data_np).unsqueeze(0).float()
            logger.debug(f"Prepared initial state tensor (1, C, H, W): {initial_state_tensor.shape}")
        except Exception as e:
            logger.error(f"Failed to convert NumPy slice {i} to tensor: {e}", exc_info=True)
            num_ics_failed += 1
            continue

        # Run the forecast using the NEW run_inference signature
        start_run = time.time()
        output_tensor = run_inference(
            model_inference=model_inference,
            initial_state_tensor=initial_state_tensor, # Pass the (1, C, H, W) tensor
            initial_time_dt=initial_time,            # <-- Pass the datetime object
            config=inference_config,
            logger=logger
        )
        end_run = time.time()

        # ... (rest of the loop: saving output, clearing cache, etc.) ...
        if output_tensor is not None:
            logger.info(f"Inference run for IC {time_label} completed in {end_run - start_run:.2f} seconds.")
            save_output(
                output_tensor=output_tensor,
                initial_time=initial_time, # Pass datetime for saving
                time_step=model_inference.time_step,
                channels=model_inference.in_channel_names,
                lat=model_inference.grid.lat,
                lon=model_inference.grid.lon,
                config=inference_config,
                output_dir=netcdf_output_dir,
                logger=logger
            )
            num_ics_processed += 1
        else:
            logger.error(f"Inference failed for IC {time_label}. No output generated.")
            num_ics_failed += 1

        if device.type == 'cuda':
             torch.cuda.empty_cache()
             logger.debug("Cleared CUDA cache.")


    # --- Final Summary ---
    logger.info("--- Inference Loop Finished ---")
    logger.info(f"Successfully processed {num_ics_processed} initial conditions.")
    if num_ics_failed > 0:
        logger.warning(f"Failed to process {num_ics_failed} initial conditions.")
    logger.info(f"Output NetCDF files saved in: {netcdf_output_dir}")
    logger.info(f"Log file saved in: {LOG_DIR}")
    logger.info("========================================================")
    logger.info(" FCNv2-SM Inference Pipeline Finished ")
    logger.info("========================================================")


















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCNv2-SM Inference Pipeline using initial conditions from a NumPy file.")

    # Input/Output paths
    parser.add_argument("--ic-file-path", type=str, default=f"/scratch/gilbreth/{USERNAME}/fcnv2/ARCO_data_73_channels/arco_data/arco_data_start_14_January_2018_end_14_January_2018_14_January_2018.npy", help="Path to the NumPy file containing initial conditions (shape: T, C, H, W).")
    # Default output path uses the dynamically generated OUTPUT_DIR
    parser.add_argument("-o", "--output-path", type=str, default=os.path.join(OUTPUT_DIR, "saved_netcdf"), help="Directory to save output NetCDF files.")

    # Inference parameters
    parser.add_argument("-sim", "--simulation-length", type=int, default=10, help="Number of autoregressive steps (forecast lead time in model steps).")
    parser.add_argument("-ef", "--output-frequency", type=int, default=1, help="Frequency (in steps) to save output states (e.g., 1 = save every step).")
    parser.add_argument("-ens", "--ensemble-members", type=int, default=4, help="Number of ensemble members (>=1).")
    parser.add_argument("-na", "--noise-amplitude", type=float, default=0.05, help="Amplitude for perturbation noise (if ensemble_members > 1). Set to 0 for no noise.")
    parser.add_argument("-ps", "--perturbation-strategy", type=str, default="gaussian", choices=["gaussian", "correlated", "none"], help="Perturbation strategy (currently uses Gaussian placeholder).")  # Note: 'correlated' uses gaussian placeholder here

    # System parameters
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (-1 for CPU).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    # Ensure the main output directory exists before potentially setting logger level
    os.makedirs(args.output_path, exist_ok=True)

    # Adjust logger level if debug flag is set
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Propagate debug level to handlers if needed
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled.")

    try:
        main(args)
    except Exception as e:
        # Use the configured logger if available
        try:
            logger.critical(f"Critical pipeline failure: {str(e)}", exc_info=True)
        except NameError:  # logger might not be defined if setup failed early
            logging.critical(f"Critical pipeline failure before logger setup: {str(e)}", exc_info=True)  # Fallback basic logging
        sys.exit(1)
