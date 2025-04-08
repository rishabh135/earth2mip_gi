
# -*- coding: utf-8 -*-
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

import xarray as xr
import os
import time
import logging # Assume logger is passed correctly
import pytz # Assume pytz is imported elsewhere
from typing import List, Dict, Optional, Callable, Any, Tuple # Type hints
import gc # Import garbage collector interface

# Need these imports if not already present at the top
import collections
import pickle
from typing import List, Dict, Optional, Callable, Any, Tuple # For type hints

# --- Plotting Imports ---
try:
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend suitable for scripts
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import Normalize # Use standard Normalize for Z500
    print("Successfully imported Matplotlib and Cartopy.")
    PLOTTING_ENABLED = True
except ImportError as e:
    print(f"Warning: Failed to import Matplotlib or Cartopy ({e}). Plotting functionality will be disabled.")
    print("Please install them ('pip install matplotlib cartopy') to enable plotting.")
    PLOTTING_ENABLED = False
    # Define dummy classes/functions if plotting is disabled to avoid NameErrors later
    class ccrs: PlateCarree = None; LambertConformal = None
    class cfeature: NaturalEarthFeature = None
    class plt: figure = None; rcParams = None; colorbar = None; close = None
    class Normalize: pass


# --- Configuration ---
USERNAME = os.getenv("USER", "gupt1075") # Use env var or default
SCRATCH_DIR = os.getenv("SCRATCH", f"/scratch/gilbreth/{USERNAME}") # Use env var or default
MODEL_REGISTRY_BASE = f"{SCRATCH_DIR}/fcnv2/"
EARTH2MIP_PATH = f"{SCRATCH_DIR}/fcnv2/earth2mip"










# --- Add earth2mip to Python path ---
if EARTH2MIP_PATH not in sys.path:
    sys.path.insert(0, EARTH2MIP_PATH)
    print(f"Added {EARTH2MIP_PATH} to Python path.")

# --- Environment Setup ---
os.environ["WORLD_SIZE"] = "1"
os.environ["MODEL_REGISTRY"] = MODEL_REGISTRY_BASE
print(f"Set MODEL_REGISTRY environment variable to: {MODEL_REGISTRY_BASE}")
# Optional: Try setting CUDA alloc conf for potential fragmentation issues
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


# --- Logging Setup ---
def setup_logging(log_dir, log_level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    # Use UTC for internal consistency, display local time if needed via formatter
    utc_tz = pytz.utc
    try:
        # Try to get local timezone for display purposes
        import tzlocal
        display_tz = tzlocal.get_localzone()
        if not isinstance(display_tz, datetime.tzinfo): # Handle fallback cases
             display_tz = utc_tz
    except ImportError:
        display_tz = utc_tz # Fallback to UTC if tzlocal not installed

    timestamp_str = datetime.datetime.now(display_tz).strftime("%d_%B_%H_%M")
    log_filename = os.path.join(log_dir, f"inference_pipeline_{timestamp_str}.log")

    class PytzFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None, tz=None):
            super().__init__(fmt, datefmt)
            self.tz = tz if tz else pytz.utc # Log time in UTC by default

        def formatTime(self, record, datefmt=None):
            # Create timezone-aware datetime object from record timestamp (UTC)
            dt_utc = datetime.datetime.fromtimestamp(record.created, tz=utc_tz)
            # Convert to desired display timezone
            dt_display = dt_utc.astimezone(self.tz)
            if datefmt:
                s = dt_display.strftime(datefmt)
            else:
                try:
                     s = dt_display.isoformat(timespec='milliseconds')
                except TypeError: # older python versions
                     s = dt_display.strftime('%d_%B_%Y_%H:%M:%S.%f')[:-3] # Milliseconds approx
                     s += dt_display.strftime('%z') # Add timezone offset
            return s

    logger = logging.getLogger("FCNv2Inference")
    logger.setLevel(log_level)
    logger.handlers.clear() # Prevent duplicate logs if run multiple times

    # File Handler (logs in display timezone)
    file_handler = logging.FileHandler(log_filename, mode="w")
    # Example format including timezone name
    file_formatter = PytzFormatter(
        "%(asctime)s [%(levelname)-8s] [%(name)s:%(lineno)d] %(message)s",
        datefmt=None, # Use ISO format with offset
        tz=display_tz
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler (logs in display timezone)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = PytzFormatter( # Simpler format for console
         "%(asctime)s [%(levelname)-8s] %(message)s",
         datefmt='%Y-%m-%d %H:%M:%S %Z', # Example console date format
         tz=display_tz
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Reduce verbosity of libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("cartopy").setLevel(logging.WARNING)


    logger.info(f"Logging configured. Level: {logging.getLevelName(logger.level)}. Display TZ: {display_tz.zone}")
    logger.info(f"Log file: {log_filename}")
    return logger

# --- Determine Output Directory and Setup Logging ---
pacific_tz = pytz.timezone("America/Los_Angeles") # Keep for consistent naming scheme if desired
timestamp = datetime.datetime.now(pacific_tz).strftime("%d_%B_%Y_%H_%M")
OUTPUT_DIR = f"/scratch/gilbreth/gupt1075/fcnv2/RESULTS_2025/ZETA_inference_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
logger = setup_logging(LOG_DIR) # Initialize logger early

logger.info(f"Using Output Directory: {OUTPUT_DIR}")
logger.info(f"Plotting Enabled: {PLOTTING_ENABLED}")

# # --- Load Environment Variables (optional) ---
# dotenv.load_dotenv()
# logger.info("Checked for .env file.")

# --- Earth-2 MIP Imports (AFTER setting env vars and sys.path) ---
try:
    from earth2mip import registry
    from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
    import earth2mip.grid
    from earth2mip import (
        ModelRegistry,
        loaders,
        model_registry,
        registry,
        schema,
        time_loop,
    )
    print("Successfully imported earth2mip components.")
except ImportError as e:
    print(f"Error importing earth2mip components: {e}")
    print("Please ensure earth2mip is installed correctly and EARTH2MIP_PATH is correct.")
    logger.critical(f"Earth2MIP import failed: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during earth2mip import: {e}")
    logger.critical(f"Unexpected Earth2MIP import error: {e}", exc_info=True)
    sys.exit(1)











# --- Add Memory Logging Utility ---
def log_gpu_memory(logger: logging.Logger, point: str = "Point", device: Optional[torch.device] = None):
    """Logs allocated and reserved GPU memory for a given device."""
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device() # Use current device if none specified
        try:
             allocated = torch.cuda.memory_allocated(device) / (1024**3)
             reserved = torch.cuda.memory_reserved(device) / (1024**3)
             # Get total memory
             total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
             free_memory = total_memory - reserved # Free is total minus reserved
             logger.info(
                 f"GPU Memory @ {point} (Device {device}): "
                 f"Allocated={allocated:.2f} GiB, Reserved={reserved:.2f} GiB, "
                 f"Free={free_memory:.2f} GiB, Total={total_memory:.2f} GiB"
             )
             # Optional: Log peak memory since last reset
             # max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
             # max_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
             # logger.debug(f"  Peak Memory: Max Allocated={max_allocated:.2f} GiB, Max Reserved={max_reserved:.2f} GiB")
        except Exception as e:
             logger.error(f"Could not get GPU memory stats for device {device}: {e}")
    else:
        logger.debug(f"GPU Memory @ {point}: CUDA not available.")


# --- Function Definitions (save_output_steps, save_full_output, plot_z500_progression) ---
# Assume save_output_steps, save_full_output (the version with np.datetime64 fix),
# and plot_z500_progression are defined correctly as in previous responses.














# --- **MODIFIED** Main Inference Function using TimeLoop (run_inference) ---
def run_inference(
    model_inference: time_loop.TimeLoop,
    initial_state_tensor: torch.Tensor, # Shape (1, C, H, W) - Single IC, CPU or GPU
    initial_time_dt: datetime.datetime, # Starting datetime object for the IC
    config: dict,                       # Configuration dictionary
    logger: logging.Logger,             # Logger instance
    save_func: Optional[Callable] = None, # Function to call for saving steps (e.g., save_output_steps)
    save_steps_config: Optional[Dict[str, Any]] = None, # Config for saving steps
    output_dir: Optional[str] = None, # Required if save_func is provided
):
    """
    Runs the autoregressive ensemble forecast using the TimeLoop interface.

    Performs INITIAL normalization on CPU to avoid GPU OOM errors.
    The main autoregressive loop (model prediction) runs on the target device (GPU).

    Handles ensemble creation, CPU normalization, perturbation, time stepping via
    the iterator, and optional output saving (intermediate or full history).

    Args:
        model_inference: An instance of earth2mip.networks.Inference (or compatible TimeLoop).
        initial_state_tensor: The initial condition for ONE time step.
                               Shape: (1, C, H, W), on CPU or GPU.
        initial_time_dt: The datetime object corresponding to the initial_state_tensor.
                         Must be timezone-aware (e.g., UTC).
        config: Dictionary containing inference parameters like 'ensemble_members',
                'simulation_length', 'noise_amplitude', etc.
        logger: A configured Python logger instance.
        save_func: A callable function to save output during the loop (intermediate saving).
                   If None, full history is collected.
        save_steps_config: Dictionary configuring which steps to save for intermediate saving.
        output_dir: The directory where save_func should save files.

    Returns:
        Optional[torch.Tensor]: If save_func is None, returns the full forecast history tensor
                                (denormalized, on CPU) with shape (E, T_out, C, H, W).
                                Returns None if intermediate saving is used OR if an error occurs.
    """
    run_start_time = time.time()
    # --- Configuration Extraction ---
    n_ensemble = config.get("ensemble_members", 1)
    simulation_length = config.get("simulation_length", 0) # Num steps AFTER t=0
    output_freq = config.get("output_frequency", 1) # Freq for collecting full history
    noise_amp = config.get("noise_amplitude", 0.0)
    pert_strategy = config.get("perturbation_strategy", "gaussian") # Placeholder

    logger.info(f"Starting inference run for IC: {initial_time_dt.isoformat()}")
    logger.info(f"Ensemble members: {n_ensemble}, Simulation steps: {simulation_length}")
    logger.info(f"Output collection/save frequency: {output_freq}")
    logger.info(f"Perturbation: Amp={noise_amp}, Strategy='{pert_strategy}'")

    # --- Validation ---
    if not isinstance(initial_time_dt, datetime.datetime):
        raise TypeError("initial_time_dt must be a datetime.datetime object.")
    if initial_time_dt.tzinfo is None or initial_time_dt.tzinfo.utcoffset(initial_time_dt) is None:
        logger.warning(f"Initial time {initial_time_dt.isoformat()} is timezone naive. Assuming UTC.")
        initial_time_dt = initial_time_dt.replace(tzinfo=datetime.timezone.utc)

    if initial_state_tensor.dim() != 4 or initial_state_tensor.shape[0] != 1:
        raise ValueError(f"Invalid initial state tensor shape: {initial_state_tensor.shape}. Expected (1, C, H, W).")

    perform_intermediate_saving = callable(save_func)
    if perform_intermediate_saving:
        if not output_dir: raise ValueError("output_dir required for intermediate saving.")
        if not save_steps_config: raise ValueError("save_steps_config required for intermediate saving.")
        logger.info(f"Intermediate saving enabled. Steps relative to current: {save_steps_config.get('steps_to_save', [])}")
    else:
        logger.info(f"Intermediate saving disabled. Collecting full history in memory (output_freq={output_freq}).")

    # --- Get Model Properties and Target Device ---
    try:
        target_device = model_inference.device # The device the model lives on (e.g., GPU)
        n_history = getattr(model_inference, 'n_history', 0)
        time_step_delta = model_inference.time_step
        all_channels = model_inference.in_channel_names
        lat = model_inference.grid.lat
        lon = model_inference.grid.lon
        logger.info(f"Model properties: Target Device={target_device}, n_history={n_history}, time_step={time_step_delta}")
        log_gpu_memory(logger, "Start of run_inference", target_device)
    except AttributeError as e:
        raise AttributeError(f"model_inference object missing required attributes: {e}")

    # --- Variables for state management ---
    initial_state_5d_gpu = None
    initial_state_5d_cpu = None
    initial_state_norm_5d_cpu = None
    initial_state_norm_5d_gpu = None
    initial_state_perturbed_norm_5d_gpu = None
    output_tensor_full = None # For collecting results if not saving intermediates

    try:
        # --- 1. Prepare Initial State Ensemble on Target Device ---
        prep_start = time.time()
        logger.debug("Preparing initial state ensemble on target device...")
        # Ensure input IC tensor is on the correct device first
        initial_state_tensor_dev = initial_state_tensor.to(target_device)
        # Create ensemble batch
        batch_tensor_4d = initial_state_tensor_dev.repeat(n_ensemble, 1, 1, 1)
        del initial_state_tensor_dev # Free memory
        # Add time dimension T = n_history + 1
        initial_state_5d_gpu = batch_tensor_4d.unsqueeze(1)
        del batch_tensor_4d # Free memory
        # Handle history padding if needed (basic repeat shown)
        if n_history > 0 and initial_state_5d_gpu.shape[1] != n_history + 1:
             logger.warning(f"Shape mismatch for history. Expected T={n_history+1}, got {initial_state_5d_gpu.shape[1]}. Assuming n_history=0 or basic repeat.")
             if initial_state_5d_gpu.shape[1] == 1:
                  logger.warning(f"Repeating initial state to match n_history={n_history}.")
                  initial_state_5d_gpu = initial_state_5d_gpu.repeat(1, n_history + 1, 1, 1, 1)
        logger.info(f"  Prepared initial state on {target_device} (E, T, C, H, W): {initial_state_5d_gpu.shape}. Time: {time.time()-prep_start:.2f}s")
        log_gpu_memory(logger, "After initial state prep (GPU)", target_device)

        # --- 2. Normalize Initial State (on CPU) ---
        norm_start = time.time()
        logger.info("Moving initial state to CPU for normalization...")
        initial_state_5d_cpu = initial_state_5d_gpu.cpu()
        logger.debug(f"  Moved state to CPU. Shape: {initial_state_5d_cpu.shape}")

        # *** Crucial: Delete GPU tensor immediately ***
        logger.debug("Deleting initial state from GPU memory...")
        del initial_state_5d_gpu
        initial_state_5d_gpu = None # Ensure reference is gone
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_gpu_memory(logger, "After moving state to CPU", target_device)

        logger.info("Performing normalization on CPU...")
        try:
            # The normalize method should work correctly on CPU tensors if implemented well
            if not hasattr(model_inference, 'normalize') or not callable(model_inference.normalize):
                 raise AttributeError("model_inference missing required 'normalize' method.")
            initial_state_norm_5d_cpu = model_inference.normalize(initial_state_5d_cpu)
            logger.info(f"  Normalization on CPU complete. Shape: {initial_state_norm_5d_cpu.shape}")
            if torch.isnan(initial_state_norm_5d_cpu).any():
                logger.warning("NaNs detected in CPU normalized state!")
        except Exception as e:
            logger.error(f"Error during CPU normalization: {e}", exc_info=True)
            raise # Propagate error

        # *** Delete the unnormalized CPU tensor ***
        logger.debug("Deleting unnormalized CPU state...")
        del initial_state_5d_cpu
        initial_state_5d_cpu = None

        logger.info("Moving normalized state back to target device...")
        initial_state_norm_5d_gpu = initial_state_norm_5d_cpu.to(target_device)
        logger.debug(f"  Moved normalized state to {target_device}. Shape: {initial_state_norm_5d_gpu.shape}")

        # *** Delete the normalized CPU tensor ***
        logger.debug("Deleting normalized CPU state...")
        del initial_state_norm_5d_cpu
        initial_state_norm_5d_cpu = None
        log_gpu_memory(logger, "After normalization (back on GPU)", target_device)
        logger.info(f"Normalization process (GPU->CPU->Norm->GPU) took: {time.time()-norm_start:.2f}s")

        # --- 3. Apply Perturbation (on Target Device) ---
        perturb_start = time.time()
        initial_state_perturbed_norm_5d_gpu = initial_state_norm_5d_gpu # Start with normalized GPU state
        if noise_amp > 0 and n_ensemble > 1:
            logger.info(f"Applying perturbation noise on {target_device} (Amp={noise_amp:.4f})")
            # Add noise directly on the target device
            noise = torch.randn_like(initial_state_norm_5d_gpu) * noise_amp
            if n_ensemble > 1: noise[0, ...] = 0 # Keep member 0 deterministic
            initial_state_perturbed_norm_5d_gpu += noise
            del noise # Free noise tensor memory
            logger.info("  Applied noise to normalized state.")
            if torch.isnan(initial_state_perturbed_norm_5d_gpu).any():
                logger.warning("NaNs detected after adding noise!")
            log_gpu_memory(logger, "After perturbation", target_device)
        else:
            logger.info("No perturbation noise applied.")
        logger.info(f"Perturbation process took: {time.time()-perturb_start:.2f}s")

        # *** Delete the intermediate non-perturbed normalized GPU state ***
        if initial_state_perturbed_norm_5d_gpu is not initial_state_norm_5d_gpu:
             logger.debug("Deleting non-perturbed normalized GPU state...")
             del initial_state_norm_5d_gpu
             initial_state_norm_5d_gpu = None


        # --- 4. Execute TimeLoop Iterator (on Target Device) ---
        loop_start_time = time.time()
        # Prepare storage for results
        output_history_buffer = collections.deque() # For intermediate saving
        output_tensors_full_history = [] # For collecting full history

        inference_times = []
        logger.info(f"Initializing TimeLoop iterator starting from {initial_time_dt.isoformat()} on {target_device}")
        logger.info(f"Target simulation steps: {simulation_length}. Iterator will run {simulation_length + 1} times.")

        model_step_counter = 0
        num_iterations_done = 0
        iterator = None # Initialize

        try:
            # *** Pass the PERTURBED, NORMALIZED state on the TARGET DEVICE ***
            iterator = model_inference(time=initial_time_dt, x=initial_state_perturbed_norm_5d_gpu)

            # Iterate simulation_length + 1 times (for t=0 up to t=simulation_length*time_step)
            for i in range(simulation_length + 1):
                iter_start_time = time.time()
                logger.debug(f"--- Iterator Step {i} (Model Step {model_step_counter}) ---")
                log_gpu_memory(logger, f"Start of iter {i}", target_device)

                # Get next state from the iterator (data_denorm expected on target_device)
                try:
                    time_out, data_denorm_dev, _ = next(iterator) # We don't need restart_state here
                    logger.debug(f"  Iterator yielded: Time={time_out.isoformat()}, Output shape={data_denorm_dev.shape}, Device={data_denorm_dev.device}")
                    num_iterations_done += 1
                except StopIteration:
                    logger.warning(f"Iterator stopped unexpectedly after {num_iterations_done} iterations.")
                    break # Exit loop
                except RuntimeError as e:
                     if "out of memory" in str(e).lower():
                          logger.error(f"CUDA OOM error during model prediction step {i}!", exc_info=True)
                          log_gpu_memory(logger, f"OOM state at iter {i}", target_device)
                     else:
                          logger.error(f"Runtime error during next(iterator) step {i}: {e}", exc_info=True)
                     raise # Re-raise the error to stop the process

                # --- Output Handling ---
                # Move result to CPU immediately to free GPU memory, ESPECIALLY if saving intermediates
                data_denorm_cpu = data_denorm_dev.cpu()
                log_gpu_memory(logger, f"After moving output to CPU iter {i}", target_device)

                if perform_intermediate_saving:
                    # Add current step data to the buffer
                    output_history_buffer.append((model_step_counter, time_out, data_denorm_cpu))
                    steps_to_save = save_steps_config.get('steps_to_save', [0])
                    max_offset = abs(min(steps_to_save)) if steps_to_save else 0
                    buffer_size = max_offset + 1
                    while len(output_history_buffer) > buffer_size:
                        output_history_buffer.popleft() # Remove oldest entry

                    # Check if we can save based on the *current* step and required offsets
                    can_save_all_steps = True
                    steps_to_save_indices = {} # step_index -> tensor
                    times_to_save = {}       # step_index -> datetime
                    for offset in steps_to_save:
                        target_step_index = model_step_counter + offset
                        found = False
                        # Search buffer for the target step index
                        for step_idx, step_time, step_data in output_history_buffer:
                            if step_idx == target_step_index:
                                steps_to_save_indices[target_step_index] = step_data
                                times_to_save[target_step_index] = step_time
                                found = True
                                break
                        if not found:
                            can_save_all_steps = False; break # Don't have history yet

                    # If all required steps are found, trigger the save
                    if can_save_all_steps and steps_to_save_indices:
                        logger.debug(f"  Triggering intermediate save for model step {model_step_counter}...")
                        try:
                            save_func( # Call the provided save function
                                data_dict=steps_to_save_indices, time_dict=times_to_save,
                                channels=all_channels, lat=lat, lon=lon, config=config,
                                output_dir=output_dir, current_model_step=model_step_counter, logger=logger
                            )
                        except Exception as save_e:
                             logger.error(f"Error during intermediate save call for step {model_step_counter}: {save_e}", exc_info=True)
                             # Decide whether to continue or stop

                else: # Collect full history in memory
                    if model_step_counter % output_freq == 0:
                        logger.debug(f"  Collecting output for model step {model_step_counter} on CPU.")
                        output_tensors_full_history.append(data_denorm_cpu)

                # --- Timing and Increment ---
                iter_end_time = time.time()
                step_duration = iter_end_time - iter_start_time
                inference_times.append(step_duration)
                logger.debug(f"  Iterator Step {i} finished in {step_duration:.3f} seconds.")
                model_step_counter += 1

            logger.info(f"Finished {num_iterations_done} iterations over TimeLoop.")
            logger.info(f"TimeLoop execution took: {time.time() - loop_start_time:.2f}s")

        except Exception as e:
            logger.error(f"Error occurred during TimeLoop iteration for IC {initial_time_dt.isoformat()}: {e}", exc_info=True)
            return None # Indicate failure

        finally:
            # Explicitly delete the iterator and the potentially large input tensor
            logger.debug("Cleaning up iterator and initial perturbed state...")
            if iterator is not None: del iterator
            if initial_state_perturbed_norm_5d_gpu is not None:
                 del initial_state_perturbed_norm_5d_gpu
                 initial_state_perturbed_norm_5d_gpu = None
            # Clear cache after the loop finishes
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            log_gpu_memory(logger, "End of TimeLoop iteration section", target_device)

        # --- 5. Combine and Return Full History (if not saving intermediates) ---
        if not perform_intermediate_saving:
            if not output_tensors_full_history:
                logger.warning("No output tensors were collected for full history!")
                return None
            logger.info(f"Stacking {len(output_tensors_full_history)} collected output tensors (on CPU)...")
            try:
                # Tensors are already on CPU
                output_tensor_full = torch.stack(output_tensors_full_history, dim=1) # Shape: (E, T_out, C, H, W)
                logger.info(f"Final aggregated output tensor shape: {output_tensor_full.shape}, Device: {output_tensor_full.device}")
                if torch.isnan(output_tensor_full).any():
                    logger.warning("NaNs detected in the final aggregated output tensor!")
                # *** Explicitly delete the list of tensors to free memory ***
                del output_tensors_full_history
                return output_tensor_full # Return the full history tensor on CPU
            except Exception as e:
                logger.error(f"Failed to stack collected output tensors: {e}", exc_info=True)
                del output_tensors_full_history # Still try to delete list
                return None
        else:
            logger.info("Intermediate saving was performed. Returning None.")
            return None

    except Exception as e:
        # Catch errors from setup stages (normalization, perturbation)
        logger.error(f"Error occurred during setup before TimeLoop for IC {initial_time_dt.isoformat()}: {e}", exc_info=True)
        return None # Indicate failure

    finally:
        # General cleanup of any remaining large tensors from setup
        logger.debug("Performing final cleanup in run_inference...")
        if initial_state_5d_gpu is not None: del initial_state_5d_gpu
        if initial_state_5d_cpu is not None: del initial_state_5d_cpu
        if initial_state_norm_5d_cpu is not None: del initial_state_norm_5d_cpu
        if initial_state_norm_5d_gpu is not None: del initial_state_norm_5d_gpu
        if initial_state_perturbed_norm_5d_gpu is not None: del initial_state_perturbed_norm_5d_gpu
        # output_tensor_full is handled within the 'if not perform_intermediate_saving' block

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_gpu_memory(logger, "End of run_inference function", target_device if 'target_device' in locals() else None)
        logger.info(f"Total time for run_inference function: {time.time() - run_start_time:.2f}s")











"""


    Smaller Chunks: The logic is broken into:

        _prepare_coordinates: Handles channel, time, lat, lon setup.

        _convert_to_numpy: Focuses solely on the torch to numpy conversion.

        _create_xarray_dataset: Manages xr.DataArray and xr.Dataset creation, including intermediate memory cleanup.

        _generate_output_filename: Creates the descriptive filename.

        _write_netcdf: Handles the actual file writing and compression.

        save_full_output: Orchestrates the calls to the helper functions.

    Detailed Comments & Logging: Each helper function and major step within them has comments. Logging is added at DEBUG and INFO levels to trace the process, including shapes, dtypes, timings, and potential warnings/errors.

    Filename: _generate_output_filename calculates the end_time based on initial_time, simulation_length, and time_step. Both start and end times (formatted as dd_Month_YYYY_HH_MM) are included in the filename, along with model name, ensemble size, and simulation length.

    Robustness & OOM Prevention (CPU):

        Checks for None return values from helpers to abort early on failure.

        Uses try...except blocks extensively.

        Explicit Memory Management: del is used immediately after large objects (input PyTorch tensor, intermediate NumPy array, intermediate DataArray) are no longer needed for the next step. gc.collect() is called to encourage Python to reclaim the memory, minimizing peak CPU RAM.

        The function still relies on the input output_tensor being on the CPU, as checked at the start.

    NetCDF Output: The final output is saved as a single NetCDF file containing all simulation steps present in the input output_tensor. The structure remains (ensemble, time, channel, lat, lon) or similar after to_dataset.

    Variable Sizes: Logging includes the shape and estimated size (nbytes) of the intermediate DataArray, giving an idea of the memory footprint before it's converted to a Dataset and potentially split into variables. The final file size is also logged.

This refactored version is more readable, easier to debug, and incorporates best practices for handling large data and potential memory constraints during the saving process on the CPU.

"""



def _prepare_coordinates(
    output_tensor_shape: Tuple[int, int, int, int, int],
    channels_list: List[str],
    initial_time: datetime.datetime,
    time_step: datetime.timedelta,
    output_freq: int,
    lat_in: np.ndarray | torch.Tensor,
    lon_in: np.ndarray | torch.Tensor,
    logger: logging.Logger,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepares and validates coordinates (channel, time, lat, lon) for the xarray dataset.

    Returns:
        Tuple containing:
        - channels_coord (np.ndarray or List[str]): Coordinate for channels.
        - channel_dim_name (str): Name for the channel dimension.
        - time_coords_np (np.ndarray): Time coordinates as datetime64[ns].
        - lat_np (np.ndarray): Latitude coordinates.
        - lon_np (np.ndarray): Longitude coordinates.
        Returns None for all if a critical error occurs.
    """
    logger.debug("Preparing coordinates...")
    n_ensemble, n_time_out, n_channels, n_lat, n_lon = output_tensor_shape

    # 1. Channel Coordinates
    logger.debug(f"  Processing channel coordinates. Expected based on list: {len(channels_list)}. Found in tensor: {n_channels}")
    if n_channels != len(channels_list):
        logger.error(f"Mismatch between channels in tensor ({n_channels}) and provided channel names ({len(channels_list)}). Using generic indices.")
        channels_coord = np.arange(n_channels).astype(str) # Use string indices for clarity
        channel_dim_name = "channel_idx"
    else:
        channels_coord = channels_list
        channel_dim_name = "channel"
    logger.debug(f"  Using channel dimension name '{channel_dim_name}' with coordinates: {channels_coord[:5]}...") # Log first few

    # 2. Time Coordinates
    logger.debug(f"  Processing time coordinates. Expected steps: {n_time_out}. Initial time: {initial_time.isoformat()}")
    time_coords_np = None
    try:
        # Ensure time_step is timedelta
        if not isinstance(time_step, datetime.timedelta):
            logger.warning(f"Model time_step is not a timedelta ({type(time_step)}), attempting conversion assuming hours.")
            try:
                actual_time_step = datetime.timedelta(hours=float(time_step))
            except ValueError:
                logger.error(f"Cannot interpret time_step '{time_step}' as hours.")
                raise TypeError("Invalid time_step type for time coordinate calculation.")
        else:
            actual_time_step = time_step
        logger.debug(f"  Using time step: {actual_time_step} and output frequency: {output_freq}")

        time_coords_py = [initial_time + i * output_freq * actual_time_step for i in range(n_time_out)]
        time_coords_np = np.array(time_coords_py, dtype='datetime64[ns]') # Convert to numpy datetime64
        logger.debug(f"  Generated {len(time_coords_np)} time coordinates (dtype: {time_coords_np.dtype}). First: {time_coords_np[0]}, Last: {time_coords_np[-1]}")

        if len(time_coords_np) != n_time_out:
             logger.warning(f"Generated {len(time_coords_np)} numpy time coordinates, expected {n_time_out}. Using anyway.")
             # Fallback not strictly needed if array creation worked, but could use indices:
             # time_coords_np = np.arange(n_time_out).astype('int64')

    except Exception as e:
        logger.error(f"Failed to create or convert time coordinates: {e}", exc_info=True)
        logger.warning("Using integer indices as fallback time coordinates.")
        time_coords_np = np.arange(n_time_out).astype('int64') # Fallback

    # 3. Spatial Coordinates (Latitude and Longitude)
    logger.debug("  Processing spatial coordinates (lat, lon)...")
    try:
        lat_np = np.asarray(lat_in.cpu().numpy() if isinstance(lat_in, torch.Tensor) else lat_in)
        lon_np = np.asarray(lon_in.cpu().numpy() if isinstance(lon_in, torch.Tensor) else lon_in)
        if lat_np.ndim != 1 or lon_np.ndim != 1:
            raise ValueError(f"Lat/Lon must be 1D arrays, got shapes {lat_np.shape} and {lon_np.shape}")
        if len(lat_np) != n_lat or len(lon_np) != n_lon:
            raise ValueError(f"Lat/Lon coordinate size mismatch. Tensor: ({n_lat}, {n_lon}), Coords: ({len(lat_np)}, {len(lon_np)})")
        logger.debug(f"  Validated Lat ({lat_np.shape}) and Lon ({lon_np.shape}) coordinates.")
    except Exception as e:
        logger.error(f"Failed to validate or convert spatial coordinates: {e}", exc_info=True)
        return None, None, None, None, None # Indicate critical failure

    return channels_coord, channel_dim_name, time_coords_np, lat_np, lon_np











def _convert_to_numpy(output_tensor_cpu: torch.Tensor, logger: logging.Logger) -> Optional[np.ndarray]:
    """Converts the CPU tensor to a NumPy array, logging time and checking NaNs."""
    logger.info("Converting full output tensor (CPU) to NumPy array...")
    logger.debug(f"Input tensor details: shape={output_tensor_cpu.shape}, dtype={output_tensor_cpu.dtype}")
    conversion_start = time.time()
    output_tensor_np = None
    try:
        # Check NaNs before potentially expensive conversion
        if torch.isnan(output_tensor_cpu).any():
           nan_count_torch = torch.count_nonzero(torch.isnan(output_tensor_cpu))
           logger.warning(f"NaNs detected in PyTorch tensor ({nan_count_torch} occurrences) before NumPy conversion!")

        output_tensor_np = output_tensor_cpu.numpy()
        conversion_end = time.time()
        logger.info(f"Converted tensor to NumPy array (shape: {output_tensor_np.shape}, dtype: {output_tensor_np.dtype}) in {conversion_end - conversion_start:.2f}s.")

        # Check NaNs again in NumPy array (might catch different issues or confirm)
        if np.isnan(output_tensor_np).any():
            nan_count_np = np.count_nonzero(np.isnan(output_tensor_np))
            logger.warning(f"NaNs confirmed/found in NumPy output array ({nan_count_np} occurrences).")

        return output_tensor_np

    except Exception as e:
         logger.error(f"Failed to convert output tensor to NumPy array: {e}", exc_info=True)
         return None # Indicate failure









def _create_xarray_dataset(
    output_tensor_np: np.ndarray,
    coords: Dict[str, Any],
    dims: List[str],
    attrs: Dict[str, Any],
    channel_dim_name: str,
    logger: logging.Logger,
) -> Optional[xr.Dataset]:
    """Creates the xarray DataArray and converts it to a Dataset."""
    logger.info("Creating xarray objects...")
    da_creation_start = time.time()
    forecast_da = None
    forecast_ds = None

    try:
        logger.debug("  Creating DataArray...")
        forecast_da = xr.DataArray(
            output_tensor_np,
            coords=coords,
            dims=dims,
            name='forecast',
            attrs=attrs
        )
        da_creation_end = time.time()
        logger.info(f"  Created DataArray in {da_creation_end - da_creation_start:.2f}s. Size: ~{forecast_da.nbytes / (1024**3):.2f} GiB")

        # --- Memory Management: Delete NumPy array now ---
        logger.debug("  Deleting NumPy array to free memory before Dataset creation...")
        del output_tensor_np
        gc.collect() # Suggest garbage collection
        # --- End Memory Management ---

        logger.debug(f"  Converting DataArray to Dataset using dimension '{channel_dim_name}'...")
        ds_creation_start = time.time()
        forecast_ds = forecast_da.to_dataset(dim=channel_dim_name)
        ds_creation_end = time.time()
        logger.info(f"  Converted to Dataset in {ds_creation_end - ds_creation_start:.2f}s.")

        # --- Memory Management: Delete DataArray now ---
        logger.debug("  Deleting intermediate DataArray...")
        del forecast_da
        gc.collect() # Suggest garbage collection
        # --- End Memory Management ---

        logger.info("xarray Dataset created successfully.")
        return forecast_ds

    except Exception as e:
        logger.error(f"Failed during xarray DataArray/Dataset creation: {e}", exc_info=True)
        # Clean up potentially created objects if error occurred mid-way
        if forecast_da is not None: del forecast_da
        if forecast_ds is not None: del forecast_ds
        # output_tensor_np might have already been deleted, or deletion failed
        gc.collect()
        return None










def _generate_output_filename(
    output_dir: str,
    model_name: str,
    n_ensemble: int,
    simulation_length: int,
    initial_time: datetime.datetime,
    end_time: datetime.datetime,
    logger: logging.Logger,
) -> str:
    """Generates the output NetCDF filename including start and end dates."""
    logger.debug("Generating output filename...")
    try:
        # Use requested format: "%d_%B_%Y_%H_%M"
        start_time_str = initial_time.strftime('%d_%B_%Y_%H_%M')
        end_time_str = end_time.strftime('%d_%B_%Y_%H_%M')
        logger.debug(f"  Formatted start time: {start_time_str}, end time: {end_time_str}")
    except ValueError:
        logger.warning("Could not format dates with '%d_%B_%Y_%H_%M', using ISO format.")
        start_time_str = initial_time.strftime('%Y%m%dT%H%M%S')
        end_time_str = end_time.strftime('%Y%m%dT%H%M%S')

    filename = (
        f"{model_name}_ens{n_ensemble}_sim{simulation_length}steps_"
        f"start_{start_time_str}_end_{end_time_str}_FULL.nc"
    )
    output_path = os.path.join(output_dir, filename)
    logger.debug(f"  Generated filename: {output_path}")
    return output_path








def _write_netcdf(
    dataset: xr.Dataset,
    output_filename: str,
    logger: logging.Logger,
) -> bool:
    """Writes the xarray Dataset to a NetCDF file with compression."""
    logger.info(f"Saving dataset to NetCDF: {output_filename}")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Define encoding for compression
    encoding = {var: {'zlib': True, 'complevel': 5, '_FillValue': np.float32(-9999.0)} for var in dataset.data_vars}
    logger.debug(f"Using encoding: {encoding}")

    save_start_time = time.time()
    try:
        dataset.to_netcdf(output_filename, encoding=encoding, engine='netcdf4')
        save_end_time = time.time()
        write_duration = save_end_time - save_start_time
        try:
            file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
            logger.info(f"NetCDF write complete. Time: {write_duration:.2f}s. File size: {file_size_mb:.2f} MB")
        except OSError as e:
            logger.error(f"Could not get file size for {output_filename}: {e}")
            logger.info(f"NetCDF write complete. Time: {write_duration:.2f}s.")
        return True # Indicate success

    except Exception as e:
        logger.error(f"Failed to write NetCDF file: {e}", exc_info=True)
        # Attempt to remove potentially corrupted file
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                logger.warning(f"Removed potentially corrupted file: {output_filename}")
            except OSError as oe:
                logger.error(f"Failed to remove corrupted file {output_filename}: {oe}")
        return False # Indicate failure











# --- Refactored Main Saving Function ---
def save_full_output(
    output_tensor: torch.Tensor, # Expects (E, T_out, C, H, W) on CPU
    initial_time: datetime.datetime,
    time_step: datetime.timedelta, # Needs time_step to build time coordinate
    channels: List[str],
    lat: np.ndarray, # Expect numpy array
    lon: np.ndarray, # Expect numpy array
    config: dict,
    output_dir: str,
    logger: logging.Logger
):
    """
    Saves the full forecast history tensor (expected on CPU) to a single NetCDF file.
    Breaks down the process into coordinate preparation, data conversion, xarray
    object creation, filename generation, and NetCDF writing, with focus on
    CPU memory management and robustness.

    Args:
        (Args remain the same as before)
    """
    if output_tensor is None:
        logger.error("Cannot save full output, tensor is None.")
        return
    if output_tensor.is_cuda:
        logger.error("save_full_output expects output_tensor on CPU, but it's on GPU. Aborting save.")
        return

    proc_start_time = time.time()
    logger.info("--- Starting Full Forecast Save Process ---")
    logger.info(f"Input tensor: shape={output_tensor.shape}, dtype={output_tensor.dtype}, device={output_tensor.device}")

    forecast_ds = None # Initialize dataset variable for cleanup
    output_tensor_np = None # Initialize numpy variable for cleanup

    try:
        # --- 1. Extract Metadata & Basic Properties ---
        logger.debug("Step 1: Extracting metadata...")
        n_ensemble, n_time_out, n_channels, n_lat, n_lon = output_tensor.shape
        output_freq_for_coords = config.get("output_frequency", 1)
        model_name = config.get("weather_model", "unknown_model")
        simulation_length = config.get("simulation_length", -1) # Get sim length for filename
        logger.debug(f"  Metadata: Ens={n_ensemble}, TimeSteps={n_time_out}, Channels={n_channels}, Lat={n_lat}, Lon={n_lon}")
        logger.debug(f"  Config: OutputFreq={output_freq_for_coords}, ModelName={model_name}, SimLength={simulation_length}")





        # --- 2. Prepare Coordinates ---
        logger.debug("Step 2: Preparing coordinates...")
        channels_coord, channel_dim_name, time_coords_np, lat_np, lon_np = _prepare_coordinates(
            output_tensor.shape, channels, initial_time, time_step, output_freq_for_coords, lat, lon, logger
        )
        if time_coords_np is None or lat_np is None or lon_np is None:
             logger.error("Coordinate preparation failed. Aborting save.")
             return # Critical failure if coordinates are bad






        # --- 3. Convert Data to NumPy ---
        logger.debug("Step 3: Converting data tensor to NumPy array...")
        output_tensor_np = _convert_to_numpy(output_tensor, logger)
        if output_tensor_np is None:
             logger.error("Tensor to NumPy conversion failed. Aborting save.")
             return

        # --- Memory Management: Delete original PyTorch tensor ---
        logger.debug("  Deleting original PyTorch tensor from memory...")
        del output_tensor
        gc.collect()
        # --- End Memory Management ---





        # --- 4. Create xarray Dataset ---
        logger.debug("Step 4: Creating xarray Dataset...")
        coords_dict = {
            'ensemble': np.arange(n_ensemble),
            'time': time_coords_np,
            channel_dim_name: channels_coord,
            'lat': lat_np,
            'lon': lon_np,
        }
        dims_list = ['ensemble', 'time', channel_dim_name, 'lat', 'lon']
        # Calculate end time for attributes
        try:
             if not isinstance(time_step, datetime.timedelta): # Recalculate actual_time_step if needed
                  actual_time_step = datetime.timedelta(hours=float(time_step))
             else: actual_time_step = time_step
             end_time_calc = initial_time + simulation_length * actual_time_step
        except Exception:
             end_time_calc = None # Handle potential errors

        attrs_dict = {
            'description': f"{model_name} full ensemble forecast output",
            'model': model_name,
            'simulation_length_steps': simulation_length,
            'output_frequency_stored': output_freq_for_coords,
            'ensemble_members': n_ensemble,
            'initial_condition_time': initial_time.isoformat(),
            'forecast_end_time': end_time_calc.isoformat() if end_time_calc else "N/A",
            'time_step_seconds': actual_time_step.total_seconds() if isinstance(actual_time_step, datetime.timedelta) else 'unknown',
            'noise_amplitude': config.get("noise_amplitude", 0.0),
            'perturbation_strategy': config.get("perturbation_strategy", "N/A"),
            'creation_date': datetime.datetime.now(pytz.utc).isoformat(),
            'pytorch_version': torch.__version__,
            'numpy_version': np.__version__,
            'xarray_version': xr.__version__,
        }
        forecast_ds = _create_xarray_dataset(
            output_tensor_np, coords_dict, dims_list, attrs_dict, channel_dim_name, logger
        )
        # output_tensor_np is deleted inside _create_xarray_dataset
        if forecast_ds is None:
             logger.error("xarray Dataset creation failed. Aborting save.")
             return






        # --- 5. Generate Filename ---
        logger.debug("Step 5: Generating output filename...")
        if end_time_calc is None:
             logger.warning("Cannot determine exact end time for filename, using simulation length.")
             # Use a fallback or simplified naming if end time calculation failed
             end_time_for_filename = initial_time + datetime.timedelta(days=999) # Placeholder clearly indicating issue
        else:
             end_time_for_filename = end_time_calc

        output_filename = _generate_output_filename(
             output_dir, model_name, n_ensemble, simulation_length, initial_time, end_time_for_filename, logger
        )





        # --- 6. Write to NetCDF ---
        logger.debug("Step 6: Writing dataset to NetCDF file...")
        success = _write_netcdf(forecast_ds, output_filename, logger)

        if success:
            logger.info("--- Full Forecast Save Process Completed Successfully ---")
        else:
            logger.error("--- Full Forecast Save Process Failed ---")

    except Exception as e:
        # Log any unexpected errors during the main flow
        logger.error(f"Unexpected error during save_full_output process: {e}", exc_info=True)

    finally:
        # Final cleanup of the largest remaining object (the Dataset)
        logger.debug("Final cleanup in save_full_output...")
        if forecast_ds is not None:
            del forecast_ds
            forecast_ds = None
        # output_tensor_np and output_tensor should have been deleted earlier if successful
        if 'output_tensor_np' in locals() and output_tensor_np is not None:
            logger.warning("NumPy tensor was not deleted earlier, attempting cleanup now.")
            del output_tensor_np
        # The original output_tensor (passed arg) might still exist if early errors happened
        # We avoid deleting args directly, deletion should happen in the caller (`main`)

        gc.collect() # Suggest final garbage collection
        total_proc_time = time.time() - proc_start_time
        logger.info(f"Total time spent in save_full_output function: {total_proc_time:.2f}s")
        logger.info("--- Exiting Full Forecast Save Process ---")
        
        













# --- Main Pipeline Function (main) ---
# Needs minor adjustments to handle potential None return from run_inference robustly
def main(args, save_steps_config, netcdf_output_dir):
    """Main pipeline execution function."""

    logger.info("========================================================")
    logger.info(" Starting FCNv2-SM Inference Pipeline from NumPy ICs")
    logger.info("========================================================")
    logger.info(f"Parsed arguments: {vars(args)}")
    logger.info(f"Save mode: {args.save_mode}")
    if args.save_mode == 'intermediate':
        logger.info(f"Intermediate save steps config: {save_steps_config}")
    logger.info(f"Plotting enabled: {PLOTTING_ENABLED}")
    logger.info(f"NetCDF output directory: {netcdf_output_dir}")

    # --- Environment and Setup ---
    if args.gpu >= 0 and torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(device) # Set default device for this process
            logger.info(f"Using GPU: {device} ({torch.cuda.get_device_name(device)})")
        except Exception as e:
            logger.error(f"Failed to set CUDA device {args.gpu}: {e}. Falling back to CPU.", exc_info=True)
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")

    log_gpu_memory(logger, "Start of main", device)

    # --- Load Model ---
    model_id = "fcnv2_sm"
    logger.info(f"Loading {model_id} model to {device}...")
    model_inference = None # Initialize
    try:
        # Reset peak memory stats before loading model
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)

        package = registry.get_model(model_id)
        if package is None: raise FileNotFoundError(f"Model package '{model_id}' not found.")
        logger.info(f"Found model package: {package.root}")

        # Load the model onto the specified device
        model_inference = fcnv2_sm_load(package, device=device, pretrained=True)
        model_inference.eval()

        log_gpu_memory(logger, "After model load", device)
        logger.info(f"{model_id} model loaded. Parameter device: {next(model_inference.parameters()).device}.") # Verify device
        logger.info(f"Input channels ({len(model_inference.in_channel_names)}): {model_inference.in_channel_names}")
        logger.info(f"Model time step: {model_inference.time_step}")
        # Ensure grid info is numpy
        if isinstance(model_inference.grid.lat, torch.Tensor): model_inference.grid.lat = model_inference.grid.lat.cpu().numpy()
        if isinstance(model_inference.grid.lon, torch.Tensor): model_inference.grid.lon = model_inference.grid.lon.cpu().numpy()

    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        logger.error(f"Check weights/means/stds files in {os.path.join(os.environ.get('MODEL_REGISTRY'), model_id)}")
        sys.exit(1)
    except _pickle.UnpicklingError as e:
        logger.error(f"Model loading failed (UnpicklingError): {e}", exc_info=False)
        logger.error("Possible weights_only=True issue. Ensure earth2mip/networks/fcnv2_sm.py uses weights_only=False.")
        sys.exit(1)
    except RuntimeError as e:
         if "out of memory" in str(e).lower():
              logger.error(f"CUDA OOM error during model loading!", exc_info=True)
              log_gpu_memory(logger, "OOM state during model load", device)
         else:
              logger.error(f"Runtime error during model loading: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Initial Conditions ---
    logger.info(f"Loading initial conditions from: {args.ic_file_path}")
    initial_conditions_np = None
    try:
        if not os.path.exists(args.ic_file_path): raise FileNotFoundError(f"IC file not found: {args.ic_file_path}")
        initial_conditions_np = np.load(args.ic_file_path) # Load into CPU memory
        logger.info(f"Loaded NumPy IC data shape: {initial_conditions_np.shape}, dtype: {initial_conditions_np.dtype}")
        if initial_conditions_np.ndim != 4: raise ValueError(f"Expected 4D ICs (T, C, H, W), got {initial_conditions_np.ndim}D")

        num_ics, num_channels, height, width = initial_conditions_np.shape
        logger.info(f"Found {num_ics} ICs. Grid: {height}x{width}. Channels: {num_channels}")

        # Validate channels and grid
        model_channels = model_inference.in_channel_names
        if num_channels != len(model_channels):
            raise ValueError(f"Channel count mismatch! Model={len(model_channels)}, File={num_channels}.")
        model_lat, model_lon = model_inference.grid.lat, model_inference.grid.lon
        if height != len(model_lat) or width != len(model_lon):
            logger.warning(f"Grid mismatch! Model={len(model_lat)}x{len(model_lon)}, File={height}x{width}.")

    except Exception as e:
        logger.error(f"Failed to load/validate IC NumPy file: {e}", exc_info=True)
        if model_inference is not None: del model_inference # Cleanup model if loaded
        sys.exit(1)






    month_to_int = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    # --- Define Timestamps for ICs ---
    ic_timestamps = []
    try:
        # (Timestamp generation logic remains the same)
        fname = os.path.basename(args.ic_file_path)
        import re
        match = re.search(r"START_(\d{1,2})_(\w{3,9})_(\d{4})_(.*?)END_(\d{1,2})_(\w{3,9})_(\d{4})", fname)
        if match:
            day_start, month_start, year_start, _, day_end, month_end, year_end = match.groups()
            base_date = datetime.datetime(int(year_start), month_to_int[month_start], int(day_start), tzinfo=pytz.utc)
            logger.info(f"Inferred base date from filename: {base_date.strftime('%Y-%B-%d')}")
        else:
            base_date = datetime.datetime(2018, 2, 7, tzinfo=pytz.utc) # Default fallback
            logger.warning(f"Using default base date: {base_date.strftime('%Y-%B-%d')}")
        ic_timestamps = [base_date + datetime.timedelta(hours=i * 6) for i in range(num_ics)]
        logger.info(f"Generated {len(ic_timestamps)} timestamps starting from {ic_timestamps[0].isoformat()}")
    except Exception as e:
        logger.error(f"Error determining timestamps: {e}. Using indices.", exc_info=True)
        ic_timestamps = list(range(num_ics))






    # --- Prepare Inference Configuration ---
    inference_config = {
        "ensemble_members": args.ensemble_members,
        "noise_amplitude": args.noise_amplitude,
        "simulation_length": args.simulation_length,
        "output_frequency": args.output_frequency,
        "weather_model": model_id,
        "perturbation_strategy": args.perturbation_strategy,
        'variables_to_save': None # For intermediate saving subsetting
    }
    logger.info(f"Inference Configuration: {inference_config}")

    # --- Run Inference for each Initial Condition ---
    num_ics_processed = 0
    num_ics_failed = 0
    total_start_time = time.time()

    for i, initial_time in enumerate(ic_timestamps):
        # --- Prepare IC Time ---
        if not isinstance(initial_time, datetime.datetime):
             logger.error(f"IC timestamp for index {i} invalid. Skipping.")
             num_ics_failed += 1; continue
        if initial_time.tzinfo is None: initial_time = initial_time.replace(tzinfo=pytz.utc)
        time_label = initial_time.isoformat()
        logger.info(f"--- Processing Initial Condition {i+1}/{num_ics}: {time_label} ---")
        log_gpu_memory(logger, f"Start of IC {i+1} Loop", device)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device) # Reset peak stats for this IC

        # --- Prepare IC Tensor (Load slice from NumPy on CPU) ---
        initial_state_tensor_cpu = None
        try:
            # Select the slice from the pre-loaded NumPy array
            ic_data_np = initial_conditions_np[i]
            # Convert directly to torch tensor on CPU
            initial_state_tensor_cpu = torch.from_numpy(ic_data_np).unsqueeze(0).float() # (1, C, H, W) on CPU
            logger.debug(f"Prepared IC tensor on CPU: {initial_state_tensor_cpu.shape}, {initial_state_tensor_cpu.dtype}")
        except Exception as e:
            logger.error(f"Failed to get/convert NumPy slice {i}: {e}", exc_info=True)
            num_ics_failed += 1; continue

        # --- Execute Inference ---
        output_tensor_full = None # Holds result if save_mode == 'full'
        run_successful = False

        try:
            if args.save_mode == 'intermediate':
                logger.info("Running inference in 'intermediate' save mode.")
                # run_inference handles moving initial_state_tensor_cpu to device internally now
                run_inference(
                    model_inference=model_inference, initial_state_tensor=initial_state_tensor_cpu,
                    initial_time_dt=initial_time, config=inference_config, logger=logger,
                    save_func=save_output_steps, save_steps_config=save_steps_config, # Pass intermediate save func
                    output_dir=netcdf_output_dir
                )
                # In intermediate mode, success is assumed if no exception was raised by run_inference
                run_successful = True # (We can't check return value)

            elif args.save_mode == 'full':
                logger.info("Running inference in 'full' save mode.")
                # run_inference handles moving initial_state_tensor_cpu to device internally
                # It returns the full tensor on CPU, or None on error.
                output_tensor_full = run_inference(
                    model_inference=model_inference, initial_state_tensor=initial_state_tensor_cpu,
                    initial_time_dt=initial_time, config=inference_config, logger=logger,
                    save_func=None, save_steps_config=None, output_dir=None # Disable intermediate saving
                )
                run_successful = output_tensor_full is not None # Success if tensor is returned
            else:
                logger.error(f"Invalid save_mode: {args.save_mode}") # Should be caught by argparse

            log_gpu_memory(logger, f"After run_inference IC {i+1}", device)




            # --- Post-Inference Processing ---
            if run_successful:
                logger.info(f"Inference run successful for IC {time_label}.")
                num_ics_processed += 1

                # Save full history if collected (output_tensor_full is already on CPU)
                if args.save_mode == 'full' and output_tensor_full is not None:
                    try:
                        logger.info("Saving collected full history tensor...")
                        save_full_output(
                            output_tensor=output_tensor_full, initial_time=initial_time,
                            time_step=model_inference.time_step, channels=model_channels,
                            lat=model_lat, lon=model_lon, config=inference_config,
                            output_dir=netcdf_output_dir, logger=logger
                        )
                        log_gpu_memory(logger, f"After saving full history IC {i+1}", device)

                        # --- Plotting ---
                        if PLOTTING_ENABLED and args.plot_z500:
                            logger.info("Generating Z500 progression plots...")
                            plot_z500_progression(
                                forecast_tensor=output_tensor_full, channels=model_channels,
                                lat=model_lat, lon=model_lon, initial_time=initial_time,
                                time_step=model_inference.time_step, output_dir=netcdf_output_dir,
                                logger=logger, config=inference_config,
                                output_freq=inference_config['output_frequency']
                            )
                            log_gpu_memory(logger, f"After plotting z500 IC {i+1}", device)
                        elif args.plot_z500:
                            logger.warning("Plotting requested but libraries not available.")

                    except Exception as post_err:
                         logger.error(f"Error during saving/plotting for IC {time_label}: {post_err}", exc_info=True)
                         # Consider if this should count as a failed IC
                    finally:
                         # *** Crucial: Delete the large output tensor from CPU RAM ***
                         logger.debug("Deleting full output tensor from CPU memory.")
                         del output_tensor_full
                         output_tensor_full = None
            else:
                # This case means run_inference returned None or raised exception captured below
                logger.error(f"Inference run failed for IC {time_label} (run_inference returned None or error occurred).")
                num_ics_failed += 1

        except Exception as run_err:
             # Catch errors raised directly by run_inference call itself
             logger.error(f"Unhandled exception during run_inference call for IC {time_label}: {run_err}", exc_info=True)
             num_ics_failed += 1
             log_gpu_memory(logger, f"After failed run_inference IC {i+1}", device)

        finally:
            # --- Cleanup IC Tensor ---
            logger.debug(f"Cleaning up CPU IC tensor for step {i+1}.")
            if initial_state_tensor_cpu is not None:
                del initial_state_tensor_cpu
                initial_state_tensor_cpu = None
            # Clear CUDA cache again after processing each IC
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_gpu_memory(logger, f"End of IC {i+1} Loop", device)
            logger.info(f"Peak GPU Memory Allocated during IC {i+1}: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GiB")
            logger.info(f"Peak GPU Memory Reserved during IC {i+1}: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GiB")


    # --- Final Summary ---
    total_end_time = time.time()
    logger.info("--------------------------------------------------------")
    logger.info(f"Total processing time: {total_end_time - total_start_time:.2f} seconds")
    logger.info(f"Successfully processed: {num_ics_processed} / {num_ics} initial conditions.")
    if num_ics_failed > 0: logger.warning(f"Failed to process: {num_ics_failed} / {num_ics} initial conditions.")
    logger.info(f"Output NetCDF files saved in: {netcdf_output_dir}")
    if PLOTTING_ENABLED and args.plot_z500 and args.save_mode == 'full' and num_ics_processed > 0:
        logger.info(f"Z500 plots saved in subdirectories within: {netcdf_output_dir}")
    logger.info(f"Log file saved in: {LOG_DIR}")
    logger.info("========================================================")
    logger.info(" FCNv2-SM Inference Pipeline Finished ")
    logger.info("========================================================")
    log_gpu_memory(logger, "End of main", device)

    # Final cleanup of large objects
    del initial_conditions_np
    if model_inference is not None: del model_inference



"""

Your provided earth2mip/networks/__init__.py code for the Inference class seems robust enough:

    normalize and denormalize already use x.device to ensure the center/scale buffers are moved to the correct device (CPU or GPU) before calculations.

    _iterate expects the input x to be on the correct device (which will be the GPU after our changes in run_inference) and performs the model call there.

Explanation of Key Changes in run_inference:

    Get Target Device: It first determines the device the model_inference object is on (target_device).

    Prepare Initial State on GPU: The initial state tensor slice is loaded from NumPy (CPU), converted to Torch (CPU), then moved to the target_device and expanded for ensemble/history (initial_state_5d_gpu).

    Move to CPU for Normalization: initial_state_5d_gpu is explicitly moved to the CPU (initial_state_5d_cpu = initial_state_5d_gpu.cpu()).

    GPU Memory Cleanup: Crucially, the large GPU tensor is deleted (del initial_state_5d_gpu) and torch.cuda.empty_cache() is called immediately after the move to CPU. This frees up GPU memory before the potentially large allocation needed for normalization happens (even though we're doing it on CPU now, it clears space for the upcoming model).

    CPU Normalization: model_inference.normalize() is called with the CPU tensor (initial_state_5d_cpu). Since the normalize method is device-aware, it works correctly on the CPU tensor.

    CPU Memory Cleanup: The unnormalized CPU tensor is deleted.

    Move Back to GPU: The resulting normalized tensor (initial_state_norm_5d_cpu) is moved back to the target_device (initial_state_norm_5d_gpu = ... .to(target_device)).

    CPU Memory Cleanup: The normalized CPU tensor is deleted.

    Perturbation on GPU: Noise is added directly to the initial_state_norm_5d_gpu on the target device.

    Iterator Call: The final, normalized, perturbed tensor on the GPU (initial_state_perturbed_norm_5d_gpu) is passed to the model_inference iterator (__call__).

    Inference Loop: The _iterate loop within model_inference proceeds entirely on the GPU.

    Output Handling: Outputs yielded by the iterator (data_denorm_dev) are immediately moved to CPU (data_denorm_cpu) to free GPU memory quickly, especially important if collecting the full history or saving intermediates frequently.

    Explicit Deletes: Added more del statements throughout run_inference and main for intermediate tensors and final results to help Python's garbage collector free memory sooner, reducing peak RAM usage on both CPU and GPU.

    Memory Logging: Enhanced log_gpu_memory and added calls at critical points (start/end of functions, before/after major steps, OOM errors) to better track usage. Added peak memory logging per IC in main.

This approach directly tackles the OOM error during normalization by offloading that specific step to the CPU, while ensuring the computationally intensive model inference remains on the GPU. The explicit memory management helps minimize peak resource usage.

"""







# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCNv2-SM Inference Pipeline using NumPy ICs.")

    # Input/Output
    # Use environment variables for defaults where possible
    default_ic_path = f"/scratch/gilbreth/gupt1075/fcnv2/ARCO_data_73_channels/data/START_07_February_2020__len_4__END_07_February_2020.npy"
    parser.add_argument("--ic-file-path", type=str, default=default_ic_path, help="Path to NumPy IC file (T, C, H, W).")
    parser.add_argument("-o", "--output-path", type=str, default=os.path.join(OUTPUT_DIR, "inference_output"), help="Directory for NetCDF output (and plot subdirs).")

    # Inference parameters
    parser.add_argument("-sim", "--simulation-length", type=int, default=3, help="Number of forecast steps (e.g., 4 steps * 6hr = 24hr).")
    parser.add_argument("-ef", "--output-frequency", type=int, default=1, help="Frequency (steps) to store outputs in 'full' mode.")
    parser.add_argument("-ens", "--ensemble-members", type=int, default=1, help="Number of ensemble members.")
    parser.add_argument("-na", "--noise-amplitude", type=float, default=0.0, help="Perturbation noise amplitude (if ens > 1).")
    parser.add_argument("-ps", "--perturbation-strategy", type=str, default="gaussian", choices=["gaussian"], help="Perturbation strategy.")

    # Saving & Plotting
    parser.add_argument("--save-mode", type=str, default="full", choices=["intermediate", "full"], help="Saving mode: 'intermediate' (recent steps) or 'full' (all steps). Plotting only works with 'full'.")
    parser.add_argument("--save-steps", type=str, default="-2,0", help="Steps to save in 'intermediate' mode (rel. to current, comma-sep).")
    parser.add_argument("--plot-z500", action='store_true', help="Generate Z500 progression plots (requires save_mode='full' and plotting libraries).")

    # System
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (-1 for CPU).")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")

    args = parser.parse_args()

    # Process save_steps
    try:
        save_steps_list = [int(step.strip()) for step in args.save_steps.split(',')]
    except ValueError:
        # Logger might not be fully configured yet if error happens early
        print(f"ERROR: Invalid --save-steps format: '{args.save_steps}'. Use comma-separated integers.")
        sys.exit(1)
    save_steps_config = {'steps_to_save': save_steps_list}

    # Check save mode for plotting
    if args.plot_z500 and args.save_mode != 'full':
         print(f"WARNING: Plotting requested (--plot-z500) but save_mode is '{args.save_mode}'. Plotting requires save_mode='full'. Disabling plotting.")
         args.plot_z500 = False

    # Output directory setup
    netcdf_output_dir = args.output_path
    # Creation happens in main after logger is ready

    # Adjust logger level if debug flag is set AFTER logger initialized
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers: handler.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled.")
    else:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers: handler.setLevel(logging.INFO)


    # Call main
    try:
        os.makedirs(netcdf_output_dir, exist_ok=True) # Create output dir now
        main(args, save_steps_config, netcdf_output_dir)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (KeyboardInterrupt).")
        sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        logger.critical(f"Critical pipeline failure in main execution: {str(e)}", exc_info=True)
        sys.exit(1)



# # -*- coding: utf-8 -*-
# import numpy as np
# import datetime 
# from datetime import timezone, timedelta
# import os
# import logging
# import argparse
# import time
# import sys
# import _pickle
# import torch
# import xarray as xr
# from dateutil.relativedelta import relativedelta
# import dotenv
# import pytz


# import numpy as np


# import torch
# import xarray as xr
# import os
# import time
# import logging # Assume logger is passed correctly
# import pytz # Assume pytz is imported elsewhere
# from typing import List, Dict, Optional, Callable, Any, Tuple # Type hints




# import os
# import sys
# import time
# import argparse

# import importlib.util
# import json
# import cdsapi
# from collections import defaultdict
# import dotenv
# import xarray as xr
# import numpy as np
# import torch
# import cdsapi
# import pandas as pd
# import zarr
# import argparse

# import importlib.util
# import json
# import logging
# import os
# import sys
# import time
# import argparse


# import matplotlib
# matplotlib.use('Agg')  # Needed for headless environments
# import matplotlib.pyplot as plt
# import dotenv
# import xarray as xr
# import numpy as np
# import torch
# import cdsapi
# import pandas as pd



# import numpy as np
# import xarray as xr
# from earth2studio.data import ARCO
# from earth2studio.models.px.sfno import VARIABLES
# import logging
# import os
# from typing import List
# from tqdm import tqdm
# import traceback




# # Need these imports if not already present at the top
# import collections
# import pickle
# from typing import List, Dict, Optional, Callable, Any, Tuple # For type hints

# # --- Plotting Imports ---
# try:
#     import matplotlib
#     matplotlib.use('Agg') # Use non-interactive backend suitable for scripts
#     import matplotlib.pyplot as plt
#     import cartopy.crs as ccrs
#     import cartopy.feature as cfeature
#     from matplotlib.colors import Normalize # Use standard Normalize for Z500
#     print("Successfully imported Matplotlib and Cartopy.")
#     PLOTTING_ENABLED = True
# except ImportError as e:
#     print(f"Warning: Failed to import Matplotlib or Cartopy ({e}). Plotting functionality will be disabled.")
#     print("Please install them ('pip install matplotlib cartopy') to enable plotting.")
#     PLOTTING_ENABLED = False
#     # Define dummy classes/functions if plotting is disabled to avoid NameErrors later
#     class ccrs:
#         PlateCarree = None
#         LambertConformal = None
#     class cfeature:
#         NaturalEarthFeature = None
#     class plt:
#         figure = None
#         rcParams = None
#         colorbar = None
#         close = None
#     class Normalize:
#         pass


# # --- Configuration ---
# USERNAME = "gupt1075"
# MODEL_REGISTRY_BASE = f"/scratch/gilbreth/gupt1075/fcnv2/"
# EARTH2MIP_PATH = f"/scratch/gilbreth/gupt1075/fcnv2/earth2mip"

# # --- Add earth2mip to Python path ---
# if EARTH2MIP_PATH not in sys.path:
#     sys.path.insert(0, EARTH2MIP_PATH)
#     print(f"Added {EARTH2MIP_PATH} to Python path.")

# # --- Environment Setup ---
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MODEL_REGISTRY"] = MODEL_REGISTRY_BASE
# print(f"Set MODEL_REGISTRY environment variable to: {MODEL_REGISTRY_BASE}")

# # --- Logging Setup ---
# def setup_logging(log_dir, log_level=logging.INFO):
#     os.makedirs(log_dir, exist_ok=True)
#     pacific_tz = pytz.timezone("America/Los_Angeles")
#     timestamp_str = datetime.datetime.now(pacific_tz).strftime("%d_%B_%H_%M_")
#     log_filename = os.path.join(log_dir, f"inference_pipeline_{timestamp_str}.log")

#     class PytzFormatter(logging.Formatter):
#         def __init__(self, fmt=None, datefmt=None, tz=None):
#             super().__init__(fmt, datefmt)
#             self.tz = tz if tz else pytz.utc

#         def formatTime(self, record, datefmt=None):
#             dt = datetime.datetime.fromtimestamp(record.created, self.tz)
#             if datefmt:
#                 return dt.strftime(datefmt)
#             else:
#                 return dt.strftime("%d-%B-%Y %H:%M:%S.%f %Z%z")

#     logger = logging.getLogger("FCNv2Inference")
#     logger.setLevel(log_level)
#     logger.handlers.clear()

#     file_handler = logging.FileHandler(log_filename, mode="w")
#     file_formatter = PytzFormatter("%(asctime)s [%(levelname)-8s] [%(name)s:%(lineno)d] %(message)s", tz=pacific_tz)
#     file_handler.setFormatter(file_formatter)
#     logger.addHandler(file_handler)

#     console_handler = logging.StreamHandler(sys.stdout)
#     console_formatter = PytzFormatter("%(asctime)s [%(levelname)-8s] %(message)s", tz=pacific_tz)
#     console_handler.setFormatter(console_formatter)
#     logger.addHandler(console_handler)

#     # Reduce verbosity of libraries
#     logging.getLogger("urllib3").setLevel(logging.WARNING)
#     logging.getLogger("matplotlib").setLevel(logging.WARNING)
#     logging.getLogger("fsspec").setLevel(logging.WARNING)
#     # Cartopy can be verbose, quiet it down if needed
#     logging.getLogger("cartopy").setLevel(logging.WARNING)


#     logger.info(f"Logging configured. Level: {logging.getLevelName(logger.level)}")
#     logger.info(f"Log file: {log_filename}")
#     return logger

# # --- Determine Output Directory and Setup Logging ---
# pacific_tz = pytz.timezone("America/Los_Angeles")
# timestamp = datetime.datetime.now(pacific_tz).strftime("%d_%B_%H_%M")
# OUTPUT_DIR = f"/scratch/gilbreth/{USERNAME}/fcnv2/RESULTS_2025/ZETA_inference_{timestamp}"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
# logger = setup_logging(LOG_DIR) # Initialize logger early

# logger.info(f"Using Output Directory: {OUTPUT_DIR}")
# logger.info(f"Plotting Enabled: {PLOTTING_ENABLED}")

# # # --- Load Environment Variables (optional) ---
# # dotenv.load_dotenv()
# # logger.info("Checked for .env file.")

# # --- Earth-2 MIP Imports (AFTER setting env vars and sys.path) ---
# try:
#     from earth2mip import registry
#     from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
#     import earth2mip.grid
#     from earth2mip import (
#         ModelRegistry,
#         loaders,
#         model_registry,
#         registry,
#         schema,
#         time_loop,
#     )
#     print("Successfully imported earth2mip components.")
# except ImportError as e:
#     print(f"Error importing earth2mip components: {e}")
#     print("Please ensure earth2mip is installed correctly and EARTH2MIP_PATH is correct.")
#     logger.critical(f"Earth2MIP import failed: {e}", exc_info=True)
#     sys.exit(1)
# except Exception as e:
#     print(f"An unexpected error occurred during earth2mip import: {e}")
#     logger.critical(f"Unexpected Earth2MIP import error: {e}", exc_info=True)
#     sys.exit(1)


# # --- Function to Save Specific Time Steps (save_output_steps) ---
# def save_output_steps(
#     data_dict: Dict[int, torch.Tensor], # Dict mapping step_index to tensor (E, C, H, W)
#     time_dict: Dict[int, datetime.datetime], # Dict mapping step_index to datetime
#     channels: List[str],
#     lat: np.ndarray,
#     lon: np.ndarray,
#     config: dict,
#     output_dir: str,
#     current_model_step: int, # The latest step index included in data_dict
#     logger: logging.Logger
# ):
#     """
#     Saves specified forecast steps (e.g., current and t-2) to a NetCDF file.
#     (Implementation remains the same as before)
#     """
#     if not data_dict or not time_dict:
#         logger.warning(f"Save request for step {current_model_step} received no data/time to save.")
#         return

#     n_ensemble = config.get("ensemble_members", 1)
#     model_name = config.get("weather_model", "unknown_model")
#     initial_time = time_dict.get(0, list(time_dict.values())[0]) # Get t=0 time if available

#     logger.info(f"Preparing to save specific steps {list(data_dict.keys())} for forecast step {current_model_step}.")

#     # --- Data Validation ---
#     expected_n_channels = len(channels)
#     first_tensor = list(data_dict.values())[0]
#     if first_tensor.shape[1] != expected_n_channels:
#          logger.error(f"Channel mismatch in save_output_steps! Expected {expected_n_channels}, got {first_tensor.shape[1]}. Channels: {channels}")
#          return # Safer to not save incorrect data

#     # --- Create Coordinates ---
#     sorted_steps = sorted(data_dict.keys())
#     time_coords = [time_dict[step] for step in sorted_steps]
#     step_coords = np.array(sorted_steps) # Coordinate for the 'time' dimension

#     # --- Assemble Data ---
#     try:
#         ref_shape = first_tensor.shape
#         for step, tensor in data_dict.items():
#              if tensor.shape[1:] != ref_shape[1:]: # Check C, H, W dims
#                   logger.error(f"Tensor shape mismatch in data_dict for step {step}. Expected {ref_shape}, got {tensor.shape}")
#                   return
#         output_tensor = torch.stack([data_dict[step] for step in sorted_steps], dim=1)
#         logger.debug(f"Stacked tensor for saving: {output_tensor.shape}")
#     except Exception as e:
#          logger.error(f"Failed to stack tensors for saving step {current_model_step}: {e}", exc_info=True)
#          return

#     # --- Variable Subsetting (Optional) ---
#     variables_to_save = config.get('variables_to_save', None)
#     if variables_to_save:
#         try:
#             var_indices = [channels.index(v) for v in variables_to_save]
#             output_tensor = output_tensor[:, :, var_indices, :, :] # Select channels along dim 2
#             channels_coord = variables_to_save # Update coords
#             logger.info(f"Selected {len(channels_coord)} variables for saving: {variables_to_save}")
#         except ValueError as e:
#             logger.error(f"Invalid variable name in 'variables_to_save': {e}. Saving all variables.")
#             channels_coord = channels # Fallback
#     else:
#         channels_coord = channels # Save all if not specified

#     n_ensemble_out, n_time_out, n_channels_out, n_lat_out, n_lon_out = output_tensor.shape # Re-check shape after potential subsetting

#     # --- Create xarray Dataset ---
#     try:
#         lat_np = lat.cpu().numpy() if isinstance(lat, torch.Tensor) else np.asarray(lat)
#         lon_np = lon.cpu().numpy() if isinstance(lon, torch.Tensor) else np.asarray(lon)
#         time_coords_np = [np.datetime64(t) for t in time_coords]

#         forecast_da = xr.DataArray(
#             output_tensor.numpy(),
#             coords={
#                 'ensemble': np.arange(n_ensemble_out),
#                 'step': step_coords,
#                 'channel': channels_coord, # Use potentially subsetted channels
#                 'lat': lat_np,
#                 'lon': lon_np,
#                 'time': ('step', time_coords_np)
#             },
#             dims=['ensemble', 'step', 'channel', 'lat', 'lon'],
#             name='forecast',
#             attrs={
#                 'description': f"{model_name} ensemble forecast output for specific steps",
#                 'forecast_step': current_model_step,
#                 'saved_steps': str(sorted_steps),
#                 'model': model_name,
#                 'simulation_length_steps': config.get("simulation_length", "N/A"),
#                 'ensemble_members': n_ensemble_out,
#                 'initial_condition_time': initial_time.isoformat() if initial_time else "N/A",
#                 'noise_amplitude': config.get("noise_amplitude", 0.0),
#                 'perturbation_strategy': config.get("perturbation_strategy", "N/A"),
#                 'creation_date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
#                 'variables_saved': channels_coord if variables_to_save else "all",
#             }
#         )
#         logger.debug("Created xarray DataArray for specific steps.")
#         forecast_ds = forecast_da.to_dataset(dim='channel')
#         logger.info(f"Converted DataArray to Dataset for saving step {current_model_step}.")

#     except Exception as e:
#         logger.error(f"Failed to create xarray Dataset for step {current_model_step}: {e}", exc_info=True)
#         return

#     # --- Define Filename and Save ---
#     ic_time_str = initial_time.strftime('%d_%B_%Y_%H%M') if initial_time else "unknownIC"
#     start_frame = sorted_steps[0]
#     end_frame = sorted_steps[-1]
#     output_filename = os.path.join(
#         output_dir,
#         f"{model_name}_ens{n_ensemble_out}_IC{ic_time_str}_"
#         f"startFrame{start_frame:04d}_endFrame{end_frame:04d}_"
#         f"currentStep{current_model_step:04d}.nc"
#     )

#     os.makedirs(output_dir, exist_ok=True)
#     logger.info(f"Saving forecast steps {list(data_dict.keys())} to: {output_filename}")

#     try:
#         encoding = {var: {'zlib': True, 'complevel': 5, '_FillValue': -9999.0} for var in forecast_ds.data_vars}
#         start_save = time.time()
#         forecast_ds.to_netcdf(output_filename, encoding=encoding, engine='netcdf4')
#         end_save = time.time()
#         file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
#         logger.info(f"Save complete for step {current_model_step}. Time: {end_save - start_save:.2f}s. Size: {file_size_mb:.2f} MB")
#     except Exception as e:
#         logger.error(f"Failed to save NetCDF file for step {current_model_step}: {e}", exc_info=True)
#         if os.path.exists(output_filename):
#             try:
#                 os.remove(output_filename)
#                 logger.warning(f"Removed potentially corrupted file: {output_filename}")
#             except OSError as oe:
#                 logger.error(f"Failed to remove corrupted file {output_filename}: {oe}")






















# """




# Summary of Changes and Reasoning:

#     Time Coordinate Conversion:

#         The core fix: After generating the list of Python datetime objects (time_coords_py), it's converted to a NumPy array of datetime64[ns] objects: time_coords_np = np.array(time_coords_py, dtype='datetime64[ns]').

#         This time_coords_np array is then used in the coords dictionary for xr.DataArray. xarray natively understands np.datetime64 and handles its serialization correctly.

#         Added robustness check for time_step type.

#         Improved fallback to use integer indices if time generation/conversion fails.

#     Filename Format:

#         Updated the strftime call: ic_time_str = initial_time.strftime('%d_%B_%Y_%H_%M').

#         Added a basic try...except ValueError around strftime in case the month name format (%B) causes issues in certain locales, falling back to ISO format.

#     Performance and Memory (CPU Focus):

#         Input Check: Added a check to ensure output_tensor is on the CPU. Saving from GPU directly is not typical and would involve implicit copies. It's better to ensure the tensor is moved to CPU before calling this function (as is done in the run_inference function when save_mode='full').

#         NumPy Conversion: The output_tensor.numpy() step is the main CPU memory allocation point. This is unavoidable. Logging is added around it.

#         Explicit del: Added del output_tensor_np, del forecast_da, and del forecast_ds after these objects are no longer needed for subsequent steps within this function. This explicitly removes the reference, potentially allowing Python's garbage collector (GC) to reclaim the memory sooner, reducing peak CPU RAM usage during the saving process. Note: This doesn't guarantee immediate memory release but is good practice.

#         Timing: Added more detailed timing logs for conversion, DataArray/Dataset creation, and NetCDF writing to help identify bottlenecks if saving is slow.

#         NaN Check: Moved the np.isnan check to after the .numpy() conversion, which is more common and potentially faster than checking on the PyTorch tensor. Added count of NaNs.

#         Fill Value Type: Explicitly set _FillValue to np.float32(-9999.0) which might save a tiny bit of space compared to float64 if the data itself is float32.

#     Robustness:

#         Kept existing try...except blocks.

#         Ensured cleanup (del) happens within a finally block to execute even if errors occur during saving.

#         Improved logging messages.

# This revised function correctly handles the time coordinates for xarray, uses the requested filename format, and includes explicit steps to manage CPU memory more proactively during the potentially demanding process of saving large forecast datasets.

# """









# # --- Function to Save Full Output History (save_full_output) ---
# def save_full_output(
#     output_tensor: torch.Tensor, # Expects (E, T_out, C, H, W) on CPU
#     initial_time: datetime.datetime,
#     time_step: datetime.timedelta, # Needs time_step to build time coordinate
#     channels: List[str],
#     lat: np.ndarray, # Expect numpy array
#     lon: np.ndarray, # Expect numpy array
#     config: dict,
#     output_dir: str,
#     logger: logging.Logger
# ):
#     """
#     Saves the full forecast history tensor (expected on CPU) to a single NetCDF file.

#     Handles time coordinate conversion to numpy.datetime64 for xarray compatibility.
#     Includes explicit deletion of large intermediate objects to potentially aid
#     garbage collection and reduce peak CPU memory usage during saving.

#     Args:
#         output_tensor: The tensor containing the full forecast history
#                        (Shape: E, T_out, C, H, W), MUST be on CPU.
#         initial_time: Datetime object for the initial condition (t=0), timezone-aware.
#         time_step: Timedelta object representing the model's time step duration.
#         channels: List of channel names corresponding to dimension C.
#         lat: Latitude coordinates (1D numpy array).
#         lon: Longitude coordinates (1D numpy array).
#         config: The main inference configuration dictionary.
#         output_dir: Directory to save the NetCDF file.
#         logger: Logger object.
#     """
#     if output_tensor is None:
#         logger.error("Cannot save full output, tensor is None.")
#         return
#     if output_tensor.is_cuda:
#         logger.error("save_full_output expects output_tensor on CPU, but it's on GPU. Aborting save.")
#         # Or move it:
#         # logger.warning("output_tensor provided to save_full_output is on GPU. Moving to CPU.")
#         # output_tensor = output_tensor.cpu()
#         return # Safer to abort if it wasn't moved earlier

#     proc_start_time = time.time()
#     logger.info("Preparing full forecast history for saving...")

#     try:
#         # --- Basic Properties ---
#         n_ensemble, n_time_out, n_channels, n_lat, n_lon = output_tensor.shape
#         output_freq_for_coords = config.get("output_frequency", 1)
#         model_name = config.get("weather_model", "unknown_model")
#         logger.debug(f"Full output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}, device: {output_tensor.device}")

#         # --- Channel Handling ---
#         if n_channels != len(channels):
#             logger.error(f"Mismatch channels in full tensor ({n_channels}) vs names ({len(channels)}). Saving with generic indices.")
#             channels_coord = np.arange(n_channels)
#             channel_dim_name = "channel_idx"
#         else:
#             channels_coord = channels
#             channel_dim_name = "channel"

#         # --- Create Time Coordinates (Convert to np.datetime64) ---
#         time_coords_np = None # Initialize
#         try:
#             # Ensure time_step is timedelta
#             if not isinstance(time_step, datetime.timedelta):
#                 logger.warning(f"Model time_step is not a timedelta ({type(time_step)}), attempting conversion assuming hours.")
#                 # Attempt conversion or raise error depending on policy
#                 try:
#                      actual_time_step = datetime.timedelta(hours=float(time_step)) # Example conversion
#                 except ValueError:
#                      logger.error(f"Cannot interpret time_step '{time_step}' as hours.")
#                      raise TypeError("Invalid time_step type for time coordinate calculation.")
#             else:
#                 actual_time_step = time_step

#             # Generate list of Python datetime objects
#             time_coords_py = [initial_time + i * output_freq_for_coords * actual_time_step for i in range(n_time_out)]

#             # *** Convert to numpy.datetime64 for xarray ***
#             time_coords_np = np.array(time_coords_py, dtype='datetime64[ns]') # Use nanosecond precision standard
#             logger.debug(f"Generated and converted {len(time_coords_np)} time coordinates for full history (dtype: {time_coords_np.dtype}).")

#             if len(time_coords_np) != n_time_out:
#                  logger.warning(f"Generated {len(time_coords_np)} numpy time coordinates, but expected {n_time_out}. There might be an issue.")
#                  # Fallback might be less useful now, but kept for robustness
#                  time_coords_np = np.arange(n_time_out).astype('int64') # Use integer index as fallback

#         except Exception as e:
#             logger.error(f"Failed to create or convert time coordinates for full history: {e}", exc_info=True)
#             logger.warning("Using integer indices as fallback time coordinates.")
#             time_coords_np = np.arange(n_time_out).astype('int64') # Fallback to simple integer index

#         # --- Ensure Lat/Lon are Numpy (already required by type hint, but double-check) ---
#         if isinstance(lat, torch.Tensor) or isinstance(lon, torch.Tensor):
#              logger.warning("Received Lat/Lon as Tensors, expected NumPy arrays. Converting.")
#              lat_np = lat.cpu().numpy() if isinstance(lat, torch.Tensor) else np.asarray(lat)
#              lon_np = lon.cpu().numpy() if isinstance(lon, torch.Tensor) else np.asarray(lon)
#         else:
#              lat_np = np.asarray(lat) # Ensure it's a numpy array
#              lon_np = np.asarray(lon)

#         # --- Convert PyTorch Tensor to NumPy array (Major CPU Memory Allocation) ---
#         logger.debug("Converting full output tensor to NumPy array...")
#         conversion_start = time.time()
#         # Check for NaNs before conversion if desired (can be slow)
#         # if torch.isnan(output_tensor).any():
#         #    logger.warning("NaNs detected in PyTorch tensor before NumPy conversion!")
#         try:
#              output_tensor_np = output_tensor.numpy()
#              conversion_end = time.time()
#              logger.info(f"Converted tensor to NumPy array (shape: {output_tensor_np.shape}, dtype: {output_tensor_np.dtype}) in {conversion_end - conversion_start:.2f}s.")
#              # Explicitly delete the large PyTorch tensor IF it's safe to do so
#              # (i.e., it won't be needed again outside this function)
#              # del output_tensor # Uncomment cautiously if memory pressure is extreme
#         except Exception as e:
#              logger.error(f"Failed to convert output tensor to NumPy array: {e}", exc_info=True)
#              return # Cannot proceed without numpy array

#         # Check for NaNs in NumPy array (more common check)
#         if np.isnan(output_tensor_np).any():
#             nan_count = np.count_nonzero(np.isnan(output_tensor_np))
#             logger.warning(f"NaNs present in the NumPy output array ({nan_count} occurrences)! Saving may proceed but data is invalid.")

#         # --- Create xarray DataArray & Dataset ---
#         logger.debug("Creating xarray DataArray for full history...")
#         da_creation_start = time.time()
#         forecast_da = None # Initialize
#         forecast_ds = None # Initialize
#         try:
#             forecast_da = xr.DataArray(
#                 output_tensor_np, # Use the NumPy array
#                 coords={
#                     'ensemble': np.arange(n_ensemble),
#                     'time': time_coords_np, # Use numpy datetime64 coords
#                     channel_dim_name: channels_coord,
#                     'lat': lat_np,
#                     'lon': lon_np,
#                 },
#                 dims=['ensemble', 'time', channel_dim_name, 'lat', 'lon'],
#                 name='forecast',
#                 attrs={
#                     # (Attributes remain the same as before)
#                     'description': f"{model_name} full ensemble forecast output",
#                     'model': model_name,
#                     'simulation_length_steps': config['simulation_length'],
#                     'output_frequency_stored': output_freq_for_coords,
#                     'ensemble_members': n_ensemble,
#                     'initial_condition_time': initial_time.isoformat(),
#                     'time_step_seconds': actual_time_step.total_seconds() if isinstance(actual_time_step, datetime.timedelta) else 'unknown',
#                     'noise_amplitude': config.get("noise_amplitude", 0.0),
#                     'perturbation_strategy': config.get("perturbation_strategy", "N/A"),
#                     'creation_date': datetime.datetime.now(pytz.utc).isoformat(),
#                     'pytorch_version': torch.__version__,
#                     'numpy_version': np.__version__,
#                     'xarray_version': xr.__version__,
#                 }
#             )
#             da_creation_end = time.time()
#             logger.info(f"Created xarray DataArray in {da_creation_end - da_creation_start:.2f}s.")

#             # Free NumPy array memory IF DataArray creation was successful
#             # and the numpy array is no longer needed directly
#             logger.debug("Deleting NumPy tensor copy...")
#             del output_tensor_np
#             output_tensor_np = None # Ensure reference is gone

#             logger.debug(f"Converting DataArray to Dataset using dimension '{channel_dim_name}'...")
#             ds_creation_start = time.time()
#             forecast_ds = forecast_da.to_dataset(dim=channel_dim_name)
#             ds_creation_end = time.time()
#             logger.info(f"Converted DataArray to Dataset in {ds_creation_end - ds_creation_start:.2f}s.")

#             # Free DataArray memory if Dataset conversion was successful
#             logger.debug("Deleting intermediate DataArray...")
#             del forecast_da
#             forecast_da = None # Ensure reference is gone

#         except Exception as e:
#             logger.error(f"Failed during xarray DataArray/Dataset creation: {e}", exc_info=True)
#             # Clean up potentially created objects
#             del forecast_da
#             del forecast_ds
#             if 'output_tensor_np' in locals() and output_tensor_np is not None: del output_tensor_np
#             return # Cannot proceed

#         # --- Define Filename & Save ---
#         # Use requested format: "%d_%B_%Y_%H_%M"
#         try:
#             ic_time_str = initial_time.strftime('%d_%B_%Y_%H_%M')
#         except ValueError: # Handle potential issues with locale/names if needed
#              logger.warning("Could not format initial time with '%d_%B_%Y_%H_%M', using ISO format.")
#              ic_time_str = initial_time.strftime('%Y%m%dT%H%M%S')

#         output_filename = os.path.join(
#             output_dir,
#             f"{model_name}_ens{n_ensemble}_sim{config['simulation_length']}_IC{ic_time_str}_FULL.nc"
#         )

#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f"Saving full forecast history to: {output_filename}")

#         # Define encoding for compression
#         encoding = {var: {'zlib': True, 'complevel': 5, '_FillValue': np.float32(-9999.0)} for var in forecast_ds.data_vars} # Use float32 for fillvalue
#         logger.debug(f"Using encoding: {encoding}")

#         start_save = time.time()
#         forecast_ds.to_netcdf(output_filename, encoding=encoding, engine='netcdf4')
#         end_save = time.time()

#         # --- Final Logging and Cleanup ---
#         try:
#             file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
#             total_proc_time = time.time() - proc_start_time
#             logger.info(f"Save complete. NetCDF write time: {end_save - start_save:.2f}s. Total function time: {total_proc_time:.2f}s. File size: {file_size_mb:.2f} MB")
#         except OSError as e:
#             logger.error(f"Could not get file size for {output_filename}: {e}")
#             total_proc_time = time.time() - proc_start_time
#             logger.info(f"Save complete. NetCDF write time: {end_save - start_save:.2f}s. Total function time: {total_proc_time:.2f}s.")

#     except Exception as e:
#         # Log the top-level error
#         logger.error(f"Failed during the save_full_output process: {e}", exc_info=True)
#         # Attempt to remove potentially corrupted file
#         if 'output_filename' in locals() and os.path.exists(output_filename):
#             try:
#                 os.remove(output_filename)
#                 logger.warning(f"Removed potentially corrupted file: {output_filename}")
#             except OSError as oe:
#                 logger.error(f"Failed to remove corrupted file {output_filename}: {oe}")

#     finally:
#         # Explicitly delete large objects created within the function to aid GC
#         logger.debug("Cleaning up large objects in save_full_output...")
#         if 'output_tensor_np' in locals() and output_tensor_np is not None:
#             del output_tensor_np
#         if 'forecast_da' in locals() and forecast_da is not None:
#             del forecast_da
#         if 'forecast_ds' in locals() and forecast_ds is not None:
#             del forecast_ds
#         # output_tensor is passed in, deleting it here might affect caller
#         # Consider if the caller needs it after this function returns. If not,
#         # delete it in the caller *after* this function returns.
#         logger.debug("Finished cleanup in save_full_output.")
        
        
        
        
        
        
        








# # --- Main Inference Function using TimeLoop (run_inference) ---
# def run_inference(
#     model_inference: time_loop.TimeLoop,
#     initial_state_tensor: torch.Tensor,
#     initial_time_dt: datetime.datetime,
#     config: dict,
#     logger: logging.Logger,
#     save_func: Optional[Callable] = save_output_steps,
#     save_steps_config: Dict[str, Any] = {'steps_to_save': [-2, 0]},
#     output_dir: Optional[str] = None,
# ):
#     """
#     Runs the autoregressive ensemble forecast using the TimeLoop interface.
#     (Implementation remains the same as before)
#     """
#     # --- Configuration Extraction ---
#     n_ensemble = config.get("ensemble_members", 1)
#     simulation_length = config.get("simulation_length", 0)
#     output_freq = config.get("output_frequency", 1)
#     noise_amp = config.get("noise_amplitude", 0.0)
#     pert_strategy = config.get("perturbation_strategy", "gaussian")

#     logger.info(f"Starting inference run for IC: {initial_time_dt.isoformat()}")
#     logger.info(f"Ensemble members: {n_ensemble}, Simulation steps: {simulation_length}")
#     logger.info(f"Output collection/save frequency: {output_freq}")
#     logger.info(f"Perturbation: Amp={noise_amp}, Strategy='{pert_strategy}' (placeholder)")

#     # --- Validation ---
#     if not isinstance(initial_time_dt, datetime.datetime):
#         raise TypeError("initial_time_dt must be a datetime.datetime object.")
#     if initial_time_dt.tzinfo is None or initial_time_dt.tzinfo.utcoffset(initial_time_dt) is None:
#         logger.warning(f"Initial time {initial_time_dt.isoformat()} is timezone naive. Assuming UTC.")
#         initial_time_dt = initial_time_dt.replace(tzinfo=datetime.timezone.utc)

#     if initial_state_tensor.dim() != 4 or initial_state_tensor.shape[0] != 1:
#         raise ValueError(f"Invalid initial state tensor shape: {initial_state_tensor.shape}. Expected (1, C, H, W).")

#     perform_intermediate_saving = callable(save_func)
#     if perform_intermediate_saving and not output_dir:
#         raise ValueError("output_dir is required for intermediate saving.")

#     # --- Get Model Properties ---
#     try:
#         device = model_inference.device
#         n_history = getattr(model_inference, 'n_history', 0)
#         time_step_delta = model_inference.time_step
#         all_channels = model_inference.in_channel_names
#         lat = model_inference.grid.lat # Expect np.ndarray
#         lon = model_inference.grid.lon # Expect np.ndarray
#         logger.info(f"Model properties: Device={device}, n_history={n_history}, time_step={time_step_delta}")
#         logger.debug(f"Input channels ({len(all_channels)}): {all_channels}")
#     except AttributeError as e:
#         raise AttributeError(f"model_inference object missing required attributes: {e}")

#     # --- 1. Prepare Initial State ---
#     logger.debug("Preparing initial state...")
#     batch_tensor_4d = initial_state_tensor.repeat(n_ensemble, 1, 1, 1).to(device)
#     initial_state_5d = batch_tensor_4d.unsqueeze(1)
#     if n_history > 0 and initial_state_5d.shape[1] != n_history + 1:
#          logger.warning(f"Shape mismatch for history. Expected T={n_history+1}, got {initial_state_5d.shape[1]}. Assuming n_history=0.")
#          if initial_state_5d.shape[1] == 1:
#               logger.warning(f"Repeating initial state to match n_history={n_history}. This might be incorrect.")
#               initial_state_5d = initial_state_5d.repeat(1, n_history + 1, 1, 1, 1)
#     logger.info(f"  Prepared initial state for TimeLoop (E, T={initial_state_5d.shape[1]}, C, H, W): {initial_state_5d.shape}")

#     # --- 2. Normalize Initial State ---
#     logger.debug("Normalizing initial 5D state...")
#     try:
#         if not hasattr(model_inference, 'normalize') or not callable(model_inference.normalize):
#              raise AttributeError("model_inference missing required 'normalize' method.")
#         initial_state_norm_5d = model_inference.normalize(initial_state_5d)
#         logger.info("  Normalized initial 5D state.")
#         if torch.isnan(initial_state_norm_5d).any():
#             logger.warning("NaNs detected in initial normalized state!")
#     except Exception as e:
#         logger.error(f"Error during initial state normalization: {e}", exc_info=True)
#         raise

#     # --- 3. Apply Perturbation ---
#     initial_state_perturbed_norm_5d = initial_state_norm_5d.clone()
#     if noise_amp > 0 and n_ensemble > 1:
#         logger.info(f"Applying perturbation noise (Amp={noise_amp:.4f}, Strategy='{pert_strategy}')")
#         noise = torch.randn_like(initial_state_perturbed_norm_5d) * noise_amp
#         if n_ensemble > 1: noise[0, ...] = 0 # Keep member 0 deterministic
#         initial_state_perturbed_norm_5d += noise
#         logger.info("  Applied Gaussian noise to initial state.")
#         if torch.isnan(initial_state_perturbed_norm_5d).any():
#             logger.warning("NaNs detected after adding noise to initial state!")
#     else:
#         logger.info("No perturbation noise applied to initial state.")

#     # --- 4. Execute TimeLoop Iterator ---
#     output_history_buffer = collections.deque()
#     output_tensors_full_history = []
#     steps_relative_to_save = sorted(save_steps_config.get('steps_to_save', [0]))
#     max_offset = abs(min(steps_relative_to_save)) if steps_relative_to_save else 0
#     buffer_size = max_offset + 1

#     inference_times = []
#     logger.info(f"Initializing TimeLoop iterator starting from {initial_time_dt.isoformat()}")
#     logger.info(f"Target simulation steps: {simulation_length}. Iterator will run {simulation_length + 1} times.")
#     if perform_intermediate_saving:
#          logger.info(f"Intermediate saving enabled. Steps relative to current to save: {steps_relative_to_save}. Buffer size: {buffer_size}")
#          logger.info(f"Output files will be saved to: {output_dir}")
#     else:
#          logger.info(f"Intermediate saving disabled. Collecting full history in memory (output_freq={output_freq}).")

#     overall_start_time = time.time()
#     model_step_counter = 0

#     try:
#         iterator = model_inference(time=initial_time_dt, x=initial_state_perturbed_norm_5d)
#         num_iterations_done = 0
#         for i in range(simulation_length + 1):
#             iter_start_time = time.time()
#             logger.debug(f"--- Iterator Step {i} (Model Step {model_step_counter}) ---")

#             try:
#                 time_out, data_denorm, _ = next(iterator)
#                 logger.debug(f"  Iterator yielded: Time={time_out.isoformat()}, Output shape={data_denorm.shape}")
#                 num_iterations_done += 1
#             except StopIteration:
#                 logger.warning(f"Iterator stopped unexpectedly after {num_iterations_done} iterations (model step {model_step_counter}). Expected {simulation_length + 1} iterations.")
#                 break
#             except Exception as iter_err:
#                 logger.error(f"Error during next(iterator) at step {i}: {iter_err}", exc_info=True)
#                 # Maybe log the state shape before error? model_inference object stores 'x' internally.
#                 try:
#                     logger.error(f"  Shape of internal state 'x' before error: {model_inference.x.shape}")
#                 except AttributeError:
#                     logger.error("  Could not retrieve internal state 'x'.")
#                 raise # Re-raise the error to stop the process for this IC

#             # --- Output Handling ---
#             data_denorm_cpu = data_denorm.cpu()

#             if perform_intermediate_saving:
#                 output_history_buffer.append((model_step_counter, time_out, data_denorm_cpu))
#                 while len(output_history_buffer) > buffer_size:
#                     output_history_buffer.popleft()

#                 can_save_all_steps = True
#                 steps_to_save_indices = {}
#                 times_to_save = {}
#                 for offset in steps_relative_to_save:
#                     target_step_index = model_step_counter + offset
#                     found = False
#                     for step_idx, step_time, step_data in output_history_buffer:
#                         if step_idx == target_step_index:
#                             steps_to_save_indices[target_step_index] = step_data
#                             times_to_save[target_step_index] = step_time
#                             found = True
#                             break
#                     if not found:
#                         can_save_all_steps = False
#                         logger.debug(f"  Cannot save yet: required step {target_step_index} (offset {offset}) not in buffer.")
#                         break

#                 if can_save_all_steps and steps_to_save_indices: # Ensure there's something to save
#                     logger.debug(f"  All required steps {list(steps_to_save_indices.keys())} found. Triggering save...")
#                     try:
#                         save_func(
#                             data_dict=steps_to_save_indices, time_dict=times_to_save,
#                             channels=all_channels, lat=lat, lon=lon, config=config,
#                             output_dir=output_dir, current_model_step=model_step_counter, logger=logger
#                         )
#                     except Exception as save_e:
#                          logger.error(f"Error during intermediate save call for step {model_step_counter}: {save_e}", exc_info=True)

#             else: # Collecting full history
#                 if model_step_counter % output_freq == 0:
#                     logger.debug(f"  Collecting output for model step {model_step_counter} in memory.")
#                     output_tensors_full_history.append(data_denorm_cpu)

#             # --- Timing and Increment ---
#             iter_end_time = time.time()
#             step_duration = iter_end_time - iter_start_time
#             inference_times.append(step_duration)
#             logger.debug(f"  Iterator Step {i} finished in {step_duration:.3f} seconds.")
#             model_step_counter += 1

#         logger.info(f"Finished {num_iterations_done} iterations over TimeLoop.")

#     except Exception as e:
#         logger.error(f"Error occurred during TimeLoop iteration for IC {initial_time_dt.isoformat()}: {e}", exc_info=True)
#         return None # Indicate failure

#     finally:
#          if torch.cuda.is_available(): torch.cuda.empty_cache()

#     overall_end_time = time.time()
#     total_duration = overall_end_time - overall_start_time
#     avg_inference_time = np.mean(inference_times) if inference_times else 0
#     logger.info(f"TimeLoop execution finished. Total time: {total_duration:.2f}s. Average step time: {avg_inference_time:.3f}s.")

#     # --- 5. Combine and Return Full History ---
#     if not perform_intermediate_saving:
#         if not output_tensors_full_history:
#             logger.warning("No output tensors were collected for full history!")
#             return None
#         logger.info(f"Stacking {len(output_tensors_full_history)} collected output tensors...")
#         try:
#             final_output_tensor = torch.stack(output_tensors_full_history, dim=1)
#             logger.info(f"Final aggregated output tensor shape: {final_output_tensor.shape}")
#             if torch.isnan(final_output_tensor).any():
#                 logger.warning("NaNs detected in the final aggregated output tensor!")
#             return final_output_tensor
#         except Exception as e:
#             logger.error(f"Failed to stack collected output tensors: {e}", exc_info=True)
#             return None
#     else:
#         logger.info("Intermediate saving was performed. Returning None.")
#         return None


# # --- ** NEW Plotting Function ** ---
# def plot_z500_progression(
#     forecast_tensor: torch.Tensor, # Full history tensor (E, T, C, H, W) on CPU
#     channels: List[str],
#     lat: np.ndarray,
#     lon: np.ndarray,
#     initial_time: datetime.datetime,
#     time_step: datetime.timedelta,
#     output_dir: str, # Base directory for NetCDF outputs
#     logger: logging.Logger,
#     config: dict,
#     output_freq: int = 1 # How often steps were stored in forecast_tensor
# ):
#     """
#     Plots the progression of the z500 field for the first ensemble member
#     over all forecasted time steps. Saves plots to a dedicated subdirectory.

#     Args:
#         forecast_tensor: The tensor containing the full forecast history
#                          (Shape: E, T, C, H, W), on CPU.
#         channels: List of channel names corresponding to dimension C.
#         lat: Latitude coordinates (1D numpy array).
#         lon: Longitude coordinates (1D numpy array).
#         initial_time: Datetime object for the initial condition (t=0).
#         time_step: Timedelta object representing the model's time step duration.
#         output_dir: The base directory where NetCDF files are saved. Plots
#                     will be saved in a subdirectory within this directory.
#         logger: Logger object.
#         config: The main inference configuration dictionary.
#         output_freq: The frequency (in model steps) at which states were
#                      stored in the forecast_tensor. Important for calculating
#                      correct forecast times.
#     """
#     if not PLOTTING_ENABLED:
#         logger.warning("Plotting libraries not found. Skipping z500 progression plot.")
#         return

#     logger.info(f"--- Starting z500 Progression Plotting for IC: {initial_time.isoformat()} ---")

#     try:
#         # --- Find z500 channel index ---
#         try:
#             z500_index = channels.index("z500")
#             logger.debug(f"Found 'z500' channel at index {z500_index}.")
#         except ValueError:
#             logger.error(f"'z500' channel not found in the provided channel list: {channels}. Cannot generate plots.")
#             return

#         # --- Extract Data ---
#         # Shape: (E, T, C, H, W) -> (E, T, H, W) for z500
#         z500_data = forecast_tensor[:, :, z500_index, :, :].numpy() # Convert to numpy
#         n_ensemble, n_time_steps_out, n_lat, n_lon = z500_data.shape

#         # Select first ensemble member for plotting
#         member_data = z500_data[0, :, :, :] # Shape: (T, H, W)
#         logger.info(f"Extracted z500 data for first ensemble member. Shape: {member_data.shape}")

#         # --- Calculate Time Coordinates for Plots ---
#         plot_times = []
#         actual_forecast_times = [] # Actual datetime objects
#         try:
#              # Calculate time for each step stored in the tensor
#              for i in range(n_time_steps_out):
#                  # The i-th stored step corresponds to model step i * output_freq
#                  model_step_number = i * output_freq
#                  forecast_time = initial_time + model_step_number * time_step
#                  lead_time_hrs = model_step_number * time_step.total_seconds() / 3600
#                  plot_times.append((model_step_number, forecast_time, lead_time_hrs))
#                  actual_forecast_times.append(forecast_time)
#              logger.debug(f"Calculated {len(plot_times)} timestamps for plotting.")
#         except Exception as e:
#              logger.error(f"Error calculating plot timestamps: {e}", exc_info=True)
#              return # Cannot proceed without times

#         # --- Create Output Subdirectory ---
#         ic_time_str = initial_time.strftime('%d_%B_%Y_%H_%M')
#         plot_subdir = os.path.join(output_dir, f"z500_plots_IC_{ic_time_str}")
#         try:
#             os.makedirs(plot_subdir, exist_ok=True)
#             logger.info(f"Saving z500 plots to subdirectory: {plot_subdir}")
#         except OSError as e:
#             logger.error(f"Failed to create plot subdirectory {plot_subdir}: {e}")
#             return

#         # --- Setup Plotting ---
#         proj = ccrs.PlateCarree() # Global plot projection
#         # Define reasonable normalization for z500 (in meters)
#         # Adjust vmin/vmax based on expected range, e.g., mid-latitude winter
#         z500_min = 4800
#         z500_max = 6000
#         norm = Normalize(vmin=z500_min, vmax=z500_max)
#         cmap = 'viridis' # Or 'jet', 'coolwarm', etc.

#         # Cartopy features (load once)
#         try:
#             countries = cfeature.NaturalEarthFeature(
#                 category="cultural", name="admin_0_countries", scale="110m",
#                 facecolor="none", edgecolor='gray'
#             )
#             coastlines = cfeature.NaturalEarthFeature(
#                 category='physical', name='coastline', scale='110m',
#                 facecolor='none', edgecolor='black'
#             )
#         except Exception as feat_err:
#             logger.warning(f"Could not load some Cartopy features: {feat_err}. Plots might look simpler.")
#             countries = None
#             coastlines = None

#         # --- Loop Through Time Steps and Plot ---
#         plot_count = 0
#         start_plot_time = time.time()
#         for t_idx in range(n_time_steps_out):
#             step_data = member_data[t_idx, :, :] # Data for this step (H, W)
#             model_step, forecast_dt, lead_hrs = plot_times[t_idx]

#             fig = plt.figure(figsize=(12, 6))
#             ax = fig.add_subplot(1, 1, 1, projection=proj)
#             ax.set_global() # Zoom out to see the whole world

#             try:
#                 # Check for NaNs before plotting
#                 if np.isnan(step_data).any():
#                     logger.warning(f"NaNs detected in z500 data for step {model_step}. Plot may be incomplete.")
#                     # Optional: fill NaNs for plotting, e.g., step_data = np.nan_to_num(step_data, nan=z500_min)

#                 img = ax.pcolormesh(
#                     lon, lat, step_data,
#                     transform=ccrs.PlateCarree(), # Data coordinates are lat/lon
#                     cmap=cmap,
#                     norm=norm
#                 )

#                 # Add features
#                 if coastlines: ax.add_feature(coastlines, linewidth=0.5)
#                 if countries: ax.add_feature(countries, linewidth=0.3)
#                 # ax.gridlines(draw_labels=True, linestyle='--') # Can be slow/clutter global plots

#                 # Add Colorbar
#                 cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
#                 cbar.set_label('Geopotential Height at 500 hPa (m)')

#                 # Add Title
#                 title = (f"Z500 Forecast (Ens Member 0)\n"
#                          f"IC: {initial_time.strftime('%d_%B_%Y_%H_%M')}"
#                          f"Forecast: {forecast_dt.strftime('%d_%B_%Y_%H_%M')}\n"
#                          f"Lead Time: {lead_hrs:.1f} hrs (Model Step {model_step})")
#                 ax.set_title(title, fontsize=10)

#                 # --- Save Figure ---
#                 plot_filename = os.path.join(plot_subdir, f"z500_step_{model_step:04d}_{forecast_dt.strftime('%d_%B_%Y_%H_%M')}.png")
#                 try:
#                     plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
#                     logger.debug(f"Saved plot: {plot_filename}")
#                     plot_count += 1
#                 except Exception as save_err:
#                     logger.error(f"Failed to save plot {plot_filename}: {save_err}")

#             except Exception as plot_err:
#                  logger.error(f"Failed to generate plot for step {model_step} ({forecast_dt.isoformat()}): {plot_err}", exc_info=True)
#             finally:
#                  plt.close(fig) # IMPORTANT: Close figure to free memory

#         end_plot_time = time.time()
#         if plot_count > 0:
#              logger.info(f"Finished plotting {plot_count}/{n_time_steps_out} z500 steps in {end_plot_time - start_plot_time:.2f} seconds.")
#         else:
#              logger.warning(f"No z500 plots were successfully generated for IC {initial_time.isoformat()}.")

#     except Exception as e:
#         logger.error(f"An unexpected error occurred during z500 plotting for IC {initial_time.isoformat()}: {e}", exc_info=True)


# # --- Add Memory Logging Utility ---
# def log_gpu_memory(logger, point="Point"):
#     if torch.cuda.is_available():
#         allocated = torch.cuda.memory_allocated() / (1024**3)
#         reserved = torch.cuda.memory_reserved() / (1024**3)
#         max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
#         max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
#         logger.info(f"GPU Memory @ {point}: Allocated={allocated:.2f} GiB, Reserved={reserved:.2f} GiB")
#         # Reset peak counters for next section if desired
#         # torch.cuda.reset_peak_memory_stats()
#     else:
#         logger.debug(f"GPU Memory @ {point}: CUDA not available.")


# # --- Updated Main Pipeline Function ---
# def main(args, save_steps_config, netcdf_output_dir): # Add new args
#     """Main pipeline execution function."""

#     logger.info("========================================================")
#     logger.info(" Starting FCNv2-SM Inference Pipeline from NumPy ICs")
#     logger.info("========================================================")
#     logger.info(f"Parsed arguments: {vars(args)}")
#     logger.info(f"Save mode: {args.save_mode}")
#     if args.save_mode == 'intermediate':
#         logger.info(f"Intermediate save steps config: {save_steps_config}")
#     logger.info(f"Plotting enabled: {PLOTTING_ENABLED}")
#     logger.info(f"NetCDF output directory: {netcdf_output_dir}")

#     # --- Environment and Setup ---
#     if args.gpu >= 0 and torch.cuda.is_available():
#         try:
#             device = torch.device(f"cuda:{args.gpu}")
#             torch.cuda.set_device(device)
#             logger.info(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(device)})")
#         except Exception as e:
#             logger.error(f"Failed to set CUDA device {args.gpu}: {e}. Falling back to CPU.", exc_info=True)
#             device = torch.device("cpu")
#     else:
#         device = torch.device("cpu")
#         logger.info("Using CPU.")

#     log_gpu_memory(logger, "Start of main")

#     # --- Load Model ---
#     model_id = "fcnv2_sm"
#     logger.info(f"Loading {model_id} model...")
#     try:
#         package = registry.get_model(model_id)
#         if package is None: raise FileNotFoundError(f"Model package '{model_id}' not found in registry.")
#         logger.info(f"Found model package: {package.root}")
#         model_inference = fcnv2_sm_load(package, device=device, pretrained=True)
#         model_inference.eval()
#         log_gpu_memory(logger, "After model load")

#         logger.info(f"{model_id} model loaded to device: {next(model_inference.parameters()).device}.")
#         logger.info(f"Input channels ({len(model_inference.in_channel_names)}): {model_inference.in_channel_names}")
#         logger.info(f"Output channels ({len(model_inference.out_channel_names)}): {model_inference.out_channel_names}")
#         logger.info(f"Model time step: {model_inference.time_step}")
#         # Ensure grid info is numpy for plotting/saving
#         if isinstance(model_inference.grid.lat, torch.Tensor): model_inference.grid.lat = model_inference.grid.lat.cpu().numpy()
#         if isinstance(model_inference.grid.lon, torch.Tensor): model_inference.grid.lon = model_inference.grid.lon.cpu().numpy()

#     except FileNotFoundError as e:
#         logger.error(f"Model loading failed: {e}", exc_info=True)
#         logger.error(f"Check weights/means/stds files in {os.path.join(os.environ.get('MODEL_REGISTRY'), model_id)}")
#         sys.exit(1)
#     except _pickle.UnpicklingError as e:
#         logger.error(f"Model loading failed (UnpicklingError): {e}", exc_info=False)
#         logger.error("Possible weights_only=True issue. Ensure earth2mip/networks/fcnv2_sm.py uses weights_only=False.")
#         sys.exit(1)
#     except Exception as e:
#         logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
#         sys.exit(1)





#     # --- Load Initial Conditions ---
#     logger.info(f"Loading initial conditions from: {args.ic_file_path}")
#     if not os.path.exists(args.ic_file_path):
#         logger.error(f"IC file not found: {args.ic_file_path}"); sys.exit(1)
#     try:
#         initial_conditions_np = np.load(args.ic_file_path)
#         logger.info(f"Loaded NumPy IC data shape: {initial_conditions_np.shape}, dtype: {initial_conditions_np.dtype}")
#         if initial_conditions_np.ndim != 4: raise ValueError(f"Expected 4D ICs (T, C, H, W), got {initial_conditions_np.ndim}D")

#         num_ics, num_channels, height, width = initial_conditions_np.shape
#         logger.info(f"Found {num_ics} ICs. Grid: {height}x{width}. Channels: {num_channels}")

#         # Validate channels and grid
#         model_channels = model_inference.in_channel_names
#         if num_channels != len(model_channels):
#             raise ValueError(f"Channel count mismatch! Model={len(model_channels)}, File={num_channels}. Model channels: {model_channels}")
#         model_lat, model_lon = model_inference.grid.lat, model_inference.grid.lon
#         if height != len(model_lat) or width != len(model_lon):
#             logger.warning(f"Grid mismatch! Model={len(model_lat)}x{len(model_lon)}, File={height}x{width}. Proceeding cautiously.")

#     except Exception as e:
#         logger.error(f"Failed to load/validate IC NumPy file: {e}", exc_info=True)
#         sys.exit(1)






#     # --- Define Timestamps for ICs ---
#     try:
#         fname = os.path.basename(args.ic_file_path)
#         import re
#         match = re.search(r"(\d{4})_(\d{2})_(\d{2})", fname) or re.search(r"(\d{8})", fname)
#         if match:
#             if len(match.groups()) == 3: year, month, day = map(int, match.groups())
#             else: date_str = match.group(1); year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
#             base_date = datetime.datetime(year, month, day, tzinfo=pytz.utc)
#             logger.info(f"Inferred base date from filename: {base_date.strftime('%d_%B_%Y_%H_%M')}")
#         else:
#             base_date = datetime.datetime(2020, 2, 7, tzinfo=pytz.utc) # Default fallback
#             logger.warning(f"Using default base date: {base_date.strftime('%d_%B_%Y_%H_%M')}")

#         ic_timestamps = [base_date + datetime.timedelta(hours=i * 6) for i in range(num_ics)] # Assuming 6hr steps in file
#         logger.info(f"Generated {len(ic_timestamps)} timestamps starting from {ic_timestamps[0].isoformat()}")

#     except Exception as e:
#         logger.error(f"Error determining timestamps: {e}. Using indices.", exc_info=True)
#         ic_timestamps = list(range(num_ics))









#     # --- Prepare Inference Configuration ---
#     inference_config = {
#         "ensemble_members": args.ensemble_members,
#         "noise_amplitude": args.noise_amplitude,
#         "simulation_length": args.simulation_length,
#         "output_frequency": args.output_frequency, # Crucial for plotting/saving time coords
#         "weather_model": model_id,
#         "perturbation_strategy": args.perturbation_strategy,
#         'variables_to_save': None # Set if subsetting needed for intermediate saves
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
#              num_ics_failed += 1; continue
#         if initial_time.tzinfo is None: initial_time = initial_time.replace(tzinfo=pytz.utc)
#         time_label = initial_time.isoformat()
#         logger.info(f"--- Processing Initial Condition {i+1}/{num_ics}: {time_label} ---")
#         log_gpu_memory(logger, f"Start of IC {i+1} Loop")

#         # --- Prepare IC Tensor ---
#         try:
#             ic_data_np = initial_conditions_np[i]
#             initial_state_tensor = torch.from_numpy(ic_data_np).unsqueeze(0).float() # (1, C, H, W)
#         except Exception as e:
#             logger.error(f"Failed to get/convert NumPy slice {i}: {e}", exc_info=True)
#             num_ics_failed += 1; continue

#         # --- Execute Inference ---
#         start_run = time.time()
#         output_tensor_full = None
#         run_successful = False

#         try:
#             if args.save_mode == 'intermediate':
#                 logger.info("Running inference in 'intermediate' save mode.")
#                 run_inference(
#                     model_inference=model_inference, initial_state_tensor=initial_state_tensor,
#                     initial_time_dt=initial_time, config=inference_config, logger=logger,
#                     save_func=save_output_steps, save_steps_config=save_steps_config,
#                     output_dir=netcdf_output_dir
#                 )
#                 run_successful = True # Assume success if no exception

#             elif args.save_mode == 'full':
#                 logger.info("Running inference in 'full' save mode.")
#                 output_tensor_full = run_inference(
#                     model_inference=model_inference, initial_state_tensor=initial_state_tensor,
#                     initial_time_dt=initial_time, config=inference_config, logger=logger,
#                     save_func=None, save_steps_config={}, output_dir=None
#                 )
#                 run_successful = output_tensor_full is not None
#             else:
#                 logger.error(f"Invalid save_mode: {args.save_mode}") # Should be caught by argparse

#             end_run = time.time()
#             log_gpu_memory(logger, f"After run_inference IC {i+1}")

#             # --- Post-Inference Processing ---
#             if run_successful:
#                 logger.info(f"Inference run for IC {time_label} completed in {end_run - start_run:.2f} seconds.")
#                 num_ics_processed += 1

#                 # Save full history if collected
#                 if args.save_mode == 'full' and output_tensor_full is not None:
#                     try:
#                         save_full_output(
#                             output_tensor=output_tensor_full, initial_time=initial_time,
#                             time_step=model_inference.time_step, channels=model_channels,
#                             lat=model_lat, lon=model_lon, config=inference_config,
#                             output_dir=netcdf_output_dir, logger=logger
#                         )
#                         log_gpu_memory(logger, f"After saving full history IC {i+1}")

#                         # <<< --- CALL PLOTTING FUNCTION --- >>>
#                         if PLOTTING_ENABLED and args.plot_z500:
#                             plot_z500_progression(
#                                 forecast_tensor=output_tensor_full, # Pass the full tensor
#                                 channels=model_channels,
#                                 lat=model_lat,
#                                 lon=model_lon,
#                                 initial_time=initial_time,
#                                 time_step=model_inference.time_step,
#                                 output_dir=netcdf_output_dir, # Base dir for plots
#                                 logger=logger,
#                                 config=inference_config,
#                                 output_freq=inference_config['output_frequency'] # Pass freq
#                             )
#                             log_gpu_memory(logger, f"After plotting z500 IC {i+1}")
#                         elif args.plot_z500:
#                              logger.warning("Plotting was requested (--plot-z500) but plotting libraries are not available.")
#                         # <<< ----------------------------- >>>

#                     except Exception as post_err:
#                          logger.error(f"Error during saving/plotting for IC {time_label}: {post_err}", exc_info=True)
#                          # Don't count as failed run, but log the error
#                     finally:
#                          del output_tensor_full # Crucial to free CPU RAM
#                          output_tensor_full = None # Ensure it's cleared
#             else:
#                 logger.error(f"Inference run failed for IC {time_label}.")
#                 num_ics_failed += 1

#         except Exception as run_err:
#              logger.error(f"Unhandled exception during run/save/plot for IC {time_label}: {run_err}", exc_info=True)
#              num_ics_failed += 1
#              end_run = time.time()
#              log_gpu_memory(logger, f"After failed run_inference IC {i+1}")
#              logger.info(f"Inference run attempt for IC {time_label} took {end_run - start_run:.2f}s before failing.")

#         # --- Cleanup GPU Cache ---
#         del initial_state_tensor
#         if torch.cuda.is_available(): torch.cuda.empty_cache()
#         log_gpu_memory(logger, f"End of IC {i+1} Loop")






#     # --- Final Summary ---
#     total_end_time = time.time()
#     logger.info(f"--- Total processing time for {num_ics} ICs: {total_end_time - total_start_time:.2f} seconds ---")
#     logger.info(f"Successfully processed {num_ics_processed} initial conditions.")
#     if num_ics_failed > 0: logger.warning(f"Failed to process {num_ics_failed} initial conditions.")
#     logger.info(f"Output NetCDF files saved in: {netcdf_output_dir}")
#     if PLOTTING_ENABLED and args.plot_z500 and args.save_mode == 'full' and num_ics_processed > 0:
#         logger.info(f"Z500 plots saved in subdirectories within: {netcdf_output_dir}")
#     logger.info(f"Log file saved in: {LOG_DIR}")
#     logger.info("========================================================")
#     logger.info(" FCNv2-SM Inference Pipeline Finished ")
#     logger.info("========================================================")
#     log_gpu_memory(logger, "End of main")


# # --- Main Execution Block ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="FCNv2-SM Inference Pipeline using NumPy ICs.")

#     # Input/Output
#     parser.add_argument("--ic-file-path", type=str, default=f"/scratch/gilbreth/gupt1075/fcnv2/ARCO_data_73_channels/data/START_07_February_2020__len_4__END_07_February_2020.npy", help="Path to NumPy IC file (T, C, H, W).")
#     parser.add_argument("-o", "--output-path", type=str, default=os.path.join(OUTPUT_DIR, "inference_output"), help="Directory for NetCDF output (and plot subdirs).")

#     # Inference parameters
#     parser.add_argument("-sim", "--simulation-length", type=int, default=1, help="Number of forecast steps (e.g., 4 steps * 6hr = 24hr).") # Increased default
#     parser.add_argument("-ef", "--output-frequency", type=int, default=1, help="Frequency (steps) to store outputs in 'full' mode.")
#     parser.add_argument("-ens", "--ensemble-members", type=int, default=1, help="Number of ensemble members.")
#     parser.add_argument("-na", "--noise-amplitude", type=float, default=0.0, help="Perturbation noise amplitude (if ens > 1).")
#     parser.add_argument("-ps", "--perturbation-strategy", type=str, default="gaussian", choices=["gaussian"], help="Perturbation strategy.")

#     # Saving & Plotting
#     parser.add_argument("--save-mode", type=str, default="full", choices=["intermediate", "full"], help="Saving mode: 'intermediate' (recent steps) or 'full' (all steps). Plotting only works with 'full'.") # Changed default to 'full' for plotting
#     parser.add_argument("--save-steps", type=str, default="-2,0", help="Steps to save in 'intermediate' mode (rel. to current, comma-sep).")
#     parser.add_argument("--plot-z500", action='store_true', help="Generate Z500 progression plots (requires save_mode='full' and plotting libraries).") # Added flag

#     # System
#     parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU).")
#     parser.add_argument("--debug", action='store_true', help="Enable debug logging.") # Use action='store_true'

#     args = parser.parse_args()

#     # Process save_steps
#     try:
#         save_steps_list = [int(step.strip()) for step in args.save_steps.split(',')]
#     except ValueError:
#         logger.critical(f"Invalid --save-steps format: '{args.save_steps}'. Use comma-separated integers.")
#         sys.exit(1)
#     save_steps_config = {'steps_to_save': save_steps_list}

#     # Check save mode for plotting
#     if args.plot_z500 and args.save_mode != 'full':
#          logger.warning(f"Plotting requested (--plot-z500) but save_mode is '{args.save_mode}'. Plotting requires save_mode='full'. Disabling plotting.")
#          args.plot_z500 = False # Override plot flag

#     # Output directory setup
#     netcdf_output_dir = args.output_path
#     os.makedirs(netcdf_output_dir, exist_ok=True)

#     # Adjust logger level if debug flag is set AFTER logger initialized
#     if args.debug:
#         logger.setLevel(logging.DEBUG)
#         for handler in logger.handlers: handler.setLevel(logging.DEBUG)
#         logger.info("Debug logging enabled.")

#     # Call main
#     try:
#         main(args, save_steps_config, netcdf_output_dir)
#     except Exception as e:
#         logger.critical(f"Critical pipeline failure in main: {str(e)}", exc_info=True)
#         sys.exit(1)
