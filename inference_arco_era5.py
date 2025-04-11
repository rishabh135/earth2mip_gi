

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
import gc
from typing import Optional
import numpy as np
# Need these imports if not already present at the top
import collections
import pickle
import pytz
from typing import List, Dict, Optional, Callable, Any, Tuple # For type hints
import torch
import re
import gc


import time
import logging
from typing import Optional, Any, Iterator, Tuple, List, Dict
import collections # For deque

from typing import Optional
import torch
import numpy as np

import time
import logging
import collections
from typing import Optional, Callable, Dict, Any, Tuple, List # Added Tuple, List
import argparse
import time
import sys
import _pickle
import torch
import xarray as xr
from dateutil.relativedelta import relativedelta
import dotenv
import pytz
import re
from typing import Optional



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
OUTPUT_DIR = f"/scratch/gilbreth/wwtung/FourCastNetV2_RESULTS_2025/ZETA_INFERENCE_{timestamp}"
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







"""

    Memory Management Review:

        The original code deleted initial_state_5d_cpu, center, scale after the (failed) normalization.

        My fix introduces center_np, scale_np, center_tensor_cpu, scale_tensor_cpu, center_reshaped_cpu, scale_reshaped_cpu.

        The finally block ensures these intermediate variables and the original initial_state_5d_cpu (which is replaced by initial_state_norm_5d_cpu) are deleted regardless of whether an error occurs during normalization (though if an error does occur, the function might exit early anyway depending on where the raise happens). This seems robust.

        The variable initial_state_norm_5d_cpu (the result) is correctly preserved for the next steps.

    Robustness Review:

        Added type checks for center_np and scale_np.

        Added shape validation against the input tensor's channel dimension.

        Added a warning for zero values in the scale tensor.

        Added try...except...finally block for better error handling and guaranteed cleanup. Catches potential AttributeError if model_inference doesn't have the stats, ValueError for shape issues, and general Exception.

    Final Code Structure Check: The fix is localized to the normalization section. The rest of the function logic (perturbation, GPU transfer, TimeLoop iteration, output collection) remains the same and should work correctly with the properly normalized initial_state_norm_5d_cpu (which is renamed to initial_state_perturbed_norm_5d_cpu after perturbation, and then transferred to GPU).

Looks like a solid plan and implementation. I will now integrate this fix into the full function provided by the user.
One more check: The deletion of initial_state_5d_cpu happens inside the finally block. The subsequent code block (perturbation) uses initial_state_norm_5d_cpu. This is correct. The variable initial_state_norm_5d_cpu is created before the finally block, so it persists. Good.

The log_gpu_mem flag isn't used in the provided snippet, but the docstring mentions it. I won't add explicit GPU memory logging unless requested, but the memory management improvements (CPU pre-processing, del calls) align with the function's goal of being memory-optimized.
The pert_strategy variable is extracted but not actually used in the perturbation block (it defaults to Gaussian noise regardless). This might be an omission in the original code, but fixing it is outside the scope of the specific error addressed. I'll leave it as is.
The error handling in the main loop seems okay (try...except StopIteration and general Exception).
The final output combination using torch.stack is standard.

The fix seems complete and addresses the specific AttributeError while enhancing robustness and maintaining the memory optimization goal.
"""

# --- Updated Function ---
def run_inference(
    model_inference,
    initial_state_tensor: torch.Tensor,
    initial_time_dt: datetime.datetime,
    config: dict,
    logger: logging.Logger,
    log_gpu_mem: bool = False, # Flag kept for consistency, implementation not added here
) -> Optional[torch.Tensor]:
    """
    Optimized and robust function to run autoregressive ensemble forecasts using TimeLoop.

    Performs initial state preparation, normalization, and perturbation on CPU
    to avoid GPU OOM errors with large ensemble/history states. Transfers data
    to GPU just before the inference loop. Includes aggressive memory management
    and detailed logging. Addresses potential issues with NumPy array handling during normalization.

    Args:
        model_inference: The loaded model with a TimeLoop interface. Expected to have
                         `center_np` and `scale_np` NumPy array attributes for normalization.
        initial_state_tensor: Initial condition tensor (1, C, H, W), expected on CPU.
        initial_time_dt: Timezone-aware datetime for the initial state.
        config: Configuration dictionary with keys like 'ensemble_members', 'simulation_length', etc.
        logger: Configured logger instance.
        log_gpu_mem: If True, logs GPU memory usage at critical points (basic implementation here).

    Returns:
        Optional[torch.Tensor]: Full forecast history (E, T_out, C, H, W) on CPU if
                                successful. None on failure.
    """
    overall_start_time = time.time()
    logger.info("=" * 50)
    logger.info(f"Starting run_inference for IC: {initial_time_dt.isoformat()}")
    logger.info("=" * 50)

    # --- Configuration Extraction ---
    n_ensemble = config.get("ensemble_members", 1)
    simulation_length = config.get("simulation_length", 0) # Num steps *after* initial state
    output_freq = config.get("output_frequency", 1)
    noise_amp = config.get("noise_amplitude", 0.0)
    pert_strategy = config.get("perturbation_strategy", "gaussian") # Note: Strategy currently only supports Gaussian

    logger.info(f"Config - Ensemble: {n_ensemble}, Sim Length: {simulation_length}, Output Freq: {output_freq}")
    logger.info(f"Config - Perturbation: Amp={noise_amp:.4e}, Strategy='{pert_strategy}'")

    # --- Validation ---
    if not isinstance(initial_state_tensor, torch.Tensor):
        logger.error(f"Input initial_state_tensor is not a PyTorch Tensor. Got type: {type(initial_state_tensor)}")
        raise TypeError("initial_state_tensor must be a PyTorch Tensor.")

    if initial_state_tensor.dim() != 4 or initial_state_tensor.shape[0] != 1:
        logger.error(f"Input IC tensor shape invalid: {initial_state_tensor.shape}. Expected (1, C, H, W).")
        raise ValueError("Invalid initial state tensor shape.")

    # Ensure input tensor is on CPU before proceeding with CPU operations
    if initial_state_tensor.is_cuda:
        logger.warning("Initial state tensor was on GPU, moving to CPU for pre-processing.")
        initial_state_tensor = initial_state_tensor.cpu()

    # Ensure float type (adjust if model uses float64)
    if not torch.is_floating_point(initial_state_tensor):
         logger.warning(f"Initial state tensor is not float type (got {initial_state_tensor.dtype}). Casting to float32.")
         initial_state_tensor = initial_state_tensor.float()

    if initial_time_dt.tzinfo is None or initial_time_dt.tzinfo.utcoffset(initial_time_dt) is None:
        logger.warning(f"Initial time {initial_time_dt.isoformat()} is timezone naive. Assuming UTC.")
        initial_time_dt = initial_time_dt.replace(tzinfo=datetime.timezone.utc)

    # --- Prepare Device ---
    device = None
    try:
        # Try accessing device directly first
        device = model_inference.device
        logger.info(f"Device retrieved from model_inference.device: {device}")
    except AttributeError:
        logger.warning("'model_inference' object has no 'device' attribute. Attempting to infer from parameters.")
        try:
            device = next(model_inference.parameters()).device
            logger.info(f"Device inferred from model parameters: {device}")
        except Exception as e:
            logger.error(f"Could not determine model device: {e}", exc_info=True)
            logger.error("Falling back to CPU.")
            device = torch.device("cpu")
    except Exception as e:
        logger.error(f"An unexpected error occurred while determining model device: {e}", exc_info=True)
        logger.error("Falling back to CPU.")
        device = torch.device("cpu")

    logger.info(f"Target device for inference: {device}")

    # --- Prepare Initial State on CPU ---
    logger.info("Preparing initial state on CPU...")
    batch_tensor_4d_cpu = None
    initial_state_5d_cpu = None
    try:
        # Repeat ensemble members
        batch_tensor_4d_cpu = initial_state_tensor.repeat(n_ensemble, 1, 1, 1) # (E, C, H, W)
        # Add time dimension (T=1 for initial state)
        initial_state_5d_cpu = batch_tensor_4d_cpu.unsqueeze(1)  # (E, 1, C, H, W)
        logger.info(f"Prepared initial state on CPU: Shape={initial_state_5d_cpu.shape}, Dtype={initial_state_5d_cpu.dtype}")
    except Exception as e:
        logger.error(f"Error preparing initial state batch: {e}", exc_info=True)
        return None # Cannot proceed
    finally:
        del batch_tensor_4d_cpu # Free intermediate tensor

    # --- Normalize on CPU ---
    logger.info("Normalizing initial state on CPU...")
    center_np = None
    scale_np = None
    center_tensor_cpu = None
    scale_tensor_cpu = None
    center_reshaped_cpu = None
    scale_reshaped_cpu = None
    initial_state_norm_5d_cpu = None # Result variable
    num_channels = -1

    try:
        # Need initial_state_5d_cpu to exist
        if initial_state_5d_cpu is None:
             raise RuntimeError("Cannot normalize, initial_state_5d_cpu was not created.")
        num_channels = initial_state_5d_cpu.shape[2] # Get expected channel count C

        # 1. Retrieve NumPy arrays from model attributes
        try:
            center_np = model_inference.center_np
            scale_np = model_inference.scale_np
        except AttributeError as e:
             logger.error(f"Model object is missing required 'center_np' or 'scale_np' attributes: {e}")
             raise AttributeError("Model missing normalization attributes.") from e

        logger.info(f"Retrieved normalization stats (NumPy): center shape {getattr(center_np, 'shape', 'N/A')}, scale shape {getattr(scale_np, 'shape', 'N/A')}")

        # 2. Validate and Convert NumPy arrays to PyTorch CPU Tensors
        if not isinstance(center_np, np.ndarray) or not isinstance(scale_np, np.ndarray):
            logger.error(f"Normalization stats are not NumPy arrays. Got types: {type(center_np)}, {type(scale_np)}")
            raise TypeError("Normalization stats (center_np, scale_np) must be NumPy arrays.")

        # Ensure they are float for calculations. Match input tensor's dtype if needed.
        tensor_dtype = initial_state_5d_cpu.dtype
        center_tensor_cpu = torch.from_numpy(center_np).to(dtype=tensor_dtype)
        scale_tensor_cpu = torch.from_numpy(scale_np).to(dtype=tensor_dtype)
        logger.info(f"Converted normalization stats to CPU Tensors: Shapes {center_tensor_cpu.shape}, {scale_tensor_cpu.shape}, Dtype {center_tensor_cpu.dtype}")

        # 3. Validate shapes - should be (C,)
        if center_tensor_cpu.dim() != 1 or scale_tensor_cpu.dim() != 1 or center_tensor_cpu.shape[0] != num_channels or scale_tensor_cpu.shape[0] != num_channels:
            logger.error(f"Normalization stats shape mismatch. Input C={num_channels}, but got center={center_tensor_cpu.shape}, scale={scale_tensor_cpu.shape}. Expected 1D tensor of size C.")
            raise ValueError(f"Normalization stats shape incompatible with input tensor's channel dimension ({num_channels}).")

        # Check for zero scale values to prevent division by zero -> NaN/Inf
        if torch.any(scale_tensor_cpu == 0):
            zero_indices = torch.where(scale_tensor_cpu == 0)[0]
            logger.warning(f"Scale tensor contains zero values at indices: {zero_indices.tolist()}. This may lead to NaN/Inf during normalization.")
            # Consider adding epsilon or raising error if this is critical
            # scale_tensor_cpu[zero_indices] = 1e-9 # Example fix: replace zeros with small epsilon

        # 4. Reshape tensors for broadcasting: (C,) -> (1, 1, C, 1, 1)
        # This aligns them with the (E, 1, C, H, W) shape of initial_state_5d_cpu
        center_reshaped_cpu = center_tensor_cpu.view(1, 1, num_channels, 1, 1)
        scale_reshaped_cpu = scale_tensor_cpu.view(1, 1, num_channels, 1, 1)
        logger.info(f"Reshaped normalization stats for broadcasting: Shapes {center_reshaped_cpu.shape}, {scale_reshaped_cpu.shape}")

        # 5. Perform normalization on CPU
        initial_state_norm_5d_cpu = (initial_state_5d_cpu - center_reshaped_cpu) / scale_reshaped_cpu
        logger.info(f"Normalization completed on CPU. Result shape: {initial_state_norm_5d_cpu.shape}")

        # Check for NaNs/Infs after normalization
        if torch.isnan(initial_state_norm_5d_cpu).any() or torch.isinf(initial_state_norm_5d_cpu).any():
            logger.warning("NaN or Inf values detected in normalized state tensor. Check input data and scale values.")


    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
         # Log specific errors caught above
         logger.error(f"Error during normalization step: {e}", exc_info=True)
         return None # Cannot proceed
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during normalization: {e}", exc_info=True)
        return None # Cannot proceed
    finally:
        # 6. Clean up intermediate variables to free memory explicitly
        del center_np, scale_np # Delete NumPy arrays
        del center_tensor_cpu, scale_tensor_cpu # Delete intermediate tensors
        del center_reshaped_cpu, scale_reshaped_cpu
        # Delete the input state that is no longer needed
        del initial_state_5d_cpu
        # Ensure garbage collection runs if needed, helps release memory faster sometimes
        # gc.collect() # Uncomment if memory pressure is extreme
        logger.info("Cleaned up intermediate normalization variables.")


    # --- Apply Perturbation on CPU ---
    initial_state_perturbed_norm_5d_cpu = None # Result variable
    if initial_state_norm_5d_cpu is None:
        logger.error("Cannot apply perturbation, normalized state is missing.")
        return None

    if noise_amp > 0 and n_ensemble > 1:
        logger.info(f"Applying perturbation noise (Amp={noise_amp:.4e}, Strategy='{pert_strategy}') on CPU")
        noise_cpu = None
        try:
            # Currently only Gaussian strategy is implemented implicitly
            if pert_strategy.lower() != 'gaussian':
                 logger.warning(f"Perturbation strategy '{pert_strategy}' not explicitly implemented, using default Gaussian noise.")
            noise_cpu = torch.randn_like(initial_state_norm_5d_cpu) * noise_amp
            # Ensure ensemble member 0 is deterministic (control forecast)
            noise_cpu[0] = 0.0
            initial_state_perturbed_norm_5d_cpu = initial_state_norm_5d_cpu + noise_cpu
            logger.info(f"Perturbation applied. Result shape: {initial_state_perturbed_norm_5d_cpu.shape}")
        except Exception as e:
             logger.error(f"Error applying perturbation: {e}", exc_info=True)
             return None # Cannot proceed
        finally:
             del noise_cpu # Free noise tensor memory
    else:
        if n_ensemble <= 1:
            logger.info("Skipping perturbation: ensemble size is 1.")
        else: # noise_amp == 0
             logger.info("Skipping perturbation: noise amplitude is 0.")
        # Pass the normalized state directly
        initial_state_perturbed_norm_5d_cpu = initial_state_norm_5d_cpu

    # Delete the no-longer-needed normalized state (it's either copied or perturbed into the new variable)
    del initial_state_norm_5d_cpu
    gc.collect() # Explicit GC call after major CPU ops before GPU transfer

    # --- Transfer to GPU ---
    logger.info(f"Transferring initial state to target device: {device}...")
    initial_state_perturbed_norm_5d_gpu = None
    if initial_state_perturbed_norm_5d_cpu is None:
        logger.error("Cannot transfer to GPU, perturbed state is missing.")
        return None

    try:
        transfer_start_time = time.time()
        initial_state_perturbed_norm_5d_gpu = initial_state_perturbed_norm_5d_cpu.to(device, non_blocking=True) # Use non_blocking for potential overlap
        # If using CUDA, synchronize only if measuring time or needing immediate use
        # torch.cuda.synchronize() # Uncomment if precise timing or immediate access needed
        transfer_duration = time.time() - transfer_start_time
        logger.info(f"Transfer to device {device} completed in {transfer_duration:.2f}s. Shape: {initial_state_perturbed_norm_5d_gpu.shape}")

        # Optional: Log GPU memory after transfer
        if log_gpu_mem and device.type == 'cuda':
            # Ensure tensor is actually on GPU before logging memory
            torch.cuda.synchronize(device=device) # Ensure transfer is complete
            free_mem, total_mem = torch.cuda.mem_get_info(device=device)
            used_mem = total_mem - free_mem
            logger.info(f"GPU Memory Usage ({device}): Used={used_mem/1024**3:.2f} GB, Free={free_mem/1024**3:.2f} GB, Total={total_mem/1024**3:.2f} GB")

    except Exception as e:
        logger.error(f"Error transferring initial state tensor to device {device}: {e}", exc_info=True)
        # Clean up CPU tensor if transfer fails and GPU tensor wasn't created
        del initial_state_perturbed_norm_5d_cpu
        return None # Cannot proceed
    finally:
        # Free the CPU copy AFTER successful transfer (or if transfer failed)
        del initial_state_perturbed_norm_5d_cpu
        gc.collect()

    # --- Initialize TimeLoop Iterator ---
    logger.info("Initializing TimeLoop iterator...")
    iterator = None
    try:
        # Pass the state already on the correct device
        iterator = model_inference(time=initial_time_dt, x=initial_state_perturbed_norm_5d_gpu)
        # The model / TimeLoop now owns the GPU tensor, we can delete our reference if desired,
        # but it might be cleaner to let it go out of scope naturally.
        # Let's delete it to be explicit about releasing control.
        del initial_state_perturbed_norm_5d_gpu
        gc.collect()
        logger.info("TimeLoop iterator initialized.")
        if log_gpu_mem and device.type == 'cuda': # Log mem after model init potentially allocates more
             torch.cuda.synchronize(device=device)
             free_mem, total_mem = torch.cuda.mem_get_info(device=device)
             used_mem = total_mem - free_mem
             logger.info(f"GPU Memory Usage ({device}) after TimeLoop init: Used={used_mem/1024**3:.2f} GB, Free={free_mem/1024**3:.2f} GB")

    except Exception as e:
        logger.error(f"Error initializing TimeLoop iterator: {e}", exc_info=True)
        # Clean up GPU tensor if iterator init fails
        if 'initial_state_perturbed_norm_5d_gpu' in locals() and initial_state_perturbed_norm_5d_gpu is not None:
            del initial_state_perturbed_norm_5d_gpu
            if device.type == 'cuda': torch.cuda.empty_cache() # Try to release GPU memory
        return None

    # --- Main Inference Loop ---
    output_tensors_cpu = [] # Store results on CPU
    inference_step_times = []
    total_steps_to_run = simulation_length # We need 'simulation_length' steps *after* the initial state.
                                          # The iterator might yield the initial state as step 0, or start from step 1.
                                          # Let's assume the iterator yields `simulation_length` future states.

    logger.info(f"Starting inference loop for {total_steps_to_run} steps...")
    try:
        # The iterator should yield `simulation_length` prediction steps
        for step_idx in range(total_steps_to_run):
            step_start_time = time.time()

            # Get next prediction step from iterator (on GPU)
            # Assume iterator yields (timestamp, data_tensor_gpu, optional_metadata)
            time_out, data_gpu, _ = next(iterator)

            # --- Critical Memory Step: Move result to CPU immediately ---
            data_cpu = data_gpu.cpu()
            output_tensors_cpu.append(data_cpu)

            # --- Critical Memory Step: Delete GPU tensor reference ---
            # Deleting data_gpu allows its memory to be potentially reused by the *next* iterator step
            del data_gpu
            if device.type == 'cuda':
                # Not strictly necessary to empty cache every step, can slow things down.
                # Only use if experiencing OOM despite deleting tensor references.
                # torch.cuda.empty_cache()
                pass

            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            inference_step_times.append(step_duration)

            # Log progress periodically
            if (step_idx + 1) % output_freq == 0 or step_idx == total_steps_to_run - 1:
                avg_step_time = np.mean(inference_step_times) if inference_step_times else 0
                logger.info(
                    f"Step {step_idx + 1}/{total_steps_to_run}: Time={time_out.isoformat()}, "
                    f"Output Shape={data_cpu.shape}, Step Time={step_duration:.3f}s, Avg Time={avg_step_time:.3f}s"
                )
                if log_gpu_mem and device.type == 'cuda':
                    torch.cuda.synchronize(device=device) # Ensure ops complete before mem check
                    free_mem, total_mem = torch.cuda.mem_get_info(device=device)
                    used_mem = total_mem - free_mem
                    logger.info(f"GPU Memory Usage ({device}) after step {step_idx+1}: Used={used_mem/1024**3:.2f} GB, Free={free_mem/1024**3:.2f} GB")

            # Check for NaNs/Infs in output (optional, can be expensive)
            # if torch.isnan(data_cpu).any() or torch.isinf(data_cpu).any():
            #     logger.error(f"NaN or Inf detected in output tensor at step {step_idx + 1}. Aborting.")
            #     raise ValueError(f"NaN/Inf detected at step {step_idx + 1}")

    except StopIteration:
        logger.warning(f"TimeLoop iterator stopped prematurely after {len(output_tensors_cpu)} steps (expected {total_steps_to_run}).")
        # Continue with the data collected so far, but log the discrepancy.
    except Exception as e:
        logger.error(f"Error during TimeLoop iteration step {len(output_tensors_cpu) + 1}: {e}", exc_info=True)
        del output_tensors_cpu # Clean up partial results
        if device.type == 'cuda': torch.cuda.empty_cache() # Attempt cleanup
        return None # Indicate failure

    finally:
        # Explicitly delete iterator to potentially trigger cleanup in the model/TimeLoop object
        del iterator
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache() # Final cleanup attempt
        logger.info("Inference loop finished.")


    # --- Combine Outputs ---
    if not output_tensors_cpu:
        logger.warning("No output tensors were collected during the inference loop.")
        return None

    logger.info(f"Combining {len(output_tensors_cpu)} collected output steps...")
    final_output_tensor = None
    try:
        # Stack along the time dimension (dim=1)
        # Input tensors have shape (E, 1, C, H, W) -> stacked shape (E, T_out, C, H, W)
        # Note: The iterator usually yields (E, 1, C, H, W), need to confirm this assumption.
        # If iterator yields (E, C, H, W), need to unsqueeze before stacking or stack differently.
        # Assuming the iterator yields (E, 1, C, H, W) matching the input structure's time dim.
        if output_tensors_cpu[0].dim() == 5 and output_tensors_cpu[0].shape[1] == 1:
             # Shape is (E, 1, C, H, W), stack directly on dim 1
             final_output_tensor = torch.cat(output_tensors_cpu, dim=1) # More memory efficient than stack for large T
        elif output_tensors_cpu[0].dim() == 4:
             # Shape is (E, C, H, W), stack along new dim 1
             logger.warning("Output tensors from iterator have 4 dims (E, C, H, W). Stacking along new time dimension (dim=1).")
             final_output_tensor = torch.stack(output_tensors_cpu, dim=1)
        else:
             logger.error(f"Unexpected dimension of output tensors: {output_tensors_cpu[0].dim()}. Cannot combine.")
             return None

        logger.info(f"Final combined output tensor shape: {final_output_tensor.shape} on CPU.")

        # Optional: De-normalize the output tensor back to original scale?
        # This would require center/scale again and careful broadcasting.
        # Not implemented here as it wasn't requested, but often a final step.
        # Example: final_output_denorm = final_output_tensor * scale_reshaped_for_output + center_reshaped_for_output

    except Exception as e:
        logger.error(f"Error combining output tensors: {e}", exc_info=True)
        return None
    finally:
        # Clean up the list of tensors
        del output_tensors_cpu
        gc.collect()

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    avg_step_time = np.mean(inference_step_times) if inference_step_times else 0
    logger.info("=" * 50)
    logger.info(f"run_inference completed for IC: {initial_time_dt.isoformat()}")
    logger.info(f"Total time: {total_duration:.2f}s")
    logger.info(f"Average inference step time: {avg_step_time:.3f}s")
    logger.info(f"Final Output Shape: {final_output_tensor.shape if final_output_tensor is not None else 'N/A'}")
    logger.info("=" * 50)

    return final_output_tensor


# def run_inference(
#     model_inference,
#     initial_state_tensor: torch.Tensor,
#     initial_time_dt: datetime.datetime,
#     config: dict,
#     logger: logging.Logger,
#     log_gpu_mem: bool = False,
# ) -> Optional[torch.Tensor]:
#     """
#     Optimized and robust function to run autoregressive ensemble forecasts using TimeLoop.

#     Performs initial state preparation, normalization, and perturbation on CPU
#     to avoid GPU OOM errors with large ensemble/history states. Transfers data
#     to GPU just before the inference loop. Includes aggressive memory management
#     and detailed logging.

#     Args:
#         model_inference: The loaded model with a TimeLoop interface.
#         initial_state_tensor: Initial condition tensor (1, C, H, W), expected on CPU.
#         initial_time_dt: Timezone-aware datetime for the initial state.
#         config: Configuration dictionary with keys like 'ensemble_members', 'simulation_length', etc.
#         logger: Configured logger instance.
#         log_gpu_mem: If True, logs GPU memory usage at critical points.

#     Returns:
#         Optional[torch.Tensor]: Full forecast history (E, T_out, C, H, W) on CPU if
#                                 successful. None on failure.
#     """
#     overall_start_time = time.time()
#     logger.info("=" * 50)
#     logger.info(f"Starting run_inference for IC: {initial_time_dt.isoformat()}")
#     logger.info("=" * 50)

#     # --- Configuration Extraction ---
#     n_ensemble = config.get("ensemble_members", 1)
#     simulation_length = config.get("simulation_length", 0)
#     output_freq = config.get("output_frequency", 1)
#     noise_amp = config.get("noise_amplitude", 0.0)
#     pert_strategy = config.get("perturbation_strategy", "gaussian")

#     logger.info(f"Config - Ensemble: {n_ensemble}, Sim Length: {simulation_length}, Output Freq: {output_freq}")
#     logger.info(f"Config - Perturbation: Amp={noise_amp:.4e}, Strategy='{pert_strategy}'")

#     # --- Validation ---
#     if initial_state_tensor.dim() != 4 or initial_state_tensor.shape[0] != 1:
#         logger.error(f"Input IC tensor shape invalid: {initial_state_tensor.shape}. Expected (1, C, H, W).")
#         raise ValueError("Invalid initial state tensor shape.")

#     if initial_time_dt.tzinfo is None or initial_time_dt.tzinfo.utcoffset(initial_time_dt) is None:
#         logger.warning(f"Initial time {initial_time_dt.isoformat()} is timezone naive. Assuming UTC.")
#         initial_time_dt = initial_time_dt.replace(tzinfo=datetime.timezone.utc)

#     # --- Prepare Device ---
#     try:
#         device = model_inference.device
#     except AttributeError:
#         device = next(model_inference.parameters()).device
#         logger.warning("Device not directly available. Fetched device from model parameters.")

#     logger.info(f"Running on device: {device}")

#     # --- Prepare Initial State on CPU ---
#     logger.info("Preparing initial state on CPU...")
#     batch_tensor_4d_cpu = initial_state_tensor.repeat(n_ensemble, 1, 1, 1)
#     initial_state_5d_cpu = batch_tensor_4d_cpu.unsqueeze(1)  # Add time dimension
#     del batch_tensor_4d_cpu  # Free memory
#     logger.info(f"Prepared initial state on CPU: {initial_state_5d_cpu.shape}")



#     # --- Normalize on CPU ---
#     logger.info("Normalizing initial state on CPU...")
#     center = model_inference.center_np
#     scale = model_inference.scale_np
#     initial_state_norm_5d_cpu = (initial_state_5d_cpu - center.unsqueeze(1)) / scale.unsqueeze(1)
#     del initial_state_5d_cpu, center, scale  # Free memory
#     logger.info("Normalization completed.")




#     # --- Apply Perturbation on CPU ---
#     if noise_amp > 0 and n_ensemble > 1:
#         logger.info(f"Applying perturbation noise (Amp={noise_amp:.4e}, Strategy='{pert_strategy}')")
#         noise_cpu = torch.randn_like(initial_state_norm_5d_cpu) * noise_amp
#         noise_cpu[0] = 0  # Ensure ensemble member 0 is deterministic
#         initial_state_perturbed_norm_5d_cpu = initial_state_norm_5d_cpu + noise_cpu
#         del noise_cpu  # Free memory
#     else:
#         logger.info("No perturbation noise applied.")
#         initial_state_perturbed_norm_5d_cpu = initial_state_norm_5d_cpu

#     del initial_state_norm_5d_cpu  # Free memory
#     logger.info(f"Perturbation completed. Shape: {initial_state_perturbed_norm_5d_cpu.shape}")

#     # --- Transfer to GPU ---
#     logger.info("Transferring initial state to GPU...")
#     initial_state_perturbed_norm_5d_gpu = initial_state_perturbed_norm_5d_cpu.to(device)
#     del initial_state_perturbed_norm_5d_cpu  # Free memory
#     logger.info("Transfer to GPU completed.")

#     # --- Initialize TimeLoop Iterator ---
#     logger.info("Initializing TimeLoop iterator...")
#     iterator = model_inference(time=initial_time_dt, x=initial_state_perturbed_norm_5d_gpu)
#     del initial_state_perturbed_norm_5d_gpu  # Free memory
#     logger.info("TimeLoop iterator initialized.")

#     # --- Main Loop ---
#     output_tensors = []
#     inference_times = []
#     try:
#         for step_idx in range(simulation_length + 1):
#             step_start_time = time.time()
#             time_out, data_gpu, _ = next(iterator)
#             data_cpu = data_gpu.cpu()  # Move data to CPU
#             output_tensors.append(data_cpu)
#             del data_gpu  # Free memory
#             step_end_time = time.time()
#             inference_times.append(step_end_time - step_start_time)

#             if step_idx % output_freq == 0:
#                 logger.info(f"Step {step_idx}: Time={time_out.isoformat()}, Output Shape={data_cpu.shape}")

#     except StopIteration:
#         logger.warning("TimeLoop iterator stopped prematurely.")
#     except Exception as e:
#         logger.error(f"Error during TimeLoop iteration: {e}", exc_info=True)
#         return None

#     # --- Combine Outputs ---
#     logger.info("Combining collected outputs...")
#     final_output_tensor = torch.stack(output_tensors, dim=1)
#     logger.info(f"Final output tensor shape: {final_output_tensor.shape}")

#     return final_output_tensor





"""


    Understand the Goal: The user wants to fix a ValueError in the save_output function. The error message unable to infer dtype on variable 'time'; xarray cannot serialize arbitrary Python objects clearly indicates that the time_coords list, which contains Python datetime objects, is causing the problem when creating the xarray.DataArray. Additionally, the user wants to ensure the time coordinate calculation is correct, assuming each simulation step is 6 hours, and wants the function to be memory optimized (though saving primarily involves CPU/disk I/O, we should handle the input tensor carefully).

    Identify the Error Source:

        Traceback points to xarray/conventions.py functions (_infer_dtype, ensure_dtype_not_object, encode_cf_variable, cf_encoder). This confirms xarray's internal encoding process is failing.

        The error message specifically mentions the time variable and "arbitrary Python objects".

        Looking at the code, time_coords is created as a list of datetime.datetime objects: time_coords.append(current_time).

        When creating the xr.DataArray, time_coords is passed directly as the coordinate values for the "time" dimension. xarray needs NumPy arrays or similar array-like objects with specific data types (like np.datetime64) for coordinates, not raw Python objects.

    Determine the Fix for the Error:

        The time_coords list needs to be converted into a NumPy array of a type xarray understands for time coordinates, specifically np.datetime64.

        This conversion should happen before creating the xr.DataArray.

        The conversion is simple: time_coords_np = np.array(time_coords, dtype='datetime64[ns]'). Using nanosecond precision ([ns]) is standard for np.datetime64 and generally well-supported.

    Address the Time Calculation Requirement:

        The user specified that each simulation step (autoregressive step) corresponds to a 6-hour time delta.

        The current code calculates time based on the output_freq and time_step. Let's re-evaluate this logic.

        The output_tensor has shape (E, T_out, C, H, W). T_out represents the number of saved time steps.

        The simulation_length from the config represents the total number of simulation steps run by the model after the initial condition.

        The output_freq determines how often a simulation step result is saved (e.g., if output_freq is 1, every step is saved; if 2, every second step is saved).

        The input time_step variable in the current function seems redundant or potentially confusing given the new requirement (each sim step = 6 hours). The calculation should only depend on the initial time, the index of the saved step, the output_freq, and the fixed 6-hour delta.

        Let's rename the fixed delta: simulation_step_delta = datetime.timedelta(hours=6).

        The time for the i-th saved step (where i goes from 0 to T_out - 1) corresponds to the i * output_freq-th simulation step.

        Therefore, the correct time calculation for the i-th element in time_coords should be: initial_time + (i * output_freq * simulation_step_delta).

        We need to ensure the loop iterates correctly to generate T_out time points. The initial time (i=0) corresponds to initial_time + 0 * simulation_step_delta = initial_time. The next saved time (i=1) corresponds to initial_time + (1 * output_freq * simulation_step_delta). The last saved time (i = T_out - 1) corresponds to initial_time + ((T_out - 1) * output_freq * simulation_step_delta).

        The current loop structure for i in range(1, n_time_out): and adding the initial time outside the loop is slightly awkward. A cleaner way is for i in range(n_time_out): and calculate the time directly.

    Address Memory Optimization:

        The main memory usage comes from output_tensor. The function receives it, presumably already computed.

        The critical step is output_tensor.numpy(). This creates a NumPy copy of the potentially large tensor. This is required by xarray.

        Can we avoid holding both the PyTorch tensor and the NumPy array simultaneously if memory is extremely tight?

            Option 1: Convert to NumPy just before passing to xr.DataArray. This is what the code does.

            Option 2: Delete the PyTorch tensor immediately after creating the NumPy array: data_np = output_tensor.numpy(); del output_tensor; gc.collect(). Then use data_np in xr.DataArray. This saves memory if the output_tensor reference passed into save_output is the only reference to that tensor data. If the calling function still holds a reference, del output_tensor only removes the local reference. This optimization is slightly risky depending on the calling context. Given the focus is fixing the save, let's stick to the standard conversion first, but add logging around it.

        GPU usage: The function explicitly moves lat and lon to CPU if they are tensors. The main output_tensor is assumed to be on the CPU already (as it likely comes from run_inference which returns the final tensor on CPU). The function should not use the GPU at all. Let's add assertions/checks to ensure input tensor is on CPU.

    Refine Logging and Robustness:

        Add more specific logs around time calculation.

        Log the final shape and type of the time_coords_np array.

        Improve error handling in time calculation (the existing try/except is okay but can be simplified with the new logic).

        Ensure actual_time_step variable is removed or repurposed correctly based on the fixed 6-hour delta.

        Check lat/lon conversion more robustly.

        Make sure pytz is imported if used (it's used in attrs for creation_date). Add import.

        Ensure os is imported. Add import.

        Ensure xarray (as xr) is imported. Add import.

        The fallback to np.arange(n_time_out) for time coordinates if calculation fails is reasonable, but it loses the actual time information. Log this clearly.

        The fallback where forecast_ds = forecast_da if to_dataset fails needs careful handling regarding encoding. Encoding should be defined based on the actual object being saved (forecast_da or forecast_ds).

        Add checks for lat/lon shapes matching H/W from the tensor.

    Structure the Code Update:

        Add necessary imports (os, pytz, xarray, numpy, datetime, torch, time, logging, gc).

        Define simulation_step_delta = datetime.timedelta(hours=6).

        Modify the time coordinate generation loop:

            Initialize an empty list time_coords.

            Loop for i in range(n_time_out):.

            Calculate current_time = initial_time + i * output_freq * simulation_step_delta.

            Append current_time to the list.

            Wrap calculation in try/except for robustness, falling back to index if necessary.

        After the loop, convert time_coords to np.array(time_coords, dtype='datetime64[ns]'). Handle the case where fallback to indices occurred (dtype will be int/float). Log the final coordinate array.

        Ensure output_tensor is on CPU. Add a check/warning.

        Perform the .numpy() conversion. Log memory usage before/after if possible (tricky without external tools, maybe just log tensor size). Consider del output_tensor after conversion if memory is critical and feasible.

        Ensure lat/lon are NumPy arrays and check shapes.

        Create xr.DataArray using the np.datetime64 time coordinates.

        Handle to_dataset conversion and fallback.

        Define encoding based on whether forecast_ds is a Dataset or DataArray.

        Perform the save operation.

        Refine logging messages throughout.

    Self-Correction/Refinement:

        The variable time_step passed into the function is now unused and misleading. Remove it from the function signature and docstring.

        The attribute time_step_seconds relies on actual_time_step. This should be updated to use the fixed 6-hour delta.

        The fallback time_coords = np.arange(n_time_out) means the coordinate type won't be datetime. xarray might handle this okay, but it's better to be explicit. If fallback happens, the 'time' coordinate won't represent actual time. The attributes should reflect this if possible.

        When converting output_tensor to numpy, explicitly log the memory footprint (e.g., output_tensor.nelement() * output_tensor.element_size()).

        In the to_dataset fallback, make sure the encoding dictionary keys match the object being saved (forecast_da.name vs variable names in forecast_ds).

Let's implement these changes.

"""



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
# def save_output(output_tensor, initial_time, time_step, channels, lat, lon, config, output_dir, logger):
#     """Saves the forecast output tensor to a NetCDF file."""

#     if output_tensor is None:
#         logger.error("Cannot save output, tensor is None.")
#         return

#     try:
#         # output_tensor shape: (E, T_out, C, H, W)
#         n_ensemble, n_time_out, n_channels, n_lat, n_lon = output_tensor.shape
#         output_freq = config.get("output_frequency", 1)

#         logger.info("Preparing output for saving...")
#         logger.debug(f"Output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
#         logger.debug(f"Number of channels: {n_channels}, Expected channels from model: {len(channels)}")

#         if n_channels != len(channels):
#             logger.error(f"Mismatch between channels in output tensor ({n_channels}) and provided channel names ({len(channels)}). Saving with generic channel indices.")
#             channels_coord = np.arange(n_channels)  # Use generic index if names mismatch
#             channel_dim_name = "channel_idx"  # Use a different name to indicate mismatch
#         else:
#             channels_coord = channels
#             channel_dim_name = "channel"

#         # Create time coordinates
#         time_coords = []
#         current_time = initial_time
#         # Add initial time (t=0)
#         time_coords.append(current_time)
#         # Generate forecast times
#         for i in range(1, n_time_out):
#             # Calculate time delta: i * output_freq * base_time_step
#             try:
#                 # Ensure time_step is timedelta
#                 if not isinstance(time_step, datetime.timedelta):
#                     logger.warning(f"Model time_step is not a timedelta ({type(time_step)}), assuming hours.")
#                     actual_time_step = datetime.timedelta(hours=time_step)
#                 else:
#                     actual_time_step = time_step

#                 # Time for the i-th saved output step (corresponds to model step i * output_freq)
#                 forecast_step_number = i * output_freq
#                 current_time = initial_time + forecast_step_number * actual_time_step
#                 time_coords.append(current_time)

#             except TypeError as te:
#                 logger.error(f"TypeError calculating time coordinates at step {i}: {te}. Check time_step type.")
#                 # Fallback to index if calculation fails
#                 time_coords = np.arange(n_time_out)
#                 break  # Stop trying to calculate time coords
#             except Exception as e:
#                 logger.error(f"Error calculating time coordinates at step {i}: {e}", exc_info=True)
#                 time_coords = np.arange(n_time_out)
#                 break

#         logger.debug(f"Generated {len(time_coords)} time coordinates. First: {time_coords[0].isoformat() if time_coords else 'N/A'}, Last: {time_coords[-1].isoformat() if len(time_coords)>0 else 'N/A'}")
#         if len(time_coords) != n_time_out:
#             logger.warning(f"Generated {len(time_coords)} time coordinates, but expected {n_time_out}. Using indices instead.")
#             time_coords = np.arange(n_time_out)

#         # Ensure lat/lon are numpy arrays on CPU
#         lat_np = lat.cpu().numpy() if isinstance(lat, torch.Tensor) else np.asarray(lat)
#         lon_np = lon.cpu().numpy() if isinstance(lon, torch.Tensor) else np.asarray(lon)

#         # Create DataArray
#         logger.debug("Creating xarray DataArray...")
#         # Check for NaNs before creating DataArray
#         if np.isnan(output_tensor.numpy()).any():
#             logger.warning("NaNs present in the output tensor before saving to NetCDF!")

#         forecast_da = xr.DataArray(
#             output_tensor.numpy(),  # Convert tensor to numpy array
#             coords={
#                 "ensemble": np.arange(n_ensemble),
#                 "time": time_coords,
#                 channel_dim_name: channels_coord,  # Use dynamic channel dimension name
#                 "lat": lat_np,
#                 "lon": lon_np,
#             },
#             dims=["ensemble", "time", channel_dim_name, "lat", "lon"],
#             name="forecast",
#             attrs={
#                 "description": f"{config['weather_model']} ensemble forecast output",
#                 "model": config["weather_model"],
#                 "simulation_length_steps": config["simulation_length"],
#                 "output_frequency_steps": output_freq,
#                 "ensemble_members": n_ensemble,
#                 "initial_condition_time": initial_time.isoformat(),
#                 "time_step_seconds": actual_time_step.total_seconds() if isinstance(actual_time_step, datetime.timedelta) else "unknown",
#                 "noise_amplitude": config["noise_amplitude"],
#                 "perturbation_strategy": config["perturbation_strategy"],
#                 "creation_date": datetime.datetime.now(pytz.utc).isoformat(),
#                 "pytorch_version": torch.__version__,
#                 "numpy_version": np.__version__,
#                 "xarray_version": xr.__version__,
#             },
#         )
#         logger.info("Created xarray DataArray.")

#         # Convert to Dataset with each channel as a variable for better compatibility
#         # Handle potential channel name issues (e.g., invalid characters for variable names)
#         logger.debug(f"Converting DataArray to Dataset using dimension '{channel_dim_name}'...")
#         try:
#             forecast_ds = forecast_da.to_dataset(dim=channel_dim_name)
#             logger.info("Converted DataArray to Dataset (channels as variables).")
#         except Exception as e:
#             logger.error(f"Failed to convert DataArray to Dataset (dim='{channel_dim_name}'): {e}. Saving as DataArray instead.", exc_info=True)
#             # Fallback: save the DataArray directly if conversion fails
#             forecast_ds = forecast_da

#         # Define output filename
#         ic_time_str = initial_time.strftime("%Y%m%d_%H%M%S")
#         # Add ensemble size and sim length to filename for clarity
#         output_filename = os.path.join(output_dir, f"{config['weather_model']}_ensemble{n_ensemble}_sim{config['simulation_length']}_{ic_time_str}.nc")

#         # Ensure output directory exists
#         os.makedirs(output_dir, exist_ok=True)

#         logger.info(f"Saving forecast output to: {output_filename}")

#         # Define encoding for compression (optional but recommended)
#         # Apply encoding only if saving as Dataset
#         encoding = {}
#         if isinstance(forecast_ds, xr.Dataset):
#             encoding = {var: {"zlib": True, "complevel": 5, "_FillValue": -9999.0} for var in forecast_ds.data_vars}  # Add FillValue
#             logger.debug(f"Applying encoding to variables: {list(forecast_ds.data_vars.keys())}")
#         elif isinstance(forecast_ds, xr.DataArray):
#             encoding = {forecast_ds.name: {"zlib": True, "complevel": 5, "_FillValue": -9999.0}}
#             logger.debug(f"Applying encoding to DataArray: {forecast_ds.name}")

#         start_save = time.time()
#         forecast_ds.to_netcdf(output_filename, encoding=encoding, engine="netcdf4")  # Specify engine
#         end_save = time.time()
#         # Check file size
#         file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
#         logger.info(f"Save complete. Time taken: {end_save - start_save:.2f} seconds. File size: {file_size_mb:.2f} MB")

#     except Exception as e:
#         logger.error(f"Failed during the save_output process: {e}", exc_info=True)
#         # Attempt to remove potentially corrupted file
#         if "output_filename" in locals() and os.path.exists(output_filename):
#             try:
#                 os.remove(output_filename)
#                 logger.warning(f"Removed potentially corrupted file: {output_filename}")
#             except OSError as oe:
#                 logger.error(f"Failed to remove corrupted file {output_filename}: {oe}")



"""


Reasoning for the Fix:

    Root Cause Confirmation: The traceback definitively shows the error occurs when checking lat_np.shape[-2]. The log confirms lat_np has shape (721,), meaning it's 1-dimensional (ndim=1). Accessing index -2 of the shape tuple for a 1D array is invalid.

    Flaw in Previous Logic: The previous if/elif structure for validation should have worked:

          
    if lat_np.ndim == 1:
        # Checks shape[0]
    elif lat_np.ndim >= 2:
        # Checks shape[-2] <--- Error happened here despite ndim being 1?

        

    IGNORE_WHEN_COPYING_START

    Use code with caution.Python
    IGNORE_WHEN_COPYING_END

    The only way the elif block (containing the faulty index [-2]) could be reached when ndim is 1 is if the code being executed did not actually contain this if/elif structure. The traceback points to line 1183 in the user's script, strongly suggesting they were running the version of the code before the ndim check was introduced, or the integration was faulty.

    The Robust Fix: Although the previous logic was likely correct, to make it absolutely foolproof and guard against potential copy-paste errors or subtle logic bugs, the fix explicitly separates the checks and avoids relying solely on the elif structure:

        It checks if lat_np.ndim == 1: and performs the shape[0] comparison.

        It then checks elif lat_np.ndim >= 2: and performs the shape[-2] comparison.

        Crucially, these checks are now mutually exclusive. If ndim is 1, the second block is never entered, preventing the IndexError.

        Added debug logging within these blocks to make it crystal clear which comparison is being performed.

    Other Improvements:

        Moved the time step validation earlier to fail faster.

        Made the coordinate dimension mismatch warning more prominent (CRITICAL).

        Improved the coordinate mapping during xr.DataArray creation for clarity and robustness with 1D/2D coordinates.

        Ensured the final finally block robustly cleans up the output_tensor reference.

This version correctly implements the logic to handle both 1D and >=2D latitude/longitude coordinate arrays, explicitly preventing the IndexError by checking the number of dimensions before attempting to access specific dimension indices.


mproved Robustness & Cleanup:

    Added .contiguous() call before .numpy() conversion for potential efficiency/safety.

    Enhanced the final finally block to ensure the output_tensor reference is cleaned up even if an error occurs before the explicit deletion step after the NumPy conversion.

    Improved the coordinate definition within xr.DataArray creation to handle 1D/2D lat/lon more explicitly for dimension naming.

    Refined the check for valid NetCDF variable names when converting channels to dataset variables.

    Ensured the time_calculation_failed flag correctly prevents time calculation if time_step is invalid.

    Made sure output_tensor is deleted or set to None after the .numpy() call to signal the reference is gone.
    
    
        Time Coordinate Error (ValueError):

        The core issue was passing a list of Python datetime objects directly to xarray.

        Fix: The time_coords_list is now explicitly converted to a NumPy array with dtype='datetime64[ns]' before creating the xr.DataArray: time_coords = np.array(time_coords_list, dtype='datetime64[ns]'). This is the format xarray expects for time coordinates.

        Added fallback to np.arange(n_time_out) if any error occurs during datetime calculation, and updated metadata attribute time_coordinate_type accordingly.

    Correct Time Calculation Logic:

        Removed the unused time_step argument.

        Defined a constant SIMULATION_STEP_TIMEDELTA = datetime.timedelta(hours=6).

        The time calculation loop now correctly calculates the time for the i-th saved output step based on the corresponding simulation step (simulation_step = i * output_freq) and the fixed 6-hour delta: current_time = initial_time + simulation_step * SIMULATION_STEP_TIMEDELTA.

    Memory Optimization and GPU Avoidance:

        Input Tensor on CPU: Added a check to ensure the input output_tensor is on the CPU. If it's on the GPU, it's moved to the CPU with a warning. Saving should not involve GPU memory.

        Explicit Memory Cleanup:

            After converting the output_tensor (PyTorch) to data_np (NumPy) using .numpy(), the original PyTorch tensor reference (output_tensor) is explicitly deleted (del output_tensor) followed by gc.collect(). This helps free up significant memory before creating the potentially large xarray object, crucial for preventing OOM errors if the tensor is large.

            Similarly, after the to_netcdf call (or if an error occurs), the xarray object (save_object) and the NumPy array (data_np) references are deleted in a finally block to ensure cleanup.

    Logging and Robustness:

        Added more detailed logging for each step (dimension extraction, time calculation, tensor conversion, xarray creation, saving).

        Logged the memory size of the PyTorch tensor before conversion.

        Added checks for NaN/Inf values in the NumPy array before creating the xarray object.

        Ensured initial_time is timezone-aware (defaulting to UTC if naive) for consistent NetCDF time attributes.

        Checked for potential mismatches between tensor dimensions (H, W) and lat/lon coordinate shapes.

        Improved error handling with try...except...finally blocks around critical operations.

        Included necessary imports (os, pytz, gc).

        Refined attribute metadata, including simulation_time_step_hours.

        Made the conversion to xr.Dataset conditional on having valid channel names to avoid potential errors during conversion or saving if channel names are not suitable as NetCDF variable names. The code now saves as a DataArray if conversion isn't feasible.

        Ensured encoding is applied correctly whether saving a Dataset or a fallback DataArray.

This revised function correctly handles the time coordinate serialization, implements the specified 6-hour time step logic, and incorporates expl

"""



def save_output(
    output_tensor: torch.Tensor,
    initial_time: datetime.datetime,
    time_step: datetime.timedelta or int or float, # Timedelta per simulation step (or numeric assumed hours)
    channels: list,
    lat: torch.Tensor or np.ndarray,
    lon: torch.Tensor or np.ndarray,
    config: dict,
    output_dir: str,
    logger: logging.Logger
):
    """
    Saves the forecast output tensor to a NetCDF file with corrected time coordinates
    using the provided time_step and robust dimension validation. Includes memory optimizations.

    Args:
        output_tensor (torch.Tensor): The forecast data tensor, expected on CPU.
                                      Shape: (E, T_out, C, H, W).
        initial_time (datetime.datetime): Timezone-aware datetime of the initial condition (t=0).
        time_step (datetime.timedelta or int or float): The time difference corresponding to
                                                        *one simulation step*. If numeric,
                                                        it's assumed to be in hours.
        channels (list): List of channel names (strings). Length must match C.
        lat (torch.Tensor or np.ndarray): Latitude coordinates (1D or >=2D).
        lon (torch.Tensor or np.ndarray): Longitude coordinates (1D or >=2D).
        config (dict): Configuration dictionary containing metadata (e.g., 'weather_model',
                       'simulation_length', 'output_frequency', 'noise_amplitude',
                       'perturbation_strategy').
        output_dir (str): Directory to save the NetCDF file.
        logger (logging.Logger): Configured logger instance.
    """
    func_start_time = time.time()
    logger.info("Starting save_output process...")

    # --- Pre-computation Validation & Setup ---
    # Moved these checks earlier to fail fast before potentially expensive operations
    if output_tensor is None:
        logger.error("Cannot save output, the provided output_tensor is None.")
        return
    if not isinstance(output_tensor, torch.Tensor):
        logger.error(f"output_tensor must be a PyTorch Tensor, but got {type(output_tensor)}. Cannot save.")
        return
    if output_tensor.dim() != 5:
         logger.error(f"Output tensor has incorrect dimensions ({output_tensor.dim()}). Expected 5 (E, T_out, C, H, W).")
         # Clean up tensor if it exists before returning
         if 'output_tensor' in locals() and output_tensor is not None: del output_tensor
         return

    # --- Validate Time Step Early ---
    actual_time_step = None # Will hold the validated timedelta object
    time_calculation_failed = False # Flag to track if time coordinates can be calculated
    time_step_units = "unknown"
    try:
        if isinstance(time_step, datetime.timedelta):
            actual_time_step = time_step
            time_step_units = "timedelta"
            logger.info(f"Using provided time_step (timedelta): {actual_time_step}")
        elif isinstance(time_step, (int, float)):
            logger.info(f"Provided time_step is numeric ({time_step}, type={type(time_step)}). Assuming units are hours.")
            actual_time_step = datetime.timedelta(hours=time_step)
            time_step_units = "assumed_hours"
            logger.info(f"Converted numeric time_step to timedelta: {actual_time_step}")
        else:
            # Raise error here to be caught by the outer block and trigger cleanup
            raise TypeError(f"Invalid time_step type provided ({type(time_step)}). Expected timedelta or numeric (hours).")
    except Exception as e:
        logger.error(f"Failed to process time_step '{time_step}': {e}. Cannot calculate time coordinates.")
        time_calculation_failed = True # Set flag to use indices later


    # --- Ensure Tensor is on CPU ---
    try:
        if output_tensor.is_cuda:
            logger.warning("Output tensor is on GPU. Moving to CPU for saving.")
            move_start = time.time()
            output_tensor = output_tensor.cpu()
            logger.info(f"Moved output tensor to CPU in {time.time() - move_start:.2f}s.")

        # --- Ensure Timezone Aware Initial Time ---
        if initial_time.tzinfo is None or initial_time.tzinfo.utcoffset(initial_time) is None:
             logger.warning(f"Initial time {initial_time.isoformat()} is timezone naive. Assuming UTC.")
             initial_time = initial_time.replace(tzinfo=datetime.timezone.utc)
        else:
             # Convert to UTC for standardization if it's not already UTC
             initial_time = initial_time.astimezone(datetime.timezone.utc)
             logger.info(f"Initial time converted to UTC: {initial_time.isoformat()}")

        # --- Process Lat/Lon Coordinates ---
        lat_np = lat.cpu().numpy() if isinstance(lat, torch.Tensor) else np.asarray(lat)
        lon_np = lon.cpu().numpy() if isinstance(lon, torch.Tensor) else np.asarray(lon)
        logger.info(f"Processed lat (shape: {lat_np.shape}, type: {type(lat_np)}, ndim: {lat_np.ndim})")
        logger.info(f"Processed lon (shape: {lon_np.shape}, type: {type(lon_np)}, ndim: {lon_np.ndim})")

        # --- Extract Dimensions and Configuration ---
        n_ensemble, n_time_out, n_channels, n_lat, n_lon = output_tensor.shape
        output_freq = config.get("output_frequency", 1)
        simulation_length = config.get("simulation_length", "N/A") # Total sim steps run

        logger.info(f"Output tensor dimensions: E={n_ensemble}, T_out={n_time_out}, C={n_channels}, H={n_lat}, W={n_lon}")
        logger.info(f"Config: Output Freq={output_freq}, Sim Length={simulation_length}")


        # --- Validate Lat/Lon Dimension Consistency (CORRECTED FIX) ---
        lat_dim_matches = False
        lon_dim_matches = False
        validation_error = False # Flag potential critical errors

        # Check Latitude
        if lat_np.ndim == 1:
            # 1D Latitude: Check length against n_lat (H dimension)
            lat_dim_matches = (lat_np.shape[0] == n_lat)
            logger.debug(f"Lat is 1D. Checking shape[0] ({lat_np.shape[0]}) against n_lat ({n_lat}). Match: {lat_dim_matches}")
        elif lat_np.ndim >= 2:
            # 2D+ Latitude: Assume it varies along the second-to-last dimension (H)
            # Safely check if the dimension index exists before accessing it
            if -2 < lat_np.ndim: # Equivalent to checking if ndim >= 2
                lat_dim_matches = (lat_np.shape[-2] == n_lat)
                logger.debug(f"Lat is {lat_np.ndim}D. Checking shape[-2] ({lat_np.shape[-2]}) against n_lat ({n_lat}). Match: {lat_dim_matches}")
            else:
                 # This case should not be reachable if ndim >= 2, but included for safety
                 logger.error(f"Logic error: lat_np.ndim ({lat_np.ndim}) is >= 2 but cannot access index -2.")
                 lat_dim_matches = False
                 validation_error = True
        else: # ndim == 0 (scalar)
            logger.warning(f"Latitude array has 0 dimensions (shape={lat_np.shape}). Cannot validate against n_lat ({n_lat}).")
            lat_dim_matches = False # Treat as mismatch

        # Check Longitude
        if lon_np.ndim == 1:
            # 1D Longitude: Check length against n_lon (W dimension)
            lon_dim_matches = (lon_np.shape[0] == n_lon)
            logger.debug(f"Lon is 1D. Checking shape[0] ({lon_np.shape[0]}) against n_lon ({n_lon}). Match: {lon_dim_matches}")
        elif lon_np.ndim >= 2:
            # 2D+ Longitude: Assume it varies along the last dimension (W)
            # Index -1 is always valid for ndim >= 1, so direct check is safe here if ndim >= 2
            lon_dim_matches = (lon_np.shape[-1] == n_lon)
            logger.debug(f"Lon is {lon_np.ndim}D. Checking shape[-1] ({lon_np.shape[-1]}) against n_lon ({n_lon}). Match: {lon_dim_matches}")
        else: # ndim == 0 (scalar)
            logger.warning(f"Longitude array has 0 dimensions (shape={lon_np.shape}). Cannot validate against n_lon ({n_lon}).")
            lon_dim_matches = False # Treat as mismatch

        # Log warning or raise error if dimensions don't match
        if validation_error: # Critical internal error
             raise RuntimeError("Internal error during lat/lon dimension validation.")
        if not lat_dim_matches or not lon_dim_matches:
            warning_msg = (f"CRITICAL WARNING: Coordinate dimension mismatch detected!\n" # Made critical
                           f"  Tensor H dimension: {n_lat}, Lat coordinate shape: {lat_np.shape} (Match: {lat_dim_matches})\n"
                           f"  Tensor W dimension: {n_lon}, Lon coordinate shape: {lon_np.shape} (Match: {lon_dim_matches})\n"
                           f"Saved NetCDF file will likely have misaligned coordinates. Please verify input lat/lon data.")
            logger.critical(warning_msg) # Use critical log level
            # Optionally raise an error to halt execution:
            # raise ValueError("Coordinate dimensions do not match tensor dimensions.")
        else:
            logger.info("Lat/Lon coordinate dimensions successfully validated against tensor dimensions.")


        # --- Validate Channel Names ---
        if n_channels != len(channels):
            logger.error(f"Mismatch between channels in output tensor ({n_channels}) and provided channel names ({len(channels)}). Saving with generic channel indices.")
            channels_coord = np.arange(n_channels)
            channel_dim_name = "channel_idx" # Use a different name to indicate mismatch
        else:
            channels_coord = channels
            channel_dim_name = "channel"
            logger.info(f"Using provided channel names: {channels}")

        # --- Create Time Coordinates (Using time_step) ---
        logger.info("Generating time coordinates...")
        time_coords = None # Initialize
        try:
            if time_calculation_failed:
                 # Skip calculation if time step is invalid from earlier check
                 raise ValueError("Cannot calculate times due to invalid time_step.")

            time_coords_list = []
            for i in range(n_time_out):
                simulation_step = i * output_freq
                current_time = initial_time + simulation_step * actual_time_step
                time_coords_list.append(current_time)

            time_coords = np.array(time_coords_list, dtype='datetime64[ns]')
            logger.info(f"Successfully generated {len(time_coords)} time coordinates as numpy array (dtype={time_coords.dtype}).")
            if len(time_coords) > 0:
                logger.debug(f"First time: {time_coords[0]}, Last time: {time_coords[-1]}")
            # Reset flag as calculation succeeded (might have been set True initially if time_step invalid)
            time_calculation_failed = False

        except Exception as e:
             # Only overwrite flag and coords if error occurs *here*
             if not time_calculation_failed: # Check if it wasn't already failed
                  logger.error(f"Error calculating time coordinates: {e}. Falling back to using numerical indices.", exc_info=True)
                  time_coords = np.arange(n_time_out)
                  time_calculation_failed = True # Set flag
                  logger.warning(f"Using time indices {time_coords} due to calculation error.")
             else:
                 # If time_step was already bad, just use indices
                 logger.error(f"Using time indices because time_step was invalid.")
                 time_coords = np.arange(n_time_out)


        # --- Prepare Data for xarray (Memory Intensive Step) ---
        logger.info("Converting output tensor (PyTorch) to NumPy array...")
        data_np = None
        try:
            tensor_memory_mb = output_tensor.nelement() * output_tensor.element_size() / (1024 * 1024)
            logger.info(f"PyTorch tensor memory size: {tensor_memory_mb:.2f} MB")
            conversion_start_time = time.time()
            if not output_tensor.is_contiguous():
                logger.debug("Output tensor is not contiguous. Calling .contiguous()...")
                output_tensor = output_tensor.contiguous()
            data_np = output_tensor.numpy() # This creates a copy
            conversion_duration = time.time() - conversion_start_time
            logger.info(f"Conversion to NumPy completed in {conversion_duration:.2f}s. NumPy array dtype: {data_np.dtype}")

            if np.isnan(data_np).any(): logger.warning("NaN values detected in the NumPy data array.")
            if np.isinf(data_np).any(): logger.warning("Infinity values detected in the NumPy data array.")

            logger.info("Deleting original PyTorch tensor reference to potentially free memory...")
            del output_tensor
            output_tensor = None # Ensure variable is None
            gc.collect()
            logger.info("PyTorch tensor reference deleted.")

        except Exception as e:
            logger.error(f"Failed to convert output tensor to NumPy array: {e}", exc_info=True)
            if data_np is not None: del data_np
            return # Cannot proceed


        # --- Create xarray Object ---
        logger.info("Creating xarray DataArray...")
        forecast_da = None
        try:
            # Define coordinate names based on dimensionality
            lat_coord_dims = ("lat",) if lat_np.ndim == 1 else ("lat", "lon")
            lon_coord_dims = ("lon",) if lon_np.ndim == 1 else ("lat", "lon")
            # Ensure dims tuple length matches ndim (handles 1D/2D cases)
            lat_coord_dims = lat_coord_dims[:lat_np.ndim]
            lon_coord_dims = lon_coord_dims[:lon_np.ndim]

            # Verify dimensions before creating DataArray
            expected_data_dims = ["ensemble", "time", channel_dim_name, "lat", "lon"]
            coord_map = {
                "ensemble": np.arange(n_ensemble),
                "time": time_coords,
                channel_dim_name: channels_coord,
                "lat": (lat_coord_dims, lat_np), # Tuple: (dimension_names, data)
                "lon": (lon_coord_dims, lon_np), # Tuple: (dimension_names, data)
            }

            forecast_da = xr.DataArray(
                data_np, # Use the numpy array
                coords=coord_map,
                dims=expected_data_dims, # Ensure dims match coord keys and data order
                name="forecast_variables",
                attrs={
                    "description": f"{config.get('weather_model', 'N/A')} ensemble forecast output",
                    "model": config.get('weather_model', 'N/A'),
                    "simulation_length_steps": simulation_length,
                    "output_frequency_steps": output_freq,
                    "ensemble_members": n_ensemble,
                    "initial_condition_time": initial_time.isoformat(),
                    "simulation_time_step_input_value": str(time_step),
                    "simulation_time_step_input_type": str(type(time_step)),
                    "simulation_time_step_units_used": time_step_units,
                    "simulation_time_step_seconds_used": actual_time_step.total_seconds() if actual_time_step else "unknown",
                    "time_coordinate_type": "datetime64[ns]" if not time_calculation_failed else "index",
                    "noise_amplitude": config.get("noise_amplitude", "N/A"),
                    "perturbation_strategy": config.get("perturbation_strategy", "N/A"),
                    "creation_date": datetime.datetime.now(pytz.utc).isoformat(),
                    "pytorch_version": torch.__version__,
                    "numpy_version": np.__version__,
                    "xarray_version": xr.__version__,
                },
            )
            logger.info("Created xarray DataArray successfully.")

        except Exception as e:
            logger.error(f"Failed to create xarray DataArray: {e}", exc_info=True)
            if data_np is not None: del data_np
            return # Cannot proceed

        # --- Convert to Dataset (Optional but Recommended) ---
        logger.info("Attempting to convert DataArray to Dataset (channels as variables)...")
        save_object = None
        try:
            valid_channel_names = True
            if channel_dim_name == "channel":
                for name in channels_coord:
                    if not isinstance(name, str) or not name or not (name[0].isalpha() or name[0] == '_') or not all(c.isalnum() or c in ['_', '-'] for c in name):
                        logger.warning(f"Channel name '{name}' might not be a valid NetCDF variable name. Fallback to saving as DataArray.")
                        valid_channel_names = False
                        break

            if channel_dim_name == "channel" and valid_channel_names:
                forecast_ds = forecast_da.to_dataset(dim=channel_dim_name)
                logger.info("Converted DataArray to Dataset (channels as variables).")
                save_object = forecast_ds
            else:
                if channel_dim_name != "channel":
                     logger.warning(f"Cannot convert to Dataset because channel dimension name is '{channel_dim_name}'. Saving as DataArray.")
                save_object = forecast_da

        except Exception as e:
            logger.error(f"Failed during attempt to convert DataArray to Dataset: {e}. Saving as DataArray instead.", exc_info=True)
            save_object = forecast_da

        # --- Define Output File Path ---
        try:
            ic_time_str = initial_time.strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(
                output_dir,
                f"{config.get('weather_model', 'model')}_"
                f"E{n_ensemble}_"
                f"T{n_time_out}_"
                f"Sim{simulation_length}_"
                f"{ic_time_str}.nc"
            )
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output filename: {output_filename}")
        except Exception as e:
            logger.error(f"Failed to create output path: {e}", exc_info=True)
            if save_object is not None: del save_object
            if data_np is not None: del data_np
            return

        # --- Define Encoding and Save to NetCDF ---
        logger.info(f"Preparing encoding and saving to NetCDF using engine 'netcdf4'...")
        encoding = {}
        fill_value = np.float32(-9999.0)
        try:
            if isinstance(save_object, xr.Dataset):
                encoding = {var: {"zlib": True, "complevel": 5, "_FillValue": fill_value} for var in save_object.data_vars}
                logger.debug(f"Applying encoding to Dataset variables: {list(save_object.data_vars.keys())}")
            elif isinstance(save_object, xr.DataArray):
                 arr_name = save_object.name if save_object.name else "data"
                 encoding = {arr_name: {"zlib": True, "complevel": 5, "_FillValue": fill_value}}
                 logger.debug(f"Applying encoding to DataArray: {arr_name}")

            start_save = time.time()
            save_object.to_netcdf(output_filename, encoding=encoding, engine="netcdf4")
            end_save = time.time()

            if os.path.exists(output_filename):
                 file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
                 logger.info(f"Save complete. Time taken: {end_save - start_save:.2f} seconds. File size: {file_size_mb:.2f} MB")
            else:
                 logger.error("Saving process via to_netcdf seemed to complete but output file not found!")

        except Exception as e:
            logger.error(f"Failed during the NetCDF saving process: {e}", exc_info=True)
            if 'output_filename' in locals() and os.path.exists(output_filename):
                try:
                    os.remove(output_filename)
                    logger.warning(f"Removed potentially corrupted file due to save error: {output_filename}")
                except OSError as oe:
                    logger.error(f"Failed to remove potentially corrupted file {output_filename}: {oe}")
        finally:
             if 'save_object' in locals() and save_object is not None: del save_object
             if 'data_np' in locals() and data_np is not None: del data_np
             gc.collect()
             logger.info("Cleaned up xarray object and NumPy data array.")

    except Exception as e:
        # Catch-all for errors during initial validation or setup
        logger.error(f"An unexpected error occurred early in the save_output function: {e}", exc_info=True)
        # No need to delete output_tensor here, final finally block handles it

    finally:
        # Final cleanup: ensure original input tensor ref is cleaned up if it exists
        if 'output_tensor' in locals() and output_tensor is not None:
            logger.debug("Cleaning up output_tensor reference in final finally block.")
            del output_tensor
            gc.collect()

        func_duration = time.time() - func_start_time
        logger.info(f"save_output function finished in {func_duration:.2f} seconds.")

    return None # Return None to indicate completion



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
        logger.info(f"Processing initial condition file: {fname}")
        
        try:
            base_date = parse_date_from_filename(fname)
            logger.info(f"Successfully parsed date: {base_date.strftime('%Y-%m-%d')}")
        except:            
            base_date = datetime.datetime(2020, 6, 22, tzinfo=pytz.utc)  # Make timezone aware (UTC is standard for IFS/ERA)
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
    logger.info(f"Output path: {args.output_path}") 
    
    
    
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
                output_dir=args.output_path,
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
    logger.info(f"Output NetCDF files saved in: {args.output_path}")
    logger.info(f"Log file saved in: {LOG_DIR}")
    logger.info("========================================================")
    logger.info(" FCNv2-SM Inference Pipeline Finished ")
    logger.info("========================================================")


















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCNv2-SM Inference Pipeline using initial conditions from a NumPy file.")

    # Input/Output paths
    parser.add_argument("--ic-file-path", type=str, default=f"/scratch/gilbreth/wwtung/ARCO_73chanel_data/data/2020/June/START_22_June_2020_END_22_June_2020.npy", help="Path to the NumPy file containing initial conditions (shape: T, C, H, W).")
    # Default output path uses the dynamically generated OUTPUT_DIR
    parser.add_argument("-o", "--output-path", type=str, default=os.path.join(OUTPUT_DIR, "saved_netcdf"), help="Directory to save output NetCDF files.")

    # Inference parameters
    parser.add_argument("-sim", "--simulation-length", type=int, default=5, help="Number of autoregressive steps (forecast lead time in model steps).")
    parser.add_argument("-ef", "--output-frequency", type=int, default=1, help="Frequency (in steps) to save output states (e.g., 1 = save every step).")
    parser.add_argument("-ens", "--ensemble-members", type=int, default=1, help="Number of ensemble members (>=1).")
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
