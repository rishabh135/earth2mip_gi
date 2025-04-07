# Paste pre-Pyrfecte# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import os
import sys
import urllib
import warnings
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from modulus.utils.zenith_angle import cos_zenith_angle



import _pickle

# import torch
# import numpy as np
# import datetime
# import logging
# from typing import Optional, Callable, Any, Iterator, Tuple
# import earth2mip.grid
# from . import time_loop # Assuming time_loop is in the same directory or correctly imported
# from . import schema # Assuming schema is correctly imported
# from . import loaders # Assuming loaders is correctly imported
# from importlib.metadata import EntryPoint
# from .types import LoaderProtocol # Assuming types is correctly imported



logging.getLogger("earth2mip.networks_init_inference").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


import earth2mip.grid
from earth2mip import (
    ModelRegistry,
    loaders,
    model_registry,
    registry,
    schema,
    time_loop,
)
from earth2mip.loaders import LoaderProtocol

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points
else:
    from importlib.metadata import EntryPoint, entry_points


__all__ = ["get_model"]


def depends_on_time(f):
    """
    A function to detect if the function `f` takes an argument `time`.

    Args:
        f: a function.

    Returns:
        bool: True if the function takes a second argument `time`, False otherwise.
    """
    # check if model is a torchscript model
    if isinstance(f, torch.jit.ScriptModule):
        return False
    else:
        import inspect

        signature = inspect.signature(f)
        parameters = signature.parameters
        return "time" in parameters


class Wrapper(torch.nn.Module):
    """Makes sure the parameter names are the same as the checkpoint"""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """x: (batch, history, channel, x, y)"""
        return self.module(*args, **kwargs)


class CosZenWrapper(torch.nn.Module):
    def __init__(self, model, lon, lat):
        super().__init__()
        self.model = model
        self.lon = lon
        self.lat = lat

    def forward(self, x, time):
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        cosz = cos_zenith_angle(time, lon_grid, lat_grid)
        cosz = cosz.astype(np.float32)
        z = torch.from_numpy(cosz).to(device=x.device)
        # assume no history
        x = torch.cat([x, z[None, None]], dim=1)
        return self.model(x)


class _SimpleModelAdapter(torch.nn.Module):
    """Takes model of (b, c, y, x) to (b, h, y, x) where h == 1"""

    def __init__(self, model, time_dependent, has_history):
        super().__init__()
        self.model = model
        self.time_dependent = time_dependent
        self.has_history = has_history

    def forward(self, x, time):
        if not self.has_history:
            x = x[:, 0]

        if self.time_dependent:
            y = self.model.forward(x, time)
        else:
            y = self.model.forward(x)

        if not self.has_history:
            y = y[:, None]

        return y



class Inference(torch.nn.Module, time_loop.TimeLoop):
    def __init__(
        self,
        model,
        center: np.array,
        scale: np.array,
        grid: earth2mip.grid.LatLonGrid,
        source: Optional[Callable] = None,
        n_history: int = 0,
        time_step=datetime.timedelta(hours=6),
        channel_names=None,
        normalize=True, # Argument name can stay normalize
    ):
        """
        Args:
            model: a model, with signature model(x, time) or model(x). With n_history == 0, x is a
                torch tensor with shape (batch, nchannel, lat, lon). With
                n_history > 0 x has the shape (batch, nchannel, lat, lon).
                `time` is a datetime object, which is passed if model.forward has time as an argument.
            center: a 1d numpy array with shape (n_channels in data) containing
                the means. The shape is NOT `len(channels)`.
            scale: a 1d numpy array with shape (n_channels in data) containing
                the stds. The shape is NOT `len(channels)`.
            source: a source function that augments the state vector (noise, nudge or other)
            grid: metadata about the grid, which should be used to pass the
                correct data to this object.
            channel_names: The names of the prognostic channels.
            n_history: whether `model` was trained with history.
            time_step: the time-step `model` was trained with.
            normalize: boolean flag to indicate if normalization should be applied by default.

        """  # noqa
        super().__init__()

        # Check if model.forward depends on time
        try:
             self.time_dependent = depends_on_time(model.forward)
        except AttributeError:
             logger.warning("Could not inspect model.forward, assuming time-independent.")
             self.time_dependent = False


        logger.info(f"Initializing earth2mip.networks.Inference.") # Changed log level

        # Wrap model if necessary (keep existing logic)
        # TODO: Review _SimpleModelAdapter if issues persist
        model = _SimpleModelAdapter(
            model, time_dependent=self.time_dependent, has_history=n_history > 0
        )

        self.model = model
        self.channel_names = channel_names if channel_names else list(range(center.shape[0]))
        self.grid = grid
        self.time_step = time_step
        self.n_history = n_history
        self.source = source

        # ****** FIX HERE: Store boolean flag with a different name ******
        self.do_normalize_flag = normalize
        logger.info(f"Normalization flag (do_normalize_flag) set to: {self.do_normalize_flag}")
        # ****** END FIX ******

        # Check shapes and print out and log all shapes of center and scale including explicity dimensions
        logger.info(f"Center shape: {center.shape}, Scale shape: {scale.shape}") # Changed log level
        print(f"Center shape: {center.shape}, Scale shape: {scale.shape}")
        logger.debug(f"Center shape: {center.shape}, Scale shape: {scale.shape}")
        
        
        
        logger.info(f"Received center shape: {center.shape}, scale shape: {scale.shape}") # Changed log level
        if len(self.channel_names) != center.shape[0] or len(self.channel_names) != scale.shape[0]:
            logger.error(f"Mismatch between number of channel names ({len(self.channel_names)}) and center/scale dimensions ({center.shape[0]}/{scale.shape[0]})")
            raise ValueError("Channel names length must match center/scale dimension")

        center_tensor = torch.from_numpy(center).float()
        scale_tensor = torch.from_numpy(scale).float()
        # Store original 1D scale/center for potential other uses if needed
        self.register_buffer("scale_org", scale_tensor.clone())
        self.register_buffer("center_org", center_tensor.clone())

        # Prepare buffers for broadcasting (C, 1, 1)
        self.register_buffer("scale", scale_tensor[:, None, None])
        self.register_buffer("center", center_tensor[:, None, None])
        logger.debug(f"Registered buffers 'center' and 'scale' with shape: {self.center.shape}")

        # infer channel names
        self.in_channel_names = self.out_channel_names = self.channel_names
        # self.channels = list(range(len(channel_names))) # Redundant if channel_names is used

    @property
    def n_history_levels(self) -> int:
        """The expected size of the second dimension"""
        return self.n_history + 1

    @property
    def device(self) -> torch.device:
        # A reliable way to get device is from a registered buffer/parameter
        return self.center.device

    # ****** FIX HERE: Add normalize method ******
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalization (mean subtraction and division by std dev) if flag is set."""
        if self.do_normalize_flag:
            logger.debug(f"Applying normalization. Input shape: {x.shape}, Center shape: {self.center.shape}, Scale shape: {self.scale.shape}")
            center = self.center.to(x.device) # Ensure buffer is on correct device
            scale = self.scale.to(x.device)   # Ensure buffer is on correct device
            if center.shape[0] != x.shape[-3]: # Check channel dim C in (B, T, C, H, W) or (B, C, H, W)
                 logger.error(f"Channel dimension mismatch during normalization! Input channels: {x.shape[-3]}, Center/Scale channels: {center.shape[0]}")
                 raise ValueError("Channel dimension mismatch during normalization")
            try:
                 # Handle both 4D (B, C, H, W) and 5D (B, T, C, H, W) inputs
                if x.ndim == 5:
                    # Normalize each time step independently
                    return (x - center.unsqueeze(1)) / scale.unsqueeze(1)
                elif x.ndim == 4:
                    return (x - center) / scale
                else:
                    logger.error(f"Unsupported tensor dimension for normalization: {x.ndim}")
                    raise ValueError(f"Unsupported tensor dimension for normalization: {x.ndim}")
            except RuntimeError as e:
                 logger.error(f"Runtime error during normalization: {e}", exc_info=True)
                 if (scale == 0).any():
                    logger.error("Found zero values in scale tensor! Division by zero likely.")
                 raise
        else:
            logger.debug("Skipping normalization because do_normalize_flag is False.")
            return x
    # ****** END ADD normalize ******

    # ****** FIX HERE: Add denormalize method ******
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies denormalization (multiplication by std dev and mean addition)."""
        logger.debug(f"Applying denormalization. Input shape: {x.shape}, Center shape: {self.center.shape}, Scale shape: {self.scale.shape}")
        # Denormalization is typically always applied to the output, regardless of the input flag
        scale = self.scale.to(x.device) # Ensure buffer is on correct device
        center = self.center.to(x.device) # Ensure buffer is on correct device
        if scale.shape[0] != x.shape[-3]: # Check channel dim C in (B, T, C, H, W) or (B, C, H, W)
            logger.error(f"Channel dimension mismatch during denormalization! Input channels: {x.shape[-3]}, Center/Scale channels: {scale.shape[0]}")
            raise ValueError("Channel dimension mismatch during denormalization")

        # Handle both 4D (B, C, H, W) and 5D (B, T, C, H, W) inputs
        if x.ndim == 5:
            return x * scale.unsqueeze(1) + center.unsqueeze(1)
        elif x.ndim == 4:
            return x * scale + center
        else:
            logger.error(f"Unsupported tensor dimension for denormalization: {x.ndim}")
            raise ValueError(f"Unsupported tensor dimension for denormalization: {x.ndim}")
    # ****** END ADD denormalize ******


    def __call__(
        self,
        time: datetime.datetime,
        x: torch.Tensor,
        restart: Optional[Any] = None,
    ) -> Iterator[Tuple[datetime.datetime, torch.Tensor, Any]]:
        """
        Args:
            x: an initial condition. has shape (B, n_history_levels,
                len(in_channel_names), Y, X). If n_history=0, shape is (B, 1, C, H, W)
                or potentially (B, C, H, W) - needs clarification/handling.
                (Y, X) should be consistent with ``grid``.
            time: the datetime to start with
            restart: if provided this restart information (typically some torch
                Tensor) can be used to restart the time loop

        Yields:
            (time, output, restart) tuples. ``output`` is a tensor with
                shape (B, len(out_channel_names), Y, X) which will be used for
                diagnostics. Restart data should encode the state of the time
                loop.
        """
        logger.debug(f"Inference __call__ started. Initial time: {time}, Input x shape: {x.shape}, Restart provided: {restart is not None}")

        # Handle potential 4D input (B, C, H, W) by adding time dimension if n_history=0
        if x.ndim == 4 and self.n_history == 0:
            x = x.unsqueeze(1) # Add time dimension -> (B, 1, C, H, W)
            logger.debug(f"Added time dimension to 4D input, new shape: {x.shape}")
        elif x.ndim != 5:
             logger.error(f"Unexpected input tensor dimension in __call__: {x.ndim}. Expected 5D (B, T, C, H, W) or 4D if n_history=0.")
             raise ValueError("Invalid input tensor dimension for __call__")

        if restart:
            # TODO: Validate restart data structure if necessary
            logger.info("Restarting inference from provided state.")
            yield from self._iterate(**restart)
        else:
            logger.info("Starting inference from initial condition.")
            # IMPORTANT: The user script now handles normalization *before* the loop.
            # So, the `x` passed here might already be normalized if the flag was true.
            # However, _iterate internally expects normalized data based on its original structure.
            # Let's normalize here *if the flag is set* to ensure _iterate gets normalized input.
            x_normalized_for_iterate = self.normalize(x) # Use the new method
            yield from self._iterate(x=x_normalized_for_iterate, time=time) # Pass normalized data


    def _iterate(self, x, time): # Removed normalize argument, assumes x is already normalized
            """Yield (time, unnormalized data, restart) tuples

            Assumes input x is ALREADY NORMALIZED.
            Restart = dictionary containing state needed to resume.
            """
            logger.debug(f"Starting _iterate loop. Initial time: {time}. Initial norm input shape: {x.shape}")
            if self.time_dependent and not time:
                raise ValueError("Time dependent models require ``time``.")

            # Ensure time is timezone-aware if possible, default to UTC?
            # Or ensure consistency with input time's awareness. For now, use as is.

            with torch.no_grad():
                # Input x shape should be (B, T, C, H, W) where T = n_history + 1
                if x.ndim != 5:
                     logger.error(f"Unexpected input tensor dimension in _iterate: {x.ndim}. Expected 5D.")
                     raise ValueError("Invalid input tensor dimension for _iterate")

                _, n_time_levels, n_channels, _, _ = x.shape
                logger.debug(f"_iterate initial state check: time_levels={n_time_levels}, channels={n_channels}")
                if n_time_levels != self.n_history + 1:
                    logger.error(f"History level mismatch! Expected {self.n_history + 1} time levels, got {n_time_levels}")
                    raise ValueError("Input tensor time dimension doesn't match n_history")

                # Yield initial time for convenience
                # Output requires denormalized data. Use the last time step x[:, -1].
                logger.debug("Yielding initial state (t=0).")
                initial_output = self.denormalize(x[:, -1]) # Use denormalize method
                restart_state = {'x': x.clone(), 'time': time} # Store current normalized state
                yield time, initial_output, restart_state

                while True:
                    loop_start_time = datetime.datetime.now()
                    current_state_norm = x # Keep track of current normalized state

                    if self.source:
                        # Apply source term (requires denormalizing, applying, renormalizing)
                        logger.debug(f"Applying source term at time {time}")
                        x_denorm = self.denormalize(current_state_norm) # Denormalize
                        dt = torch.tensor(self.time_step.total_seconds(), device=self.device, dtype=x_denorm.dtype)
                        source_update_denorm = self.source(x_denorm, time) # Apply source
                        x_denorm += source_update_denorm # Add update
                        current_state_norm = self.normalize(x_denorm) # Renormalize
                        logger.debug("Source term applied and state renormalized.")

                    # Model prediction step
                    logger.debug(f"Running model prediction for time step starting at {time}. Input shape: {current_state_norm.shape}")
                    # Pass time only if model expects it
                    model_input_args = (current_state_norm, time) if self.time_dependent else (current_state_norm,)
                    next_state_norm = self.model(*model_input_args) # model is the _SimpleModelAdapter
                    logger.debug(f"Model prediction complete. Output shape: {next_state_norm.shape}")

                    # Update time
                    time = time + self.time_step

                    # Update state for next iteration (x becomes the new state)
                    x = next_state_norm

                    # Create restart state for this step (store normalized state)
                    restart_state = {'x': x.clone(), 'time': time}

                    # Prepare output (denormalize the last time step)
                    logger.debug("Denormalizing output for yielding.")
                    output_denorm = self.denormalize(x[:, -1]) # Use denormalize method

                    loop_end_time = datetime.datetime.now()
                    logger.debug(f"Yielding state for time {time}. Loop step took {(loop_end_time-loop_start_time).total_seconds():.3f}s")
                    yield time, output_denorm, restart_state


# Assuming _default_inference remains largely the same, but ensure it passes
# the correct 'normalize' boolean flag (or relies on its default=True)
# when creating the Inference object if that's intended.
# It currently doesn't explicitly pass 'normalize', so it will use the default True.





def _default_inference(package, metadata: schema.Model, device):
    """
    Creates an Inference object using model package, metadata, and device.
    Enhanced with preprocessing for center/scale arrays and robust validation.
    """
    logger.info("Starting _default_inference setup.")

    # Load model using appropriate loader
    if metadata.architecture == "pickle":
        loader = loaders.pickle
    elif metadata.architecture_entrypoint:
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group='earth2mip.loaders', name=metadata.architecture_entrypoint)
            if not eps:
                raise ImportError(f"Entry point {metadata.architecture_entrypoint} not found.")
            ep = next(iter(eps))  # Get first entry point
        except ImportError:
            ep = EntryPoint(name=None, group=None, value=metadata.architecture_entrypoint)

        loader: LoaderProtocol = ep.load()
    else:
        raise NotImplementedError(f"Unsupported architecture type: {metadata.architecture}")

    logger.info(f"Using loader: {loader}")
    model = loader(package, pretrained=True)
    logger.info("Model loaded successfully.")

    # Load center and scale arrays
    center_path = package.get("global_means.npy")
    scale_path = package.get("global_stds.npy")
    logger.info(f"Loading center from: {center_path}")
    logger.info(f"Loading scale from: {scale_path}")

    try:
        center_data = np.load(center_path)
        scale_data = np.load(scale_path)
        logger.info(f"Loaded center array with shape: {center_data.shape}")
        logger.info(f"Loaded scale array with shape: {scale_data.shape}")

        # Squeeze unnecessary dimensions
        center_data = np.squeeze(center_data)  # Remove dimensions of size 1
        scale_data = np.squeeze(scale_data)  # Remove dimensions of size 1
        logger.info(f"Squeezed center shape: {center_data.shape}")
        logger.info(f"Squeezed scale shape: {scale_data.shape}")

    except Exception as e:
        logger.error(f"Failed to load or preprocess center/scale arrays: {e}", exc_info=True)
        raise

    # Validate consistency between metadata and processed arrays
    num_channels_metadata = len(metadata.in_channels_names)
    num_channels_center = center_data.shape[0]
    num_channels_scale = scale_data.shape[0]

    if num_channels_metadata != num_channels_center or num_channels_metadata != num_channels_scale:
        logger.error(
            f"Channel mismatch detected! "
            f"Metadata channels: {num_channels_metadata}, "
            f"Center channels: {num_channels_center}, "
            f"Scale channels: {num_channels_scale}"
        )
        raise ValueError(
            "Mismatch between metadata channel count and center/scale dimensions."
        )

    logger.info("Channel count validation passed.")

    # Ensure input/output channels match
    assert (
        metadata.in_channels_names == metadata.out_channels_names
    ), "Input and Output channels must match for this inference wrapper."

    # Create Inference object
    logger.info("Creating Inference object.")
    try:
        inference = Inference(
            model=model,
            channel_names=metadata.in_channels_names,
            center=center_data,
            scale=scale_data,
            grid=earth2mip.grid.from_enum(metadata.grid),
            n_history=metadata.n_history or 0,
            time_step=metadata.time_step,
            normalize=True,
        )
        inference.to(device)
        logger.info(f"Inference object created successfully on device: {device}")
        return inference
    except ValueError as e:
        logger.error(f"Inference object creation failed due to ValueError: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Inference object creation failed due to unexpected error: {e}", exc_info=True)
        raise



# class Inference(torch.nn.Module, time_loop.TimeLoop):
#     def __init__(
#         self,
#         model,
#         center: np.array,
#         scale: np.array,
#         grid: earth2mip.grid.LatLonGrid,
#         source: Optional[Callable] = None,
#         n_history: int = 0,
#         time_step=datetime.timedelta(hours=6),
#         channel_names=None,
#         normalize=True,
#     ):
#         """
#         Args:
#             model: a model, with signature model(x, time) or model(x). With n_history == 0, x is a
#                 torch tensor with shape (batch, nchannel, lat, lon). With
#                 n_history > 0 x has the shape (batch, nchannel, lat, lon).
#                 `time` is a datetime object, which is passed if model.forward has time as an argument.
#             center: a 1d numpy array with shape (n_channels in data) containing
#                 the means. The shape is NOT `len(channels)`.
#             scale: a 1d numpy array with shape (n_channels in data) containing
#                 the stds. The shape is NOT `len(channels)`.
#             source: a source function that augments the state vector (noise, nudge or other)
#             grid: metadata about the grid, which should be used to pass the
#                 correct data to this object.
#             channel_names: The names of the prognostic channels.
#             n_history: whether `model` was trained with history.
#             time_step: the time-step `model` was trained with.

#         """  # noqa
#         super().__init__()
#         self.time_dependent = depends_on_time(model.forward)
        
#         logger.warning(f"inside eart2mip/earth2mip/networks.__init__.py this is the main inference call")
#         # TODO probably delete this line
#         # if not isinstance(model, modulus.Module):
#         #     model = Wrapper(model)

#         # TODO extract this to another place
#         model = _SimpleModelAdapter(
#             model, time_dependent=self.time_dependent, has_history=n_history > 0
#         )

#         self.model = model
#         self.channel_names = channel_names
#         self.grid = grid
#         self.time_step = time_step
#         self.n_history = n_history
#         self.source = source
#         self.normalize = normalize

#         logger.warning(f" center: {center.shape} scale {scale.shape} channel_names {self.channel_names} ")
        
#         center = torch.from_numpy(center).float()
#         scale = torch.from_numpy(scale).float()
#         self.register_buffer("scale_org", scale)
#         self.register_buffer("center_org", center)
#         # infer channel names
#         self.in_channel_names = self.out_channel_names = channel_names
#         self.channels = list(range(len(channel_names)))
#         self.register_buffer("scale", scale[:, None, None])
#         self.register_buffer("center", center[:, None, None])

#     @property
#     def n_history_levels(self) -> int:
#         """The expected size of the second dimension"""
#         return self.n_history + 1

#     @property
#     def device(self) -> torch.device:
#         return self.scale.device

#     def __call__(
#         self,
#         time: datetime.datetime,
#         x: torch.Tensor,
#         restart: Optional[Any] = None,
        
#     ) -> Iterator[Tuple[datetime.datetime, torch.Tensor, Any]]:
#         """
#         Args:
#             x: an initial condition. has shape (B, n_history_levels,
#                 len(in_channel_names), Y, X).  (Y, X) should be consistent with
#                 ``grid``.
#             time: the datetime to start with
#             restart: if provided this restart information (typically some torch
#                 Tensor) can be used to restart the time loop

#         Yields:
#             (time, output, restart) tuples. ``output`` is a tensor with
#                 shape (B, len(out_channel_names), Y, X) which will be used for
#                 diagnostics. Restart data should encode the state of the time
#                 loop.
#         """
#         if restart:
#             yield from self._iterate(**restart)
#         else:
#             yield from self._iterate(x=x, time=time)


#     def _iterate(self, x, normalize=True, time=None):
#             """Yield (time, unnormalized data, restart) tuples

#             restart = (time, unnormalized data)
#             """
#             if self.time_dependent and not time:
#                 raise ValueError("Time dependent models require ``time``.")
#             time = time or datetime.datetime(1900, 1, 1)
#             with torch.no_grad():
#                 # drop all but the last time point
#                 # remove channels

#                 _, n_time_levels, n_channels, _, _ = x.shape
#                 logging.warning(f" __init__.py iterate funciton time with removing normalizing from x : {time}   input_shape: {x.shape}  time_levels: {n_time_levels}  n_channels: { n_channels} ")
#                 assert n_time_levels == self.n_history + 1  # noqa

#                 # if normalize:
#                 #     x = (x - self.center) / self.scale

#                 # yield initial time for convenience
#                 restart = dict(x=x, normalize=False, time=time)
#                 yield time, self.scale * x[:, -1] + self.center, restart

#                 while True:
#                     if self.source:
#                         x_with_units = x * self.scale + self.center
#                         dt = torch.tensor(self.time_step.total_seconds())
#                         x += self.source(x_with_units, time) / self.scale * dt
#                     x = self.model(x, time)
#                     time = time + self.time_step

#                     # create args and kwargs for future use
#                     restart = dict(x=x, normalize=False, time=time)
#                     out = self.scale * x[:, -1] + self.center
                
#                     yield time, out, restart


# def _default_inference(package, metadata: schema.Model, device):
#     if metadata.architecture == "pickle":
#         loader = loaders.pickle
#     elif metadata.architecture_entrypoint:
#         ep = EntryPoint(name=None, group=None, value=metadata.architecture_entrypoint)
#         loader: LoaderProtocol = ep.load()
#     else:
#         raise NotImplementedError()

#     model = loader(package, pretrained=True)

#     center_path = package.get("global_means.npy")
#     scale_path = package.get("global_stds.npy")

#     assert metadata.in_channels_names == metadata.out_channels_names  # noqa

#     inference = Inference(
#         model=model,
#         channel_names=metadata.in_channels_names,
#         center=np.load(center_path),
#         scale=np.load(scale_path),
#         grid=earth2mip.grid.from_enum(metadata.grid),
#         n_history=metadata.n_history,
#         time_step=metadata.time_step,
#     )
#     inference.to(device)
#     return inference



def _load_package_builtin(package, device, name) -> time_loop.TimeLoop:
    group = "earth2mip.networks"
    entrypoints = entry_points(group=group)

    names_found = []
    for entry_point in entrypoints:
        names_found.append(entry_point.name)
        if entry_point.name == name:
            inference_loader = entry_point.load()
            return inference_loader(package, device=device)
    raise ValueError(f"{name} not in {names_found}.")


def _load_package(package, metadata, device, normalize) -> time_loop.TimeLoop:
    # Attempt to see if Earth2 MIP has entry point registered already
    # Read meta data from file if not present
    package.path = "/scratch/gilbreth/gupt1075/fcnv2/earth2mip/earth2mip/networks/fcnv2"
    logger.warning(
        f" Inside load package  in new model {package.get('name')}   path {package.path} \n metadata {package.get('metadata.json')}  \n directory :  {dir(package)} "
    )
    if metadata is None:
        local_path = package.get("metadata.json")
        with open(local_path) as f:
            metadata = schema.Model.parse_raw(f.read())

    if metadata.entrypoint:
        ep = EntryPoint(name=None, group=None, value=metadata.entrypoint.name)
        inference_loader = ep.load()
        return inference_loader(package, device=device, **metadata.entrypoint.kwargs)
    else:
        warnings.warn("No loading entry point found, using default inferencer")
        return _default_inference(package, metadata, device=device, normalize=normalize)


def get_model(
    model: str,
    registry: ModelRegistry = registry,
    device="cpu",
    normalize=True,
    metadata: Optional[schema.Model] = None,
) -> time_loop.TimeLoop:
    """
    Function to construct an inference model and load the appropriate
    checkpoints from the model registry

    Parameters
    ----------
    model : The model name to open in the ``registry``. If a url is passed (e.g.
        s3://bucket/model), then this location will be opened directly.
        Supported urls protocols include s3:// for PBSS access, and file:// for
        local files.
    registry: A model registry object. Defaults to the global model registry
    metadata: If provided, this model metadata will be used to load the model.
        By default this will be loaded from the file ``metadata.json`` in the
        model package.
    device: the device to load on, by default the 'cpu'


    Returns
    -------
    Inference model


    """
    url = urllib.parse.urlparse(model)
    # write out all arguments and attributes of registry object to logger file

    logger.warning(
        f" Model name inside init.get_model() {model}    path:     {registry.path}  \n     weight path: {registry.get_weight_path} \n { dir(registry) } "
    )

    logger.warning(f" >>>  url_scheme {url.scheme}  model: {model}")

    logger.warning(f" Model name inside init.get_model metadata:  {metadata} ")

    if url.scheme == "e2mip":
        package = registry.get_model(model)
        return _load_package_builtin(package, device, name=url.netloc)
    elif url.scheme == "":
        package = registry.get_model(model)
        return _load_package(package, metadata, device, normalize)
    else:
        package = model_registry.Package(root=model, seperator="/")
        return _load_package(package, metadata, device, normalize)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def persistence(package, pretrained=True):
    model = Identity()
    center = np.zeros((3))
    scale = np.zeros((3))
    grid = earth2mip.grid.equiangular_lat_lon_grid(721, 1440)
    return Inference(
        model,
        channel_names=["a", "b", "c"],
        center=center,
        scale=scale,
        grid=grid,
        n_history=0,
    )


