# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

import argparse
import json
import logging

logging.getLogger("batch_inference").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

import os, math
import sys
from datetime import datetime, timedelta
from typing import Any, Optional
import pandas as pd

from netCDF4 import Dataset


from torchinfo import summary
import cftime
import numpy as np
import torch
import tqdm
import xarray
from modulus.distributed.manager import DistributedManager
from netCDF4 import Dataset as DS

import earth2mip.grid

__all__ = ["run_inference"]

# need to import initial conditions first to avoid unfortunate
# GLIBC version conflict when importing xarray. There are some unfortunate
# issues with the environment.
from earth2mip import initial_conditions, regrid, time_loop
from earth2mip._channel_stds import channel_stds
from earth2mip.ensemble_utils import (
    generate_bred_vector,
    generate_noise_correlated,
    generate_noise_grf,
)
from earth2mip.netcdf import initialize_netcdf, update_netcdf
from earth2mip.networks import get_model
from earth2mip.schema import EnsembleRun, PerturbationStrategy
from earth2mip.time_loop import TimeLoop

# logger = logger.getLogger("inference")


def get_checkpoint_path(rank, batch_id, path):
    directory = os.path.join(path, f"{rank}")
    filename = f"{batch_id}.pth"
    return os.path.join(directory, filename)


def save_restart(restart, rank, batch_id, path):
    path = get_checkpoint_path(rank, batch_id, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Saving restart file to {path}.")
    torch.save(restart, path)


def run_ensembles(
    *,
    n_steps: int,
    weather_event,
    model: TimeLoop,
    perturb,
    x,
    nc,
    domains,
    n_ensemble: int,
    batch_size: int,
    rank: int,
    output_frequency: int,
    output_grid: Optional[earth2mip.grid.LatLonGrid],
    date_obj: datetime,
    restart_frequency: Optional[int],
    output_path: str,
    restart_initial_directory: str = "",
    progress: bool = True,
):
    if not output_grid:
        output_grid = model.grid

    regridder = regrid.get_regridder(model.grid, output_grid).to(model.device)

    diagnostics = initialize_netcdf(nc, domains, output_grid, n_ensemble, model.device)
    initial_time = date_obj
    time_units = initial_time.strftime("hours since %Y-%m-%d %H:%M:%S")
    nc["time"].units = time_units
    nc["time"].calendar = "standard"
    logger.warning(f"  time_units {time_units}  n_ensembles {n_ensemble}, batch_size {batch_size} ")
    for batch_id in range(0, n_ensemble, batch_size):
        logger.info(
            f"ensemble members {batch_id+1}-{batch_id+batch_size}/{n_ensemble}"
        )
        batch_size = min(batch_size, n_ensemble - batch_id)


        x = x.repeat(batch_size, 1, 1, 1, 1)

        logger.warning(f" SKIPPING Perturb before perturb x-> {x.shape}   rank:  {rank}  batch_id {batch_id}")
        # x = perturb(x, rank, batch_id, model.device)
        
        
        # restart_dir = weather_event.properties.restart

        # TODO: figure out if needed
        # if restart_dir:
        #     path = get_checkpoint_path(rank, batch_id, restart_dir)
        #     # TODO use logger
        #     logger.info(f"Loading from restart from {path}")
        #     kwargs = torch.load(path)
        # else:
        #     kwargs = dict(
        #         x=x,
        #         normalize=False,
        #         time=time,
        #     )

        iterator = model(initial_time, x)
        
        # out_sum = summary(model, input_data=[initial_time.map(pd.Timedelta.to_pytimedelta), x], mode="train", col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], depth=4)
        # logger.warning(" model_summary: {} \n".format(out_sum))

    
        logger.warning(f" >> run_ensemble in inference_ensemble running iterator for model for times: {initial_time} and with x {x.shape} \n ")
    

        # Check if stdout is connected to a terminal
        if sys.stderr.isatty() and progress:
            iterator = tqdm.tqdm(iterator, total=n_steps)

        time_count = -1

        # for time, data, restart in iterator:

        for k, (time, data, _) in enumerate(iterator):
            # if restart_frequency and k % restart_frequency == 0:
            #     save_restart(
            #         restart,
            #         rank,
            #         batch_id,
            #         path=os.path.join(output_path, "restart", time.isoformat()),
            #     )

            # Saving the output
            if output_frequency and k % output_frequency == 0:
                time_count += 1
                logger.warning(f" >> Saving data at step {k} of {n_steps}.")
                nc["time"][time_count] = cftime.date2num(time, nc["time"].units)
                update_netcdf(
                    regridder(data),
                    diagnostics,
                    domains,
                    batch_id,
                    time_count,
                    model.grid,
                    model.out_channel_names,
                )

            if k == n_steps:
                break

        # if restart_frequency is not None:
        #     save_restart(
        #         restart,
        #         rank,
        #         batch_id,
        #         path=os.path.join(output_path, "restart", "end"),
        #     )


def main(config=None, nc_file_path=None):
    logger.warning(
        f" Inside inference_ensemble and using standard args with weather_model config: {config} "
    )

    if config is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("config")
        parser.add_argument(
            "--weather_model",
            default=None,
        )
        args = parser.parse_args()
        config = args.config

    # If config is a file
    if os.path.exists(config):
        config: EnsembleRun = EnsembleRun.parse_file(config)
    # If string, assume JSON string
    elif isinstance(config, str):
        config: EnsembleRun = EnsembleRun.parse_obj(json.loads(config))
    # Otherwise assume parsable obj
    else:
        raise ValueError(
            f"Passed config parameter {config} should be valid file or JSON string"
        )

    # if args and args.weather_model:
    #     config.weather_model = args.weather_model

    # Set up parallel

    
    
    
    logger.warning(
        f" Inside inference_ensemble insitialuzed distributed manager setting parallel trainig with config {config} "
    )
    # DistributedManager.initialize()
    # device = DistributedManager().device

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # logger.warning(f" device {device}")
    
    group = torch.distributed.group.WORLD

    # logger.warning(f" device {device} group {group}")
    

    logger.info(f"Earth-2 MIP config loaded {config}")
    logger.info(f"Loading model onto device {device}")
    model = get_model(config.weather_model, device=device)
    logger.info("Constructing initializer data source")
    perturb = get_initializer(
        model,
        config,
    )
    logger.info("Running inference")
    original_xr = run_inference(model, config, perturb, group, nc_file_path=nc_file_path)
    return original_xr

def get_initializer(
    model,
    config,
):
    def perturb(x, rank, batch_id, device):
        shape = x.shape
        if config.perturbation_strategy == PerturbationStrategy.gaussian:
            noise = config.noise_amplitude * torch.normal(
                torch.zeros(shape), torch.ones(shape)
            ).to(device)
        elif config.perturbation_strategy == PerturbationStrategy.correlated:
            noise = generate_noise_correlated(
                shape,
                reddening=config.noise_reddening,
                device=device,
                noise_amplitude=config.noise_amplitude,
            )
        elif config.perturbation_strategy == PerturbationStrategy.spherical_grf:
            noise = generate_noise_grf(
                shape,
                model.grid,
                sigma=config.grf_noise_sigma,
                alpha=config.grf_noise_alpha,
                tau=config.grf_noise_tau,
            ).to(device)
        elif config.perturbation_strategy == PerturbationStrategy.bred_vector:
            noise = generate_bred_vector(
                x,
                model,
                config.noise_amplitude,
                time=config.weather_event.properties.start_time,
            )
        elif config.perturbation_strategy == PerturbationStrategy.none:
            return x
        if rank == 0 and batch_id == 0:  # first ens-member is deterministic
            noise[0, :, :, :, :] = 0

        # When field is not in known normalization dictionary set scale to 0
        scale = []
        for i, channel in enumerate(model.in_channel_names):
            if channel in channel_stds:
                scale.append(channel_stds[channel])
            else:
                scale.append(0)
        scale = torch.tensor(scale, device=x.device)

        if config.perturbation_channels is None:
            x += noise * scale[:, None, None]
        else:
            channel_list = model.in_channel_names
            indices = torch.tensor(
                [
                    channel_list.index(channel)
                    for channel in config.perturbation_channels
                    if channel in channel_list
                ]
            )
            x[:, :, indices, :, :] += (
                noise[:, :, indices, :, :] * scale[indices, None, None]
            )
        return x

    return perturb


def run_basic_inference(
    model: time_loop.TimeLoop,
    n: int,
    data_source: Any,
    time: datetime,
):
    
    x = initial_conditions.get_initial_condition_for_model(model, data_source, time)

    """Run a basic inference"""
    logger.warning(
        f" BASIC inference_ensemble using a basic inference model: {model}, data_source: {data_source}  time: {time} with initial_conditions {x.shape} "
    )

    arrays = []
    times = []
    for k, (time, data, _) in enumerate(model(time, x)):
        arrays.append(data.cpu().numpy())
        times.append(time)
        if k == n:
            break

    stacked = np.stack(arrays)
    coords = dict(lat=model.grid.lat, lon=model.grid.lon)
    coords["channel"] = model.out_channel_names
    coords["time"] = times
    
    logger.warning(f" ran inference for model for times: {times} and for channels {model.out_channel_names} with stacked np_arrays output {stacked.shape}")
    
    return xarray.DataArray(
        stacked, dims=["time", "history", "channel", "lat", "lon"], coords=coords
    )


def run_inference(
    model: TimeLoop,
    config: EnsembleRun,
    perturb: Any = None,
    group: Any = None,
    nc_file_path=None,
    progress: bool = True,
    # TODO add type hints
    data_source: Any = None,
):
    """Run an ensemble inference for a given config and a perturb function

    Args:
        group: the torch distributed group to use for the calculation
        progress: if True use tqdm to show a progress bar
        data_source: a Mapping object indexed by datetime and returning an
            xarray.Dataset object.
    """
    if not perturb:
        perturb = get_initializer(model, config)

    if not group and torch.distributed.is_initialized():
        group = torch.distributed.group.WORLD

    weather_event = config.get_weather_event()


    if not data_source:
        logger.warning(f">> inside data source model.in_channel_names: {model.in_channel_names} ")
        data_source = initial_conditions.get_data_source(
            model.in_channel_names,
            initial_condition_source=weather_event.properties.initial_condition_source,
            netcdf=weather_event.properties.netcdf,
        )
        logger.warning(f" >> data_source_cds: {type(data_source)}")

    date_obj = weather_event.properties.start_time

    def datetime_to_netcdf_time(start_time, base_time):
        base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S.%f")
        diff = start_time - base_time
        return diff.total_seconds() / 3600 # Convert to hours
        
    
   

    def index_netcdf_in_chunks(file_path, start_time, k, delta_t=timedelta(hours=6), chunk_size=1000):
        # Open the NetCDF file
        with Dataset(file_path) as nc_file:
            # Get the time variable
            time_var = nc_file.variables['time']
            time_list = time_var[:].tolist()
            # Convert the time list to a list of datetime objects
            time_list = [datetime(1900, 1, 1) + timedelta(hours=t) for t in time_list]
            
            logger.warning(f" time_var {time_var.shape} time_list {len(time_list)}  ")
            # Find the index of the start time
            
            start_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - start_time))
            # Calculate the end index
            end_index = min(start_index + k, len(time_list))
            logger.warning(f"  {start_index}   {end_index} ")
            
            # Calculate the number of chunks needed
            num_chunks = int(math.ceil((end_index - start_index) / chunk_size))
            # Initialize empty lists to store the time and variable data
            time_data = []
            var_data = []
            # Loop through the chunks
            
            for i in range(num_chunks):
                # Calculate the indices for the current chunk
                chunk_start = start_index + i * chunk_size
                chunk_end = min(start_index + (i + 1) * chunk_size, end_index)
                # Slice the time and variable data for the current chunk
                time_chunk = time_var[chunk_start:chunk_end]
                var_chunk = nc_file.variables['z'][chunk_start:chunk_end]
                # Append the chunk data to the lists
                time_data.append(time_chunk)
                var_data.append(var_chunk)
            # Concatenate the chunk data into arrays
            time_data = np.asarray(time_data, dtype=np.float32)
            var_data = np.asarray(var_data, dtype=np.float32)
            logger.warning(f" time_data {time_data.shape}  var_data {var_data.shape}")
            # Return the sliced time and variable data
            return time_data, var_data


    start_time = datetime(2020, 1, 1, 0, 0, 0)
    end_time = datetime(2023, 9, 1, 23, 59, 59)
    username = "gupt1075"
    tmp_path =  f"/scratch/gilbreth/{username}/fcnv2/cds_files_batch/"
    original_dir_path = f"{tmp_path}" + f"NETCDF_{start_time.strftime('%Y-%m-%d')}_to_{end_time.strftime('%Y-%m-%d')}_" + "ERA5-pl-z500.25.nc" 

    num_steps_frames= 7
    
    time_slice, var_slice= index_netcdf_in_chunks(original_dir_path , start_time, num_steps_frames)
    # Open the NetCDF4 file
    # nc_file = netCDF4.Dataset( original_dir_path , 'r')
    # x_shape torch.Size([1, 1, 73, 721, 1440]) 
    # Get the dimensions of the data variable
    logger.warning(f" >> FINAL_X_batch {var_slice.shape} ")
    input_frames =  torch.from_numpy(var_slice).transpose(1, 0).to(model.device)
    logger.warning(f" loading CDS files and calling intiial_conditiosn from inside inference_ensemble.py date_obj {date_obj}, initial_conditions: {input_frames.shape}  >>  {type(input_frames)}")

    dist = DistributedManager()
    n_ensemble_global = config.ensemble_members
    n_ensemble = n_ensemble_global // dist.world_size
    if n_ensemble == 0:
        logger.warning("World size is larger than global number of ensembles.")
        n_ensemble = n_ensemble_global

    # Set random seed
    seed = config.seed
    torch.manual_seed(seed + dist.rank)
    np.random.seed(seed + dist.rank)

    if config.output_dir:
        date_str = "{:%Y_%m_%d_%H_%M_%S}".format(date_obj)
        name = weather_event.properties.name
        output_path = (
            f"{config.output_dir}/"
            f"Output.{config.weather_model}."
            f"{name}.{date_str}"
        )
    else:
        output_path = config.output_path

    if not os.path.exists(output_path):
        # Avoid race condition across ranks
        os.makedirs(output_path, exist_ok=True)

    if dist.rank == 0:
        # Only rank 0 copies config files over
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            f.write(config.json())

    group_rank = torch.distributed.get_group_rank(group, dist.rank)



    for idx, frame in enumerate(input_frames):
        x = input_frames[idx:idx+1,].unsqueeze(0)
        output_file_path = os.path.join( output_path, f"{start_time.strftime('%d_%B_%Y')}__timedelta_{idx}__" + nc_file_path)
        logger.warning(f"idx {idx}  x {x.shape}    output_file_path {output_file_path} ")
        with DS(output_file_path, "w", format="NETCDF4") as nc:
            # assign global attributes
            nc.model = config.weather_model
            nc.config = config.json()
            nc.weather_event = weather_event.json()
            nc.date_created = datetime.now().isoformat()
            nc.history = " ".join(sys.argv)
            nc.institution = "Purdue"
            nc.Conventions = "CF-1.10"

            run_ensembles(
                weather_event=weather_event,
                model=model,
                perturb=perturb,
                nc=nc,
                domains=weather_event.domains,
                x=x,
                n_ensemble=n_ensemble,
                n_steps=config.simulation_length,
                output_frequency=config.output_frequency,
                batch_size=config.ensemble_batch_size,
                rank=dist.rank,
                date_obj=date_obj,
                restart_frequency=config.restart_frequency,
                output_path=output_path,
                output_grid=(
                    earth2mip.grid.from_enum(config.output_grid)
                    if config.output_grid
                    else None
                ),
                progress=progress,
            )
    if torch.distributed.is_initialized():
        torch.distributed.barrier(group)

    logger.info(f"Ensemble forecast finished, saved to: {output_file_path}")
    return None

if __name__ == "__main__":
    main()
