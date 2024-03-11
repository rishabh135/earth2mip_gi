import netCDF4 as nc


import argparse
import json
import logging
import os, re
import sys
from datetime import datetime, timedelta
from typing import Any, Optional
import pandas as pd

from torchinfo import summary
import cftime
import numpy as np
import torch
torch.cuda.empty_cache()
import tqdm
import xarray
from modulus.distributed.manager import DistributedManager
from netCDF4 import Dataset as DS
import math 
import earth2mip.grid

from earth2mip.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch, unlog_tp_torch



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


import dotenv
import xarray
from geopy import geocoders

username = "gupt1075"

#  added for reading custom earth2mip codebase
sys.path.append(f"/scratch/gilbreth/{username}/fcnv2/earth2mip")



# Get the current date and time
now = datetime.now()

# Format the date to get the day and month
day_month = now.strftime("%B_%d_")

# Format the current date and time with a detailed format
now_time_fully_formatted = now.strftime("%B_%d_%Y_%H_%M_%S")

# Create the logs directory if it doesn't exist
os.makedirs(f"/scratch/gilbreth/{username}/fcnv2/logs/", exist_ok=True)
# Configure the logging settings
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",  # Set the log message format
    datefmt="%m/%d/%Y %H:%M:%S",  # Set the date format for log timestamps
    level=logging.INFO,  # Set the logging level to INFO
    filename=f"/scratch/gilbreth/{username}/fcnv2/logs/{day_month}_parse_netcdf.log",  # Set the log file path
)



dotenv.load_dotenv()

# With the enviroment variables set now we import Earth-2 MIP
from earth2mip import batch_inference_ensemble, registry
from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load

from earth2mip.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch, unlog_tp_torch



logger = logging.getLogger("ACC_calculator")





def index_netcdf_in_chunks(file_path, start_time, k, delta_t=timedelta(hours=6), chunk_size=1000):
    # Open the NetCDF file
    with DS(file_path) as nc_file:
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
        logging.warning(f"  {start_index}   {end_index} ")
        
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





file_name_start = datetime(2020, 1, 1, 0, 0, 0)
file_name_end = datetime(2023, 9, 1, 23, 59, 59)
username = "gupt1075"
tmp_path =  f"/scratch/gilbreth/{username}/fcnv2/cds_files_batch/"
original_dir_path = f"{tmp_path}" + f"NETCDF_{file_name_start.strftime('%Y-%m-%d')}_to_{file_name_end.strftime('%Y-%m-%d')}_" + "ERA5-pl-z500.25.nc" 


start_time = datetime(2020, 1, 3, 0, 0, 0)
logging.warning(f" date_obj = {start_time}")
n_initial_conditions =  12
simulation_length = 187
# num_steps_frames= number_of_frames  + 5
time_slice, original_np_array= index_netcdf_in_chunks(original_dir_path , start_time, n_initial_conditions )
    
    




def output_netcdf(file_path, start_time, k, delta_t=timedelta(hours=6), chunk_size=1000):
    with DS(file_path) as nc_file:
        # Get the time variable
        time_var = nc_file.variables['time']
        time_list = time_var[:].tolist()
        # # Convert the time list to a list of datetime objects
        time_list = [datetime(1900, 1, 1) + timedelta(hours=t) for t in time_list]
        
        logger.warning(f" Time_var: {time_var}  time_list: {time_list} ")
        
        logger.warning(f" Variables: ")
        # for var in nc_file.variables:
        #     logger.warning(f"{var}")

        # List out the shape of each variable
        logger.warning(f"\n Variable shapes: ")
        for var in nc_file.variables:
            logger.warning( f" Var_name: {var}  shape:  {nc_file.variables[var].shape} ")

    return
    # logger.warning(f" time_var {time_var.shape} time_list {len(time_list)}  ")
    # # Find the index of the start time
    
    # start_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - start_time))
    # # Calculate the end index
    # end_index = min(start_index + k, len(time_list))
    # logging.warning(f"  {start_index}   {end_index} ")
    
    # # Calculate the number of chunks needed
    # num_chunks = int(math.ceil((end_index - start_index) / chunk_size))
    # # Initialize empty lists to store the time and variable data
    # time_data = []
    # var_data = []
    # # Loop through the chunks
    
    # for i in range(num_chunks):
    #     # Calculate the indices for the current chunk
    #     chunk_start = start_index + i * chunk_size
    #     chunk_end = min(start_index + (i + 1) * chunk_size, end_index)
    #     # Slice the time and variable data for the current chunk
    #     time_chunk = time_var[chunk_start:chunk_end]
    #     var_chunk = nc_file.variables['z'][chunk_start:chunk_end]
    #     # Append the chunk data to the lists
    #     time_data.append(time_chunk)
    #     var_data.append(var_chunk)
    # # Concatenate the chunk data into arrays
    # time_data = np.asarray(time_data, dtype=np.float32)
    # var_data = np.asarray(var_data, dtype=np.float32)
    # logger.warning(f" time_data {time_data.shape}  var_data {var_data.shape}")
    # # Return the sliced time and variable data
    # return time_data, var_data



nc_files_predicted = f"/scratch/gilbreth/{username}/fcnv2/output/batch_inference/proposal"

output_netcdf(file_path, start_time, delta_t=timedelta(hours=24), chunk_size=1000):