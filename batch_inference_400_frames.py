import importlib.util
import json
import logging
import os,re
from tqdm import tqdm
import sys
from datetime import datetime, timedelta
import numpy as np

import configparser
import seaborn as sns
from scipy.stats import sem
from glob import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

#  added for reading the correct login creds for cdsapi
configur = configparser.ConfigParser()
username = "gupt1075"	
configur.read( f"/scratch/gilbreth/{username}/fcnv2/config.ini")

import dotenv
import xarray
from geopy import geocoders

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
    filename=f"/scratch/gilbreth/{username}/fcnv2/logs/{day_month}_saved_ics.log",  # Set the log file path
)



dotenv.load_dotenv()

# With the enviroment variables set now we import Earth-2 MIP
from earth2mip import batch_inference_ensemble, registry, batch_inf_ensemble_saved_ics
from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load

from earth2mip.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch, unlog_tp_torch





logging.warning("Fetching model package...")
package = registry.get_model("fcnv2")




logging.warning("loading FCNv2 small model, this can take a bit")
# sfno_inference_model = fcnv2_sm_load(package)
cds_api = os.path.join(f"/scratch/gilbreth/{username}/fcnv2/earth2mip/", ".cdsapirc")
logging.warning(f" right now in {os.getcwd()} and creating .cdsapirc in  ")

if not os.path.exists(cds_api):
    # uid = input("Enter in CDS UID (e.g. 123456): ")

    uid = configur.get('login','uid')
    key = configur.get('login','key')
    
    # key = input("Enter your CDS API key (e.g. 12345678-1234-1234-1234-123456123456): ")
    # Write to config file for CDS library
    with open(cds_api, "w") as f:
        f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
        f.write(f"key: {uid}:{key}\n")





config = {
    "ensemble_members": 1,
    "noise_amplitude": 0.05,
    "simulation_length": 240,
    "n_initial_conditions" : 5,
    "weather_event": {
        "properties": {
            "name": "Globe",
            "start_time": "2020-02-09 00:00:00",
            "initial_condition_source": "cds",
        },
        "domains": [
            {
                "name": "global",
                "type": "Window",
                "diagnostics": [{"type": "raw", "channels": ["z500"]}],
            }
        ],
    },
    "output_path": f"/scratch/gilbreth/{username}/fcnv2/output/batch_inference/march_12_proposal",
    "output_frequency": 1,
    "weather_model": "fcnv2_sm",
    "seed": 12345,
    "use_cuda_graphs": False,
    "ensemble_batch_size": 1,
    "autocast_fp16": False,
    "perturbation_strategy": "correlated",
    "noise_reddening": 2.0,
}





# from metrics import (
#     unweighted_acc_torch_channels,
#     weighted_acc_masked_torch_channels,
#     weighted_acc_torch_channels,
#     weighted_rmse_torch_channels,
# )













start_time = datetime.strptime(
    config["weather_event"]["properties"]["start_time"], "%Y-%m-%d %H:%M:%S"
).strftime("%d_%B_%Y")

output_path = config["output_path"]
domains = config["weather_event"]["domains"][0]["name"]
var_computed = config["weather_event"]["domains"][0]["diagnostics"][0]["channels"][0]


nc_file_path = (
    f"var_{var_computed}_starting_at_{start_time}"
    + ".nc"
)


simulation_length = config["simulation_length"]
n_initial_conditions = config["n_initial_conditions"]


config_str = json.dumps(config)
# acc_numpy_arr =  batch_inference_ensemble.main(config_str, nc_file_path)



acc_numpy_arr =  batch_inf_ensemble_saved_ics.main(config_str, nc_file_path)



acc_numpy_arr = np.abs(acc_numpy_arr)
logging.warning(f" >>> ACC_NUMPY_Arr shape {acc_numpy_arr.shape} ")

#  saving numpy file of output_tensors
np.save(f"{output_path}/saved_on_{now_time_fully_formatted}_starting_time__{start_time}_with_{simulation_length}.npy", acc_numpy_arr)
    


def plt_acc(acc_numpy_arr, fld="z500", default_timedelta=6, start_year=2018):
    mu1 = acc_numpy_arr.mean(axis=0)
    
    # Compute the total number of hours based on the array shape and default timedelta
    total_hours = acc_numpy_arr.shape[1] * default_timedelta
    
    sigma1 = acc_numpy_arr.std(axis=0)
    # ci = np.percentile(acc_numpy_arr, 95, axis=0)
    
    
    r, c = acc_numpy_arr.shape
    colors = cm.rainbow(np.linspace(0, 1, r)) # generate r different colors
    for i in range(r):
        plt.plot( range(0, total_hours, default_timedelta) , acc_numpy_arr[i, :], color=colors[i]) # plot each line curve with a different color
    
    
    # acc_mean = np.mean(acc_numpy_arr, axis=0)
    # logging.warning(f" >>> MU  {mu1.shape} {sigma1.shape} ")
    # plot it!
    # fig, ax = plt.subplots(1)
    # plt.plot( range(0, total_hours, default_timedelta) , acc_mean, "-", lw=2, label='Anomaly Correlation Coefficient (ACC) value')
    
    
    # Compute the standard error of the mean (sem) at each time point
    # sem_vals = sem(acc_numpy_arr, axis=0)
    
    # Plot the 95% confidence interval for the mean values
    # plt.fill_between(range(0, total_hours, default_timedelta), acc_mean - 1.96*sem_vals, acc_mean + 1.96*sem_vals, alpha=0.2, label='95% CI')
      
    # ax.fill_between(  acc_mean, mu1+sigma1, mu1-sigma1, alpha=0.2)
    
    plt.title(f"Acc plot for all {total_hours} hours  ")

    plt.xlabel(f'num of hours starting from {start_time}')
    plt.ylabel('Anomaly Correlation Coefficient (ACC)  value')
    # Add a legend to the plot
    plt.legend()
    # ax.grid()
    plt.savefig(f"{output_path}/{now_time_fully_formatted}_ACC_plot_z500_starting_at_{start_time}_with_simulation_length_{total_hours/default_timedelta}frames.png")




def plot_ci_seaborn(data):
    np.save(f"{output_path}/numpy_file_{start_time}_with_{simulation_length}_.npy", acc_numpy_arr)
    # Calculate the 95th percentile confidence interval for each frame
    ci = np.percentile(data, 95, axis=1)
    lower, upper = ci[:, np.newaxis], ci[:, np.newaxis]
    # Calculate the mean for each frame
    mean = np.mean(data, axis=1)[:, np.newaxis]
    # Create a line plot of the mean values
    sns.lineplot(x=np.arange(mean.shape[0]), y=mean, label='Mean')
    # Shade the area between the lower and upper confidence intervals
    plt.fill_between(np.arange(mean.shape[0]), lower, upper, alpha=0.2, label='95% CI')
    plt.legend()
    plt.savefig(f"{output_path}/ACC_seaborn_plot_z500_{start_time}_with_dates_.png")
    




def plot_time_series(arr, filepath, fld="z500", default_timedelta=6, start_year=2018):
    # Compute the mean across the rows of the time series at each of the total_hours time points
    means = np.mean(arr, axis=0)

    # Compute the total number of hours based on the array shape and default timedelta
    total_hours = arr.shape[1] * default_timedelta

    # Plot the mean values
    plt.plot(range(0, total_hours, default_timedelta), means, label=f'Mean of {fld} across {total_hours} hours for start_year {start_year}')
    
    # Compute the standard error of the mean (sem) at each time point
    sem_vals = sem(arr, axis=0)
    
    # Plot the 95% confidence interval for the mean values
    plt.fill_between(range(0, total_hours, default_timedelta), means - 1.96*sem_vals, means + 1.96*sem_vals, alpha=0.2, label='95% CI')
    
    # Set the x-axis label with the start time
    plt.xlabel(f'Number of hours starting from {start_year}')
    
    # Set the y-axis label
    plt.ylabel('Anomaly Correlation Coefficient (ACC) value')
    
    # Add a legend to the plot
    plt.legend()
    
    # # Display the plot
    # plt.show()
    
    # Save the plot to a file with the specified filepath and DPI
    plt.savefig(f"{filepath}.png", dpi=200)
    return



# removing warmup steps: 
# acc_numpy_arr = acc_numpy_arr[:, 7:]

# plt_acc(acc_numpy_arr)


#plot_time_series(acc_numpy_arr, os.path.join( "/scratch/gilbreth/gupt1075/fcnv2/output/" , f"plot_acc_var_z500_with_nics_{simulation_length}"), fld="z500", default_timedelta=6, start_year=2020)


# plot_ci_seaborn(acc_numpy_arr)



# def open_ensemble(f, domain, chunks={"time": 1}):
#     time = xarray.open_dataset(f).time
#     root = xarray.open_dataset(f, decode_times=False)
#     ds = xarray.open_dataset(f, chunks=chunks, group=domain)
#     ds.attrs = root.attrs
#     return ds.assign_coords(time=time)


# logging.warning(
#     f"Saving ensembled output as a nc file with domains: {domains} and var_computed {var_computed}"
# )
# ensemble_members = config["ensemble_members"]

# ds = open_ensemble(
#     os.path.join(output_path, nc_file_path),
#     domains,
# )

# logging.warning(
#     f" >>>  ds.shape {ds}  start_time {start_time}  var_computed {var_computed}  \n ds keys : {ds.keys()} "
# )



# from earth2mip.inference_medium_range import score_deterministic

# scores = score_deterministic(time_loop,
#     data_source=data_source,
#     n=10,
#     initial_times=[datetime.datetime(2018, 1, 1)],
#     # fill in zeros for time-mean, will typically be grabbed from data.
#     time_mean=np.zeros((7, 721, 1440))
# )
# >>>