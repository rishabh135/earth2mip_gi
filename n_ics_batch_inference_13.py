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
    filename=f"/scratch/gilbreth/{username}/fcnv2/logs/{day_month}_iterative_saved_ics.log",  # Set the log file path
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





    
    
    
    # acc_mean = np.mean(acc_numpy_arr, axis=0)
    # logging.warning(f" >>> MU  {mu1.shape} {sigma1.shape} ")
    # plot it!
    # fig, ax = plt.subplots(1)
    
    
    # Compute the standard error of the mean (sem) at each time point
    # sem_vals = sem(acc_numpy_arr, axis=0)
    
    # Plot the 95% confidence interval for the mean values
    # plt.fill_between(range(0, total_hours, default_timedelta), acc_mean - 1.96*sem_vals, acc_mean + 1.96*sem_vals, alpha=0.2, label='95% CI')
      
    # ax.fill_between(  acc_mean, mu1+sigma1, mu1-sigma1, alpha=0.2)






# from metrics import (
#     unweighted_acc_torch_channels,
#     weighted_acc_masked_torch_channels,
#     weighted_acc_torch_channels,
#     weighted_rmse_torch_channels,
# )









    


def plt_acc(acc_numpy_arr, st_time, fld="z500", default_timedelta=6, start_year=2020):
    # acc_numpy_arr shape (13,18)
    # Compute the total number of hours based on the array shape and default timedelta
    means = np.mean(acc_numpy_arr, axis=0)

    # Compute the total number of hours based on the array shape and default timedelta
    total_hours = acc_numpy_arr.shape[1] * default_timedelta

    # Plot the mean values
    plt.plot(range(0, total_hours, default_timedelta), means, label=f'Mean of {fld} across {total_hours} hours starting_at_{st_time }')
    
    # Compute the standard error of the mean (sem) at each time point
    sem_vals = sem(acc_numpy_arr, axis=0)
    
    # Plot the 95% confidence interval for the mean values
    plt.fill_between(range(0, total_hours, default_timedelta), means - 1.96*sem_vals, means + 1.96*sem_vals, alpha=0.2, label='95% CI')
    
    plt.title(f"Acc plot for all {total_hours} hours  ")

    plt.xlabel(f'num of hours starting from {st_time}')
    plt.ylabel('Anomaly Correlation Coefficient (ACC)  value')
    # Add a legend to the plot
    # plt.legend()
    # ax.grid()
    plt.savefig(f"{output_path}/{now_time_fully_formatted}_ACC_plot_z500_starting_at_{st_time}_with_simulation_length_{total_hours/default_timedelta}frames.png")
    return
    
    
n_ics = 13
n_ics_start_time = datetime(2020, 1, 2, 0, 0, 0)
time_list = [n_ics_start_time + timedelta(hours=6*(i+1)) for i  in range(n_ics)]
logging.warning(f" >>> time_list {time_list} ") 
output_acc_list = []
for idx, ctime in  enumerate(time_list):
    start_time = ctime.strftime("%Y-%m-%d %H:%M:%S")

    config = {
        "ensemble_members": 1,
        "noise_amplitude": 0.05,
        "simulation_length": 240,
        "n_initial_conditions": 1, 
        "weather_event": {
            "properties": {
                "name": "Globe",
                "start_time": start_time,
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
        "output_path": f"/scratch/gilbreth/{username}/fcnv2/output/n_ics_iterative_final_proposal",
        "output_frequency": 1,
        "weather_model": "fcnv2_sm",
        "seed": 12345,
        "use_cuda_graphs": False,
        "ensemble_batch_size": 1,
        "autocast_fp16": False,
        "perturbation_strategy": "correlated",
        "noise_reddening": 2.0,
    }






    output_path = config["output_path"]
    domains = config["weather_event"]["domains"][0]["name"]
    var_computed = config["weather_event"]["domains"][0]["diagnostics"][0]["channels"][0]


    nc_file_path = (f'var_{var_computed}_starting_at_{ctime.strftime("%Y_%m_%d_%H")}_.nc')


    simulation_length = config["simulation_length"]

    config_str = json.dumps(config)
    logging.warning(f" >>> start_time {start_time} and config_str {config_str} ")

    acc_numpy_arr =  batch_inf_ensemble_saved_ics.main(config_str, nc_file_path)
    output_acc_list.append(acc_numpy_arr)
    logging.warning(f" >>> ACC_NUMPY_Arr shape {acc_numpy_arr.shape} ")
    #  saving numpy file of shape (config.simulation_length, )
    

final_acc_array = np.stack(output_acc_list, axis=0)
logging.warning(f">> final_acc_array_shape {final_acc_array.shape} ")
np.save(f"{output_path}/saved_on_{now_time_fully_formatted}_starting_time__{n_ics_start_time}_with_{simulation_length}.npy", final_acc_array ) 
   
plt_acc(final_acc_array, st_time=n_ics_start_time, fld="z500")









# def plot_ci_seaborn(data):
#     np.save(f"{output_path}/numpy_file_{start_time}_with_{simulation_length}_.npy", acc_numpy_arr)
#     # Calculate the 95th percentile confidence interval for each frame
#     ci = np.percentile(data, 95, axis=1)
#     lower, upper = ci[:, np.newaxis], ci[:, np.newaxis]
#     # Calculate the mean for each frame
#     mean = np.mean(data, axis=1)[:, np.newaxis]
#     # Create a line plot of the mean values
#     sns.lineplot(x=np.arange(mean.shape[0]), y=mean, label='Mean')
#     # Shade the area between the lower and upper confidence intervals
#     plt.fill_between(np.arange(mean.shape[0]), lower, upper, alpha=0.2, label='95% CI')
#     plt.legend()
#     plt.savefig(f"{output_path}/ACC_seaborn_plot_z500_{start_time}_with_dates_.png")
    




# def plot_time_series(arr, filepath, fld="z500", default_timedelta=6, start_year=2018):
#     # Compute the mean across the rows of the time series at each of the total_hours time points
#     means = np.mean(arr, axis=0)

#     # Compute the total number of hours based on the array shape and default timedelta
#     total_hours = arr.shape[1] * default_timedelta

#     # Plot the mean values
#     plt.plot(range(0, total_hours, default_timedelta), means, label=f'Mean of {fld} across {total_hours} hours for start_year {start_year}')
    
#     # Compute the standard error of the mean (sem) at each time point
#     sem_vals = sem(arr, axis=0)
    
#     # Plot the 95% confidence interval for the mean values
#     plt.fill_between(range(0, total_hours, default_timedelta), means - 1.96*sem_vals, means + 1.96*sem_vals, alpha=0.2, label='95% CI')
    
#     # Set the x-axis label with the start time
#     plt.xlabel(f'Number of hours starting from {start_year}')
    
#     # Set the y-axis label
#     plt.ylabel('Anomaly Correlation Coefficient (ACC) value')
    
#     # Add a legend to the plot
#     plt.legend()
    
#     # # Display the plot
#     # plt.show()
    
#     # Save the plot to a file with the specified filepath and DPI
#     plt.savefig(f"{filepath}.png", dpi=200)
#     return

