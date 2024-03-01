import importlib.util
import json
import logging
import os,re
from tqdm import tqdm
import sys
from datetime import datetime
import numpy as np

import configparser
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
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

os.makedirs(f"/scratch/gilbreth/{username}/fcnv2/logs/", exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=f"/scratch/gilbreth/{username}/fcnv2/logs/batch_Metrics_{day_month}.log",
)


dotenv.load_dotenv()

# With the enviroment variables set now we import Earth-2 MIP
from earth2mip import batch_inference_ensemble, registry
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
    "simulation_length": 6,
    "simulated_frames" : 100,
    "weather_event": {
        "properties": {
            "name": "Globe",
            "start_time": "2020-03-01 00:00:00",
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
    "output_path": f"/scratch/gilbreth/{username}/fcnv2/output/batch_inference/z500",
    "output_frequency": 1,
    "weather_model": "fcnv2_sm",
    "seed": 12345,
    "use_cuda_graphs": False,
    "ensemble_batch_size": 1,
    "autocast_fp16": False,
    "perturbation_strategy": "correlated",
    "noise_reddening": 2.0,
}


start_time = datetime.strptime(
    config["weather_event"]["properties"]["start_time"], "%Y-%m-%d %H:%M:%S"
).strftime("%d_%B_%Y")

output_path = config["output_path"]
domains = config["weather_event"]["domains"][0]["name"]
var_computed = config["weather_event"]["domains"][0]["diagnostics"][0]["channels"][0]


nc_file_path = (
    f"var_{var_computed}_starting_at_{start_time}_ensemble_{config['simulation_length']}"
    + ".nc"
)


simulation_length = config["simulation_length"]

config_str = json.dumps(config)
acc_numpy_arr =  batch_inference_ensemble.main(config_str, nc_file_path)



logging.warning(f" all the configuration as sent to inference_ensemble {config_str} ")





def plt_acc(acc_numpy_arr):
    np.save(f"{output_path}/numpy_file_{start_time}_with_{simulation_length}_.npy", acc_numpy_arr)
    mu1 = acc_numpy_arr.mean(axis=0)
    sigma1 = acc_numpy_arr.std(axis=0)
    acc_mean = np.mean(acc_numpy_arr, axis=0)
    logging.warning(f" >>> MU  {mu1.shape} {sigma1.shape} ")
    # plot it!
    fig, ax = plt.subplots(1)
    ax.plot( [0 + i*6 for i in range(simulation_length+1)] , acc_mean, "-", lw=2, label='Anomaly Correlation Coefficient (ACC) value')
    ax.fill_between(  acc_mean, mu1+sigma1, mu1-sigma1, alpha=0.2)
    
    ax.set_title(f"Acc plot for all {simulation_length} frames ")
    ax.legend(loc='upper left')
    ax.set_xlabel(f'num of hours starting from {start_time}')
    ax.set_ylabel('Anomaly Correlation Coefficient (ACC)  value')
    # ax.grid()
    plt.savefig(f"{output_path}/ACC_plot_z500_{start_time}_with_dates_.png")




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
    



# plt_acc(acc_numpy_arr)
plot_ci_seaborn(acc_numpy_arr)


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