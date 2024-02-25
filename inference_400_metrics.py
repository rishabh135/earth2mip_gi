import importlib.util
import json
import logging
import os,re
from tqdm import tqdm
import sys
from datetime import datetime
import numpy as np

import configparser

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
    filename=f"/scratch/gilbreth/{username}/fcnv2/logs/400_Metrics_{day_month}.log",
)


dotenv.load_dotenv()

# With the enviroment variables set now we import Earth-2 MIP
from earth2mip import inference_ensemble, registry
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
    "simulation_length": 730,
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
    "output_path": f"/scratch/gilbreth/{username}/fcnv2/output/z500_2020_03_01",
    "output_frequency": 1,
    "weather_model": "fcnv2_sm",
    "seed": 12345,
    "use_cuda_graphs": False,
    "ensemble_batch_size": 1,
    "autocast_fp16": False,
    "perturbation_strategy": "correlated",
    "noise_reddening": 2.0,
}



# Option 1: Use config file and CLI (use this outside a notebook)
# with open('./01_config.json', 'w') as f:
#     json.dump(config, f)
# ! python3 -m earth2mip.inference_ensemble 01_config.json


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


config_str = json.dumps(config)
__ =  inference_ensemble.main(config_str, nc_file_path)
logging.warning(f" all the configuration as sent to inference_ensemble {config_str} ")


def open_ensemble(f, domain, chunks={"time": 1}):
    time = xarray.open_dataset(f).time
    root = xarray.open_dataset(f, decode_times=False)
    ds = xarray.open_dataset(f, chunks=chunks, group=domain)
    ds.attrs = root.attrs
    return ds.assign_coords(time=time)









logging.warning(
    f"Saving ensembled output as a nc file with domains: {domains} and var_computed {var_computed}"
)
ensemble_members = config["ensemble_members"]

predicted_ds = open_ensemble(
    os.path.join(output_path, nc_file_path),
    domains,
)



original_dir_path = "/scratch/gilbreth/gupt1075/fcnv2/cds_files_batch/ERA5-pl-z500.25.nc"



# def find_files_with_suffix(suffix, directory):
#     """
#     Finds all files in the given directory and its subdirectories that have the specified suffix.
#     """
#     # Create an empty list to store the matching files
#     matching_files = []
#     logging.warning(f" inside find files with {directory}    ")
#     for path, subdirs, files in os.walk(directory):
#         for name in files:
#             logging.warning( f" >>  " + os.path.join(path, name))

#             if name.endswith(suffix):
#                 # If it does, add the full path of the file to the list
#                 matching_files.append(os.path.join(directory, name))

#     # Return the list of matching files
#     return matching_files


logging.warning(f" predicted_ds_keys {predicted_ds.keys()} \n  >>> predicted_ds.attrs {predicted_ds.attrs}  \n\n\n ************ original_ds_keys")
predicted_data = predicted_ds.z500[-1]





import netCDF4
import numpy as np
import torch

# Open the NetCDF4 file
nc_file = netCDF4.Dataset( original_dir_path , 'r')
# Get the dimensions of the data variable
var_shape = nc_file.variables["z"].shape

logging.warning(f" >>>  {nc_file.variables}   >>  {nc_file.variables.keys() }   << {var_shape} ")
# Define the chunk size
chunk_size = 1000

# Initialize an empty list to store the chunk data
chunk_data_list = []

# Loop through the chunks and read the data
for i in range(int(np.ceil(var_shape[0] / chunk_size))):
    if(i > 1):
        break
    start_index = i * chunk_size
    end_index = min((i + 1) * chunk_size, var_shape[0])
    chunk_data = nc_file.variables['z'][start_index:end_index, :, :]
    chunk_data_list.append(chunk_data)

# Close the NetCDF4 file
nc_file.close()

# Convert the list of chunk data to a NumPy array
original_data = np.concatenate(chunk_data_list, axis=0)[:predicted_data.shape[0]]
# logging.warning(f" data_array {original_data.shape} ")




logging.warning(
    f" >>>  predicted_data {predicted_data.shape}  original_data {original_data.shape} domains {domains}")


acc_list = []
for idx in  range(predicted_data.shape[0]):
    val = original_data[idx]
    val2 = predicted_data[idx]
    tmp_original_data = np.expand_dims(val, axis=0)
    tmp_pred_data = np.expand_dims(val2, axis=0)
    # logging.warning(f" idx : {idx}  original_data : {tmp_original_data.shape}   predicted_data[idx] : {tmp_pred_data.shape}  ")
    acc_list.append(weighted_acc(tmp_pred_data, tmp_original_data, weighted = True))
        
    
acc_list = np.asarray(acc_list)

mu1 = acc_list.mean()
sigma1 = acc_list.std()

# mu2 = X2.mean(axis=1)
# sigma2 = X2.std(axis=1)


logging.warning(f" ACC values {acc_list}  mu1: {mu1}  sigma1 {sigma1}")
# plot it!
fig, ax = plt.subplots(1)
ax.plot( np.arange(0,predicted_data.shape[0]), acc_list, lw=2, label='Anomaly Correlation Coefficient (ACC)  value')

ax.fill_between(  acc_list, mu1+sigma1, mu1-sigma1, facecolor='C0', alpha=0.4)
ax.set_title(f"Acc plot for all {predicted_data.shape[0]} frames for z500 variable  starting at {start_time}")
ax.legend(loc='upper left')
ax.set_xlabel('num steps')
ax.set_ylabel('Anomaly Correlation Coefficient (ACC)  value')
# ax.grid()
plt.savefig(f"{output_path}/ACC_values_plot_z500.png")


logging.warning(f" >>>  predicted_data.shape {predicted_data.shape }  original_data.shape {original_data.shape }  \n  acc: {acc_list}   ")



# countries = cfeature.NaturalEarthFeature(
#     category="cultural",
#     name="admin_0_countries",
#     scale="50m",
#     facecolor="none",
#     edgecolor="black",
# )


# plt.close("all")
# lead_time = np.array(
#     (pd.to_datetime(ds.time) - pd.to_datetime(ds.time)[0]).total_seconds() / 3600
# )
# nyc_lat = 40
# nyc_lon = 360 - 74
# NYC = ds.sel(lon=nyc_lon, lat=nyc_lat)
# print(f" NYC shape: {NYC.z500.shape} ")
# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111)
# ax.set_title("Ensemble members of {domains}")
# logging.warning(
#     f" plotting ensemble of domains {domains} and var_computed {var_computed} "
# )
# ax.plot(lead_time, NYC.z500.T)
# ax.set_ylabel("z500 [m/s]")

# ax = fig.add_subplot(312)
# ax.set_title("deviation from ensemble mean")
# ax.plot(lead_time, NYC.t2m.T - NYC.t2m.mean("ensemble"))
# ax.set_ylabel("u10m [m/s]")

# ax = fig.add_subplot(313)
# ax.set_title("ensemble spread")
# ax.plot(lead_time, NYC.t2m.std("ensemble"))
# ax.set_xlabel("lead_time [h]")
# ax.set_ylabel("std u10m [m/s]")



# plt.tight_layout()
# plt.savefig(f"{output_path}/new_york_{var_computed}.png")

# # %%
# # Next, lets plot some fields of surface temperature. Since we have an ensemble of
# # predictions, lets display the first ensemble member, which is deterministic member,
# # and also the last ensemble member and the ensemmble standard deviation. One or both of
# # the perturbed members may look a little noisy, thats because our noise amplitude is
# # maybe too high. Try lowering the amplitude in the config or changing pertibation type
# # to see what happens.

# # %%
# plt.close("all")
# fig = plt.figure(figsize=(15, 10))
# plt.rcParams["figure.dpi"] = 100
# proj = ccrs.LambertConformal(central_longitude=nyc_lon, central_latitude=nyc_lat)

# data = ds.z500[0, -1, :, :]
# norm = TwoSlopeNorm(vmin=220, vcenter=290, vmax=320)
# ax = fig.add_subplot(131, projection=proj)
# ax.set_title("First ensemble member z500 ")
# img = ax.pcolormesh(
#     ds.lon, ds.lat, data, transform=ccrs.PlateCarree(), norm=norm, cmap="seismic"
# )
# ax.coastlines(linewidth=1)
# ax.add_feature(countries, edgecolor="black", linewidth=0.25)
# plt.colorbar(img, ax=ax, shrink=0.40, norm=mcolors.CenteredNorm(vcenter=0))
# gl = ax.gridlines(draw_labels=True, linestyle="--")

# data = ds.z500[-1, -1, :, :]
# norm = TwoSlopeNorm(vmin=220, vcenter=290, vmax=320)
# ax = fig.add_subplot(132, projection=proj)
# plt.rcParams["figure.dpi"] = 100
# proj = ccrs.LambertConformal(central_longitude=nyc_lon, central_latitude=nyc_lat)
# ax.set_title("Last ensemble member t2m (K)")
# img = ax.pcolormesh(
#     ds.lon, ds.lat, data, transform=ccrs.PlateCarree(), norm=norm, cmap="seismic"
# )
# ax.coastlines(linewidth=1)
# ax.add_feature(countries, edgecolor="black", linewidth=0.25)
# plt.colorbar(img, ax=ax, shrink=0.40, norm=mcolors.CenteredNorm(vcenter=0))
# gl = ax.gridlines(draw_labels=True, linestyle="--")

# ds_ensemble_std = ds.std(dim="ensemble")
# data = ds_ensemble_std.z500[-1, :, :]
# # norm = TwoSlopeNorm(vmin=data.min().values, vcenter=5, vmax=data.max().values)
# proj = ccrs.LambertConformal(central_longitude=nyc_lon, central_latitude=nyc_lat)
# ax = fig.add_subplot(133, projection=proj)
# ax.set_title("ensemble z500 (K)")
# img = ax.pcolormesh(ds.lon, ds.lat, data, transform=ccrs.PlateCarree(), cmap="seismic")
# ax.coastlines(linewidth=1)
# ax.add_feature(countries, edgecolor="black", linewidth=0.25)
# plt.colorbar(img, ax=ax, shrink=0.40, norm=mcolors.CenteredNorm(vcenter=0))
# gl = ax.gridlines(draw_labels=True, linestyle="--")
# plt.savefig(f"{output_path}/gloabl_z500.png")

# %%
# We can also show a map of the ensemble mean of the 10 meter zonal winds (using some
# Nvidia style coloring!)

# %%


# def Nvidia_cmap():
#     colors = ["#8946ff", "#ffffff", "#00ff00"]
#     cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
#     return cmap


# plt.close("all")
# ds_ensemble_mean = ds.mean(dim="ensemble")
# data = ds_ensemble_mean.u10m[-1, :, :]
# fig = plt.figure(figsize=(9, 6))
# plt.rcParams["figure.dpi"] = 100

# gn = geocoders.GeoNames()
# cleveland_lat, cleveland_lon = gn.geocode("Cleveland, OH 44106")[1]

# # proj = ccrs.LambertConformal(central_longitude=nyc_lon, central_latitude=nyc_lat)
# proj = ccrs.OSGSB(central_longitude=cleveland_lon, central_latitude=cleveland_lat)

# #  lambert conformal perspective, polar perspective
# ax = fig.add_subplot(111, projection=proj)
# ax.set_title("ens. mean 10 meter zonal wind [m/s]")
# img = ax.pcolormesh(
#     ds.lon,
#     ds.lat,
#     data,
#     transform=ccrs.PlateCarree(),
#     cmap=Nvidia_cmap(),
#     vmin=-20,
#     vmax=20,
# )
# ax.coastlines(linewidth=1)
# ax.add_feature(countries, edgecolor="black", linewidth=0.25)
# plt.colorbar(img, ax=ax, shrink=0.40, norm=mcolors.CenteredNorm(vcenter=0))
# gl = ax.gridlines(draw_labels=True, linestyle="--")
# plt.savefig(f"{output_path}/gloabl_mean_zonal_wind_contour.png")


# def global_average(ds):
#     cos_lat = np.cos(np.deg2rad(ds.lat))
#     return ds.weighted(cos_lat).mean(["lat", "lon"])


# ds_ensemble_std = global_average(ds.std(dim="ensemble"))
# plt.close("all")
# plt.figure()
# plt.plot(lead_time, ds_ensemble_std.u10m)
# plt.xlabel("lead time [k]")
# plt.ylabel("u10m std [m/s]")
# plt.savefig(f"{output_path}/gloabl_std_zonal_surface_wind.png")
