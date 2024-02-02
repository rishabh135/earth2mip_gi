import importlib.util
import json
import logging
import os
import sys
from datetime import datetime

import dotenv
import xarray

# (u'Cleveland, OH, US', (41.4994954, -81.6954088))


# spec = importlib.util.spec_from_file_location(
#     "earth2mip", "/scratch/gilbreth/gupt1075/fcnv2/earth2mip"
# )
# foo = importlib.util.module_from_spec(spec)
# sys.modules["earth2mip"] = foo
# spec.loader.exec_module(foo)

sys.path.append("/scratch/gilbreth/gupt1075/fcnv2/earth2mip")

# Get the current date and time
now = datetime.now()
# Format the date to get the day and month
day_month = now.strftime("%B_%d_")

os.makedirs("/scratch/gilbreth/gupt1075/fcnv2/logs/", exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=f"/scratch/gilbreth/gupt1075/fcnv2/logs/inference_{day_month}.log",
)


dotenv.load_dotenv()

# With the enviroment variables set now we import Earth-2 MIP
from earth2mip import inference_ensemble, registry
from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load

logging.warning("Fetching model package...")


package = registry.get_model("fcnv2")


logging.warning("loading FCNv2 small model, this can take a bit")
# sfno_inference_model = fcnv2_sm_load(package)
cds_api = os.path.join("/scratch/gilbreth/gupt1075/fcnv2/earth2mip/", ".cdsapirc")
logging.warning(f" right now in {os.getcwd()} and creating .cdsapirc in  ")

if not os.path.exists(cds_api):
    # uid = input("Enter in CDS UID (e.g. 123456): ")

    # key = input("Enter your CDS API key (e.g. 12345678-1234-1234-1234-123456123456): ")
    # Write to config file for CDS library
    with open(cds_api, "w") as f:
        f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
        f.write(f"key: {uid}:{key}\n")


config = {
    "ensemble_members": 4,
    "noise_amplitude": 0.05,
    "simulation_length": 10,
    "weather_event": {
        "properties": {
            "name": "Globe",
            "start_time": "2019-06-01 00:00:00",
            "initial_condition_source": "cds",
        },
        "domains": [
            {
                "name": "global",
                "type": "Window",
                "diagnostics": [{"type": "raw", "channels": ["t2m", "u10m"]}],
            }
        ],
    },
    "output_path": "/scratch/gilbreth/gupt1075/fcnv2/output/01_ensemble_notebook",
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


config_str = json.dumps(config)
inference_ensemble.main(config_str)
logging.warning(f" all the configuration as sent to inference_ensemble {config_str} ")


def open_ensemble(f, domain, chunks={"time": 1}):
    time = xarray.open_dataset(f).time
    root = xarray.open_dataset(f, decode_times=False)
    ds = xarray.open_dataset(f, chunks=chunks, group=domain)
    ds.attrs = root.attrs
    return ds.assign_coords(time=time)


output_path = config["output_path"]
domains = config["weather_event"]["domains"][0]["name"]
logging.warning(f"Saving ensembled output as a nc file with domains: {domains}")
ensemble_members = config["ensemble_members"]
ds = open_ensemble(os.path.join(output_path, "ensemble_out_0.nc"), domains)

logging.warning(f" >>>  ds.shape {ds}   \n ds keys : {ds.keys()} ")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

countries = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_0_countries",
    scale="50m",
    facecolor="none",
    edgecolor="black",
)


plt.close("all")
lead_time = np.array(
    (pd.to_datetime(ds.time) - pd.to_datetime(ds.time)[0]).total_seconds() / 3600
)
nyc_lat = 40
nyc_lon = 360 - 74
NYC = ds.sel(lon=nyc_lon, lat=nyc_lat)
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(311)
ax.set_title("Ensemble members")
ax.plot(lead_time, NYC.u10m.T)
ax.set_ylabel("u10m [m/s]")

ax = fig.add_subplot(312)
ax.set_title("deviation from ensemble mean")
ax.plot(lead_time, NYC.t2m.T - NYC.t2m.mean("ensemble"))
ax.set_ylabel("u10m [m/s]")

ax = fig.add_subplot(313)
ax.set_title("ensemble spread")
ax.plot(lead_time, NYC.t2m.std("ensemble"))
ax.set_xlabel("lead_time [h]")
ax.set_ylabel("std u10m [m/s]")
plt.tight_layout()
plt.savefig(f"{output_path}/new_york_zonal_winds.png")

# %%
# Next, lets plot some fields of surface temperature. Since we have an ensemble of
# predictions, lets display the first ensemble member, which is deterministic member,
# and also the last ensemble member and the ensemmble standard deviation. One or both of
# the perturbed members may look a little noisy, thats because our noise amplitude is
# maybe too high. Try lowering the amplitude in the config or changing pertibation type
# to see what happens.

# %%
plt.close("all")
fig = plt.figure(figsize=(15, 10))
plt.rcParams["figure.dpi"] = 100
proj = ccrs.LambertConformal(central_longitude=nyc_lon, central_latitude=nyc_lat)

data = ds.t2m[0, -1, :, :]
norm = TwoSlopeNorm(vmin=220, vcenter=290, vmax=320)
ax = fig.add_subplot(131, projection=proj)
ax.set_title("First ensemble member t2m (K)")
img = ax.pcolormesh(
    ds.lon, ds.lat, data, transform=ccrs.PlateCarree(), norm=norm, cmap="seismic"
)
ax.coastlines(linewidth=1)
ax.add_feature(countries, edgecolor="black", linewidth=0.25)
plt.colorbar(img, ax=ax, shrink=0.40, norm=mcolors.CenteredNorm(vcenter=0))
gl = ax.gridlines(draw_labels=True, linestyle="--")

data = ds.t2m[-1, -1, :, :]
norm = TwoSlopeNorm(vmin=220, vcenter=290, vmax=320)
ax = fig.add_subplot(132, projection=proj)
plt.rcParams["figure.dpi"] = 100
proj = ccrs.LambertConformal(central_longitude=nyc_lon, central_latitude=nyc_lat)
ax.set_title("Last ensemble member t2m (K)")
img = ax.pcolormesh(
    ds.lon, ds.lat, data, transform=ccrs.PlateCarree(), norm=norm, cmap="seismic"
)
ax.coastlines(linewidth=1)
ax.add_feature(countries, edgecolor="black", linewidth=0.25)
plt.colorbar(img, ax=ax, shrink=0.40, norm=mcolors.CenteredNorm(vcenter=0))
gl = ax.gridlines(draw_labels=True, linestyle="--")

ds_ensemble_std = ds.std(dim="ensemble")
data = ds_ensemble_std.t2m[-1, :, :]
# norm = TwoSlopeNorm(vmin=data.min().values, vcenter=5, vmax=data.max().values)
proj = ccrs.LambertConformal(central_longitude=nyc_lon, central_latitude=nyc_lat)
ax = fig.add_subplot(133, projection=proj)
ax.set_title("ensemble std  t2m (K)")
img = ax.pcolormesh(ds.lon, ds.lat, data, transform=ccrs.PlateCarree(), cmap="seismic")
ax.coastlines(linewidth=1)
ax.add_feature(countries, edgecolor="black", linewidth=0.25)
plt.colorbar(img, ax=ax, shrink=0.40, norm=mcolors.CenteredNorm(vcenter=0))
gl = ax.gridlines(draw_labels=True, linestyle="--")
plt.savefig(f"{output_path}/gloabl_surface_temp_contour.png")

# %%
# We can also show a map of the ensemble mean of the 10 meter zonal winds (using some
# Nvidia style coloring!)

# %%


def Nvidia_cmap():
    colors = ["#8946ff", "#ffffff", "#00ff00"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    return cmap


plt.close("all")
ds_ensemble_mean = ds.mean(dim="ensemble")
data = ds_ensemble_mean.u10m[-1, :, :]
fig = plt.figure(figsize=(9, 6))
plt.rcParams["figure.dpi"] = 100
proj = ccrs.LambertConformal(central_longitude=nyc_lon, central_latitude=nyc_lat)
#  lambert conformal perspective, polar perspective
ax = fig.add_subplot(111, projection=proj)
ax.set_title("ens. mean 10 meter zonal wind [m/s]")
img = ax.pcolormesh(
    ds.lon,
    ds.lat,
    data,
    transform=ccrs.PlateCarree(),
    cmap=Nvidia_cmap(),
    vmin=-20,
    vmax=20,
)
ax.coastlines(linewidth=1)
ax.add_feature(countries, edgecolor="black", linewidth=0.25)
plt.colorbar(img, ax=ax, shrink=0.40, norm=mcolors.CenteredNorm(vcenter=0))
gl = ax.gridlines(draw_labels=True, linestyle="--")
plt.savefig(f"{output_path}/gloabl_mean_zonal_wind_contour.png")


def global_average(ds):
    cos_lat = np.cos(np.deg2rad(ds.lat))
    return ds.weighted(cos_lat).mean(["lat", "lon"])


ds_ensemble_std = global_average(ds.std(dim="ensemble"))
plt.close("all")
plt.figure()
plt.plot(lead_time, ds_ensemble_std.u10m)
plt.xlabel("lead time [k]")
plt.ylabel("u10m std [m/s]")
plt.savefig(f"{output_path}/gloabl_std_zonal_surface_wind.png")
