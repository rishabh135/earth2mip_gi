import cdsapi



import torch
import numpy as np
import h5py
import os, sys, re
import logging
from datetime import datetime






# Get the current date and time
now = datetime.now()
# Format the date to get the day and month
day_month = now.strftime("%B_%d_")
username = "gupt1075"
os.makedirs(f"/scratch/gilbreth/{username}/fcnv2/logs/", exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=f"/scratch/gilbreth/{username}/fcnv2/logs/z_means_{day_month}.log",
)



c = cdsapi.Client()

CHANNELS = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "r50",
    "r100",
    "r150",
    "r200",
    "r250",
    "r300",
    "r400",
    "r500",
    "r600",
    "r700",
    "r850",
    "r925",
    "r1000",
]

global_path = "/scratch/gilbreth/gupt1075/fcnv2/cds_files_batch/73_channels"

for year in range(2019, 2022):
    for month in range(1, 13):
        for day in range(1, 32):
            for time in ["00:00", "06:00", "12:00", "18:00"]:
                try:
                    c.retrieve(
                        'reanalysis-era5-pressure-levels',
                        {
                            'product_type': 'reanalysis',
                            'format': 'netcdf',
                            'variable': CHANNELS,
                            'pressure_level': [
                                '50', '100', '150', '200', '250', '300',
                                '400', '500', '600', '700', '850', '925', '1000',
                            ],
                            'year': str(year),
                            'month': '{:02d}'.format(month),
                            'day': '{:02d}'.format(day),
                            'time': time,
                        },
                        f'{global_path}/era5_{year}_month_{month:02d}_day_{day:02d}_{time.replace(":", "")}.nc'
                    )
                    print(f'Downloaded: {global_path}/era5_{year}_{month:02d}_{day:02d}_{time.replace(":", "")}.nc')
                except Exception as e:
                    print(f'Error downloading: era5_{year}_{month:02d}_{day:02d}_{time.replace(":", "")}.nc')
                    print(e)
