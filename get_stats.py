

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


years = [1979, 1989, 1999, 2004, 2010]

z_means = np.zeros((1,1,1,1))
z_stds = np.zeros((1,1,1,1))
# time_means = np.zeros((1,21,721, 1440))

for ii, year in enumerate(years):
    with h5py.File('/scratch/gilbreth/wwtung/TEMP_FourCastNet/data/train/'+ str(year) + '.h5', 'r') as f:

        keys = list(f.keys())
        logging.warning(f" keys: {keys}   shape: {f['fields'][:, 0:1].shape} ")
        rnd_idx = np.random.randint(0, 1460-500)
        z_means += np.mean(f['fields'][rnd_idx:rnd_idx+500, 0:1 ], keepdims=True, axis = (0,2,3))
        z_stds += np.var(f['fields'][rnd_idx:rnd_idx+500, 0:1], keepdims=True, axis = (0,2,3))

global_means = z_means/len(years)
global_stds = np.sqrt(z_stds/len(years))
# time_means = time_means/len(years)

np.save('/scratch/gilbreth/gupt1075/fcnv2/earth2mip/fcnv2_sm/z_means.npy', z_means)
np.save('/scratch/gilbreth/gupt1075/fcnv2/earth2mip/fcnv2_sm/z_stds.npy', z_stds)
# np.save('/pscratch/sd/s/shas1693/data/era5/time_means.npy', time_means)

print("means: ", z_means)
print("stds: ", z_stds)







