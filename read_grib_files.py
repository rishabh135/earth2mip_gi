
import pygrib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
import numpy as np

import importlib.util
import json
import logging
import os
import sys
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
    filename=f"/scratch/gilbreth/{username}/fcnv2/logs/400_frames_{day_month}_grib.log",
)

 
plt.figure(figsize=(12,8))

# Open the GRIB file
grib = "/scratch/gilbreth/gupt1075/fcnv2/cds_ics/01_January_2020/reanalysis-era5-pressure-levels_f3f31473c2d9c030f97313b7aa93ea229a715388abb0c1fa6bdb0f66d63ba48a.grib"
output_path = "/scratch/gilbreth/gupt1075/fcnv2/cds_ics/01_January_2020/reanalysis-era5-pressure-levels_f3"
grbs = pygrib.open(grib)



# # Get the number of messages in the file
# num_messages = grbs.messages_count


# Loop through all the messages and print the shapes of the variables
for grb in grbs:
    # Get the number of fields in the message
    num_fields = len(grb.values)
    
    # Loop through all the fields and print the shapes of the variables
    for j in range(num_fields):
        # Get the shape of the variable
        shape = grb.values[j].shape
        
        # Print the shape of the variable
        logging.warning(f'Field {j} shape: {shape}')
 
 
 
 
 
 
 
grb = grbs.select()[0]
data = grb.values
 

 
logging.warning(f"")
 
# need to shift data grid longitudes from (0..360) to (-180..180)
lons = np.linspace(float(grb['longitudeOfFirstGridPointInDegrees']), \
float(grb['longitudeOfLastGridPointInDegrees']), int(grb['Ni']) )
lats = np.linspace(float(grb['latitudeOfFirstGridPointInDegrees']), \
float(grb['latitudeOfLastGridPointInDegrees']), int(grb['Nj']) )
data, lons = shiftgrid(180., data, lons, start=False)
grid_lon, grid_lat = np.meshgrid(lons, lats) #regularly spaced 2D grid
 
m = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180.,llcrnrlat=lats.min(),urcrnrlat=lats.max(), resolution='c')
 
x, y = m(grid_lon, grid_lat)
 
cs = m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.gist_stern_r)
 
m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
 
plt.colorbar(cs,orientation='vertical', shrink=0.5)
plt.title('CAMS AOD forecast') # Set the name of the variable to plot
plt.savefig(f'{output_path}_plotting.png') # Set the output file name