#!/usr/bin/env python
import cdsapi
from datetime import datetime, timedelta
import logging, sys, os
c = cdsapi.Client()



# Set the start time, end time, and timedelta
start_time = datetime(2020, 3, 1, 0, 0, 0)
end_time = datetime(2023, 9, 1, 23, 59, 59)
timedelta = timedelta(hours=6)
current_time = start_time
timestamps = []
while(current_time <= end_time):
    timestamps.append(current_time)
    current_time += timedelta
    

# Create an empty list to store
# Get the current date and time
now = datetime.now()
# Format the date to get the day and month
day_month = now.strftime("%B_%d_")
username = "gupt1075"
output_path =  f"/scratch/gilbreth/{username}/fcnv2/cds_files_batch/"
os.makedirs(output_path, exist_ok=True)
 
c.retrieve('reanalysis-era5-complete', {
    # Removed obsolete comments and streamlined the explanation
    'date'    :  f"{start_time.strftime('%Y-%m-%d')}/to/{end_time.strftime('%Y-%m-%d')}",  # Specify the range of dates to download
    'levelist': '500',              # pressure Levels 
    'levtype' : 'pl',                        # Model levels
    'param'   : '129',                       # Parameter code for geopotential
    'stream'  : 'oper',                      # Operational data stream
    'time'    : '00/06/12/18',               # Times of day (in shorthand notation)
    'type'    : 'an',                        # Analysis type
    'area'    : 'global',              # Geographical subarea: North/West/South/East
    'grid'    : '0.25/0.25',                   # Grid resolution: latitude/longitude
    'format'  : 'netcdf',                    # Output format, requires 'grid' to be specified
}, f"{output_path}/ERA5-pl-z500.25.nc")        # Output filename