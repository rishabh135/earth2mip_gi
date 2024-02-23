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
}, f"{output_path}" + f"_{start_time.strftime('%Y-%m-%d')}_to_{end_time.strftime('%Y-%m-%d')}_" + "/ERA5-pl-z500.25.nc")        # Output filename






# module load netcdf
# $ ncdump -c ERA5-pl-z500.25.nc
# netcdf ERA5-pl-z500.25 {
# dimensions:
# 	longitude = 1440 ;
# 	latitude = 721 ;
# 	time = 16 ;
# variables:
# 	float longitude(longitude) ;
# 		longitude:units = "degrees_east" ;
# 		longitude:long_name = "longitude" ;
# 	float latitude(latitude) ;
# 		latitude:units = "degrees_north" ;
# 		latitude:long_name = "latitude" ;
# 	int time(time) ;
# 		time:units = "hours since 1900-01-01 00:00:00.0" ;
# 		time:long_name = "time" ;
# 		time:calendar = "gregorian" ;
# 	short z(time, latitude, longitude) ;
# 		z:scale_factor = 0.167655930504479 ;
# 		z:add_offset = 52726.9220314097 ;
# 		z:_FillValue = -32767s ;
# 		z:missing_value = -32767s ;
# 		z:units = "m**2 s**-2" ;
# 		z:long_name = "Geopotential" ;
# 		z:standard_name = "geopotential" ;