""" Download and Process the COBE SST dataset

Monthly Sea-Surface Temperature from 1981-2010

Raw data is in NetCDF format. It consists of

This script converts the data to a pandas DataFrame + csv, aggregating to
  nearest 10 degree latitude and 10 degree longitude.
  Therefore, the new data set size is:

Data Source:
    https://www.esrl.noaa.gov/psd/data/gridded/data.cobe.html#temp
    COBE SST data provided by the NOAA/OAR/ESRL PSD, Boulder, Colorado, USA,
    from their Web site at http://www.esrl.noaa.gov/psd/

"""
import urllib
import os
import sys
import netCDF4
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# SETTINGS
PATH_TO_DATA_FOLDER = "data"
PATH_TO_DATA = os.path.join(PATH_TO_DATA_FOLDER, "sst.mon.mean.nc")
PATH_TO_OUTPUT = os.path.join(PATH_TO_DATA_FOLDER, "coarse_sst.csv")
LON_STEP_SIZE = 10
LAT_STEP_SIZE = 10
DROP_MISSING = True
SUBSET = False

def main():
    print("Checking if data is in " + PATH_TO_DATA)
    # Download data to "./data/" folder if not already
    if not os.path.isdir(PATH_TO_DATA_FOLDER):
        os.makedirs(PATH_TO_DATA_FOLDER)
    if not os.path.isfile(PATH_TO_DATA):
        print("Downloading Data... This may take a while")
        urllib.request.urlretrieve(
            url = "ftp://ftp.cdc.noaa.gov/Datasets/COBE/sst.mon.mean.nc",
            filename = PATH_TO_DATA,
            reporthook = reporthook,
            )

    # Import using netCDF4
    sst_nc = netCDF4.Dataset(PATH_TO_DATA)

    lon = sst_nc.variables['lon'][:]
    lat = sst_nc.variables['lat'][:]
    original_lon_lat = np.vstack([a.flat for a in np.meshgrid(lon, lat)]).T

    coarse_lon = lon[LON_STEP_SIZE//2::LON_STEP_SIZE]
    coarse_lat = lat[LAT_STEP_SIZE//2::LAT_STEP_SIZE]
    coarse_lon_lat = np.vstack([a.flat for a in
                                np.meshgrid(coarse_lon, coarse_lat)]).T

    days_since_1891 = sst_nc.variables['time'][:]
    dates = pd.to_datetime("1891-1-1") + \
            pd.to_timedelta(days_since_1891, unit="d")
    if SUBSET:
        dates = dates[0:12]

    # Coarsen data for each month using scipy.interpolate.griddata
    coarse_sst_df = pd.DataFrame()
    print("Coarsening SST Data by Month")
    for index, date in enumerate(dates):
        print("Month {0} of {1}: {2}".format(index, dates.size, date))
        sst_t = sst_nc.variables['sst'][index,:].ravel()
        sst_t.data[sst_t.mask] = np.nan
        coarse_sst_t = griddata(original_lon_lat, sst_t, coarse_lon_lat)

        np_data = np.stack(
            [coarse_sst_t, coarse_lon_lat[:,0], coarse_lon_lat[:,1]]).T
        df = pd.DataFrame(data = np_data, columns = ['sst', 'lon', 'lat'])
        if DROP_MISSING:
            df = df.dropna()
        df = df.round({'sst': 4})
        df['date'] = date
        coarse_sst_df = coarse_sst_df.append(df, ignore_index=True)

    # Save using Pandas
    print("Saving to CSV")
    coarse_sst_df.to_csv(PATH_TO_OUTPUT, index = False)
    return

## UTILS
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

# MAIN SCRIPT
if __name__ == "__main__":
    main()
    print("... Done")

