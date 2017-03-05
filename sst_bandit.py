"""

Implementation of Bandit for Sea-Surface Temperature

"""

from bandit import Bandit
import numpy as np
import pandas as pd


class SST_Bandit(Bandit):
    """ Sea-Surface Temperature Bandit

    Args:
        sst_df (pd.DataFrame): data frame for SST Data
            Has columns:
                sst - mean sea-surface temperature
                lon - longitude
                lat - latitude
                date - month/date
            Requires lon,lat pairs to be the same across all dates
        pulls_per_month (int): number of pull for each month

    Attributes:
        num_arms - number of arms
        lon - array of arm longitudes (length num_arms)
        lat - array of arm latitudes (length num_arms)
        arm_names - list of strings: (lon, lat)
        dates - array of month dates in sst_df
        iteration - integer, number of times bandit has been pulled


    """
    def __init__(self, sst_df, pulls_per_month=1):
        self._check_sst_df(sst_df)
        self._setup_arms(sst_df)
        self.pulls_per_month = pulls_per_month
        self.iteration = 0
        return

    @property
    def num_arms(self):
        return self._num_arms

    @property
    def arm_names(self):
        return self._arm_names

    def _check_sst_df(self, sst_df):
        # Check input format of sst_df
        if not isinstance(sst_df, pd.DataFrame):
            raise TypeError("sst_df must be a pd.DataFrame")

        # Check sst_df has appropriate columns
        columns = ['sst', 'lon', 'lat', 'date']
        for col in columns:
            if col not in sst_df.columns:
                raise ValueError("sst_df is missing col: {0}".format(col))
        return

    def _setup_arms(self, sst_df):
        # Get one month's data
        sst_df['date'] = pd.to_datetime(sst_df['date'])
        first_date = sst_df['date'].iloc[0]
        first_date_df = sst_df[sst_df['date'] == first_date]

        # Number arms is Number of rows (Lon, Lat) Tuples
        self._num_arms = first_date_df.shape[0]
        self._arm_names = first_date_df.apply(
                lambda row: "({lon}, {lat})".format(
                    lon = row['lon'],
                    lat = row['lat']),
                axis = 1).tolist()

        # Define attributes
        self.lon = np.array(first_date_df['lon'].tolist())
        self.lat = np.array(first_date_df['lat'].tolist())
        self.dates = sst_df['date'].sort_values().unique()
        self.sst_df = sst_df
        return

    def _check_arm_index(self, arm_index):
        # Check format of arm_index
        if not isinstance(arm_index, int):
            raise TypeError("arm_index must be an int")
        if arm_index < 0 or arm_index >= self.num_arms:
            raise ValueError("Invalid arm_index: {0}".format(arm_index))
        return

    def pull_arm(self, arm_index):
        self._check_arm_index(arm_index)

        # Get current date:
        date_index = self.iteration // self.pulls_per_month
        if(date_index >= self.dates.size):
            raise ValueError("No more data")
        current_date = self.dates[date_index]

        # Get arm lon, lat
        arm_lon = self.lon[arm_index]
        arm_lat = self.lat[arm_index]

        # Return SST for arm_index at current date
        df = self.sst_df[self.sst_df['date'] == current_date]
        sst = df[(df['lon'] == arm_lon) & (df['lat'] == arm_lat)]['sst'].iloc[0]

        # Increment iteration
        self.iteration += 1

        return sst

if __name__ == "__main__":
    sst_df = pd.read_csv("./data/coarse_sst.csv")
    my_bandit = SST_Bandit(sst_df, 1)
    def to_profile():
        for _ in range(100):
            print(my_bandit.pull_arm(0))










