import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class RsTsLoader(Dataset):
    def __init__(self, locations, input_data, input_dates, win_size, 
                 use_indices=0, 
                 info=""):
        self.locations = locations
        self.input_data = input_data
        self.input_dates = input_dates
        self.win_size = win_size # == seq_len
        self.use_indices = use_indices
        self.info = info

        ### get the correct DOY values
        ### write list comprehension to extract the day of the year for each date in input_dates
        ### and overwrite it
        self.input_dates = np.array([date.timetuple().tm_yday for date in self.input_dates])

        ### to simplify the workflow and keep it similar to training script
        ### we compute the cumulative doy and write them on self.input_dates
        ### we can use a while loop iterating over the self.input_dates array
        ### to add 365 everytime the value of i+1 is not increasing compared to the i-th value
        i = 0
        while i < len(self.input_dates) - 1:
            if self.input_dates[i + 1] <= self.input_dates[i]:
                self.input_dates[i + 1:] += 365
            i += 1

    def __len__(self):
        return self.locations.shape[0]  # number of samples in the dataset

    def __getitem__(self, idx):

        ts = self.input_data[self.locations[idx], :, :]

        ### create a local copy of self.input_dates
        ### this is to avoid code crashing because self.input_dates gets modified across workers
        local_input_dates = np.copy(self.input_dates)

        ### remove observations where any band in ts is np.nan
        ### don't forget to remove the corresponding DOY values as well
        ### to keep them in sync with the observations
        valid_obs_arr = ~np.isnan(ts).any(axis=1) # True means valid values, False means masked values
        ts = ts[valid_obs_arr, :]
        local_input_dates = local_input_dates[valid_obs_arr]

        ### get number of observations for further processing
        ts_length = ts.shape[0]
        
        ### day of year
        doy = np.zeros((self.win_size,), dtype=int)

        ### BOA reflectances
        ts_origin = np.zeros((self.win_size, ts.shape[1]))

        ### we always use the last seq_len observations (not random sampling from sequence)
        ts_origin[:ts_length, :] = ts[:ts_length, :]

        ### assign doy values
        doy[:ts_length] = local_input_dates
        
        ### at this point, we have a prepared time series (ts_origin) and DOY values
        ### here is what we want to create from that: 
        ### a time series ts_origin of shape [self.win_size, bands] and 
        ### a time series doy of shape [self.win_size]
        ### note that self.win_size is the maximum padded sequence length for every sample, which is provided by the user
        ### instead of end padding of the whole time series, we want end padding for each of the (up to) 4 years of the time series
        ### so that the goal is a ts_origin like: [(BOA reflectances first year), 0, 0, 0, ..., (BOA reflectances second year), 0 , 0, 0, ...]
        ### where each year's values have the same length, and the DOY values are also padded accordingly
        ### the doy array should contain only values from 1 until 365 (each observation's day of the year)
        ### instead of the cumulative doy values (cumulative doy means 1 until 365 in first year, 366 until 730 in second year, etc.)
        ### the years can be separated by the doy values before adjusting them to "normal" doy values
        ### the goal is to have a time series of shape [self.win_size, bands] and a doy array of shape [self.win_size]
        
        ### define the number of days in a year and the period length
        days_in_year = 365
        period = self.win_size // 4  ### assuming period is defined as self.win_size divided by 4

        ### initialize lists to store the split data
        years = []
        doy_years = []

        ### split the original time series and DOY values into years
        for i in range(4):
            start_day = i * days_in_year + 1
            end_day = (i + 1) * days_in_year
            mask = (doy >= start_day) & (doy <= end_day)
            year_data = ts_origin[mask]
            doy_data = doy[mask] - start_day + 1  # Adjust DOY to be within 1 to 365

            ### if there are no observations for the current year, create an empty array
            if len(year_data) == 0:
                year_data = np.zeros((period, ts_origin.shape[1]), dtype=ts_origin.dtype)
                doy_data = np.zeros(period, dtype=doy.dtype)
            years.append(year_data)
            doy_years.append(doy_data)

        ### ensure that each year has the same length by padding with zeros if necessary
        for i in range(4):
            if len(years[i]) < period:
                padding_length = period - len(years[i])
                years[i] = np.pad(years[i], ((0, padding_length), (0, 0)), mode='constant', constant_values=0)
                doy_years[i] = np.pad(doy_years[i], (0, padding_length), mode='constant', constant_values=0)
            else:
                years[i] = years[i][:period]
                doy_years[i] = doy_years[i][:period]

        ### convert lists to numpy arrays
        years = np.array(years)
        doy_years = np.array(doy_years)

        ### now we have two arrays of shape [num_years, period, bands] and [num_years, period]
        ### we want to concatenate them to get a single array of shape [self.win_size, bands] and [self.win_size]
        years_reshaped = years.reshape(-1, years.shape[2])
        doy_years_reshaped = doy_years.reshape(-1)
        
        ### replace the original ts_origin and doy with the new ones
        ts_origin = years_reshaped
        doy = doy_years_reshaped

        ### observation: idx is not ordered, so effectively DataLoader class from pytorch
        ### shuffles the data implicitly, 
        ### meaning that I need to keep track of the original order of the samples myself
        ### this is why I also output the idx value to use it later
        return (ts_origin, doy)