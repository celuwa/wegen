import calendar
import pandas as pd
import numpy as np

def get_window(doy, year=2020, size=7):
    if calendar.isleap(year):
        wrap = 366
    else:
        wrap = 365
    return (np.arange(int(doy - np.floor(size/2))-1,  int(doy + np.floor(size/2))) % wrap) + 1

def get_transitions(states_timeseries):
    transition = [i for i in zip(states_timeseries.values[:-1], states_timeseries.values[1:])]
    return pd.Series(transition, index=states_timeseries.index[:-1])
