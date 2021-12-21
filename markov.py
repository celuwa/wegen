from mc_helper import *
import pandas as pd
import numpy as np

# SET REALIZATIONS
REALIZATIONS = 200
# SET YEARS
YEARS = 50; Y0 = 2025 # START YEAR
EXTHRESH = 0.95
DRTHRESH = 0.15
SIMULATED_DATES = pd.date_range("{}-01-01".format(Y0), "{}-12-31".format(Y0+YEARS-1), freq="D")
# READ DAILY DATA
daily_data = pd.read_csv("../output/daily_precipitation.csv", index_col=[0], parse_dates=True).iloc[:, 0]
# READ ARRAY OF YEAR NAMES FOR EACH SIMULATED YEAR
# SELECT ONE OF THE REALIZATIONS
# THE SHAPE SHOULE BE (50, 100) AFTER THIS SELECTION
year_names = np.load("../output/resampled_years.npy")[:, 50, :]
# TURN INTO DATAFRAME, TO EASE SUBSETTING
year_names_df = pd.DataFrame(year_names, index=np.unique(SIMULATED_DATES.year), columns=["O_{}".format(i+1) for i in range(year_names.shape[-1])], dtype=int)
# COMPUTE FREQUENCIES OF STATE TRANSITIONS
historical_frequencies, historical_states = get_frequencies(daily_data, ex_thresh=EXTHRESH, dr_thresh=DRTHRESH)
# simulated state
simulated_state_array = np.full(shape=(SIMULATED_DATES.shape[0], REALIZATIONS), fill_value=np.nan)
print("simulating...")
for r in range(REALIZATIONS):
    print(r)
    for y_ix in range(YEARS):
        year = Y0+y_ix
        # print(year)
        # GET THE SIMILAR YEARS
        available_years, repeats = np.unique(year_names_df.loc[year].values, return_counts=True)
        # GATHER FREQUENCIES IN THESE YEARS
        wegen_frequencies = get_frequencies_multiple(year=available_years, reps=repeats, daily_timeseries=daily_data, ex_thresh=EXTHRESH, dr_thresh=DRTHRESH)
    # SAMPLE DAILY DATA BASED ON SIMLAR YEARS
    # wegen_frequencies, wegen_states = get_frequencies(daily_data, ex_thresh=0.80, dr_thresh=0.3)
        sim_state = simulate_year(year, frequencies=wegen_frequencies)
        curr_ix = np.where(SIMULATED_DATES.year==year)
        simulated_state_array[curr_ix, r] = sim_state
# # SAVE THE SIMULATIONS
simulated_state_ts = pd.DataFrame(simulated_state_array, index=SIMULATED_DATES,
                                columns=["S_{}".format(r+1) for r in range(REALIZATIONS)])
simulated_state_ts.to_csv("../output/simulated_rainfall_states.csv")
# SAVE THE HISTORICAL STATES
historical_states.to_csv("../output/historical_rainfall_states.csv")
