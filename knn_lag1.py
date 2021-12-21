import pandas as pd
import numpy as np
from dwg_helper import *
import time
# READ SIMULATED STATES
simulated_states = pd.read_csv("../output/simulated_rainfall_states.csv", index_col=[0], parse_dates=True, dtype=np.int32)
# READ HISTORICAL DATA DAILY
historical_data = pd.read_csv("../output/daily_precipitation.csv", index_col=[0], parse_dates=True)#[:-1]
# READ HISTORICAL STATES
historical_states = pd.read_csv("../output/historical_rainfall_states.csv", index_col=[0], parse_dates=True, dtype=np.int32)
# RESHAPE HISTORICAL DATA TO MATCH THE TRANSITIONS
historical_data_reshape = historical_data[:-1]
historical_states_reshape = historical_states[:-1]
# CREATE TRANSITIONS
historical_transitions = get_transitions(states_timeseries=historical_states.loc[:, "0"])
# RUN THE SIMULATION
print("simulating...")
# SOME THINGS WE CAN COMPUTE OUTSIDE THE FOR LOOP
index_years = simulated_states.index.year[:-1]
index_doys = simulated_states.index.dayofyear[:-1]
index_months = simulated_states.index.month[:-1]
historical_mean = historical_data_reshape.mean().values#.groupby(historical_data_reshape.index.month).mean()
historical_std = historical_data_reshape.mean().values#.groupby(historical_data_reshape.index.month).std()#.values[0]
current_variable_value = historical_mean#.values[0]
####################################################
historical_years = historical_states_reshape.index.year[:]
historical_doy = historical_states_reshape.index.dayofyear[:]
####################################################
noMatch = np.full(shape=simulated_states.shape, fill_value=np.nan)[:-1, :]
selectDate = np.full(shape=simulated_states.shape, fill_value=np.nan, dtype='object')[:-1, :]
selectVariable = np.full(shape=simulated_states.shape, fill_value=np.nan)[:-1, :]
####################################################
for r in range(simulated_states.shape[1])[:]:
    simulated_transitions = get_transitions(states_timeseries=simulated_states.loc[:, "S_{}".format(r+1)])
    t0 = time.time()
    for ix in range(simulated_transitions.shape[0])[:]:
        # GET_CURRENT_STATE  # GET FUTURE STATE
        sim_trans = simulated_transitions.iloc[ix]
        year = index_years[ix]
        doy = index_doys[ix]
        month = index_months[ix]
        # GET WINDOW DOY
        window = get_window(doy=doy, year=year, size=15)
        condition = np.in1d(historical_doy, window)
        historical_trans_subset = historical_transitions[condition]
        # AVAILABLE YEARS SUBSET AND DOY WINDOW SUBSET HISTORICAL DATA
        historical_data_subset = historical_data_reshape[condition]
        # print(historical_trans_subset.shape, historical_data_subset.shape)
        # CHECK FOR MATCHES IN TRANSITIONS
        matchcriterion1 = np.in1d([i[0] for i in historical_trans_subset.values], sim_trans[0])
        matchcriterion2 = np.in1d([i[1] for i in historical_trans_subset.values], sim_trans[1])
        matches = matchcriterion1&matchcriterion2
        # print(sum(matches))
        # IF NO MATCH
        if sum(matches) == 0:
            noMatch[ix, r] = 1
            current_variable_value = current_variable_value
            pass
        else:
            matchData = historical_data_subset[matches].values[:, 0]
            matchDate = historical_data_subset[matches].index
            mu = historical_mean; sigma = historical_std
            mdiff1 = current_variable_value - mu
            mdiff2 = matchData - mu
            distances = np.sqrt(((mdiff1-mdiff2)**2)/sigma)
            # print(matchData, current_variable_value)
            k = matchData.shape[0]
            # print(k)
            s_ix = np.argsort(distances)[:k]
            sortData = matchData[s_ix]
            sortDate = matchDate[s_ix]
            weights = (1/(np.arange(k)+1)) / np.sum((1/(np.arange(k)+1)))
            selection_index = np.random.choice(k, 1, p=weights)
            selectDate[ix, r] = sortDate[selection_index][0]
            selectVariable[ix, r] = sortData[selection_index][0]
            current_variable_value = sortData[selection_index][0]
    #         print(selection_index, sortData[selection_index][0], str(sortDate[selection_index][0]))
    # RETURN RESULTS
    t1 = time.time()
    print("r: {} took {:.03f}s".format(r, t1-t0))
pd.DataFrame(selectVariable, index=simulated_transitions.index).to_csv("../output/simulated_daily_precip_.csv")
pd.DataFrame(selectDate, index=simulated_transitions.index).to_csv("../output/simulated_dates_.csv")
