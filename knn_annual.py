import numpy as np
import pandas as pd

# READ THE ANNUAL RAINFALL DATA (NORM, DETRENDED)
historical_df = pd.read_csv("../output/wavelet_io.csv", index_col=[0]).loc[:, ["data"]]
# GET THE VALUES OF THE SIMULATED ANNUAL RAINFALL (NORM, DETRENDED)
simulated_ts = pd.read_csv("../output/simulated_annual_timeseries.csv", index_col=[0]).values
# GET THE NORAMLIZED DETRENDED VALUES OF ANNUAL RAINFALL
historical_data = historical_df.values
# RESHAPE THE SIMULATION DATA
# FROM (YEARS, REALIZATION) --> (YEARS, REALIZATION, 1)
sim_reshape = np.expand_dims(simulated_ts, -1)
# RESHAPE HISTORICAL DATA
# FROM (DATA_LENGTH, 1) --> (1, REALIZATION, DATA_LENGTH)
hist_reshape = np.swapaxes(np.expand_dims(np.tile(historical_data, sim_reshape.shape[1]), -1), 0, -1)
# COMPUTE EUDLIDEAN DISTANCE TO SELECT CLOSEST YEARS
eucld_dist = np.sqrt((hist_reshape - sim_reshape)**2)
# DEFINE K NEAREST
k = int(np.floor(np.sqrt(historical_data.shape[0])))
# SELECT NEAREST K
hist_indices = np.argsort(eucld_dist, axis=-1)[:, :, :k]
# EXPAND HISTORICAL INDEX (DATA_LENGTH, 1)
# FROM (DATA_LENGTH, 1) --> (YEARS, REALIZATION, DATA_LENGTH)
historical_index = historical_df.index.values.reshape(-1, 1)
years_array = np.swapaxes(np.tile(np.expand_dims(np.tile(historical_index, sim_reshape.shape[1]), -1), sim_reshape.shape[0]), 0, -1)
# USE INDEX OF NEAREST K TO SUBSET ARRAY OF HISTORICAL INDEX
subset = np.take_along_axis(years_array, hist_indices, axis=-1)
# COMPUTE WEIGHTS USING RANK
dist_rank_matrix = np.repeat(np.repeat(np.expand_dims(np.expand_dims(np.arange(k)+1, 0), 0), axis=1, repeats=100), axis=0, repeats=50)
weights = dist_rank_matrix[:, :, ::-1] / np.sum(dist_rank_matrix, axis=-1, keepdims=True)
# USE THE WEIGHTS TO SAMPLE THE SUBSET OF YEARS
seed = 1
rng = np.random.default_rng()
resample_years = np.full(shape=(subset.shape[0], subset.shape[1], 100), fill_value=np.nan)
for y in range(subset.shape[0])[:]: # LOOP OVER YEARS
    for r in range(subset.shape[1])[:]: # LOOP OVER REALIZATIONS
        resample_years[y, r, :] = rng.choice(a=subset[y, r, :], size=100, replace=True, p=weights[y, r, :])
# SAVE NUMPY ARRAY OF RESAMPLED YEAR NAMES
np.save("../output/resampled_years.npy", arr=resample_years, allow_pickle=False)
