#
import statsmodels.api as sm
import matplotlib.pyplot as plt
from wavelets import WaveletAnalysis
import numpy as np
import netCDF4 as nc
import pandas as pd
from wavelet_helpers import *
from ar_helpers import *

# READ FILE
data = pd.read_csv("../data/climate/climate_timeseries_daily.csv", parse_dates=True, index_col=[0])
# SET REALIZATIONS AND YEARS TO SIMULATE
REALIZATIONS = 100; YEARS = 50
# SET RANDOM SEED WITH THE NUMBER OF YEARS
np.random.seed(YEARS)
# SELECT THE VARIABLE FOR ANALYSIS
dat_daily = data.loc[:, "pr"]
# REDUCE TO ANNUAL
dat_annual = dat_daily.groupby(dat_daily.index.year).sum()
# DETREND AND NORMALIZE ANNUAL SERIES
dat_detrended_norm = normalize(detrend(dat_annual))
# GET WAVELET COMPONENTS
wavelet_components, power, signif, scales, periods = get_wavelet_components(dat_detrended_norm)
# SELECT ARMA ORDER
p, d = select_arma_order(wavelet_components, check=10, ic="aic")
# FIT ARMA MODEL
ar_model = sm.tsa.ARIMA(endog=wavelet_components, order=(p, 0, d), trend="n").fit()
# GENERATE SAMPLES USING MODEL
### FITTING THE AR MODEL REQUIRES SOME BURNIN
### SO ESTIMATE A BURNIN PERIOD FOR THE RESIDUALS
CUT = int(wavelet_components.shape[0]*0.15)
ar_model_residuals = ar_model.resid[CUT:]
component_model = sm.tsa.arma_generate_sample(ar=np.r_[1, -1*ar_model.arparams], ma=np.r_[1, ar_model.maparams], nsample=(YEARS, REALIZATIONS), burnin=50, scale=ar_model_residuals.std())
# COMPUTE WAVELET MODEL ERROR
wavelet_residuals = dat_detrended_norm - wavelet_components
# GENERATE WAVELET MODEL ERROR
wavelet_residual_model = np.random.normal(size=component_model.shape, loc=wavelet_residuals.mean(), scale=wavelet_residuals.std())
# COMBINE SIMLUATED SAMPLES AND RESIDUALS
simulated_annual_timeseries = component_model + wavelet_residual_model
# SAVE OUTPUTS
print("saving")
pd.DataFrame(np.array([power, signif, periods]).T, columns=["power", "signif", "periods"]).to_csv("../output/wavelet_analysis_results.csv")
pd.DataFrame(np.array([dat_detrended_norm, wavelet_components, wavelet_residuals]).T, columns=["data", "wavelet", "residual"], index=dat_annual.index).to_csv("../output/wavelet_io.csv", index_label=["year"])
pd.DataFrame(component_model, columns=["R_{}".format(int(i)) for i in range(1,component_model.shape[1]+1)]).to_csv("../output/modeled_components.csv")
pd.DataFrame(wavelet_residual_model, columns=["R_{}".format(int(i)) for i in range(1,wavelet_residual_model.shape[1]+1)]).to_csv("../output/modeled_residuals.csv")
pd.DataFrame(simulated_annual_timeseries, columns=["R_{}".format(int(i)) for i in range(1,simulated_annual_timeseries.shape[1]+1)]).to_csv("../output/simulated_annual_timeseries.csv")
dat_daily.to_csv("../output/daily_precipitation.csv", index_label=["date"])
