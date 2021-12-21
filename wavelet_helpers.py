import numpy as np
from sklearn import linear_model
from wavelets import WaveletAnalysis

def detrend(timeseries):
    model = linear_model.LinearRegression()
    x = np.arange(timeseries.shape[0]).reshape(-1, 1)
    y = timeseries.values.reshape(-1, 1)
    model.fit(x, y)
    yhat = model.predict(x)
    detrended = y - yhat + y.mean()
    return detrended[:, 0]

def normalize(timeseries):
    return (timeseries - timeseries.mean()) / timeseries.std()

def get_wavelet_components(timeseries, dt=1.0):
    print("computing wavelet components")
    wa = WaveletAnalysis(timeseries, dt=dt)
    signif = get_wavelet_significance(timeseries) / wa.scales
    power = np.nanmean(wa.wavelet_power, 1) / wa.scales
    signif_scales = np.where(power>signif)[0]
    if len(signif_scales) == 0:
        print("no significance\n ... returning transformed wavelet")
        components = components = np.nanmean(wa.wavelet_transform.real[:, :], 0)
    else:
        components = np.nanmean(wa.wavelet_transform.real[signif_scales, :], 0)
    return components, power, signif, wa.scales, wa.fourier_periods

def get_wavelet_significance(timeseries, size=1000, sig=0.95):
    _al = np.corrcoef(timeseries[:-1], timeseries[1:])[0][1]
    seed = 1
    np.random.seed(seed)
    white_noise = np.random.normal(size=(timeseries.shape[0], size))
    red_noise = np.full(shape=white_noise.shape, fill_value=np.nan)
    red_noise[0, :] = white_noise[0, :]
    red_noise[1:, :] = (_al * white_noise[:-1, :]) + ((1 - _al)**0.5)*white_noise[1:, :]
    theor_spec = np.array([np.nanmean(WaveletAnalysis(red_noise[:, i], dt=1).wavelet_power, 1) for i in range(red_noise.shape[1])])
    return np.quantile(theor_spec, axis=0, q=sig)
