import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

def select_arma_order(timeseries, check=10, ic="aic"):
    print("selecting AR order")
    res = sm.tsa.arma_order_select_ic(y=timeseries, max_ar=check, max_ma=check, trend="n", ic=ic)
    p, d = res["{}_min_order".format(ic)]
    return (p, d)
