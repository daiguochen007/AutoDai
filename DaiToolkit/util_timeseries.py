import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import yfinance as yf
from pylab import mpl

#mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False  #负号'-'显示方块的问题


def fit_distribution(num_list, dis_type="norm", plot=True, bins=100, title=None):
    """
    fit given series to given distribution
    series na omitted
    return fitted parameters: MLE, location(mean), scale(variance)

    support:
        normal
        student t
        gamma
        pareto
        powerlaw
        beta
    """
    x = np.linspace(min(num_list), max(num_list), bins)

    if dis_type == "norm":
        params = scipy.stats.norm.fit(num_list)
        pdf_fitted = scipy.stats.norm.pdf(x, *params)
    elif dis_type == "t":
        params = scipy.stats.t.fit(num_list)
        pdf_fitted = scipy.stats.t.pdf(x, *params)
    elif dis_type == "gamma":
        params = scipy.stats.gamma.fit(num_list)
        pdf_fitted = scipy.stats.gamma.pdf(x, *params)
    elif dis_type == "pareto":
        params = scipy.stats.pareto.fit(num_list)
        pdf_fitted = scipy.stats.pareto.pdf(x, *params)
    elif dis_type == "powerlaw":
        params = scipy.stats.powerlaw.fit(num_list)
        pdf_fitted = scipy.stats.powerlaw.pdf(x, *params)
    elif dis_type == "beta":
        params = scipy.stats.beta.fit(num_list)
        pdf_fitted = scipy.stats.beta.pdf(x, *params)
    else:
        raise Exception("Distribution type not exist!")

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(x, pdf_fitted, 'b-')
        plt.hist(num_list, bins=bins, density=True, alpha=.3)
        title_str = "Distribution Type: " + dis_type + " | Parameters: " + str([round(y, 4) for y in params])
        if title is None:
            plt.title(title_str)
        else:
            plt.title(title+" | "+title_str)
        plt.grid(ls="--", alpha=0.7)
        plt.show()

    return params


def timeseries_distri_fit(ts_px, resample="", dis_type="norm", plot=True, bins=100):
    """
    fit return distribution

    :param ts_px:
    :param resample:
    :param dis_type:
    :param plot:
    :param bins:
    :return:
    """
    rescales = {"D": 252, "W": 52, "M": 12, "Q": 4, "Y": 1}
    if resample != "D":
        ts_ret = ts_px / ts_px.shift(int(round(252 / rescales[resample]))) - 1.0
    else:
        ts_ret = ts_px / ts_px.shift(1) - 1.0
    ts_ret = ts_ret.dropna(how="any")
    return fit_distribution(ts_ret, dis_type=dis_type, plot=plot, bins=bins, title="Resample: "+resample)


def timeseries_distri_stats(ts_px):
    """
    get 4 moments for each sample
    skew/kurt are central moments / scale free

    :param ts_px:pd.Series
    :return:df_tbl
    """
    intervals = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    rescales = {"Daily": 252, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Yearly": 1}
    resamples = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
    stat_tbl = {x: {} for x in intervals}

    for interval in intervals:
        ts_ret = ts_px / ts_px.shift(int(round(252 / rescales[interval]))) - 1.0
        ts_ret = ts_ret.dropna(how="any")
        stat_tbl[interval]["# Data Points"] = len(ts_ret)
        stat_tbl[interval]["Mean(Annual)"] = ts_ret.mean() * rescales[interval]
        stat_tbl[interval]["Std(Annual)"] = ts_ret.std() * math.sqrt(rescales[interval])
        stat_tbl[interval]["Var(Annual)"] = ts_ret.var() * rescales[interval]
        stat_tbl[interval]["Skew"] = ts_ret.skew()
        stat_tbl[interval]["Kurt"] = ts_ret.kurtosis()

    stat_tbl = pd.DataFrame(stat_tbl)[intervals].T
    stat_tbl = stat_tbl[["# Data Points", "Mean(Annual)", "Std(Annual)", "Var(Annual)", "Skew", "Kurt"]]
    stat_tbl.index = [x + " Ret" for x in stat_tbl.index]
    return stat_tbl


if __name__ == "__main__":
    df_raw = yf.Ticker("^GSPC").history(period="max")
    # basic stats
    print(timeseries_distri_stats(ts_px=df_raw["Close"]))
    # ret distribution
    timeseries_distri_fit(df_raw["Close"], resample="D", dis_type="t", plot=True, bins=300)
