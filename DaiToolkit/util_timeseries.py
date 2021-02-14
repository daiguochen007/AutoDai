import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import yfinance as yf
from pylab import mpl
from sklearn import linear_model

# mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False  # 负号'-'显示方块的问题


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
            plt.title(title + " | " + title_str)
        plt.grid(ls="--", alpha=0.7)
        plt.show()

    return params


def timeseries_fit_ret_distri(ts_px, freq="", dis_type="norm", plot=True, bins=100):
    """
    fit return distribution from price timeseries

    :param ts_px:
    :param resample:
    :param dis_type:
    :param plot:
    :param bins:
    :return:
    """
    rescales = {"Daily": 252, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Yearly": 1}
    if freq != "D":
        ts_ret = ts_px / ts_px.shift(int(round(252 / rescales[freq]))) - 1.0
    else:
        ts_ret = ts_px / ts_px.shift(1) - 1.0
    ts_ret = ts_ret.dropna(how="any")
    return fit_distribution(ts_ret, dis_type=dis_type, plot=plot, bins=bins, title="Freq: " + freq)


def timeseries_ret_distri_stats(ts_px, plot=True):
    """
    return 4 moments for each sample
    skew/kurt are central moments / scale free

    plot: plot moment - sampling graph

    :param ts_px:pd.Series
    :return:df_tbl
    """
    freqs = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    rescales = {"Daily": 252, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Yearly": 1}
    stat_tbl = {x: {} for x in freqs}

    for freq in freqs:
        ts_ret = ts_px / ts_px.shift(int(round(252 / rescales[freq]))) - 1.0
        ts_ret = ts_ret.dropna(how="any")
        stat_tbl[freq]["# Data Points"] = len(ts_ret)
        stat_tbl[freq]["Mean(Annual)"] = ts_ret.mean() * rescales[freq]
        stat_tbl[freq]["Std(Annual)"] = ts_ret.std() * math.sqrt(rescales[freq])
        stat_tbl[freq]["Var(Annual)"] = ts_ret.var() * rescales[freq]
        stat_tbl[freq]["Skew"] = ts_ret.skew()
        stat_tbl[freq]["Kurt"] = ts_ret.kurtosis()

    stat_tbl = pd.DataFrame(stat_tbl)[freqs].T
    stat_tbl = stat_tbl[["# Data Points", "Mean(Annual)", "Std(Annual)", "Var(Annual)", "Skew", "Kurt"]]
    stat_tbl.index = [x + " Ret" for x in stat_tbl.index]

    if plot:
        plot_res = {"Mean(Annual)": {}, "Std(Annual)": {}, "Skew": {}, "Kurt": {}}
        ret_scale = min(252, len(ts_px))
        for i in range(1, ret_scale, 1):
            ts_ret = ts_px / ts_px.shift(i) - 1.0
            plot_res["Mean(Annual)"][i] = ts_ret.mean() * 252 / i
            plot_res["Std(Annual)"][i] = ts_ret.std() * math.sqrt(252 / i)
            plot_res["Skew"][i] = ts_ret.skew()
            plot_res["Kurt"][i] = ts_ret.kurtosis()
        plot_res = pd.DataFrame(plot_res)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('Moments for Return Scales (1 - ' + str(ret_scale) + ')')
        ax1.plot(plot_res.index, plot_res["Mean(Annual)"])
        ax1.set_title("Mean(Annual)")
        ax2.plot(plot_res.index, plot_res["Std(Annual)"], 'tab:orange')
        ax2.set_title("Std(Annual)")
        ax3.plot(plot_res.index, plot_res["Skew"], 'tab:green')
        ax3.set_title("Skew")
        ax4.plot(plot_res.index, plot_res["Kurt"], 'tab:red')
        ax4.set_title("Kurt")
        plt.show()

    return stat_tbl


def timeseries_tail_ana(ts_px, freqs=['Daily'], tail_level=0.001, plot=True):
    """
    return

    :param ts_px:
    :param freq:
    :param tail_level: if <=1   perc of data
                       if >1  num of data points
    :return:df_tail_stat
    """
    rescales = {"Daily": 252, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Yearly": 1}
    res = {}
    for freq in freqs:
        ts_ret = ts_px / ts_px.shift(int(round(252 / rescales[freq]))) - 1.0
        ts_ret = ts_ret.dropna(how="any")

        df_righttail = pd.DataFrame(sorted(ts_ret[ts_ret > 0].values), columns=["ret"])
        df_lefttail = pd.DataFrame(sorted(ts_ret[ts_ret < 0].values * -1), columns=["ret"])

        def fit_tail_distri(df_tail, tail_level):
            df_tail["p"] = [1.0 - float(m) / (len(df_tail) + 1) for m in df_tail.index]
            df_tail["ret_log"] = df_tail["ret"].apply(math.log)
            df_tail["p_log"] = df_tail["p"].apply(math.log)

            if tail_level <= 1:
                tail_filter = (1 - tail_level) * len(df_tail)
            else:
                tail_filter = df_tail.index[-1 * tail_level]

            df_tail["tail_tag"] = [True if x >= tail_filter else False for x in df_tail.index]
            lm_x = df_tail.loc[df_tail["tail_tag"], "ret_log"].values
            lm_y = df_tail.loc[df_tail["tail_tag"], "p_log"].values
            lm = linear_model.LinearRegression()
            lm.fit([[m] for m in lm_x], lm_y)
            tail_alpha = lm.coef_[0]
            lm_y_pred = lm.predict([[m] for m in lm_x])
            return [lm_x, lm_y_pred, tail_alpha]

        lm_x_r, lm_y_pred_r, tail_alpha_r = fit_tail_distri(df_righttail, tail_level)
        lm_x_l, lm_y_pred_l, tail_alpha_l = fit_tail_distri(df_lefttail, tail_level)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Return Tail Analysis (Freq = ' + freq + ")")
            ax1.plot(df_lefttail["ret_log"].values, df_lefttail["p_log"].values, ls='-', marker='o', alpha=0.3,
                     markersize=2)
            ax1.plot(lm_x_l, lm_y_pred_l, ls='-', alpha=0.8)
            ax1.set_title("Left Tail (Alpha = " + str(round(tail_alpha_l, 2)) + ")")
            ax2.plot(df_righttail["ret_log"].values, df_righttail["p_log"].values, ls='-', marker='o', alpha=0.3,
                     markersize=2)
            ax2.plot(lm_x_r, lm_y_pred_r, ls='-', alpha=0.8)
            ax2.set_title("Right Tail (Alpha = " + str(round(tail_alpha_r, 2)) + ")")
            plt.show()

        res[freq] = {'left_tail_alpha': tail_alpha_l, 'right_tail_alpha': tail_alpha_r,
                     'left_tail_#dp': len(df_lefttail), 'right_tail_#dp': len(df_righttail),
                     'left_tail_alpha_fit#dp': len(lm_x_l),
                     'right_tail_alpha_fit#dp': len(lm_x_r)}
    res = pd.DataFrame(res).T
    return res


def timeseries_var_ana(ts_px, days=[1, 5, 10, 20, 30], var_level=[0.995, 0.99, 0.95, 0.9]):
    var_res = {}
    for d in days:
        ts_ret = ts_px / ts_px.shift(d) - 1.0
        ts_ret = ts_ret.dropna(how="any")
        day_label = str(d) + "d"
        var_res[day_label] = {}
        var_res[day_label]["#dp"] = len(ts_ret)
        var_res[day_label]["min"] = ts_ret.min()
        var_res[day_label]["max"] = ts_ret.max()
        var_res[day_label]["min_date"] = ts_ret.idxmin()
        var_res[day_label]["max_date"] = ts_ret.idxmax()
        for var in var_level:
            var_res[day_label]["left " + str(var * 100) + "%"] = ts_ret.quantile(1 - var)
            var_res[day_label]["right " + str(var * 100) + "%"] = ts_ret.quantile(var)

    var_res = pd.DataFrame(var_res).T
    var_res = var_res[["#dp", "min"] + ["left " + str(var * 100) + "%" for var in sorted(var_level)[::-1]] + [
        "right " + str(var * 100) + "%" for var in sorted(var_level)] + ["max", "min_date", "max_date"]]
    return var_res




if __name__ == "__main__":
    df_raw = yf.Ticker("^GSPC").history(period="max") #sp500
    df_raw = yf.Ticker("000001.SS").history(period="max")  #上证综指

    # basic stats
    df = timeseries_ret_distri_stats(ts_px=df_raw["Close"], plot=True)
    # ret distribution
    fit_params = timeseries_fit_ret_distri(df_raw["Close"], freq="Daily", dis_type="t", plot=True, bins=300)
    # tail stats
    df_tail_stat = timeseries_tail_ana(df_raw["Close"], freqs=['Daily', "Weekly", "Monthly", "Quarterly", "Yearly"],
                                       tail_level=0.01, plot=False)
    # var stats
    df_var_stat = timeseries_var_ana(df_raw["Close"], days=[1, 5, 10, 20, 30, 60, 252],
                                     var_level=[0.995, 0.99, 0.95, 0.9])
