import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import yfinance as yf
from pylab import mpl
from sklearn import linear_model

from DaiToolkit.util_akshare import akshare_get_history, aksharelocal_get_history
from DaiToolkit.util_portfolio import perf_stats
from DaiToolkit.util_tushare import tushare_get_history, tusharelocal_get_history

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


def timeseries_fit_ret_distri(ts_px, freq=5, dis_type="norm", plot=True, bins=100):
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
    if type(freq) == str:
        ts_ret = ts_px / ts_px.shift(int(round(252 / rescales[freq]))) - 1.0
    else:
        ts_ret = ts_px / ts_px.shift(freq) - 1.0
    ts_ret = ts_ret.dropna(how="any")
    return fit_distribution(ts_ret, dis_type=dis_type, plot=plot, bins=bins, title="Freq: " + str(freq))


def timeseries_ret_distri_stats(ts_px, log_ret=True, plot=True, plot_max_freq=252):
    """
    return 4 moments for each sample
    skew/kurt are central moments / scale free

    plot: plot moment - sampling graph

    :param ts_px:pd.Series daily price
    :return:df_tbl
    """
    freqs = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    rescales = {"Daily": 252, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Yearly": 1}
    stat_tbl = {x: {} for x in freqs}

    for freq in freqs:
        ts_ret = ts_px / ts_px.shift(int(round(252 / rescales[freq])))
        if log_ret:
            ts_ret = ts_ret.apply(math.log)
        else:
            ts_ret = ts_ret - 1.0
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
        ret_scale = min(plot_max_freq, len(ts_px))
        for i in range(1, ret_scale, 1):
            ts_ret = ts_px / ts_px.shift(i)
            if log_ret:
                ts_ret = ts_ret.apply(math.log)
            else:
                ts_ret = ts_ret - 1.0
            plot_res["Mean(Annual)"][i] = ts_ret.mean() * 252 / i
            plot_res["Std(Annual)"][i] = ts_ret.std() * math.sqrt(252 / i)
            plot_res["Skew"][i] = ts_ret.skew()
            plot_res["Kurt"][i] = ts_ret.kurtosis()
        plot_res = pd.DataFrame(plot_res)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('Moments for Return Scales (1d - ' + str(ret_scale) + 'd)')
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


def timeseries_ret_distri_stats_ndays(ts_px, ndays=[1, 5, 10, 20, 40, 60, 120], log_ret=True):
    """
    return 4 moments for each sample
    skew/kurt are central moments / scale free

    plot: plot moment - sampling graph

    :param ts_px:pd.Series daily price
    :return:df_tbl
    """
    stat_tbl = {str(x) + 'd': {} for x in ndays}
    for nday in ndays:
        ts_ret = ts_px / ts_px.shift(nday)
        if log_ret:
            ts_ret = ts_ret.apply(math.log)
        else:
            ts_ret = ts_ret - 1.0
        ts_ret = ts_ret.dropna(how="any")
        stat_tbl[str(nday) + 'd']["# Data Points"] = len(ts_ret)
        stat_tbl[str(nday) + 'd']["Mean"] = ts_ret.mean()
        stat_tbl[str(nday) + 'd']["Std"] = ts_ret.std()
        stat_tbl[str(nday) + 'd']["Skew"] = ts_ret.skew()
        stat_tbl[str(nday) + 'd']["Kurt"] = ts_ret.kurtosis()
        stat_tbl[str(nday) + 'd']["Min"] = ts_ret.min()
        stat_tbl[str(nday) + 'd']["Max"] = ts_ret.max()

        # assume normal var randomize
        stat_tbl[str(nday) + 'd']["FatTail:Tail Start"] = ts_ret.std() * math.sqrt(0.5 * (5 + math.sqrt(17)))
        stat_tbl[str(nday) + 'd']["FatTail:Shoulder Start"] = ts_ret.std() * math.sqrt(0.5 * (5 - math.sqrt(17)))

    stat_tbl = pd.DataFrame(stat_tbl)[[str(x) + 'd' for x in ndays]].T
    stat_tbl = stat_tbl[["# Data Points", "Mean", "Std", "Skew", "Kurt", "Min", "Max", "FatTail:Tail Start", "FatTail:Shoulder Start"]]
    stat_tbl.index = [x + " Ret" for x in stat_tbl.index]
    return stat_tbl


def timeseries_volsig_diff(ts_px, short_term=1, long_term=5, log_ret=True):
    """
    return short term annual vol - long term annual vol
    plot: plot moment - sampling graph

    :param ts_px:pd.Series daily price
    :return:vol diff
    """
    ts_ret_s = ts_px / ts_px.shift(short_term)
    ts_ret_l = ts_px / ts_px.shift(long_term)

    if log_ret:
        ts_ret_s = ts_ret_s.apply(math.log)
        ts_ret_l = ts_ret_l.apply(math.log)
    else:
        ts_ret_s = ts_ret_s - 1.0
        ts_ret_l = ts_ret_l - 1.0

    return ts_ret_s.std() * math.sqrt(252 / short_term) - ts_ret_l.std() * math.sqrt(252 / long_term)


def timeseries_tail_ana_regular(ts_px, freqs=['Daily'], tail_level=0.001, plot=True):
    """
    return tail analysis

    :param ts_px:
    :param freq: list of ['Daily',"Weekly","Monthly","Quarterly","Yearly"]
    :param tail_level: if <=1   perc of data points
                       if >1    num of data points
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
            df_tail["p_log"] = df_tail["p"].apply(lambda x: math.log(x) if x > 0 else float("nan"))

            tail_vol = np.std(df_tail["ret"].to_list() + (df_tail["ret"] * -1).to_list())
            df_tail['norm_p'] = [(1 - scipy.stats.norm.cdf(x, loc=0, scale=tail_vol)) * 2 for x in df_tail["ret"]]
            df_tail['norm_p_log'] = df_tail["norm_p"].apply(
                lambda x: float("nan") if x == 0 else math.log(x) if math.log(x) >= df_tail["p_log"].min() else float(
                    "nan"))

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
            return [lm_x, lm_y_pred, tail_alpha, tail_vol]

        lm_x_r, lm_y_pred_r, tail_alpha_r, tail_vol_r = fit_tail_distri(df_righttail, tail_level)
        lm_x_l, lm_y_pred_l, tail_alpha_l, tail_vol_l = fit_tail_distri(df_lefttail, tail_level)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Return Tail Analysis (Freq = ' + freq + ")")
            ax1.plot(df_lefttail["ret_log"].values, df_lefttail["p_log"].values, ls='-', marker='o', alpha=0.3,
                     markersize=2, label='actual')
            ax1.plot(df_lefttail["ret_log"].values, df_lefttail["norm_p_log"].values, ls='-', alpha=0.5, label='norm')
            ax1.plot(lm_x_l, lm_y_pred_l, ls='-', alpha=0.8, label='power law fit')
            ax1.legend()
            ax1.set_title("Left Tail (Alpha = " + str(round(tail_alpha_l, 2)) + ")")
            ax2.plot(df_righttail["ret_log"].values, df_righttail["p_log"].values, ls='-', marker='o', alpha=0.3,
                     markersize=2, label='actual')
            ax2.plot(df_righttail["ret_log"].values, df_righttail["norm_p_log"].values, ls='-', alpha=0.5, label='norm')
            ax2.plot(lm_x_r, lm_y_pred_r, ls='-', alpha=0.8, label='power law fit')
            ax2.legend()
            ax2.set_title("Right Tail (Alpha = " + str(round(tail_alpha_r, 2)) + ")")
            plt.show()

        res[freq] = {'left_tail_alpha': tail_alpha_l, 'right_tail_alpha': tail_alpha_r,
                     'left_tail_#dp': len(df_lefttail), 'right_tail_#dp': len(df_righttail),
                     'left_tail_alpha_fit#dp': len(lm_x_l),
                     'right_tail_alpha_fit#dp': len(lm_x_r),
                     'left_tail_volatility(Annualized)': tail_vol_l * math.sqrt(rescales[freq]),
                     'right_tail_volatility(Annualized)': tail_vol_r * math.sqrt(rescales[freq])}
    res = pd.DataFrame(res)
    return res


def timeseries_tail_ana(ts_px, ret_freq=5, tail_start=None, plot=True):
    """
    return tail analysis

    :param ts_px:
    :param freq: list of ['Daily',"Weekly","Monthly","Quarterly","Yearly"]
    :param tail_level: if <=1   perc of data points
                       if >1    num of data points
    :return:df_tail_stat
    """
    ts_ret = ts_px / ts_px.shift(ret_freq) - 1.0
    ts_ret = ts_ret.dropna(how="any")

    df_righttail = pd.DataFrame(sorted(ts_ret[ts_ret > 0].values), columns=["ret"])
    df_lefttail = pd.DataFrame(sorted(ts_ret[ts_ret < 0].values * -1), columns=["ret"])

    def fit_tail_distri(df_tail, tail_filter):
        tail_vol = np.std(df_tail["ret"].to_list() + (df_tail["ret"] * -1).to_list())
        if tail_filter is None:
            tail_filter = tail_vol * math.sqrt(0.5 * (5 + math.sqrt(17)))

        df_tail["p"] = [1.0 - float(m) / (len(df_tail) + 1) for m in df_tail.index]
        df_tail["ret_log"] = df_tail["ret"].apply(math.log)
        df_tail["p_log"] = df_tail["p"].apply(lambda x: math.log(x) if x > 0 else float("nan"))

        df_tail['norm_p'] = [(1 - scipy.stats.norm.cdf(x, loc=0, scale=tail_vol)) * 2 for x in df_tail["ret"]]
        df_tail['norm_p_log'] = df_tail["norm_p"].apply(
            lambda x: float("nan") if x == 0 else math.log(x) if math.log(x) >= df_tail["p_log"].min() else float(
                "nan"))

        df_tail["tail_tag"] = [True if x >= tail_filter else False for x in df_tail["ret"]]
        lm_x = df_tail.loc[df_tail["tail_tag"], "ret_log"].values
        lm_y = df_tail.loc[df_tail["tail_tag"], "p_log"].values
        lm = linear_model.LinearRegression()
        lm.fit([[m] for m in lm_x], lm_y)
        tail_alpha = lm.coef_[0]
        lm_y_pred = lm.predict([[m] for m in lm_x])
        return [lm_x, lm_y_pred, tail_alpha, tail_vol, tail_filter]

    lm_x_r, lm_y_pred_r, tail_alpha_r, tail_vol_r, tail_start_r = fit_tail_distri(df_righttail, tail_start)
    lm_x_l, lm_y_pred_l, tail_alpha_l, tail_vol_l, tail_start_l = fit_tail_distri(df_lefttail, tail_start)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(str(ret_freq) + 'd Return Tail Analysis')
        ax1.plot(df_lefttail["ret_log"].values, df_lefttail["p_log"].values, ls='-', marker='o', alpha=0.3,
                 markersize=2, label='actual')
        ax1.plot(df_lefttail["ret_log"].values, df_lefttail["norm_p_log"].values, ls='-', alpha=0.5, label='norm')
        ax1.plot(lm_x_l, lm_y_pred_l, ls='-', alpha=0.8, label='power law fit')
        ax1.legend()
        ax1.set_title("Left Tail (Alpha = " + str(round(tail_alpha_l, 2)) + ")")
        ax2.plot(df_righttail["ret_log"].values, df_righttail["p_log"].values, ls='-', marker='o', alpha=0.3,
                 markersize=2, label='actual')
        ax2.plot(df_righttail["ret_log"].values, df_righttail["norm_p_log"].values, ls='-', alpha=0.5, label='norm')
        ax2.plot(lm_x_r, lm_y_pred_r, ls='-', alpha=0.8, label='power law fit')
        ax2.legend()
        ax2.set_title("Right Tail (Alpha = " + str(round(tail_alpha_r, 2)) + ")")
        plt.show()

    res = {'left_tail_alpha': tail_alpha_l,
           'right_tail_alpha': tail_alpha_r,
           '#dp_left_tail': len(df_lefttail),
           '#dp_right_tail': len(df_righttail),
           '#dp_fit_left_tail_alpha': len(lm_x_l),
           '#dp_fit_right_tail_alpha': len(lm_x_r),
           'left_tail_vol': tail_vol_l,
           'right_tail_vol': tail_vol_r,
           'left_tail_vol_ann': tail_vol_l * math.sqrt(252 / ret_freq),
           'right_tail_vol_ann': tail_vol_r * math.sqrt(252 / ret_freq),
           'left_tail_start': tail_start_l,
           'right_tail_start': tail_start_r,
           }
    res = pd.DataFrame(res, index=[str(ret_freq) + 'd Ret']).T
    return res


def timeseries_moment_rolling_ana(ts_px, ret_freq=5, rolling_window=252):
    """
    return rolling moments
    """
    ts_ret = ts_px / ts_px.shift(ret_freq) - 1.0
    ts_ret = ts_ret.dropna(how="any")

    ts_1m = ts_ret.rolling(rolling_window).mean()
    ts_2m = ts_ret.rolling(rolling_window).std()
    ts_3m = ts_ret.rolling(rolling_window).skew()
    ts_4m = ts_ret.rolling(rolling_window).apply(scipy.stats.kurtosis)

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, sharex=True)
    fig.suptitle(str(ret_freq) + 'd Return Rolling 1-4 Moments (Window=' + str(rolling_window) + 'd)')
    ax0.plot(ts_px, color='black')
    ax0.set_title("Performance")
    ax0.grid(ls='--', alpha=0.5)
    ax1.plot(ts_1m, color='grey')
    ax1.grid(ls='--', alpha=0.5)
    ax1.set_title("Mean")
    ax2.plot(ts_2m)
    ax2.grid(ls='--', alpha=0.5)
    ax2.set_title("Std")
    ax3.plot(ts_3m, color='darkgreen')
    ax3.grid(ls='--', alpha=0.5)
    ax3.set_title("Skew")
    ax4.plot(ts_4m, color='orange')
    ax4.grid(ls='--', alpha=0.5)
    ax4.set_title("Kurt")
    plt.show()


def timeseries_var_ana(ts_px, days=[1, 5, 10, 20, 30], var_level=[0.995, 0.99, 0.95]):
    """
    return VAR/CVAR analysis

    :param ts_px:
    :param days:
    :param var_level:
    :return: df stats
    """
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
        var_res[day_label]["start_date"] = ts_ret.index.min()
        var_res[day_label]["end_date"] = ts_ret.index.max()
        for var in var_level:
            var_res[day_label]["VAR_left_" + str(var * 100) + "%"] = ts_ret.quantile(1 - var)
            var_res[day_label]["VAR_right_" + str(var * 100) + "%"] = ts_ret.quantile(var)

            var_res[day_label]["CVAR_left_" + str(var * 100) + "%"] = ts_ret[ts_ret < ts_ret.quantile(1 - var)].mean()
            var_res[day_label]["CVAR_right_" + str(var * 100) + "%"] = ts_ret[ts_ret > ts_ret.quantile(var)].mean()
        # Max Drawdown
        ts_ret_mdd = ts_px / ts_px.rolling(d + 1).max() - 1.0
        ts_ret_mdd = ts_ret_mdd.dropna(how="any")
        var_res[day_label]["max_drawdown"] = ts_ret_mdd.min()
        var_res[day_label]["max_drawdown_date"] = ts_ret_mdd.idxmin()

    var_res = pd.DataFrame(var_res)
    var_res = var_res.loc[["min"] + ["VAR_left_" + str(var * 100) + "%" for var in sorted(var_level)[::-1]] + [
        "VAR_right_" + str(var * 100) + "%" for var in sorted(var_level)] + ["max", "min_date", "max_date"] + [
                              "CVAR_left_" + str(var * 100) + "%" for var in sorted(var_level)[::-1]] + [
                              "CVAR_right_" + str(var * 100) + "%" for var in sorted(var_level)] + [
                              "max_drawdown", "max_drawdown_date", "#dp", "start_date", "end_date"]]

    return var_res


def timeseries_rebalance_ana(ts_px_series, rebal_freqs=[5], rebal_ratio=1, rebal_hurdle=0, rebal_type="mean reverse",
                             rebal_anchor="no rebalance", weight_bound=[0, 1], long_term_growth="implied",
                             start_posperc=0.5, riskfree_rate=0.03, plot=True):
    """
    single timeseries rebalance analysis -> within freq period -> lower pos when price up / higher when price down -> close out when period ends

    :param ts_px_series: pd series "close"
    :param rebal_freqs: 每n天进行一次再平衡（调回 rebal_anchor）, list of n trading days
    :param rebal_ratio: 调仓轻重 rebalance position %change / daily return (5 means when market increase 1%, lower position by 5%)
    :param rebal_type: 'mean reverse': 均值回归
                       'trend': 动量/趋势
    :param rebal_anchor: 'no rebalance': 再平衡到初始静态增长状态
                         'fixed weight': 再平衡到固定配比，如40-60
                         'long term': 再平衡到长期增长配比
    :param long_term_growth: 'implied': 隐含长期增长率 (最终价格/初始价格，年化)
                              number: 0.05 假设年化增速5%
    :param rebal_freqs: list of n trading days
    :param rebal_hurdle: only rebalance if abs(today's return) > rebal_hurdle
    :param start_posperc: allocate % to asset in the beginning, rest invest in cash
    :param riskfree_rate: use short term cash rate
    :return: portfolio stats
    """
    rebal_freqs_str = [str(x) for x in sorted(rebal_freqs)]
    start_cashperc = 1 - start_posperc
    if rebal_type == "mean reverse":
        rebal_type_direction = -1
    elif rebal_type == "trend":
        rebal_type_direction = 1
    else:
        raise Exception("rebal_type support 'mean reverse' and 'trend'")
    w_min, w_max = weight_bound

    # rebalancing
    ts_px = pd.DataFrame(ts_px_series).dropna(how="any").copy()
    ts_px['date'] = ts_px.index
    ts_px['days_accrual'] = (ts_px['date'] - ts_px['date'].shift(1)).apply(lambda x: x.days if not pd.isna(x) else 0)
    ts_px["ret_1d"] = ts_px["close"] / ts_px["close"].shift(1) - 1.0
    ts_px["ret_1d_riskfree"] = ts_px['days_accrual'] / 365 * riskfree_rate
    ts_px["pos_nav"] = ts_px["close"] / ts_px["close"][0]
    ts_px["cash_nav"] = (ts_px["ret_1d_riskfree"] + 1).cumprod()
    ts_px["#"] = range(0, len(ts_px), 1)
    ts_px["port_nav_no_rebal"] = ts_px["pos_nav"] * start_posperc + ts_px["cash_nav"] * start_cashperc
    ts_px["no_rebal_pos_weight"] = ts_px["pos_nav"] * start_posperc / ts_px["port_nav_no_rebal"]

    if long_term_growth == 'implied':
        compound_ret_daily = ts_px["pos_nav"][-1] ** (1 / ts_px['days_accrual'].sum())
    else:
        compound_ret_daily = (1 + long_term_growth) ** (1 / 365)
    ts_px["pos_longterm_imp_nav"] = ts_px['days_accrual'].cumsum().apply(lambda x: compound_ret_daily ** x)

    for rf in rebal_freqs_str:
        ts_px["rebal_tag" + rf] = ts_px["#"].apply(lambda x: True if x % int(rf) == 0 else False)
        ts_px["rebal_tag" + rf][-1] = True
        res = {}
        res[ts_px.index[0]] = {}
        res[ts_px.index[0]]["pos_val_before_rebal" + rf] = start_posperc
        res[ts_px.index[0]]["cash_val_before_rebal" + rf] = start_cashperc
        res[ts_px.index[0]]["total_val_before_rebal" + rf] = start_posperc + start_cashperc
        res[ts_px.index[0]]["rebal_pos_weight" + rf] = start_posperc
        res[ts_px.index[0]]["rebal_cash_weight" + rf] = start_cashperc
        res[ts_px.index[0]]["pos_val_after_rebal" + rf] = start_posperc
        res[ts_px.index[0]]["cash_val_after_rebal" + rf] = start_cashperc
        res[ts_px.index[0]]["rebal_enhance_signal" + rf] = 0

        res[ts_px.index[0]]["pos_val_before_rebal_anchor" + rf] = start_posperc
        res[ts_px.index[0]]["cash_val_before_rebal_anchor" + rf] = start_cashperc
        res[ts_px.index[0]]["rebal_pos_weight_anchor" + rf] = start_posperc
        res[ts_px.index[0]]["rebal_cash_weight_anchor" + rf] = start_cashperc
        res[ts_px.index[0]]["pos_val_after_rebal_anchor" + rf] = start_posperc
        res[ts_px.index[0]]["cash_val_after_rebal_anchor" + rf] = start_cashperc
        res[ts_px.index[0]]["total_val_before_rebal_anchor" + rf] = start_posperc + start_cashperc

        for dt, dtm1 in zip(ts_px.index[1:], ts_px.index[:-1]):
            res[dt] = {}
            res[dt]["pos_val_before_rebal" + rf] = res[dtm1]["pos_val_after_rebal" + rf] * (1 + ts_px.loc[dt, "ret_1d"])
            res[dt]["cash_val_before_rebal" + rf] = res[dtm1]["cash_val_after_rebal" + rf] * (1 + ts_px.loc[dt, "ret_1d_riskfree"])
            res[dt]["total_val_before_rebal" + rf] = res[dt]["pos_val_before_rebal" + rf] + res[dt]["cash_val_before_rebal" + rf]

            res[dt]["pos_val_before_rebal_anchor" + rf] = res[dtm1]["pos_val_after_rebal_anchor" + rf] * (1 + ts_px.loc[dt, "ret_1d"])
            res[dt]["cash_val_before_rebal_anchor" + rf] = res[dtm1]["cash_val_after_rebal_anchor" + rf] * (1 + ts_px.loc[dt, "ret_1d_riskfree"])
            res[dt]["total_val_before_rebal_anchor" + rf] = res[dt]["pos_val_before_rebal_anchor" + rf] + res[dt]["cash_val_before_rebal_anchor" + rf]

            if ts_px.loc[dt, "rebal_tag" + rf]:
                # rebalance trigged -> return to rebal anchor weight
                if rebal_anchor == "no rebalance":
                    res[dt]["rebal_pos_weight" + rf] = ts_px.loc[dt, "pos_nav"] * start_posperc / ts_px.loc[dt, "port_nav_no_rebal"]
                elif rebal_anchor == "fixed weight":
                    res[dt]["rebal_pos_weight" + rf] = start_posperc
                elif rebal_anchor == "long term":
                    res[dt]["rebal_pos_weight" + rf] = ts_px.loc[dt, "pos_longterm_imp_nav"] * start_posperc / (
                            ts_px.loc[dt, "pos_longterm_imp_nav"] * start_posperc + ts_px.loc[dt, "cash_nav"] * start_cashperc)
                else:
                    raise Exception("rebal_anchor support 'no rebalance','fixed weight','long term'")
                res[dt]["rebal_cash_weight" + rf] = 1 - res[dt]["rebal_pos_weight" + rf]
                res[dt]["rebal_enhance_signal" + rf] = 0

                # anchor
                res[dt]['rebal_pos_weight_anchor' + rf] = res[dt]["rebal_pos_weight" + rf]
                res[dt]["rebal_cash_weight_anchor" + rf] = res[dt]["rebal_cash_weight" + rf]
            else:
                # between rebalance freq -> enhance rebalance if price moved much
                # mean reverse: when market increase 1%, lower position by 1% * rebal_ratio, no short no leverage, postion in [0,1]
                origin_weight = res[dt]["pos_val_before_rebal" + rf] / res[dt]["total_val_before_rebal" + rf]
                if ts_px.loc[dt, "ret_1d"] > rebal_hurdle:
                    rebal_ratio_curr = rebal_ratio
                    rebal_enhance_signal = rebal_type_direction
                elif ts_px.loc[dt, "ret_1d"] < -1 * rebal_hurdle:
                    rebal_ratio_curr = rebal_ratio
                    rebal_enhance_signal = -1 * rebal_type_direction
                else:
                    rebal_ratio_curr = 0
                    rebal_enhance_signal = 0

                if origin_weight >= 1 or origin_weight <= 0:
                    # rebal_ratio_curr = 0
                    rebal_enhance_signal = 0

                res[dt]["rebal_pos_weight" + rf] = origin_weight + rebal_type_direction * ts_px.loc[dt, "ret_1d"] * rebal_ratio_curr
                res[dt]["rebal_pos_weight" + rf] = max(min(w_max, res[dt]["rebal_pos_weight" + rf]), w_min)
                res[dt]["rebal_cash_weight" + rf] = 1 - res[dt]["rebal_pos_weight" + rf]
                res[dt]["rebal_enhance_signal" + rf] = res[dtm1]["rebal_enhance_signal" + rf] + rebal_enhance_signal

                # anchor
                res[dt]['rebal_pos_weight_anchor' + rf] = res[dt]["pos_val_before_rebal_anchor" + rf] / res[dt]["total_val_before_rebal_anchor" + rf]
                res[dt]["rebal_cash_weight_anchor" + rf] = 1 - res[dt]["rebal_pos_weight_anchor" + rf]

            res[dt]["pos_val_after_rebal" + rf] = res[dt]["total_val_before_rebal" + rf] * res[dt]["rebal_pos_weight" + rf]
            res[dt]["cash_val_after_rebal" + rf] = res[dt]["total_val_before_rebal" + rf] * res[dt]["rebal_cash_weight" + rf]

            res[dt]["pos_val_after_rebal_anchor" + rf] = res[dt]["total_val_before_rebal_anchor" + rf] * res[dt]["rebal_pos_weight_anchor" + rf]
            res[dt]["cash_val_after_rebal_anchor" + rf] = res[dt]["total_val_before_rebal_anchor" + rf] * res[dt]["rebal_cash_weight_anchor" + rf]

        res = pd.DataFrame(res).T
        ts_px = ts_px.merge(res, left_index=True, right_index=True, how='left')

        ts_px["port_nav_rebal" + rf] = ts_px["pos_val_after_rebal" + rf] + ts_px["cash_val_after_rebal" + rf]
        ts_px["port_nav_rebal_anchor" + rf] = ts_px["pos_val_after_rebal_anchor" + rf] + ts_px["cash_val_after_rebal_anchor" + rf]

        ts_px["port_rebal_excess_ret" + rf] = (ts_px["port_nav_rebal" + rf] / ts_px["port_nav_rebal" + rf].shift(1)) - (
                ts_px["port_nav_rebal_anchor" + rf] / ts_px["port_nav_rebal_anchor" + rf].shift(1))
        ts_px["port_rebal_excess_ret" + rf] = ts_px["port_rebal_excess_ret" + rf].fillna(0)
        ts_px["port_rebal_excess_cumret" + rf] = (ts_px["port_rebal_excess_ret" + rf] + 1).cumprod()

        ts_px["port_rebal_anchor_excess_ret" + rf] = (ts_px["port_nav_rebal_anchor" + rf] / ts_px["port_nav_rebal_anchor" + rf].shift(1)) - (
                ts_px["port_nav_no_rebal"] / ts_px["port_nav_no_rebal"].shift(1))
        ts_px["port_rebal_anchor_excess_ret" + rf] = ts_px["port_rebal_anchor_excess_ret" + rf].fillna(0)
        ts_px["port_rebal_anchor_excess_cumret" + rf] = (ts_px["port_rebal_anchor_excess_ret" + rf] + 1).cumprod()

    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        fig.suptitle(
            'Advanced Rebalance Analysis\n(rebalance freq:' + ','.join([x + 'd' for x in rebal_freqs_str]) + ' | type:' + rebal_type + ' | hurdle:' + \
            str(int(rebal_hurdle * 100)) + '% | anchor:' + rebal_anchor + ' | assume pos ann ret:' + str(
                round((compound_ret_daily ** 365 - 1) * 100, 2)) + '%)')
        for rf in rebal_freqs_str:
            ax1.plot(ts_px.index, ts_px["port_nav_rebal" + rf], '-', label='rebal enhance ' + str(rf) + 'd', alpha=0.7)
            ax1.plot(ts_px.index, ts_px["port_nav_rebal_anchor" + rf], '-', label='rebal anchor ' + str(rf) + 'd', alpha=0.3)
        ax1.plot(ts_px.index, ts_px["port_nav_no_rebal"], '-', color="dimgrey", label='no rebal', alpha=0.9)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_title('Portfolio NAV compare')
        ax1.grid(ls="--", alpha=0.5)
        for rf in rebal_freqs_str:
            ax2.plot(ts_px.index, ts_px["port_rebal_excess_cumret" + rf] - 1, '-', label='rebal ' + str(rf) + 'd enhance-anchor', alpha=0.9)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_title('Cumulative excess return (enhance - anchor)')
        ax2.grid(ls="--", alpha=0.5)
        for rf in rebal_freqs_str:
            ax3.plot(ts_px.index, ts_px["port_rebal_anchor_excess_cumret" + rf] - 1, '-', label='rebal ' + str(rf) + 'd anchor-no rebal', alpha=0.9)
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.set_title('Cumulative excess return (anchor - no rebalance)')
        ax3.grid(ls="--", alpha=0.5)
        ymin, ymax = ax3.get_ylim()
        ax3.set_ylim(min(ymin, -0.01), max(ymax, 0.01))
        ax4.plot(ts_px.index, ts_px["no_rebal_pos_weight"], '-', color="dimgrey", label='no rebal', alpha=0.3)
        for rf in rebal_freqs_str:
            ax4.plot(ts_px.index, ts_px["rebal_pos_weight_anchor" + rf], '--', alpha=0.3, label='rebal anchor ' + str(rf) + 'd')
            ax4.plot(ts_px.index, ts_px["rebal_pos_weight" + rf], '-', alpha=0.7, label='rebal ' + str(rf) + 'd')
        ax4.set_title('Position Weights')
        ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    # performance stats
    df_stats = {}
    for col in ["pos_nav", "port_nav_no_rebal"] + ["port_nav_rebal" + rf for rf in rebal_freqs_str] + ["port_nav_rebal_anchor" + rf for rf in rebal_freqs_str]:
        df_stats[col] = perf_stats((ts_px[col] / ts_px[col].shift(1) - 1).dropna(how='any'))
    for col in ["port_rebal_excess_ret" + rf for rf in rebal_freqs_str]:
        df_stats[col] = perf_stats(ts_px[col].dropna(how='any'))
    df_stats = pd.DataFrame(df_stats)
    df_stats = df_stats[["port_nav_no_rebal"] + ["port_nav_rebal" + rf for rf in rebal_freqs_str] + [
        "port_nav_rebal_anchor" + rf for rf in rebal_freqs_str] + ["port_rebal_excess_ret" + rf for rf in rebal_freqs_str] + ["pos_nav"]]
    rename_dict = {"port_nav_no_rebal": "Portfolio(no rebal)", "pos_nav": "Asset(100% hold)"}
    rename_dict.update({"port_nav_rebal" + rf: "Portfolio(" + rf + "d rebal enhance)" for rf in rebal_freqs_str})
    rename_dict.update({"port_nav_rebal_anchor" + rf: "Portfolio(" + rf + "d rebal anchor)" for rf in rebal_freqs_str})
    rename_dict.update({"port_rebal_excess_ret" + rf: "Enhance Excess Ret(" + rf + "d rebal)" for rf in rebal_freqs_str})
    df_stats = df_stats.rename(rename_dict, axis=1).T
    return df_stats


def timeseries_port_rebal_ana(df_port_secs, start_weight, rebal_freq=5, rebal_enhance_ratio=0.5,
                              rebal_anchor="no rebalance", rebal_enhance_hurdle=0, weight_bound=[0, 1],
                              longterm_growth="implied", plot=True):
    """
    组合再平衡分析 - 均值回复型再平衡增强，隔n日回到再平衡基准

    :param df_port_secs: df - list of security px
    :param start_weight: initial weight vector
    :param rebal_freq: num
    :param rebal_enhance_ratio: factor for moving weight when px move
    :param rebal_anchor: 'no rebalance': 再平衡到初始静态增长状态
                         'fixed weight': 再平衡到固定配比，如40-60
                         'long term': 再平衡到长期增长配比
    :param longterm_growth: 'fixed' or annualized [growth1, growth2...]
    :param weight_bound: boundary when rebalancing
    :param plot: True/False
    :return: df_stats
    """

    df_port_secs = df_port_secs.sort_index()
    df_port_secs = df_port_secs / df_port_secs.iloc[0, :].values

    sec_list = df_port_secs.columns
    rf = str(rebal_freq) + 'd'
    w_min, w_max = weight_bound

    # rebalancing
    df_port_secs['date'] = df_port_secs.index
    df_port_secs['days_accrual'] = (df_port_secs['date'] - df_port_secs['date'].shift(1)).apply(lambda x: x.days if not pd.isna(x) else 0)
    for sec in sec_list:
        df_port_secs[sec + "_ret_1d"] = df_port_secs[sec] / df_port_secs[sec].shift(1) - 1.0
        df_port_secs[sec + "_ret_1d"] = df_port_secs[sec + "_ret_1d"].fillna(0)
    df_port_secs["#"] = range(0, len(df_port_secs), 1)

    #### no rebalance / static hold
    df_port_secs["port_nav_no_rebal"] = df_port_secs[sec_list].dot(start_weight)
    #### fixed weight / rebalance to fixed weight daily
    df_port_secs["port_ret_fixed_weight"] = df_port_secs[[sec + "_ret_1d" for sec in sec_list]].dot(start_weight)
    df_port_secs["port_nav_fixed_weight"] = (df_port_secs["port_ret_fixed_weight"] + 1).cumprod()

    #### long term growth
    if longterm_growth == 'implied':
        longterm_growth = df_port_secs[sec_list].iloc[-1, :] ** (365 / df_port_secs['days_accrual'].sum()) - 1
        longterm_growth = longterm_growth.values
    longterm_growth_daily = np.array([(1 + x) ** (1 / 365) for x in longterm_growth])
    for sec, compound_ret_daily in zip(sec_list, longterm_growth_daily):
        df_port_secs[sec + "_longterm_imp_nav"] = df_port_secs['days_accrual'].cumsum().apply(lambda x: compound_ret_daily ** x)

    #### rebalance cycle
    df_port_secs["rebal_tag_" + rf] = df_port_secs["#"].apply(lambda x: True if x % rebal_freq == 0 else False)
    df_port_secs["rebal_tag_" + rf][-1] = True

    res = {}
    res[df_port_secs.index[0]] = {}
    for sec, w in zip(sec_list, start_weight):
        res[df_port_secs.index[0]][sec + "_val_before_rebal_" + rf] = w
        res[df_port_secs.index[0]][sec + "_rebal_weight_" + rf] = w
        res[df_port_secs.index[0]][sec + "_val_after_rebal_" + rf] = w

        res[df_port_secs.index[0]][sec + "_val_before_rebal_anchor_" + rf] = w
        res[df_port_secs.index[0]][sec + "_rebal_weight_anchor_" + rf] = w
        res[df_port_secs.index[0]][sec + "_val_after_rebal_anchor_" + rf] = w

    res[df_port_secs.index[0]]["port_val_before_rebal_" + rf] = sum(start_weight)
    res[df_port_secs.index[0]]["port_val_after_rebal_" + rf] = sum(start_weight)
    res[df_port_secs.index[0]]["port_val_before_rebal_anchor_" + rf] = sum(start_weight)
    res[df_port_secs.index[0]]["port_val_after_rebal_anchor_" + rf] = sum(start_weight)

    for dt, dtm1 in zip(df_port_secs.index[1:], df_port_secs.index[:-1]):
        res[dt] = {}
        for sec in sec_list:
            res[dt][sec + "_val_before_rebal_" + rf] = res[dtm1][sec + "_val_after_rebal_" + rf] * (1 + df_port_secs.loc[dt, sec + "_ret_1d"])
            res[dt][sec + "_val_before_rebal_anchor_" + rf] = res[dtm1][sec + "_val_after_rebal_anchor_" + rf] * (1 + df_port_secs.loc[dt, sec + "_ret_1d"])

        res[dt]["port_val_before_rebal_" + rf] = sum([res[dt][sec + "_val_before_rebal_" + rf] for sec in sec_list])
        res[dt]["port_val_before_rebal_anchor_" + rf] = sum([res[dt][sec + "_val_before_rebal_anchor_" + rf] for sec in sec_list])

        if df_port_secs.loc[dt, "rebal_tag_" + rf]:
            # rebalance trigged -> return to rebal anchor weight
            if rebal_anchor == "no rebalance":
                for sec, w in zip(sec_list, start_weight):
                    res[dt][sec + "_rebal_weight_" + rf] = df_port_secs.loc[dt, sec] * w / df_port_secs.loc[dt, "port_nav_no_rebal"]
            elif rebal_anchor == "fixed weight":
                for sec, w in zip(sec_list, start_weight):
                    res[dt][sec + "_rebal_weight_" + rf] = w
            elif rebal_anchor == "long term":
                longterm_nav = df_port_secs.loc[dt, [sec + "_longterm_imp_nav" for sec in sec_list]].dot(start_weight)
                for sec, w in zip(sec_list, start_weight):
                    res[dt][sec + "_rebal_weight_" + rf] = df_port_secs.loc[dt, sec + "_longterm_imp_nav"] * w / longterm_nav
            else:
                raise Exception("rebal_anchor support 'no rebalance','fixed weight','long term'")
            # anchor
            for sec in sec_list:
                res[dt][sec + '_rebal_weight_anchor_' + rf] = res[dt][sec + "_rebal_weight_" + rf]
        else:
            # between rebalance freq -> enhance rebalance if price moved much
            # mean reverse: when market increase 1%, lower position by 1% * rebal_ratio
            origin_weight = np.array([res[dt][sec + "_val_before_rebal_" + rf] / res[dt]["port_val_before_rebal_" + rf] for sec in sec_list])
            curr_ret = df_port_secs.loc[dt, [sec + "_ret_1d" for sec in sec_list]].values
            if curr_ret.max() - curr_ret.min() > rebal_enhance_hurdle:
                rebal_ratio_curr = rebal_enhance_ratio
            else:
                rebal_ratio_curr = 0
            new_weight = [x - rebal_ratio_curr * y for x, y in zip(origin_weight, curr_ret)]
            new_weight = [x / sum(new_weight) for x in new_weight]
            # limit to weight constraint
            new_weight = [max(w_min, min(x, w_max)) for x in new_weight]
            new_weight = [x / sum(new_weight) for x in new_weight]

            for sec, w in zip(sec_list, new_weight):
                res[dt][sec + "_rebal_weight_" + rf] = w
                res[dt][sec + "_rebal_weight_anchor_" + rf] = res[dt][sec + "_val_before_rebal_anchor_" + rf] / res[dt]["port_val_before_rebal_anchor_" + rf]

        for sec in sec_list:
            res[dt][sec + "_val_after_rebal_" + rf] = res[dt]["port_val_before_rebal_" + rf] * res[dt][sec + "_rebal_weight_" + rf]
            res[dt][sec + "_val_after_rebal_anchor_" + rf] = res[dt]["port_val_before_rebal_anchor_" + rf] * res[dt][sec + "_rebal_weight_anchor_" + rf]

        res[dt]["port_val_after_rebal_" + rf] = sum([res[dt][sec + "_val_after_rebal_" + rf] for sec in sec_list])
        res[dt]["port_val_after_rebal_anchor_" + rf] = sum([res[dt][sec + "_val_after_rebal_anchor_" + rf] for sec in sec_list])

    res = pd.DataFrame(res).T
    df_port_secs = df_port_secs.merge(res, left_index=True, right_index=True, how='left')
    df_port_secs["port_rebal_excess_ret_" + rf] = (df_port_secs["port_val_after_rebal_" + rf] / df_port_secs["port_val_after_rebal_" + rf].shift(1)) - \
                                                  (df_port_secs["port_val_after_rebal_anchor_" + rf] / df_port_secs["port_val_after_rebal_anchor_" + rf].shift(
                                                      1))
    df_port_secs["port_rebal_excess_ret_" + rf] = df_port_secs["port_rebal_excess_ret_" + rf].fillna(0)
    df_port_secs["port_rebal_excess_cumret_" + rf] = (df_port_secs["port_rebal_excess_ret_" + rf] + 1).cumprod()

    df_port_secs["port_rebal_anchor_excess_ret_" + rf] = (df_port_secs["port_val_after_rebal_anchor_" + rf] / df_port_secs[
        "port_val_after_rebal_anchor_" + rf].shift(1)) - \
                                                         (df_port_secs["port_nav_no_rebal"] / df_port_secs["port_nav_no_rebal"].shift(1))
    df_port_secs["port_rebal_anchor_excess_ret_" + rf] = df_port_secs["port_rebal_anchor_excess_ret_" + rf].fillna(0)
    df_port_secs["port_rebal_anchor_excess_cumret_" + rf] = (df_port_secs["port_rebal_anchor_excess_ret_" + rf] + 1).cumprod()

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.suptitle(
            'Portfolio Rebalance Analysis\n(rebalance freq:' + rf + ' | anchor:' + rebal_anchor + ' | hurdle:' + str(rebal_enhance_hurdle * 100) + '%)')

        for sec in sec_list[1:]:
            ax1.plot(df_port_secs.index, df_port_secs[sec], '-', alpha=0.2)
        ax1.plot(df_port_secs.index, df_port_secs["port_val_after_rebal_" + rf], '-', color="blueviolet", label='rebal enhance ' + rf, alpha=0.9)
        ax1.plot(df_port_secs.index, df_port_secs["port_val_after_rebal_anchor_" + rf], '-', color="orange", label='rebal anchor ' + rf, alpha=0.9)
        ax1.plot(df_port_secs.index, df_port_secs["port_nav_no_rebal"], '-', color="dimgrey", label='no rebal', alpha=0.7)
        ax1.plot(df_port_secs.index, df_port_secs["port_nav_fixed_weight"], '-', color="darkgreen", label='fixed weight', alpha=0.7)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_title('Portfolio NAV compare')
        ax1.grid(ls="--", alpha=0.5)

        ax2.plot(df_port_secs.index, df_port_secs["port_rebal_excess_cumret_" + rf] - 1, '-', label='rebal ' + rf + ' enhance-anchor', alpha=0.9)
        ax2.plot(df_port_secs.index, df_port_secs["port_rebal_anchor_excess_cumret_" + rf] - 1, '-', label='rebal ' + rf + ' anchor-no rebal', alpha=0.9)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_title('Cumulative excess return')
        ax2.grid(ls="--", alpha=0.5)

        ts = df_port_secs[sec_list[0] + "_rebal_weight_anchor_" + rf].copy()
        ax3.plot(df_port_secs.index, ts, '-', label='anchor', alpha=0.6)
        for sec in sec_list[1:]:
            ts += df_port_secs[sec + "_rebal_weight_anchor_" + rf]
            ax3.plot(df_port_secs.index, ts, '-', alpha=0.6)
        ax3.stackplot(df_port_secs.index, df_port_secs[[sec + "_rebal_weight_" + rf for sec in sec_list]].T, alpha=0.4, labels=["enhance"])
        ax3.set_title('Position Weights')
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    # performance stats
    best_sec = sec_list[list(longterm_growth).index(max(longterm_growth))]
    worst_sec = sec_list[list(longterm_growth).index(min(longterm_growth))]
    df_stats = {}
    df_stats['rebal enhance'] = perf_stats(
        (df_port_secs["port_val_after_rebal_" + rf] / df_port_secs["port_val_after_rebal_" + rf].shift(1) - 1).dropna(how='any'))
    df_stats['rebal anchor'] = perf_stats(
        (df_port_secs["port_val_after_rebal_anchor_" + rf] / df_port_secs["port_val_after_rebal_anchor_" + rf].shift(1) - 1).dropna(how='any'))
    df_stats['no rebal'] = perf_stats((df_port_secs["port_nav_no_rebal"] / df_port_secs["port_nav_no_rebal"].shift(1) - 1).dropna(how='any'))
    df_stats['enhance-anchor'] = perf_stats(df_port_secs["port_rebal_excess_ret_" + rf].dropna(how='any'))
    df_stats['anchor-no rebal'] = perf_stats(df_port_secs["port_rebal_anchor_excess_ret_" + rf].dropna(how='any'))
    df_stats['fixed weight'] = perf_stats(df_port_secs["port_ret_fixed_weight"].dropna(how='any'))
    df_stats['best sec'] = perf_stats(df_port_secs[best_sec + "_ret_1d"].dropna(how='any'))
    df_stats['worst sec'] = perf_stats(df_port_secs[worst_sec + "_ret_1d"].dropna(how='any'))
    df_stats = pd.DataFrame(df_stats)
    df_stats = df_stats[['rebal enhance', "rebal anchor", 'no rebal', 'enhance-anchor', 'anchor-no rebal',
                         'fixed weight', 'best sec', 'worst sec']]
    df_stats = df_stats.T
    return df_stats


def get_history_data(security_id, source='tushare', market=None):
    '''
    return daily ohlc data, some need adjustments
    '''
    if source == 'yahoo':
        df_raw = yf.Ticker(security_id).history(period="max")
        df_raw.columns = [x.lower() for x in df_raw.columns]
        if security_id == '600519.SS':  # 贵州茅台
            df_raw.loc[df_raw.index == '2018-10-29', ['open', 'high', 'low', 'close']] = [534.82] * 4  # 贵州茅台数据错误 (少一个跌停板)
            df_raw.loc[df_raw.index <= '2006-05-24', ['open', 'high', 'low', 'close']] *= 23.75 / 27.86  # 贵州茅台数据错误（红利复权错误）
    elif source == 'tushare':
        df_raw = tushare_get_history(security_id)
    elif source == 'akshare':
        df_raw = akshare_get_history(security_id, market=market, adjust='qfq')
    elif source == 'local':
        try:
            df_raw = tusharelocal_get_history(security_id.split(".")[0])
        except:
            df_raw = aksharelocal_get_history(security_id, market)
    else:
        raise Exception('Source ' + source + ' not supported!')
    df_raw = df_raw.sort_index(ascending=True)
    return df_raw


if __name__ == "__main__":
    df_raw = get_history_data("^GSPC", source='yahoo')  # sp500
    df_raw = get_history_data("^IXIC", source='yahoo')  # 纳斯达克
    df_raw = get_history_data("^N225", source='yahoo')  # 日经225
    df_raw = get_history_data("^FTSE", source='yahoo')  # 英国FTSE100指數
    df_raw = get_history_data("TSLA", source='yahoo')  # TSLA
    df_raw = get_history_data("000001.SS", source='yahoo')  # 上证综指
    df_raw = get_history_data("^HSI", source='yahoo')  # 恒生指数
    df_raw = get_history_data("399001.SZ", source='yahoo')  # 深证成指

    df_raw = get_history_data("sh000016", source='akshare', market='INDEX')  # 上证50
    df_raw = get_history_data("sh000300", source='akshare', market='INDEX')  # 沪深300

    df_raw = pd.read_excel('C:/Users/Dai/Desktop/investment/data/crypto/BTC.xlsx', 'BTC')  # BTC
    df_raw.columns = [x.lower() for x in df_raw.columns]
    df_raw.index = df_raw['Date']

    # basic stats
    df_basic_stats = timeseries_ret_distri_stats(ts_px=df_raw["close"], log_ret=False, plot=True, plot_max_freq=252)
    df_basic_stats = timeseries_ret_distri_stats_ndays(ts_px=df_raw["close"], ndays=[1, 5, 10, 20, 40, 60, 120], log_ret=False)
    df_basic_stats.to_clipboard()

    # ret distribution
    # fit_params = timeseries_fit_ret_distri(df_raw["close"], freq="Weekly", dis_type="norm", plot=True, bins=300)
    fit_params = timeseries_fit_ret_distri(df_raw["close"], freq=40, dis_type="norm", plot=True, bins=300)

    # tail stats
    df_tail = timeseries_tail_ana(df_raw["close"], ret_freq=5, tail_start=None, plot=True)

    df_tail_stat = pd.DataFrame()
    for i in range(1, 41, 1):
        df_tail_stat = pd.concat([df_tail_stat, timeseries_tail_ana(df_raw["close"], ret_freq=i, tail_start=None, plot=False).T], axis=0)
        print('tail ana ' + str(i) + 'd finished')

    # var stats
    df_var_stat = timeseries_var_ana(df_raw["close"], days=[1, 5, 10, 20, 40, 60, 120], var_level=[0.995, 0.99])
    df_var_stat.to_clipboard()

    # rolling moments
    timeseries_moment_rolling_ana(df_raw["close"], ret_freq=20, rolling_window=1000)

    # rebalance stats
    df_rebal_stats = timeseries_rebalance_ana(ts_px_series=df_raw["close"], rebal_freqs=[5, 20, 60],
                                              rebal_anchor="fixed weight", long_term_growth="implied",
                                              rebal_ratio=0.5, rebal_type="mean reverse", rebal_hurdle=0.0, start_posperc=0.5,
                                              riskfree_rate=0.03, plot=True)
    df_rebal_stats.to_clipboard()
