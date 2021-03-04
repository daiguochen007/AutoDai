"""
Analysis for timeseries data

1. basic stats (1-4 moments)
2. return distribution fit (t distriubtion)
3. tail analysis (power law alpha)
4. var/cvar analysis
5. rebalancing analysis (cash ~ asset)

To do:
jinja template for presentation purpose

"""
import numpy as np
import pandas as pd

import DaiToolkit as tk

##############################
# market simulation
##############################
# mkt type 1 (flat growth)
ts_px = np.arange(1, 10.01, 0.002)
ts_px = pd.Series(ts_px)

# mkt type 2 (flat growth + normal noise)
ts_px = np.arange(1, 10.01, 0.002)
ts_px = [x + np.random.normal(loc=0.0, scale=1.0) / 20 for x in ts_px]
ts_px = pd.Series(ts_px)

# mkt type 3 (geo random walk 1)
ts_px = np.random.normal(loc=0.005, scale=1.0, size=10000) / 50
ts_px = (10 ** ts_px).cumprod()
ts_px = pd.Series(ts_px)

# mkt type 4 (arith random walk 2)
ts_px = np.random.normal(loc=0.005, scale=1.0, size=10000) / 50
ts_px = 1 + ts_px.cumsum()
ts_px = pd.Series(ts_px)

# mkt type 5 (random jump)
ts_px = np.random.normal(loc=0.005, scale=1.0, size=10) / 20
ts_px = (10 ** ts_px).cumprod()
ts_px = [[x] * y for x, y in zip(ts_px, np.random.poisson(1000, size=10))]
ts_px = [y for x in ts_px for y in x]
ts_px = pd.Series(ts_px)

# set df: date index
ts_px = ts_px / ts_px[0]
ts_px = pd.DataFrame(ts_px, columns=['close'])
ts_px.index = pd.date_range(end='2021-03-01', periods=len(ts_px))

# mkt type 6 (real)
ts_px = tk.get_history_data("601318.SH", source='tushare')  # 中国平安
ts_px = tk.get_history_data("600519.SH", source='tushare')  # 贵州茅台
ts_px = tk.get_history_data("000651.SZ", source='tushare')  # 格力电器
ts_px = tk.get_history_data("600016.SH", source='tushare')  # 民生银行
ts_px = tk.get_history_data("600383.SH", source='tushare')  # 金地集团
ts_px = tk.get_history_data("600887.SH", source='tushare')  # 伊利股份
ts_px = tk.get_history_data("000002.SZ", source='tushare')  # 万科A
ts_px = tk.get_history_data("00700", source='local')        # 腾讯
ts_px = tk.get_history_data("00966", source='local')        # 中国太平

##############################
# analysis
##############################
ts_px['close'].plot(title='price graph')
# params = tk.timeseries_fit_ret_distri(ts_px, freq="Daily", dis_type="t", plot=True, bins=300)
df_stats = tk.timeseries_ret_distri_stats(ts_px=ts_px['close'], log_ret=True, plot=True, plot_max_freq=90)
# df_var_stat = tk.timeseries_var_ana(ts_px["close"], days=[1, 5, 10, 20, 30, 60, 252], var_level=[0.995, 0.99])
df_tail_stat = tk.timeseries_tail_ana(ts_px["close"], freqs=["Weekly"], tail_level=0.05, plot=True)

df_rebal = tk.timeseries_rebalance_ana(ts_px_series=ts_px['close'], rebal_freqs=[10],
                                        rebal_anchor="no rebalance", long_term_growth="implied",weight_bound=[-0.2,1.2],
                                        rebal_ratio=0.5, rebal_type="mean reverse", rebal_hurdle=0.0, start_posperc=0.5,
                                        riskfree_rate=0.03, plot=True)

# relationship between vol spread and rebalance enhance return
rebal_term_list = [5, 10, 20, 30, 60]
df_rebal_ana = {t: {"vol spread": tk.timeseries_volsig_diff(ts_px['close'], 1, t, True)} for t in rebal_term_list}
df = tk.timeseries_rebalance_ana(ts_px_series=ts_px['close'], rebal_freqs=rebal_term_list,
                                  rebal_anchor="fixed weight", long_term_growth="implied",weight_bound=[-0.2,1.2],
                                  rebal_ratio=0.5, rebal_type="mean reverse", rebal_hurdle=0.0, start_posperc=0.5,
                                  riskfree_rate=0.03, plot=False)
for t in rebal_term_list:
    df_rebal_ana[t]['rebal excess ret(annual)'] = df.loc["Enhance Excess Ret(" + str(t) + "d rebal)", "Annual return"]
df_rebal_ana = pd.DataFrame(df_rebal_ana).T
df_rebal_ana.plot.scatter("vol spread", 'rebal excess ret(annual)')