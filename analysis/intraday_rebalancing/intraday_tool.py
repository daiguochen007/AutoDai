# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 22:43:44 2019

@author: Dai
"""

import DaiToolkit as tk

###### 整合交割单
tk.db_update_tradingrecord()

####### 交易rebalancing tool 1 (rough match)
# df_rebalance = tk.get_intraday_rebalancing_smy(15)

###### 交易rebalancing tool 2 (accurate match)
df_rebalance = tk.get_intraday_rebalancing_smy_accu(15, drop_amt_below=4000)
df_pair_unmatched_smy, df_pair_matched_smy = tk.rebalancing_match_algo(df_rebalance, commission=0.002,
                                                                       drop_amt_below=4000)

###### plot rebalancing covered/uncovered smy
tk.plot_rebal_smy(df_pair_matched_smy)
tk.plot_rebal_unmatched_smy(df_pair_unmatched_smy)

###### 获取历史某日持仓
df_histpos = tk.get_historypos_date("20201208", download_data=False)

###### 历史回报对比
df_perfsmy = tk.compare_perf("20200801", "20200922", highlight_seclist=["600383", "601318", "000069", "600919"],
                             download_data=True, benchmark_sec="601318")
