# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:23:55 2019

注： 手续费 净佣金	 印花税	 过户费	 结算费	 其他费

手续费 = 净佣金 + 其他费
总佣金 = 净佣金 + 其他费 + 印花税 + 过户费 + 结算费

@author: Dai
"""

from datetime import datetime

import numpy as np
import pandas as pd

import DaiToolkit as tk

###########################################################
## 
##   Trade Record Data
##
###########################################################
df_allrecord = tk.get_stocktraderecord(autoadj=True)
accounting_type = "FIFO"
df_match_res, err_df_sec = tk.match_all_trading_record(df_allrecord, accounting_type)

###########################################################
## 
##   Plots
##
###########################################################
# tk.TradeSmyPlot_commission(df_allrecord)
# tk.TradeSmyPlot_cashjournal(df_allrecord)
tk.TradeSmyPlot_closepospnl(df_match_res, accounting_type)

################## 持仓盈亏统计
df_currpos = tk.currpos_smy(df_match_res)
df_smy = tk.overall_pnlsmy(df_match_res)

### 期限统计表
df_curr_pos = df_match_res[(df_match_res["PnL"] != df_match_res["PnL"]) & (df_match_res["买入数量"] >= 1)].copy()
df_curr_pos["持仓期限(" + accounting_type + ")"] = [(datetime.today() - x).days for x in df_curr_pos["买入日期"]]
df_curr_pos["持仓期限(" + accounting_type + ")"] = ["超过1年" if x >= 365 else "1个月至1年" if x >= 31 else "1个月以内" for x in df_curr_pos["持仓期限(" + accounting_type + ")"]]
df_curr_pos["证券代码"] = df_curr_pos.index
df_curr_pos_termsmy = pd.pivot_table(df_curr_pos, index=["证券代码", "持仓期限(" + accounting_type + ")"],
                                     values=['买入金额', '买入佣金', '买入数量'], aggfunc=np.sum)
df_curr_pos_termsmy['买入价格'] = df_curr_pos_termsmy['买入金额'] / df_curr_pos_termsmy['买入数量']
print(">>> 未平仓持仓期限统计:")
print(df_curr_pos_termsmy)
