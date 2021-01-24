# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:55:34 2018

@author: Dai
"""

ROOT_PATH = "C:/Users/Dai/Desktop/investment"

import os
import sys

sys.path.append(os.path.abspath(ROOT_PATH + "/__py__/toolkit"))
import pandas as pd
from excel_toolkit import *

df_allsec = pd.read_excel(ROOT_PATH + "/股票/证监会行业分类/证监会行业分类2018三季度.xlsx", "Sheet1")
sectors = list(set(df_allsec['门类名称及代码'].values))

res = {k: pd.DataFrame() for k in ["Asset Top 1", "Asset% Top 1", "Equity Top 1", "Equity% Top 1"]}

for sector in sectors:
    pass
    # sector = u"房地产业"
    excel_path = ROOT_PATH + "/股票/证监会行业分类/行业集中度/证监会一级行业/" + sector + ".xlsx"
    df = pd.read_excel(excel_path, "Top n")

    temp = df.loc[["Asset Top 1"]].copy()
    temp.index = [sector]
    res["Asset Top 1"] = pd.concat([res["Asset Top 1"], temp], axis=0)

    temp = df.loc[["Asset% Top 1"]].copy()
    temp.index = [sector]
    res["Asset% Top 1"] = pd.concat([res["Asset% Top 1"], temp], axis=0)

    temp = df.loc[["Equity Top 1"]].copy()
    temp.index = [sector]
    res["Equity Top 1"] = pd.concat([res["Equity Top 1"], temp], axis=0)

    temp = df.loc[["Equity% Top 1"]].copy()
    temp.index = [sector]
    res["Equity% Top 1"] = pd.concat([res["Equity% Top 1"], temp], axis=0)

    print(sector + " finished!")

