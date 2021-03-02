# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 01:04:21 2018

@author: Dai
"""

ROOT_PATH = "C:/Users/Dai/Desktop/investment"

import os
import sys

sys.path.append(os.path.abspath(ROOT_PATH + "/__py__/toolkit"))
import pandas as pd
from excel_toolkit import *

df_allsec = pd.read_excel(ROOT_PATH + "/股票/申万行业分类/申万行业分类.xlsx", "Sheet1")
df_allsec["代码"] = ["0" * (6 - len(str(x))) + str(x) for x in df_allsec["代码"]]
sectors = list(set(df_allsec['申万三级行业'].values))

################################################################## avg summary

res = pd.DataFrame()
for sector in sectors:
    pass
    excel_path = ROOT_PATH + "/股票/申万行业分类/行业杜邦分析/申万三级行业avg/" + sector + "_avg.xlsx"
    df_sector = pd.read_excel(excel_path, "Table")
    df_sector = df_sector.loc[
        ["净资产收益率(avg)", "总资产净利润率(avg)", "归属母公司净利润占比(avg)", "权益乘数(avg)", "营业净利润率(avg)", "总资产周转率(avg)",
         "资产负债率(avg)"]].copy()
    cols = sorted(df_sector.columns)[-5:]
    temp = pd.DataFrame(df_sector[cols].mean(axis=1), columns=[sector]).T
    temp["# Years"] = len(cols)
    res = pd.concat([res, temp], axis=0)
    print("[" + sector + "] loaded!")

excel_outpath = ROOT_PATH + "/股票/申万行业分类/行业杜邦分析/申万三级行业 summary.xlsx"
res.to_excel(excel_outpath)

################################################################## median summary

res = pd.DataFrame()
for sector in sectors:
    pass
    excel_path = ROOT_PATH + "/股票/申万行业分类/行业杜邦分析/申万三级行业median/" + sector + "_median.xlsx"
    df_sector = pd.read_excel(excel_path, "Table")
    df_sector = df_sector.loc[
        ["净资产收益率(median)", "总资产净利润率(median)", "归属母公司净利润占比(median)", "权益乘数(median)", "营业净利润率(median)",
         "总资产周转率(median)", "资产负债率(median)"]].copy()
    cols = sorted(df_sector.columns)[-5:]
    temp = pd.DataFrame(df_sector[cols].mean(axis=1), columns=[sector]).T
    temp["# Years"] = len(cols)
    res = pd.concat([res, temp], axis=0)
    print("[" + sector + "] loaded!")

excel_outpath = ROOT_PATH + "/股票/申万行业分类/行业杜邦分析/申万三级行业 summary.xlsx"
res.to_excel(excel_outpath)

################################################################## top5 summary

res = pd.DataFrame()
for sector in sectors:
    pass
    excel_path = ROOT_PATH + "/股票/申万行业分类/行业杜邦分析/申万三级行业top5/" + sector + "_top5.xlsx"
    df = pd.read_excel(excel_path, "Table")
    df_sector = pd.DataFrame()
    for idx in ["净资产收益率", "总资产净利润率", "归属母公司净利润占比", "权益乘数", "营业净利润率", "总资产周转率", "资产负债率"]:
        pass
        df_sector = pd.concat([df_sector, pd.DataFrame(df.loc[[idx]].mean(axis=0), columns=[idx + "(avg)"]).T], axis=0)
    cols = sorted(df_sector.columns)[-5:]
    temp = pd.DataFrame(df_sector[cols].mean(axis=1), columns=[sector]).T
    temp["# Years"] = len(cols)
    res = pd.concat([res, temp], axis=0)
    print("[" + sector + "] loaded!")

excel_outpath = ROOT_PATH + "/股票/申万行业分类/行业杜邦分析/申万三级行业 summary.xlsx"
res.to_excel(excel_outpath)
