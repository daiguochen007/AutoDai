# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 22:14:47 2018

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
sectors = list(set(df_allsec['申万一级行业'].values))
TICKERS_dict = df_allsec[['名称', '代码']].values
TICKERS_dict = {x[0]: x[1] for x in TICKERS_dict}


def add_linesgraph(workbook, worksheet, sheet_name='Top n', ncols=5, title="", line=list(range(2, 5)), insert_cell="A1",
                   num_format="0.00%"):
    """
    """
    chart1 = workbook.add_chart({'type': 'line', })
    for l in line:
        chart1.add_series({
            'name': "='" + sheet_name + "'!$" + excel_colnum_str(ncols) + "$" + str(l),
            'categories': "='" + sheet_name + "'!$B$1:$" + excel_colnum_str(ncols - 1) + '$1',
            'values': "='" + sheet_name + "'!$B$" + str(l) + ':$' + excel_colnum_str(ncols - 1) + '$' + str(l),
            'marker': {'type': 'square', 'size': 5, 'fill': {'color': 'black'}},
            'line': {}
        })
    chart1.set_title({'name': title})
    # chart1.set_x_axis({'reverse': True})
    chart1.set_y_axis({'num_format': num_format})
    chart1.set_legend({'position': 'bottom'})
    worksheet.insert_chart(insert_cell, chart1, {'x_offset': 0, 'y_offset': 0, 'x_scale': 1.46, 'y_scale': 1.5})


for sector in sectors:
    pass
    secname_list = list(df_allsec.loc[df_allsec['申万一级行业'] == sector, "名称"].values)

    excel_outpath = ROOT_PATH + "/股票/申万行业分类/行业杜邦分析/" + sector + "_top5.xlsx"
    if os.path.exists(excel_outpath):
        print("[" + sector + "] File Exist / skipped!")
        continue

    sec_asset_tbl = pd.DataFrame()
    for secname in secname_list:
        pass
        excel_path = ROOT_PATH + "/data/163_sec_fundamental/" + TICKERS_dict[secname] + ".xlsx"
        df_sec = pd.read_excel(excel_path, "资产负债表")
        df_sec.index = [x.strip() for x in df_sec["报告日期"]]
        temp = df_sec.loc[["资产总计(万元)"]].copy()
        temp.index = [secname]
        del temp["报告日期"]
        sec_asset_tbl = pd.concat([sec_asset_tbl, temp], axis=0)
        print("[" + secname + "] loaded!")

    topn_secname_list = list(sec_asset_tbl.iloc[:, -1].sort_values(ascending=False)[:5].index)

    sec_tbl = pd.DataFrame()
    for secname in topn_secname_list:
        pass
        excel_path = ROOT_PATH + "/data/163_sec_fundamental/" + TICKERS_dict[secname] + ".xlsx"
        df_sec = pd.read_excel(excel_path, "杜邦分析")
        df_sec.index = [x.strip() for x in df_sec["报告日期"]]
        temp = df_sec.loc[["净资产收益率", "总资产净利润率", "归属母公司净利润占比", "权益乘数", "营业净利润率", "总资产周转率", "资产负债率"]].copy()
        temp["报告日期"] = secname
        sec_tbl = pd.concat([sec_tbl, temp], axis=0)
        print("[" + secname + "] loaded!")

    sec_tbl = sec_tbl[[x for x in sec_tbl.columns if "-" in x] + ["报告日期"]]
    num_sec = len(topn_secname_list)

    sec_tbl = sec_tbl.sort_index()
    # annual only
    sec_tbl = sec_tbl[[x for x in sec_tbl.columns if "-" not in x or x.split("-")[1] == "12"]]

    print(sector + " Output to excel...")
    # output
    excel_outpath = ROOT_PATH + "/股票/申万行业分类/行业杜邦分析/" + sector + "_top5.xlsx"
    import xlsxwriter

    workbook = xlsxwriter.Workbook(excel_outpath)
    worksheet = workbook.add_worksheet("Table")
    xlsxwriter_dftoExcel(sec_tbl, worksheet, index=True, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("图表")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(sec_tbl.columns) + 1, title="净资产收益率ROE",
                   line=list(range(2, 2 + num_sec)), insert_cell="A23", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(sec_tbl.columns) + 1, title="总资产收益率ROA",
                   line=list(range(2 + 2 * num_sec, 2 + 3 * num_sec)), insert_cell="L1", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(sec_tbl.columns) + 1, title="归母净利润占比",
                   line=list(range(2 + num_sec, 2 + 2 * num_sec)), insert_cell="L23", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(sec_tbl.columns) + 1, title="权益乘数",
                   line=list(range(2 + 4 * num_sec, 2 + 5 * num_sec)), insert_cell="L45", num_format="0.00")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(sec_tbl.columns) + 1, title="总资产周转率",
                   line=list(range(2 + 3 * num_sec, 2 + 4 * num_sec)), insert_cell="W1", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(sec_tbl.columns) + 1, title="营业净利润率",
                   line=list(range(2 + 5 * num_sec, 2 + 6 * num_sec)), insert_cell="W23", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(sec_tbl.columns) + 1, title="资产负债率",
                   line=list(range(2 + 6 * num_sec, 2 + 7 * num_sec)), insert_cell="W45", num_format="0.00%")

    workbook.close()

    # format
    print(sector + " Format excel...")
    import win32com.client as win32

    excel = win32.Dispatch('Excel.Application')
    excel.Visible = True
    wb = excel.Workbooks.Open(excel_outpath, ReadOnly='False')

    for worksheet in wb.Sheets:
        if worksheet.Name == "Table":
            excel_tbl_bscfmt(worksheet, sec_tbl, index=True)
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()
            window = wb.Windows(1)
            window.SplitRow = 1
            window.FreezePanes = True

        if worksheet.Name == "图表":
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 70

    wb.Close(True)
