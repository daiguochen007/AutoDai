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

df_allsec = pd.read_excel(ROOT_PATH + "/股票/证监会行业分类/证监会行业分类2018三季度.xlsx", "Sheet1")
sectors = list(set(df_allsec['门类名称及代码'].values))
TICKERS_dict = df_allsec[['上市公司简称', '上市公司代码']].values
TICKERS_dict = {x[0]: "0" * (6 - len(str(x[1]))) + str(x[1]) for x in TICKERS_dict}


def add_linesgraph(workbook, worksheet, sheet_name='Top n', ncols=5, title="", line=list(range(2, 5)), insert_cell="A1",
                   num_format="0.00%"):
    """
    """
    chart1 = workbook.add_chart({'type': 'line', })
    for l in line:
        chart1.add_series({
            'name': "='" + sheet_name + "'!$A$" + str(l),
            'categories': "='" + sheet_name + "'!$B$1:$" + excel_colnum_str(ncols) + '$1',
            'values': "='" + sheet_name + "'!$B$" + str(l) + ':$' + excel_colnum_str(ncols) + '$' + str(l),
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
    # sector = u"居民服务、修理和其他服务业(O)"
    excel_path = ROOT_PATH + "/股票/证监会行业分类/行业集中度/证监会一级行业/" + sector + ".xlsx"
    df = pd.read_excel(excel_path, "资产占比")
    secname_list = list(df.iloc[:, -1].sort_values(ascending=False).index)

    excel_outpath = ROOT_PATH + "/股票/证监会行业分类/行业杜邦分析/" + sector + "_avg.xlsx"
    if os.path.exists(excel_outpath):
        print("[" + sector + "] File Exist / skipped!")
        continue

    sec_tbl = pd.DataFrame()
    for secname in secname_list:
        pass
        excel_path = ROOT_PATH + "/data/163_sec_fundamental/" + TICKERS_dict[secname] + ".xlsx"
        df_sec = pd.read_excel(excel_path, "杜邦分析")
        df_sec.index = [x.strip() for x in df_sec["报告日期"]]
        temp = df_sec.loc[["净资产收益率", "总资产净利润率", "归属母公司净利润占比", "权益乘数", "营业净利润率", "总资产周转率", "资产负债率"]].copy()
        del temp["报告日期"]
        sec_tbl = pd.concat([sec_tbl, temp], axis=0)
        print("[" + secname + "] loaded!")

    res_tbl = pd.DataFrame()
    for measure in ["净资产收益率", "总资产净利润率", "归属母公司净利润占比", "权益乘数", "营业净利润率", "总资产周转率", "资产负债率"]:
        res_tbl = pd.concat(
            [res_tbl, pd.DataFrame(sec_tbl[sec_tbl.index == measure].mean(axis=0), columns=[measure + "(avg)"]).T],
            axis=0)
        res_tbl = pd.concat(
            [res_tbl, pd.DataFrame(sec_tbl[sec_tbl.index == measure].std(axis=0), columns=[measure + "(std)"]).T],
            axis=0)
        res_tbl.loc[measure + "(avg+std)"] = res_tbl.loc[measure + "(avg)"] + res_tbl.loc[measure + "(std)"]
        res_tbl.loc[measure + "(avg-std)"] = res_tbl.loc[measure + "(avg)"] - res_tbl.loc[measure + "(std)"]

    # annual only
    res_tbl = res_tbl[[x for x in res_tbl.columns if "-" not in x or x.split("-")[1] == "12"]]

    print(sector + " Output to excel...")
    # output
    excel_outpath = ROOT_PATH + "/股票/证监会行业分类/行业杜邦分析/" + sector + "_avg.xlsx"
    import xlsxwriter

    workbook = xlsxwriter.Workbook(excel_outpath)
    worksheet = workbook.add_worksheet("Table")
    xlsxwriter_dftoExcel(res_tbl, worksheet, index=True, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("图表")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(res_tbl.columns) + 1, title="净资产收益率ROE",
                   line=[2, 2 + 2, 2 + 3], insert_cell="A23", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(res_tbl.columns) + 1, title="总资产收益率ROA",
                   line=[6, 6 + 2, 6 + 3], insert_cell="L1", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(res_tbl.columns) + 1, title="归母净利润占比",
                   line=[10, 10 + 2, 10 + 3], insert_cell="L23", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(res_tbl.columns) + 1, title="权益乘数",
                   line=[14, 14 + 2, 14 + 3], insert_cell="L45", num_format="0.00")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(res_tbl.columns) + 1, title="总资产周转率",
                   line=[22, 22 + 2, 22 + 3], insert_cell="W1", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(res_tbl.columns) + 1, title="营业净利润率",
                   line=[18, 18 + 2, 18 + 3], insert_cell="W23", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Table', ncols=len(res_tbl.columns) + 1, title="资产负债率",
                   line=[26, 26 + 2, 26 + 3], insert_cell="W45", num_format="0.00%")

    workbook.close()

    # format
    print(sector + " Format excel...")
    import win32com.client as win32

    excel = win32.Dispatch('Excel.Application')
    excel.Visible = True
    wb = excel.Workbooks.Open(excel_outpath, ReadOnly='False')

    for worksheet in wb.Sheets:
        if worksheet.Name == "Table":
            excel_tbl_bscfmt(worksheet, res_tbl, index=True)
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
