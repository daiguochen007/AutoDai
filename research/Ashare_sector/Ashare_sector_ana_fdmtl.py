# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:26:07 2018

check concentration of sectors

@author: Dai
"""

ROOT_PATH = "C:/Users/Dai/Desktop/investment"

import os
import sys

sys.path.append(os.path.abspath(ROOT_PATH + "/__py__/toolkit"))
import pandas as pd
from excel_toolkit import *

df_allsec = pd.read_excel(ROOT_PATH + "/股票/证监会行业分类/证监会行业分类2018三季度.xlsx", "Sheet1")
temp = df_allsec[['行业大类名称', '上市公司代码', '上市公司简称']].values
sector_map = {}
for x in temp:
    x[1] = "0" * (6 - len(str(x[1]))) + str(x[1])
    if x[0] in list(sector_map.keys()):
        sector_map[x[0]][x[1]] = x[2]
    else:
        sector_map[x[0]] = {x[1]: x[2]}

for sector, TICKERS_dict in list(sector_map.items()):
    pass
    # sector = u"交通运输、仓储和邮政业(G)"
    dir_path = ROOT_PATH + "/data/163_sec_fundamental"
    excel_outpath = ROOT_PATH + "/股票/证监会行业分类/行业集中度/" + sector + ".xlsx"
    #    if os.path.exists(excel_outpath):
    #        print u"["+sector+u"] File Exist / skipped!"
    #        continue

    df_totalasset = pd.DataFrame()
    df_totalequity = pd.DataFrame()

    for seccode, secname in list(TICKERS_dict.items()):
        pass
        excel_path = dir_path + "/" + seccode + ".xlsx"
        df = pd.read_excel(excel_path, "资产负债表")
        df.index = [x.strip() for x in df["报告日期"]]
        df["报告日期"] = secname
        df_totalasset = pd.concat([df_totalasset, df.loc[["资产总计(万元)"]]], axis=0)
        df_totalequity = pd.concat([df_totalequity, df.loc[["归属于母公司股东权益合计(万元)"]]], axis=0)
        print("Read sec [" + secname + "] finished!")

    df_totalasset.index = df_totalasset["报告日期"]
    df_totalequity.index = df_totalequity["报告日期"]

    del df_totalasset["报告日期"]
    del df_totalequity["报告日期"]
    df_totalasset_perc = df_totalasset.copy()
    df_totalequity_perc = df_totalequity.copy()

    topnasset = {}
    topnequity = {}
    topnasset_perc = {}
    topnequity_perc = {}

    for c in df_totalasset_perc.columns:
        df_totalasset_perc[c] = df_totalasset_perc[c] / df_totalasset_perc[c].sum()
        df_totalequity_perc[c] = df_totalequity_perc[c] / df_totalequity_perc[c].sum()

        topnasset_perc[c] = sorted([x for x in sorted(df_totalasset_perc[c]) if x == x])[::-1][:5]
        topnasset_perc[c] += [float("nan")] * (5 - len(topnasset_perc[c]))
        topnequity_perc[c] = sorted([x for x in sorted(df_totalequity_perc[c]) if x == x])[::-1][:5]
        topnequity_perc[c] += [float("nan")] * (5 - len(topnequity_perc[c]))

        topnasset[c] = sorted([x for x in sorted(df_totalasset[c]) if x == x])[::-1][:5]
        topnasset[c] += [float("nan")] * (5 - len(topnasset[c]))
        topnequity[c] = sorted([x for x in sorted(df_totalequity[c]) if x == x])[::-1][:5]
        topnequity[c] += [float("nan")] * (5 - len(topnequity[c]))

    topnasset = pd.DataFrame(topnasset, index=["Asset Top " + str(x) for x in range(1, 6)])
    topnequity = pd.DataFrame(topnequity, index=["Equity Top " + str(x) for x in range(1, 6)])
    topnasset_perc = pd.DataFrame(topnasset_perc, index=["Asset% Top " + str(x) for x in range(1, 6)])
    topnequity_perc = pd.DataFrame(topnequity_perc, index=["Equity% Top " + str(x) for x in range(1, 6)])


    def add_areagraph(workbook, worksheet, sheet_name='资产占比', ncols=5, title="", line=list(range(2, 5)), insert_cell="A1",
                      num_format="0.00%"):
        """
        """
        chart1 = workbook.add_chart({'type': 'area', 'subtype': 'percent_stacked'})
        for l in line:
            chart1.add_series({
                'name': '=' + sheet_name + '!$A$' + str(l),
                'categories': '=' + sheet_name + '!$B$1:$' + excel_colnum_str(ncols) + '$1',
                'values': '=' + sheet_name + '!$B$' + str(l) + ':$' + excel_colnum_str(ncols) + '$' + str(l),
            })
        chart1.set_title({'name': title})
        # chart1.set_x_axis({'reverse': True})
        chart1.set_y_axis({'num_format': num_format})
        chart1.set_legend({'position': 'bottom'})
        worksheet.insert_chart(insert_cell, chart1, {'x_offset': 0, 'y_offset': 0, 'x_scale': 3.1, 'y_scale': 1.5})


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
        worksheet.insert_chart(insert_cell, chart1, {'x_offset': 0, 'y_offset': 0, 'x_scale': 1.5, 'y_scale': 1.5})


    print(sector + " Output to excel...")
    # output
    excel_outpath = ROOT_PATH + "/股票/证监会行业分类/行业集中度/" + sector + ".xlsx"
    import xlsxwriter

    workbook = xlsxwriter.Workbook(excel_outpath)
    worksheet = workbook.add_worksheet("资产")
    xlsxwriter_dftoExcel(df_totalasset, worksheet, index=True, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("权益")
    xlsxwriter_dftoExcel(df_totalequity, worksheet, index=True, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("资产占比")
    xlsxwriter_dftoExcel(df_totalasset_perc, worksheet, index=True, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("权益占比")
    xlsxwriter_dftoExcel(df_totalequity_perc, worksheet, index=True, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("Top n")
    xlsxwriter_dftoExcel(topnasset, worksheet, index=True, header=True, startcol=0, startrow=0)
    cr = 5 + 3
    xlsxwriter_dftoExcel(topnasset_perc, worksheet, index=True, header=True, startcol=0, startrow=cr)
    cr += 5 + 3
    xlsxwriter_dftoExcel(topnequity, worksheet, index=True, header=True, startcol=0, startrow=cr)
    cr += 5 + 3
    xlsxwriter_dftoExcel(topnequity_perc, worksheet, index=True, header=True, startcol=0, startrow=cr)
    worksheet = workbook.add_worksheet("图表")
    add_areagraph(workbook, worksheet, sheet_name='资产占比', ncols=len(df_totalasset_perc.columns) + 1,
                  title=sector + "-资产占比", line=list(range(2, 2 + len(df_totalasset_perc))),
                  insert_cell="A1", num_format="0.0%")
    add_areagraph(workbook, worksheet, sheet_name='权益占比', ncols=len(df_totalequity_perc.columns) + 1,
                  title=sector + "-权益占比", line=list(range(2, 2 + len(df_totalequity_perc))),
                  insert_cell="A23", num_format="0.0%")
    add_linesgraph(workbook, worksheet, sheet_name='Top n', ncols=len(df_totalasset_perc.columns) + 1,
                   title=sector + "-Top 5 Asset", line=list(range(2, 7)), insert_cell="A45", num_format="#,##0")
    add_linesgraph(workbook, worksheet, sheet_name='Top n', ncols=len(df_totalasset_perc.columns) + 1,
                   title=sector + "-Top 5 Asset (%)", line=list(range(10, 15)), insert_cell="M45", num_format="0.00%")
    add_linesgraph(workbook, worksheet, sheet_name='Top n', ncols=len(df_totalasset_perc.columns) + 1,
                   title=sector + "-Top 5 Equity", line=list(range(18, 23)), insert_cell="A67", num_format="#,##0")
    add_linesgraph(workbook, worksheet, sheet_name='Top n', ncols=len(df_totalasset_perc.columns) + 1,
                   title=sector + "-Top 5 Equity (%)", line=list(range(26, 31)), insert_cell="M67", num_format="0.00%")

    workbook.close()

    # format
    print(sector + " Format excel...")
    import win32com.client as win32

    excel = win32.Dispatch('Excel.Application')
    excel.Visible = True
    wb = excel.Workbooks.Open(excel_outpath, ReadOnly='False')

    for worksheet in wb.Sheets:
        if worksheet.Name == "资产":
            excel_tbl_bscfmt(worksheet, df_totalasset, index=True)
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()
            window = wb.Windows(1)
            window.SplitRow = 1
            window.FreezePanes = True

        if worksheet.Name == "权益":
            excel_tbl_bscfmt(worksheet, df_totalequity, index=True)
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()
            window = wb.Windows(1)
            window.SplitRow = 1
            window.FreezePanes = True

        if worksheet.Name == "资产占比":
            excel_tbl_bscfmt(worksheet, df_totalasset_perc, index=True)
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()
            window = wb.Windows(1)
            window.SplitRow = 1
            window.FreezePanes = True

        if worksheet.Name == "权益占比":
            excel_tbl_bscfmt(worksheet, df_totalequity_perc, index=True)
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()
            window = wb.Windows(1)
            window.SplitRow = 1
            window.FreezePanes = True

        if worksheet.Name == "Top n":
            excel_tbl_bscfmt(worksheet, topnasset, index=True, startcol=0, startrow=0)
            cr = 5 + 3
            excel_tbl_bscfmt(worksheet, topnasset_perc, index=True, startcol=0, startrow=cr)
            cr += 5 + 3
            excel_tbl_bscfmt(worksheet, topnequity, index=True, startcol=0, startrow=cr)
            cr += 5 + 3
            excel_tbl_bscfmt(worksheet, topnequity_perc, index=True, startcol=0, startrow=cr)
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90

        if worksheet.Name == "图表":
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 70

    wb.Worksheets("图表").Activate()
    wb.Close(True)
