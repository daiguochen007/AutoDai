# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:23:55 2019

注： 手续费 净佣金	 印花税	 过户费	 结算费	 其他费

手续费 = 净佣金 + 其他费
总佣金 = 净佣金 + 其他费 + 印花税 + 过户费 + 结算费

@author: Dai
"""

import os
import sys

sys.path.append(os.path.abspath("C:/Users/Dai/Desktop/investment/__py__"))
import toolkit as tk
import pandas as pd

record_folder = tk.ROOT_PATH + "/股票/个人研究/交割单/"
record_list = os.listdir(record_folder)
record_list = sorted([x for x in record_list if x[-4:] == ".xls"])

df_allrecord = pd.DataFrame()
for file_name in record_list:
    pass
    df_record = pd.read_excel(record_folder + file_name, file_name[:8])
    df_record["证券代码"] = [x if x != x else "0" * (6 - len(str(int(x)))) + str(int(x)) for x in df_record["证券代码"]]
    df_allrecord = pd.concat([df_allrecord, df_record], axis=0)
    print(file_name + " finished!")

excel_path = record_folder + "交割单汇总.xlsx"
# output
import xlsxwriter

workbook = xlsxwriter.Workbook(excel_path)
worksheet = workbook.add_worksheet("Data")
tk.xlsxwriter_dftoExcel(df_allrecord, worksheet, index=False, header=True, startcol=0, startrow=0)
workbook.close()

# format
print("Format excel...")
import win32com.client as win32

excel = win32.DispatchEx('Excel.Application')
excel.Visible = True
wb = excel.Workbooks.Open(excel_path, ReadOnly='False')

for worksheet in wb.Sheets:
    if worksheet.Name == "Data":
        tk.excel_tbl_bscfmt(worksheet, df_allrecord)
        # worksheet.Range("B2:B9").NumberFormat = "#,##0;[Red](#,##0)"
        worksheet.Activate()
        excel.ActiveWindow.Zoom = 80
        worksheet.Columns.AutoFit()
        window = wb.Windows(1)
        window.SplitRow = 1
        window.FreezePanes = True

wb.Worksheets("Data").Activate()
wb.Close(True)
