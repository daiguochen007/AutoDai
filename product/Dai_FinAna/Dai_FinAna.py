# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 00:04:49 2018

@author: Dai
"""

try:
    import os
    # import sys
    import DaiToolkit as tk
    import threading
    import pythoncom
    import datetime
    import tkinter as tkgui
    import tkinter.ttk
    import pandas as pd
except Exception as e:
    print("Import Error: " + str(e))
    ret = input("Press Any Key to Continue...")


def gui_download_func():
    """
    download sec fdmtl data and analyze
    """
    global TICKERS_dict
    seccode = entrySeccode.get(1.0, tkgui.END)[:-1]
    dir_path = entryPath.get(1.0, tkgui.END)[:-1]
    secname = TICKERS_dict[seccode] if seccode in list(TICKERS_dict.keys()) else ""

    def internal_func(seccode, secname, dir_path):
        """ get fdmtl data online, perf analysis ad save """
        pythoncom.CoInitialize()
        try:
            frame_sec_buttondwnfdmtl.config(state="disabled", text="基本面数据下载中...", bg="#66b3ff")
            frame_output_text.insert(tkgui.END, "----- 下载中: " + seccode + "|" + secname + " -----\n")
            tk.download_finfdtml_excel(seccode, secname, dir_path)
            frame_output_text.insert(tkgui.END, "----- 下载完成! " + seccode + "|" + secname + " -----\n")
        except Exception as e:
            frame_output_text.insert(tkgui.END, "----- Error! " + str(e) + " -----\n")
        frame_sec_buttondwnfdmtl.config(state="normal", text="下载基本面数据（网易财经）", bg="#e6f7ff")
        pythoncom.CoUninitialize()

    if secname == "":
        frame_output_text.insert(tkgui.END, "----- Error! [" + seccode + "] 不在本地证券列表中 -----\n")
    else:
        th = threading.Thread(target=internal_func, args=(seccode, secname, dir_path))
        th.setDaemon(True)  # 守护线程
        th.start()
    return


def download_history():
    """"""
    global TICKERS_dict
    seccode = entrySeccode.get(1.0, tkgui.END)[:-1]
    secname = TICKERS_dict[seccode] if seccode in list(TICKERS_dict.keys()) else ""

    def internal_func():
        frame_sec_buttondwnts.config(state="disabled", text="时间序列数据下载中...", bg="#66b3ff")
        frame_output_text.insert(tkgui.END, "----- 开始下载 [" + secname + "] 历史数据 -----\n")
        tk.tushare_get_history(seccode)
        frame_output_text.insert(tkgui.END, "----- [" + secname + "] 历史数据下载完成! -----\n")
        frame_sec_buttondwnts.config(state="normal", text="下载时间序列数据（Tushare）", bg="#e6f7ff")

    if secname == "":
        frame_output_text.insert(tkgui.END, "----- Error! [" + seccode + "] 不在本地证券列表中 -----\n")
    else:
        th = threading.Thread(target=internal_func)
        th.setDaemon(True)  # 守护线程
        th.start()


def download_history_batch():
    global TICKERS_dict
    exl_seclist_path = entryPath_batch.get(1.0, tkgui.END)[:-1]
    df_seclist = pd.read_excel(exl_seclist_path, "Sheet1", dtype=str)

    def internal_func():
        frame_sec_buttondwnts_batch.config(state="disabled", text="批量时间序列数据下载中...", bg="#66b3ff")
        for seccode in df_seclist["证券代码"].values:
            secname = TICKERS_dict[seccode] if seccode in list(TICKERS_dict.keys()) else ""
            frame_output_text.insert(tkgui.END, "----- 开始下载 [" + seccode + "|" + secname + "] 历史数据 -----\n")
            try:
                tk.tushare_get_history(seccode)
                frame_output_text.insert(tkgui.END, "----- [" + seccode + "|" + secname + "] 历史数据下载完成! -----\n")
            except:
                frame_output_text.insert(tkgui.END, "----- [" + seccode + "|" + secname + "] 下载出错! -----\n")
        frame_sec_buttondwnts_batch.config(state="normal", text="批量下载时间序列数据（Tushare）", bg="#e6f7ff")

    th = threading.Thread(target=internal_func)
    th.setDaemon(True)  # 守护线程
    th.start()


def gui_refresh_label(event):
    global TICKERS_dict
    seccode = entrySeccode.get(1.0, tkgui.END)[:-1]
    secname_loc = TICKERS_dict[seccode] if seccode in list(TICKERS_dict.keys()) else ""
    frame_sec_labelsecname.config(text=secname_loc)
    # print seccode
    return


def download_latest_seclist():
    global TICKERS_dict

    def internal_func():
        frame_output_text.insert(tkgui.END,
                                 "----- 开始下载 " + datetime.datetime.now().strftime("%Y%m%d") + " 证券列表 -----\n")
        pythoncom.CoInitialize()
        frame_allmkt_dwnseclist.config(state="disabled", text="最新股票列表下载中...", bg="#ffb066")
        global TICKERS_dict
        tk.tushare_getallsec_basics()
        TICKERS_dict = tk.tusharelocal_get_allsecnames()
        frame_allmkt_dwnseclist.config(state="normal", text="下载最新股票列表", bg="#fff2e6")
        pythoncom.CoUninitialize()
        frame_output_text.insert(tkgui.END,
                                 "----- " + datetime.datetime.now().strftime("%Y%m%d") + " 证券列表下载完成! -----\n")

    th = threading.Thread(target=internal_func)
    th.setDaemon(True)
    th.start()


def download_latest_profitdata():
    def internal_func():
        pythoncom.CoInitialize()
        frame_allmkt_dwnqtreps.config(state="disabled", text="最新季度盈利数据下载中...", bg="#ffb066")
        frame_output_text.insert(tkgui.END, "----- 开始下载季报盈利数据 -----\n")
        tk.tushare_get_nearestqtr_profitdata()
        frame_output_text.insert(tkgui.END, "----- 季报盈利数据下载完成! -----\n")
        frame_allmkt_dwnqtreps.config(state="normal", text="下载最新季度盈利数据", bg="#fff2e6")
        pythoncom.CoUninitialize()

    th = threading.Thread(target=internal_func)
    th.setDaemon(True)
    th.start()


def consolidate_tradingrecords():
    global df_traderecord

    def internal_func():
        global df_traderecord
        frame_output_text_tradesmy.insert(tkgui.END, "----- 开始加载交易记录 -----\n")
        pythoncom.CoInitialize()
        frame_tradesmy_btn_constrades.config(state="disabled", text='加载交割单', bg="#ffb066")
        tk.db_update_tradingrecord()
        frame_tradesmy_btn_constrades.config(state="normal", text='加载交割单', bg="#fff2e6")
        pythoncom.CoUninitialize()
        # load to global
        df_traderecord = tk.get_stocktraderecord(autoadj=True)
        frame_output_text_tradesmy.insert(tkgui.END, "----- 交易记录载入完成! -----\n")

    th = threading.Thread(target=internal_func)
    th.setDaemon(True)
    th.start()


def TradeSmyPlot_cashjournal():
    global df_traderecord

    def internal_func():
        global df_traderecord
        if df_traderecord is not None:
            frame_output_text_tradesmy.insert(tkgui.END, "----- 开始出入金分析 -----\n")
            frame_tradeana_btn_cashjnl.config(state="disabled", text='出入金分析', bg="#66b3ff")
            tk.TradeSmyPlot_cashjournal(df_traderecord)
            frame_tradeana_btn_cashjnl.config(state="normal", text='出入金分析', bg="#e6f7ff")
            frame_output_text_tradesmy.insert(tkgui.END, "----- 出入金分析完成! -----\n")
        else:
            frame_output_text_tradesmy.insert(tkgui.END, "----- [Error]交易记录未加载! -----\n")

    th = threading.Thread(target=internal_func)
    th.setDaemon(True)
    th.start()


def TradeSmyPlot_commission():
    global df_traderecord

    def internal_func():
        global df_traderecord
        if df_traderecord is not None:
            frame_output_text_tradesmy.insert(tkgui.END, "----- 开始手续费分析 -----\n")
            frame_tradeana_btn_commsmy.config(state="disabled", text='手续费分析', bg="#66b3ff")
            tk.TradeSmyPlot_commission(df_traderecord)
            frame_tradeana_btn_commsmy.config(state="normal", text='手续费分析', bg="#e6f7ff")
            frame_output_text_tradesmy.insert(tkgui.END, "----- 手续费分析完成! -----\n")
        else:
            frame_output_text_tradesmy.insert(tkgui.END, "----- [Error]交易记录未加载! -----\n")

    th = threading.Thread(target=internal_func)
    th.setDaemon(True)
    th.start()


def TradeSmyPlot_closepospnlLIFO():
    global df_traderecord

    def internal_func():
        global df_traderecord
        if df_traderecord is not None:
            frame_output_text_tradesmy.insert(tkgui.END, "----- 开始平仓pnl(LIFO)分析 -----\n")
            frame_tradeana_btn_pnlsmyLIFO.config(state="disabled", text='平仓pnl分析(LIFO)', bg="#66b3ff")
            df_match_res, err_df_sec = tk.match_all_trading_record(df_traderecord, "LIFO")
            tk.TradeSmyPlot_closepospnl(df_match_res, "LIFO")
            frame_tradeana_btn_pnlsmyLIFO.config(state="normal", text='平仓pnl分析(LIFO)', bg="#e6f7ff")
            frame_output_text_tradesmy.insert(tkgui.END, "----- 平仓pnl(LIFO)分析完成! -----\n")
        else:
            frame_output_text_tradesmy.insert(tkgui.END, "----- [Error]交易记录未加载! -----\n")

    th = threading.Thread(target=internal_func)
    th.setDaemon(True)
    th.start()


def TradeSmyPlot_closepospnlFIFO():
    global df_traderecord

    def internal_func():
        global df_traderecord
        if df_traderecord is not None:
            frame_output_text_tradesmy.insert(tkgui.END, "----- 开始平仓pnl(FIFO)分析 -----\n")
            frame_tradeana_btn_pnlsmyFIFO.config(state="disabled", text='平仓pnl分析(FIFO)', bg="#66b3ff")
            df_match_res, err_df_sec = tk.match_all_trading_record(df_traderecord, "FIFO")
            tk.TradeSmyPlot_closepospnl(df_match_res, "FIFO")
            frame_tradeana_btn_pnlsmyFIFO.config(state="normal", text='平仓pnl分析(FIFO)', bg="#e6f7ff")
            frame_output_text_tradesmy.insert(tkgui.END, "----- 平仓pnl(FIFO)分析完成! -----\n")
        else:
            frame_output_text_tradesmy.insert(tkgui.END, "----- [Error]交易记录未加载! -----\n")

    th = threading.Thread(target=internal_func)
    th.setDaemon(True)
    th.start()


if __name__ == "__main__":
    ########################################### main
    ##  Global Var:
    TICKERS_dict = tk.tusharelocal_get_allsecnames()
    secname = ""
    df_traderecord = None

    mainwindow = tkgui.Tk()
    mainwindow.title('股票交易辅助工具 v1.0')
    mainwindow.minsize(550, 250)
    mainwindow.resizable(False, False)
    mainwindow.bind("<Key>", gui_refresh_label)

    #### menu
    menubar = tkgui.Menu(mainwindow)
    funcmenu = tkgui.Menu(menubar, tearoff=0)
    funcmenu.add_command(label="加载交割单", command=consolidate_tradingrecords)
    funcmenu.add_separator()
    funcmenu.add_command(label="退出", command=mainwindow.quit)
    menubar.add_cascade(label="菜单", menu=funcmenu)

    #### mainwindow
    tabControl = tkinter.ttk.Notebook(mainwindow)
    tab_datadwnload = tkgui.Frame(tabControl)
    tabControl.add(tab_datadwnload, text='  数据下载  ')
    tab_tradesmytool = tkgui.Frame(tabControl)
    tabControl.add(tab_tradesmytool, text='  交易统计  ')
    tabControl.pack(expand=1, fill='both')

    ################################# data dwnload tab
    # === all mkt data dwnload
    frame_allmkt = tkgui.LabelFrame(tab_datadwnload, text='全市场数据', fg="#005ce6")
    frame_allmkt.grid(row=0, columnspan=7, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
    frame_allmkt_dwnseclist = tkgui.Button(frame_allmkt, text='下载最新股票列表', width=30, bg="#fff2e6", fg="#1a0d00",
                                           command=download_latest_seclist)
    frame_allmkt_dwnseclist.grid(row=0, column=0, padx=5, pady=2)
    frame_allmkt_dwnqtreps = tkgui.Button(frame_allmkt, text='下载最新季度盈利数据', width=30, bg="#fff2e6", fg="#1a0d00",
                                          command=download_latest_profitdata)
    frame_allmkt_dwnqtreps.grid(row=0, column=1, padx=5, pady=2)

    # === sec data dwnload
    frame_sec = tkgui.LabelFrame(tab_datadwnload, text='个股数据', fg="#005ce6")
    frame_sec.grid(row=2, columnspan=7, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)

    tkgui.Label(frame_sec, text='   股票代码：').grid(row=1, column=0, sticky='E', padx=5, pady=2)
    entrySeccode = tkgui.Text(frame_sec, height=1, width=50)
    entrySeccode.grid(row=1, column=1, sticky='W', padx=5, pady=2)

    tkgui.Label(frame_sec, text='   股票名称：').grid(row=2, column=0, sticky='E', padx=5, pady=2)
    frame_sec_labelsecname = tkgui.Label(frame_sec, text=secname, width=50, anchor="w")
    frame_sec_labelsecname.grid(row=2, column=1, sticky='W', padx=5, pady=2)

    tkgui.Label(frame_sec, text=' 储存文件夹：').grid(row=3, column=0, sticky='E', padx=5, pady=2)
    entryPath = tkgui.Text(frame_sec, height=1, width=50)
    entryPath.grid(row=3, column=1, sticky='W', padx=5, pady=2)

    dir_path = str(os.path.dirname(os.path.realpath(__file__)))
    entryPath.insert(tkgui.END, dir_path)

    frame_sec_buttondwnfdmtl = tkgui.Button(frame_sec, text='下载基本面数据（网易财经）', width=60, bg="#e6f7ff", fg="#00111a",
                                            command=gui_download_func)
    frame_sec_buttondwnfdmtl.grid(row=4, columnspan=7, padx=5, pady=2)
    frame_sec_buttondwnts = tkgui.Button(frame_sec, text='下载时间序列数据（Tushare）', width=60, bg="#e6f7ff", fg="#00111a",
                                         command=download_history)
    frame_sec_buttondwnts.grid(row=5, columnspan=7, padx=5, pady=2)

    frame_secbatch = tkgui.LabelFrame(tab_datadwnload, text='批量个股数据', fg="#005ce6")
    frame_secbatch.grid(row=5, columnspan=7, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
    tkgui.Label(frame_secbatch, text='读取证券列表：').grid(row=0, column=0, sticky='E', padx=5, pady=2)
    entryPath_batch = tkgui.Text(frame_secbatch, height=1, width=49)
    entryPath_batch.grid(row=0, column=1, sticky='W', padx=5, pady=2)
    entryPath_batch.insert(tkgui.END, dir_path + "/batch_seclist.xlsx")
    frame_sec_buttondwnts_batch = tkgui.Button(frame_secbatch, text='批量下载时间序列数据（Tushare）', width=60, bg="#e6f7ff",
                                               fg="#00111a", command=download_history_batch)
    frame_sec_buttondwnts_batch.grid(row=1, columnspan=7, padx=5, pady=2)

    # === output box
    frame_output = tkgui.LabelFrame(tab_datadwnload, text='运行输出', fg="#005ce6")
    frame_output.grid(row=0, column=9, columnspan=2, rowspan=8, sticky='NS', padx=5, pady=5)
    frame_output_text = tkgui.Text(frame_output, width=50)
    frame_output_text.grid(row=0, sticky='W', padx=5, pady=2)

    ################################# trade smy tab
    frame_tradesmy = tkgui.LabelFrame(tab_tradesmytool, text='交割单', fg="#005ce6")
    frame_tradesmy.grid(row=0, columnspan=7, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
    frame_tradesmy_btn_constrades = tkgui.Button(frame_tradesmy, text='加载交割单', width=63, bg="#fff2e6", fg="#1a0d00",
                                                 command=consolidate_tradingrecords)
    frame_tradesmy_btn_constrades.grid(row=0, column=0, padx=5, pady=2)

    frame_tradeana = tkgui.LabelFrame(tab_tradesmytool, text='证券交易分析', fg="#005ce6")
    frame_tradeana.grid(row=1, columnspan=7, sticky='W', padx=5, pady=5, ipadx=5, ipady=5)
    frame_tradeana_btn_cashjnl = tkgui.Button(frame_tradeana, text='出入金分析', width=63, bg="#e6f7ff", fg="#00111a",
                                              command=TradeSmyPlot_cashjournal)
    frame_tradeana_btn_cashjnl.grid(row=1, columnspan=2, padx=5, pady=2)
    frame_tradeana_btn_commsmy = tkgui.Button(frame_tradeana, text='手续费分析', width=63, bg="#e6f7ff", fg="#00111a",
                                              command=TradeSmyPlot_commission)
    frame_tradeana_btn_commsmy.grid(row=2, columnspan=2, padx=5, pady=2)
    frame_tradeana_btn_pnlsmyLIFO = tkgui.Button(frame_tradeana, text='平仓pnl分析(LIFO)', width=30, bg="#e6f7ff",
                                                 fg="#00111a", command=TradeSmyPlot_closepospnlLIFO)
    frame_tradeana_btn_pnlsmyLIFO.grid(row=3, column=0, padx=5, pady=2)
    frame_tradeana_btn_pnlsmyFIFO = tkgui.Button(frame_tradeana, text='平仓pnl分析(FIFO)', width=30, bg="#e6f7ff",
                                                 fg="#00111a", command=TradeSmyPlot_closepospnlFIFO)
    frame_tradeana_btn_pnlsmyFIFO.grid(row=3, column=1, padx=5, pady=2)

    # === output box
    frame_output_tradesmy = tkgui.LabelFrame(tab_tradesmytool, text='运行输出', fg="#005ce6")
    frame_output_tradesmy.grid(row=0, column=9, columnspan=2, rowspan=8, sticky='NS', padx=5, pady=5)
    frame_output_text_tradesmy = tkgui.Text(frame_output_tradesmy, width=50)
    frame_output_text_tradesmy.grid(row=0, sticky='W', padx=5, pady=2)

    mainwindow.config(menu=menubar)
    mainwindow.mainloop()
