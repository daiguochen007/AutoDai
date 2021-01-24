# -*- coding: utf-8 -*-
"""
证券交易：
  A股： 
       手续费 净佣金 印花税 过户费 结算费 其他费
       印花税单边收取 0.1%
       手续费 = 净佣金 + 其他费
       总佣金 = 净佣金 + 其他费 + 印花税 + 过户费 + 结算费
       总佣金单边 买入 0.05% 卖出 0.15%
  
 沪港通：
       印花税	手续费	股份交收费	交易费	前台费	交易系统使用费 交易征费 其他费
       持有收取港股通组合费 0.8%% （香港结算）
       总佣金 = 印花税 + 手续费 + 股份交收费 + 交易费 + 前台费 + 交易系统使用费 + 交易征费 + 其他费
       总佣金单边 0.3% - 0.35%
       
Pending：
   合并交易记录
   
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools

from DaiToolkit import util_basics
from DaiToolkit import util_database
from DaiToolkit import util_portfolio
from DaiToolkit import util_tushare
from functools import reduce

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
import warnings

warnings.filterwarnings("ignore")


#####################################################################################
# source data process
#####################################################################################

def db_update_tradingrecord():
    """
    检索并将新的交割单存入数据库
    
    A股交易记录 + 沪港通交易记录
    """
    #######################################################################################
    # A shares
    #######################################################################################
    if not util_database.db_table_isexist("tradingrecord_ashares"):
        record_maxdate = 19900101
    else:
        record_maxdate = util_database.db_query("select max(交割日期) from tradingrecord_ashares").values[0][0]

    # load new files:
    record_folder = util_basics.PROJECT_DATA_PATH + "/trading_record/AShares/"
    record_list = os.listdir(record_folder)
    record_list = sorted([x for x in record_list if x[-4:] == ".xls"])
    record_list = [x for x in record_list if int(x[:-4]) >= int(record_maxdate)]

    df_allrecord_new = pd.DataFrame()
    for file_name in record_list:
        pass
        df_record = pd.read_excel(record_folder + file_name, file_name[:8])
        df_record["证券代码"] = [x if x != x else "0" * (6 - len(str(int(x)))) + str(int(x)) for x in df_record["证券代码"]]
        df_allrecord_new = pd.concat([df_allrecord_new, df_record], axis=0)
        print("加载数据：" + file_name)
    df_allrecord_new = df_allrecord_new[df_allrecord_new["交割日期"] > record_maxdate]
    if len(df_allrecord_new) > 0:
        util_database.db_table_update(df_allrecord_new, "tradingrecord_ashares", if_exists='append')
        print("A股交易记录导入数据库，交易日期[" + str(df_allrecord_new["交割日期"].min()) + " - " + str(
            df_allrecord_new["交割日期"].max()) + "]")
    else:
        print("无新A股交易记录导入")
    #
    record_mindate = util_database.db_query("select min(交割日期) from tradingrecord_ashares").values[0][0]
    record_maxdate = util_database.db_query("select max(交割日期) from tradingrecord_ashares").values[0][0]
    record_num = util_database.db_query("select count(交割日期) from tradingrecord_ashares").values[0][0]
    print("A股交易记录更新完成，数据库交易日期[" + str(record_mindate) + " - " + str(record_maxdate) + "]，总计记录" + str(
        record_num) + "条")

    #######################################################################################
    # HK shares
    #######################################################################################
    if not util_database.db_table_isexist("tradingrecord_hkshares"):
        record_maxdate = 19900101
    else:
        record_maxdate = util_database.db_query("select max(交收日期) from tradingrecord_hkshares").values[0][0]

    # load new files:
    record_folder = util_basics.PROJECT_DATA_PATH + "/trading_record/HKShares/"
    record_list = os.listdir(record_folder)
    record_list = sorted([x for x in record_list if x[-4:] == ".xls"])
    record_list = [x for x in record_list if int(x[:-4]) >= int(record_maxdate)]

    df_allrecord_new = pd.DataFrame()
    for file_name in record_list:
        pass
        df_record = pd.read_excel(record_folder + file_name, file_name[:8], 1)
        df_record["证券代码"] = [x if x != x else "0" * (5 - len(str(int(x)))) + str(int(x)) for x in df_record["证券代码"]]
        df_allrecord_new = pd.concat([df_allrecord_new, df_record], axis=0)
        print("加载数据：" + file_name)
    df_allrecord_new = df_allrecord_new[df_allrecord_new["交收日期"] > record_maxdate]
    if len(df_allrecord_new) > 0:
        util_database.db_table_update(df_allrecord_new, "tradingrecord_hkshares", if_exists='append')
        print("沪港通交易记录导入数据库，交收日期[" + str(df_allrecord_new["交收日期"].min()) + " - " + str(
            df_allrecord_new["交收日期"].max()) + "]")
    else:
        print("无新沪港通交易记录导入")
    #
    record_mindate = util_database.db_query("select min(交收日期) from tradingrecord_hkshares").values[0][0]
    record_maxdate = util_database.db_query("select max(交收日期) from tradingrecord_hkshares").values[0][0]
    record_num = util_database.db_query("select count(交收日期) from tradingrecord_hkshares").values[0][0]
    print("沪港通交易记录更新完成，数据库交收日期[" + str(record_mindate) + " - " + str(record_maxdate) + "]，总计记录" + str(
        record_num) + "条")


#####################################################################################
# data loader
#####################################################################################

def get_stocktraderecord(autoadj=True):
    """
    get stockrecord from local
    
    autoadj:
        新股
        转债转股
    """
    df_allrecord = util_database.db_query("SELECT * FROM tradingrecord_ashares")
    df_allrecord["总佣金"] = df_allrecord[["净佣金", "其他费", "印花税", "过户费", "结算费"]].sum(axis=1)
    df_allrecord["交割日期"] = pd.DatetimeIndex(list(map(str, df_allrecord["交割日期"])))

    if autoadj:
        df_newstock = get_newIPO_trades()
        df_cvbond = get_cbconv_trades()
        df_allrecord["业务类型Origin"] = df_allrecord["业务类型"]
        df_allrecord = dataprepare_adjIPO(df_allrecord, df_newstock)
        df_allrecord = dataprepare_adjconvts(df_allrecord, df_cvbond)

    return df_allrecord


def get_newIPO_trades():
    """
    只统计已入账的新股
    
    return sample: 
    证券代码	证券名称	业务类型	成交价格
    603505	金石资源	新股入帐	3.74
    """

    df_allrecord = get_stocktraderecord(autoadj=False)
    df_newstock = df_allrecord.loc[df_allrecord["业务类型"].isin(["新股入帐", "配售缴款"]), ["证券代码", "证券名称", "业务类型", "成交价格"]]
    df_newstock.index = list(range(len(df_newstock.index)))

    id_psjk = list(df_newstock[df_newstock["业务类型"] == "配售缴款"].index)
    id_xgrz = list(df_newstock[df_newstock["业务类型"] == "新股入帐"].index)
    id_psjk = id_psjk[:len(id_xgrz)]
    df_newstock_res = []
    for i, j in zip(id_psjk, id_xgrz):
        df_newstock_res.append({"证券代码": df_newstock.loc[j, "证券代码"],
                                "证券名称": df_newstock.loc[i, "证券名称"],
                                "业务类型": "新股入帐",
                                "成交价格": df_newstock.loc[i, "成交价格"]})
    df_newstock_res = pd.DataFrame(df_newstock_res)
    df_newstock_res["证券名称"] = df_newstock_res["证券名称"].fillna("")
    df_newstock_res = df_newstock_res[["证券代码", "证券名称", "业务类型", "成交价格"]]

    return df_newstock_res


def get_cbconv_trades():
    """
    return sample: 
    转债代码	转债名称	转股零款证券代码	转股代码	转股名称
    113001	中行转债	         191001	601988	中国银行

    """
    df_allrecord = get_stocktraderecord(autoadj=False)
    df_cbconv = df_allrecord.loc[
        df_allrecord["业务类型"].isin(["转股零款", "债券转股回售转出", "转股入帐"]), ["证券代码", "证券名称", "业务类型"]]
    df_cbconv.index = list(range(len(df_cbconv.index)))

    id_zglk = list(df_cbconv[df_cbconv["业务类型"] == "转股零款"].index)
    id_zzdm = list(df_cbconv[df_cbconv["业务类型"] == "债券转股回售转出"].index)
    id_zgdm = list(df_cbconv[df_cbconv["业务类型"] == "转股入帐"].index)

    df_cbconv_res = []
    for i, j, k in zip(id_zglk, id_zzdm, id_zgdm):
        df_cbconv_res.append({"转债代码": df_cbconv.loc[j, "证券代码"],
                              "转债名称": df_cbconv.loc[j, "证券名称"],
                              "转股零款证券代码": df_cbconv.loc[i, "证券代码"],
                              "转股零款证券名称": df_cbconv.loc[i, "证券名称"],
                              "转股代码": df_cbconv.loc[k, "证券代码"],
                              "转股名称": df_cbconv.loc[k, "证券名称"]
                              })

    df_cbconv_res = pd.DataFrame(df_cbconv_res)
    df_cbconv_res = df_cbconv_res[["转债代码", "转债名称", "转股零款证券代码", "转股零款证券名称", "转股代码", "转股名称"]]
    df_cbconv_res = df_cbconv_res.drop_duplicates("转债代码")
    return df_cbconv_res


#####################################################################################
# Intraday Tool
#####################################################################################

def get_intraday_rebalancing_smy(hist_days=7):
    """
    用于计算日内交易的rebalancing相对位置
    粗略match
    """
    ############################ load 交割单
    df_allrecord = get_stocktraderecord(autoadj=False)
    df_buysellrecord = df_allrecord[df_allrecord["业务类型"].isin(["证券买入", "证券卖出"])]

    date_set = sorted(list(set(df_buysellrecord["交割日期"])))

    df_matched_all = pd.DataFrame()
    for curr_date in date_set[::-1][:hist_days]:
        pass
        df_buysell_currdate = df_buysellrecord[df_buysellrecord["交割日期"] == curr_date]
        df_buy_currdate = df_buysell_currdate[df_buysell_currdate["业务类型"].isin(["证券买入"])].copy()
        df_buy_currdate.index = list(range(len(df_buy_currdate)))
        df_sell_currdate = df_buysell_currdate[df_buysell_currdate["业务类型"].isin(["证券卖出"])].copy()
        df_sell_currdate.index = list(range(len(df_sell_currdate)))

        df_buy_matched = pd.DataFrame()
        df_sell_matched = pd.DataFrame()

        while len(df_buy_currdate) > 0 or len(df_sell_currdate) > 0:
            if len(df_buy_currdate) > 0:
                buy_ind = df_buy_currdate.index[-1]
                df_curr_buy = df_buy_currdate.loc[[buy_ind], ["证券代码", "证券名称", "成交价格", "成交数量", "成交金额"]].copy()
                amt_buy = df_curr_buy["成交金额"].values[0]
                df_buy_currdate = df_buy_currdate.drop([buy_ind])
            else:
                df_curr_buy = pd.DataFrame(index=[0], columns=["证券代码", "证券名称", "成交价格", "成交数量", "成交金额"])
                amt_buy = 0

            if len(df_sell_currdate) > 0:
                sell_ind = df_sell_currdate.index[-1]
                df_curr_sell = df_sell_currdate.loc[[sell_ind], ["证券代码", "证券名称", "成交价格", "成交数量", "成交金额"]].copy()
                amt_sell = df_curr_sell["成交金额"].values[0]
                df_sell_currdate = df_sell_currdate.drop([sell_ind])
            else:
                df_curr_sell = pd.DataFrame(index=[0], columns=["证券代码", "证券名称", "成交价格", "成交数量", "成交金额"])
                df_curr_sell["成交金额"] = amt_buy
                amt_sell = 0

            if amt_buy == 0:
                df_curr_buy["成交金额"] = float("nan")  # amt_sell
                df_buy_matched = pd.concat([df_buy_matched, df_curr_buy])  # empty
                df_sell_matched = pd.concat([df_sell_matched, df_curr_sell], axis=0)
            elif float(amt_sell) / amt_buy < 1.2 and float(amt_sell) / amt_buy > 0.8:
                df_buy_matched = pd.concat([df_buy_matched, df_curr_buy], axis=0)
                df_sell_matched = pd.concat([df_sell_matched, df_curr_sell], axis=0)
            else:
                df_buy_matched = pd.concat([df_buy_matched, df_curr_buy], axis=0)
                temp = pd.DataFrame(index=[0], columns=["证券代码", "证券名称", "成交价格", "成交数量", "成交金额"])
                temp["成交金额"] = float("nan")  # amt_buy
                df_sell_matched = pd.concat([df_sell_matched, temp], axis=0)
                if amt_sell > 0:
                    df_sell_matched = pd.concat([df_sell_matched, df_curr_sell], axis=0)
                    temp = pd.DataFrame(index=[0], columns=["证券代码", "证券名称", "成交价格", "成交数量", "成交金额"])
                    temp["成交金额"] = float("nan")  # amt_sell
                    df_buy_matched = pd.concat([df_buy_matched, temp], axis=0)

        df_buy_matched.index = [curr_date] * len(df_buy_matched)
        df_sell_matched.index = [curr_date] * len(df_sell_matched)

        df_buy_matched.columns = ["（买入）" + x for x in df_buy_matched.columns]
        df_sell_matched.columns = ["（卖出）" + x for x in df_sell_matched.columns]

        df_matched = pd.concat([df_buy_matched, df_sell_matched], axis=1)
        df_matched_all = pd.concat([df_matched_all, df_matched], axis=0)
        print("Date [" + str(curr_date) + "] finished!")

    ############################ tushare download curr px
    df_matched_all["（买入）证券代码"] = ["0" * (6 - len(str(int(x)))) + str(int(x)) if x == x else x for x in df_matched_all["（买入）证券代码"]]
    df_matched_all["（卖出）证券代码"] = ["0" * (6 - len(str(int(x)))) + str(int(x)) if x == x else x for x in df_matched_all["（卖出）证券代码"]]

    seccode_list = list(
        set(list(df_matched_all.dropna(how="any")["（买入）证券代码"]) + list(df_matched_all.dropna(how="any")["（卖出）证券代码"])))
    price_dict = {}
    for k in seccode_list:
        price_dict[k] = float(util_tushare.ts.get_realtime_quotes(k)["price"][0])
        print(k + " 最新价格: " + str(price_dict[k]))

    df_matched_all["（买入）相对涨幅"] = list(map(
        lambda x, y: (price_dict[y] / float(x) - 1) * 100 if y in list(price_dict.keys()) else float("nan"),
        df_matched_all['（买入）成交价格'], df_matched_all["（买入）证券代码"]))
    df_matched_all["（卖出）相对涨幅"] = list(map(
        lambda x, y: (price_dict[y] / float(x) - 1) * 100 if y in list(price_dict.keys()) else float("nan"),
        df_matched_all['（卖出）成交价格'], df_matched_all["（卖出）证券代码"]))
    df_matched_all["净相对涨幅"] = df_matched_all["（买入）相对涨幅"] - df_matched_all["（卖出）相对涨幅"]

    df_matched_all["（买入）相对涨幅"] = [str(round(x, 2)) + "%" if x == x else x for x in df_matched_all["（买入）相对涨幅"]]
    df_matched_all["（卖出）相对涨幅"] = [str(round(x, 2)) + "%" if x == x else x for x in df_matched_all["（卖出）相对涨幅"]]
    df_matched_all["净相对涨幅"] = [str(round(x, 2)) + "%" if x == x else x for x in df_matched_all["净相对涨幅"]]

    return df_matched_all


def get_intraday_rebalancing_smy_accu(hist_days=7, drop_amt_below=0):
    """
    用于计算日内交易的rebalancing相对位置
    精确 match
    """
    ############################ load 交割单
    df_allrecord = get_stocktraderecord(autoadj=False)
    df_buysellrecord = df_allrecord[df_allrecord["业务类型"].isin(["证券买入", "证券卖出"])]

    date_set = sorted(list(set(df_buysellrecord["交割日期"])))

    curr_date_list = date_set[::-1][:hist_days]
    df_buysell_currdate = df_buysellrecord[df_buysellrecord["交割日期"].isin(curr_date_list)]
    df_buy_currdate = df_buysell_currdate[df_buysell_currdate["业务类型"].isin(["证券买入"])].copy()
    df_buy_currdate.index = list(range(len(df_buy_currdate)))
    df_sell_currdate = df_buysell_currdate[df_buysell_currdate["业务类型"].isin(["证券卖出"])].copy()
    df_sell_currdate.index = list(range(len(df_sell_currdate)))

    df_buy_matched = pd.DataFrame()
    df_sell_matched = pd.DataFrame()
    while len(df_buy_currdate) > 0 or len(df_sell_currdate) > 0:
        # check buy sell last record
        if len(df_buy_currdate) > 0:
            buy_ind = df_buy_currdate.index[-1]
            df_curr_buy = df_buy_currdate.loc[[buy_ind], ["交割日期", "证券代码", "证券名称", "成交价格", "成交数量", "成交金额"]].copy()
            amt_buy = df_curr_buy["成交金额"].values[0]
        else:
            df_curr_buy = pd.DataFrame(index=[0], columns=["交割日期", "证券代码", "证券名称", "成交价格", "成交数量", "成交金额"])
            amt_buy = 0
            buy_ind = None

        if len(df_sell_currdate) > 0:
            sell_ind = df_sell_currdate.index[-1]
            df_curr_sell = df_sell_currdate.loc[
                [sell_ind], ["交割日期", "证券代码", "证券名称", "成交价格", "成交数量", "成交金额"]].copy()
            amt_sell = df_curr_sell["成交金额"].values[0]
        else:
            df_curr_sell = pd.DataFrame(index=[0], columns=["交割日期", "证券代码", "证券名称", "成交价格", "成交数量", "成交金额"])
            df_curr_sell["成交金额"] = amt_buy
            amt_sell = 0
            sell_ind = None

        if amt_buy == 0:
            df_curr_buy["成交金额"] = float("nan")  # amt_sell
            df_buy_matched = pd.concat([df_buy_matched, df_curr_buy])  # empty
            df_sell_matched = pd.concat([df_sell_matched, df_curr_sell], axis=0)
            if buy_ind is not None:
                df_buy_currdate = df_buy_currdate.drop([buy_ind])
            if sell_ind is not None:
                df_sell_currdate = df_sell_currdate.drop([sell_ind])

        elif amt_sell == 0:
            df_curr_sell["成交金额"] = float("nan")  # amt_sell
            df_sell_matched = pd.concat([df_sell_matched, df_curr_sell])  # empty
            df_buy_matched = pd.concat([df_buy_matched, df_curr_buy], axis=0)
            if buy_ind is not None:
                df_buy_currdate = df_buy_currdate.drop([buy_ind])
            if sell_ind is not None:
                df_sell_currdate = df_sell_currdate.drop([sell_ind])

        elif float(amt_sell) == float(amt_buy):
            df_buy_matched = pd.concat([df_buy_matched, df_curr_buy], axis=0)
            df_sell_matched = pd.concat([df_sell_matched, df_curr_sell], axis=0)

            if buy_ind is not None:
                df_buy_currdate = df_buy_currdate.drop([buy_ind])
            if sell_ind is not None:
                df_sell_currdate = df_sell_currdate.drop([sell_ind])

        elif float(amt_buy) > float(amt_sell):
            factor = float(amt_buy) / float(amt_sell)
            df_curr_buy.loc[:, ["成交数量", "成交金额"]] = df_curr_buy.loc[:, ["成交数量", "成交金额"]] / factor
            df_buy_currdate.loc[[buy_ind], ["成交数量", "成交金额"]] *= (1.0 - 1.0 / factor)

            df_buy_matched = pd.concat([df_buy_matched, df_curr_buy], axis=0)
            df_sell_matched = pd.concat([df_sell_matched, df_curr_sell], axis=0)
            if sell_ind is not None:
                df_sell_currdate = df_sell_currdate.drop([sell_ind])

        elif float(amt_sell) > float(amt_buy):
            factor = float(amt_sell) / float(amt_buy)
            df_curr_sell.loc[:, ["成交数量", "成交金额"]] = df_curr_sell.loc[:, ["成交数量", "成交金额"]] / factor
            df_sell_currdate.loc[[sell_ind], ["成交数量", "成交金额"]] *= (1.0 - 1.0 / factor)

            df_buy_matched = pd.concat([df_buy_matched, df_curr_buy], axis=0)
            df_sell_matched = pd.concat([df_sell_matched, df_curr_sell], axis=0)
            if buy_ind is not None:
                df_buy_currdate = df_buy_currdate.drop([buy_ind])

    df_buy_matched.columns = ["（买入）" + x for x in df_buy_matched.columns]
    df_sell_matched.columns = ["（卖出）" + x for x in df_sell_matched.columns]
    df_buy_matched.index = list(range(len(df_buy_matched)))
    df_sell_matched.index = list(range(len(df_sell_matched)))
    df_matched_all = pd.concat([df_buy_matched, df_sell_matched], axis=1)

    ############################ tushare download curr px
    df_matched_all["（买入）证券代码"] = ["0" * (6 - len(str(int(x)))) + str(int(x)) if x == x else x for x in df_matched_all["（买入）证券代码"]]
    df_matched_all["（卖出）证券代码"] = ["0" * (6 - len(str(int(x)))) + str(int(x)) if x == x else x for x in df_matched_all["（卖出）证券代码"]]

    seccode_list = list(
        set(list(df_matched_all.dropna(how="any")["（买入）证券代码"]) + list(df_matched_all.dropna(how="any")["（卖出）证券代码"])))
    price_dict = {}
    for k in seccode_list:
        price_dict[k] = float(util_tushare.ts.get_realtime_quotes(k)["price"][0])
        print(k + " 最新价格: " + str(price_dict[k]))

    df_matched_all["（买入）相对涨幅"] = list(map(
        lambda x, y: (price_dict[y] / float(x) - 1) * 100 if y in list(price_dict.keys()) else float("nan"),
        df_matched_all['（买入）成交价格'], df_matched_all["（买入）证券代码"]))
    df_matched_all["（卖出）相对涨幅"] = list(map(
        lambda x, y: (price_dict[y] / float(x) - 1) * 100 if y in list(price_dict.keys()) else float("nan"),
        df_matched_all['（卖出）成交价格'], df_matched_all["（卖出）证券代码"]))
    df_matched_all["净相对涨幅"] = df_matched_all["（买入）相对涨幅"] - df_matched_all["（卖出）相对涨幅"]

    df_matched_all["（买入）相对涨幅"] = [str(round(x, 2)) + "%" if x == x else x for x in df_matched_all["（买入）相对涨幅"]]
    df_matched_all["（卖出）相对涨幅"] = [str(round(x, 2)) + "%" if x == x else x for x in df_matched_all["（卖出）相对涨幅"]]
    df_matched_all["净相对涨幅"] = [str(round(x, 2)) + "%" if x == x else x for x in df_matched_all["净相对涨幅"]]

    # drop amount below
    df_matched_all = df_matched_all[
        [True if x >= drop_amt_below or x != x else False for x in df_matched_all["（卖出）成交金额"]]]
    return df_matched_all


def rebalancing_match_algo(df_rebalance_, commission=0.002, drop_amt_below=0):
    """
    df_rebalance = get_intraday_rebalancing_smy_accu(hist_days=7,drop_amt_below=0)
    
    df_rebalance order: 从前往后，前面的时间戳更晚
    
    """
    print("分拆rebalancing股票对...")
    df_rebalance = df_rebalance_.copy()
    df_rebalance["pairkey"] = ["-".join([y if y == y else "NAN" for y in x]) for x in
                               df_rebalance[['（买入）证券代码', '（卖出）证券代码']].values]
    df_rebalance["pairkey_nodirection"] = ["-".join([y if y == y else "NAN" for y in sorted(x)]) for x in
                                           df_rebalance[['（买入）证券代码', '（卖出）证券代码']].values]

    df_pair_matched = pd.DataFrame()
    df_pair_unmatched = pd.DataFrame()
    for pair in set(df_rebalance["pairkey_nodirection"]):
        pass
        pair_reverse = "-".join([pair.split("-")[1], pair.split("-")[0]])
        if pair == pair_reverse:
            """买卖相同sec"""
            df_pair = df_rebalance[df_rebalance["pairkey"] == pair]
            df_pair_matched = pd.concat([df_pair_matched, df_pair], axis=0)
        else:
            """买卖不同sec"""
            df_pair = df_rebalance[df_rebalance["pairkey"] == pair]
            df_pair_reverse = df_rebalance[df_rebalance["pairkey"] == pair_reverse]

            while len(df_pair) > 0 or len(df_pair_reverse) > 0:
                if len(df_pair) == 0:
                    df_pair_unmatched = pd.concat([df_pair_unmatched, df_pair_reverse], axis=0)
                    df_pair_reverse = pd.DataFrame()

                elif len(df_pair_reverse) == 0:
                    df_pair_unmatched = pd.concat([df_pair_unmatched, df_pair], axis=0)
                    df_pair = pd.DataFrame()

                else:
                    pair_ind = df_pair['（买入）成交金额'].index[0]
                    pait_rev_ind = df_pair_reverse['（买入）成交金额'].index[0]

                    pair_amt = df_pair.loc[pair_ind, '（买入）成交金额']
                    pair_rev_amt = df_pair_reverse.loc[pait_rev_ind, '（买入）成交金额']

                    if pair_amt == pair_rev_amt:
                        if pair_ind <= pait_rev_ind:
                            df_pair_matched = pd.concat([df_pair_matched, df_pair.loc[[pair_ind], :]], axis=0)
                            df_pair_matched = pd.concat([df_pair_matched, df_pair_reverse.loc[[pait_rev_ind], :]],
                                                        axis=0)
                        else:
                            df_pair_matched = pd.concat([df_pair_matched, df_pair_reverse.loc[[pait_rev_ind], :]],
                                                        axis=0)
                            df_pair_matched = pd.concat([df_pair_matched, df_pair.loc[[pair_ind], :]], axis=0)

                        df_pair = df_pair.drop([pair_ind])
                        df_pair_reverse = df_pair_reverse.drop([pait_rev_ind])

                    elif pair_amt > pair_rev_amt:
                        # process
                        factor = float(pair_amt) / float(pair_rev_amt)
                        df_pair_m = df_pair.loc[[pair_ind], :].copy()
                        df_pair_m.loc[pair_ind, ['（买入）成交数量', '（买入）成交金额', '（卖出）成交数量', '（卖出）成交金额']] /= factor
                        df_pair.loc[pair_ind, ['（买入）成交数量', '（买入）成交金额', '（卖出）成交数量', '（卖出）成交金额']] *= (
                                    1 - 1.0 / factor)

                        if pair_ind <= pait_rev_ind:
                            df_pair_matched = pd.concat([df_pair_matched, df_pair_m], axis=0)
                            df_pair_matched = pd.concat([df_pair_matched, df_pair_reverse.loc[[pait_rev_ind], :]],
                                                        axis=0)
                            df_pair_reverse = df_pair_reverse.drop([pait_rev_ind])
                        else:
                            df_pair_matched = pd.concat([df_pair_matched, df_pair_reverse.loc[[pait_rev_ind], :]],
                                                        axis=0)
                            df_pair_reverse = df_pair_reverse.drop([pait_rev_ind])
                            df_pair_matched = pd.concat([df_pair_matched, df_pair_m], axis=0)

                    else:
                        # process
                        factor = float(pair_rev_amt) / float(pair_amt)
                        df_pair_revm = df_pair_reverse.loc[[pait_rev_ind], :].copy()
                        df_pair_revm.loc[pait_rev_ind, ['（买入）成交数量', '（买入）成交金额', '（卖出）成交数量', '（卖出）成交金额']] /= factor
                        df_pair_reverse.loc[pait_rev_ind, ['（买入）成交数量', '（买入）成交金额', '（卖出）成交数量', '（卖出）成交金额']] *= (
                                    1 - 1.0 / factor)

                        if pair_ind <= pait_rev_ind:
                            df_pair_matched = pd.concat([df_pair_matched, df_pair.loc[[pair_ind], :]], axis=0)
                            df_pair = df_pair.drop([pair_ind])
                            df_pair_matched = pd.concat([df_pair_matched, df_pair_revm], axis=0)
                        else:
                            df_pair_matched = pd.concat([df_pair_matched, df_pair_revm], axis=0)
                            df_pair_matched = pd.concat([df_pair_matched, df_pair.loc[[pair_ind], :]], axis=0)
                            df_pair = df_pair.drop([pair_ind])

    df_pair_unmatched = df_pair_unmatched.drop(['pairkey', 'pairkey_nodirection'], axis=1)
    df_pair_unmatched = df_pair_unmatched.sort_index(axis=0)
    df_pair_unmatched = df_pair_unmatched[df_pair_unmatched['（买入）成交金额'] >= drop_amt_below]
    df_pair_unmatched_smy = df_pair_unmatched.copy()
    df_pair_unmatched_smy['净相对涨幅(扣除手续费)'] = df_pair_unmatched_smy['净相对涨幅'].apply(
        lambda x: float(x.strip("%")) * 0.01 - commission)
    df_pair_unmatched_smy['调仓相对盈亏(扣除手续费)'] = df_pair_unmatched_smy['净相对涨幅(扣除手续费)'] * df_pair_unmatched_smy[
        '（买入）成交金额']

    ##### analysis / matched rebalancing
    print("开始计算matched股票对盈亏...")
    df_pair_matched["Wash"] = [True if x.split("-")[0] == x.split("-")[1] else False for x in df_pair_matched['pairkey']]
    df_pair_matched_nonwash = df_pair_matched[df_pair_matched["Wash"] == False]
    df_pair_matched_wash = df_pair_matched[df_pair_matched["Wash"] == True]
    df_pair_matched_smy = []
    for i in range(len(df_pair_matched_nonwash) / 2):
        temp = {}
        df_pair_curr = df_pair_matched_nonwash.iloc[[2 * i, 2 * i + 1], :]
        temp['套利证券名称'] = "-".join(sorted([df_pair_curr['（买入）证券名称'].values[0], df_pair_curr['（卖出）证券名称'].values[0]]))
        temp['成交金额'] = df_pair_curr['（买入）成交金额'].values[0]
        temp['套利涨幅'] = (df_pair_curr['（卖出）成交价格'].values[0] / df_pair_curr['（买入）成交价格'].values[1] - 1.0) - \
                        (df_pair_curr['（买入）成交价格'].values[0] / df_pair_curr['（卖出）成交价格'].values[1] - 1.0)
        temp['套利涨幅(扣除手续费)'] = temp['套利涨幅'] - commission
        temp['套利PnL'] = temp['套利涨幅'] * temp['成交金额']
        temp['套利PnL(扣除手续费)'] = temp['套利涨幅(扣除手续费)'] * temp['成交金额']
        temp['套利开仓日期'] = min(df_pair_curr[['（买入）交割日期', '（卖出）交割日期']].values.flatten())
        temp['套利平仓日期'] = max(df_pair_curr[['（买入）交割日期', '（卖出）交割日期']].values.flatten())
        df_pair_matched_smy.append(temp)

    for i in range(len(df_pair_matched_wash)):
        temp = {}
        df_pair_curr = df_pair_matched_wash.iloc[[i], :]
        temp['套利证券名称'] = "-".join(sorted([df_pair_curr['（买入）证券名称'].values[0], df_pair_curr['（卖出）证券名称'].values[0]]))
        temp['成交金额'] = df_pair_curr['（买入）成交金额'].values[0]
        temp['套利涨幅'] = (df_pair_curr['（卖出）成交价格'].values[0] / df_pair_curr['（买入）成交价格'].values[0] - 1.0)
        temp['套利涨幅(扣除手续费)'] = temp['套利涨幅'] - commission
        temp['套利PnL'] = temp['套利涨幅'] * temp['成交金额']
        temp['套利PnL(扣除手续费)'] = temp['套利涨幅(扣除手续费)'] * temp['成交金额']
        temp['套利开仓日期'] = min(df_pair_curr[['（买入）交割日期', '（卖出）交割日期']].values.flatten())
        temp['套利平仓日期'] = max(df_pair_curr[['（买入）交割日期', '（卖出）交割日期']].values.flatten())
        df_pair_matched_smy.append(temp)

    df_pair_matched_smy = pd.DataFrame(df_pair_matched_smy)
    df_pair_matched_smy = df_pair_matched_smy[['套利平仓日期', '套利开仓日期', '套利证券名称', '成交金额', '套利涨幅',
                                               '套利涨幅(扣除手续费)', '套利PnL', '套利PnL(扣除手续费)']]
    df_pair_matched_smy = df_pair_matched_smy.sort_values(by='套利平仓日期', ascending=False)
    df_pair_matched_smy['套利涨幅'] = [str(round(x * 100.0, 2)) + "%" if x == x else x for x in df_pair_matched_smy['套利涨幅']]
    df_pair_matched_smy['套利涨幅(扣除手续费)'] = [str(round(x * 100.0, 2)) + "%" if x == x else x for x in df_pair_matched_smy['套利涨幅(扣除手续费)']]
    df_pair_matched_smy.index = list(range(len(df_pair_matched_smy)))
    df_pair_matched_smy = df_pair_matched_smy[df_pair_matched_smy['成交金额'] >= drop_amt_below]

    return df_pair_unmatched_smy, df_pair_matched_smy


def plot_rebal_smy(df_pair_matched_smy):
    """
    df_pair_unmatched_smy, df_pair_matched_smy = tk.rebalancing_match_algo(df_rebalance,commission = 0.002,drop_amt_below=0)

    plot df_pair_matched_smy
    
    cols: 
        [u'套利平仓日期', u'套利开仓日期', u'套利证券名称', u'成交金额', 
         u'套利涨幅', u'套利涨幅(扣除手续费)', u'套利PnL', u'套利PnL(扣除手续费)']
    """
    df_rebal_plot = pd.pivot_table(df_pair_matched_smy, index='套利平仓日期', values=['成交金额', '套利PnL(扣除手续费)'],
                                   aggfunc=np.sum)
    df_rebal_plot["累计套利PnL(扣除手续费) "] = df_rebal_plot['套利PnL(扣除手续费)'].cumsum()
    df_rebal_plot["平均盈利% "] = df_rebal_plot['套利PnL(扣除手续费)'] / df_rebal_plot['成交金额'] * 100.0
    # df_rebal_plot.index = pd.DatetimeIndex([str(int(x)) for x in df_rebal_plot.index])
    start_date = min(df_rebal_plot.index).strftime("%Y-%m-%d")
    end_date = max(df_rebal_plot.index).strftime("%Y-%m-%d")

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plot covered pnl by timeseries
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(df_rebal_plot.index, df_rebal_plot["累计套利PnL(扣除手续费) "], marker="o", label="累计套利PnL(扣除手续费) ")
    ax2.bar(df_rebal_plot.index, df_rebal_plot['平均盈利% '], color="green", alpha=0.6, label='平均盈利% ')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('盈利(元)')
    ax2.set_ylabel('Return %')
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax1.grid(ls='--', alpha=0.6)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.35, -0.2))
    ax2.legend(loc='lower center', bbox_to_anchor=(0.6, -0.2))
    plt.title("Rebalance PnL Summary " + start_date + " to " + end_date)
    plt.show()

    # plot covered pnl by security
    df_pair_matched_smy_plt = df_pair_matched_smy.copy()
    df_pair_matched_smy_plt['（买入）证券名称'] = df_pair_matched_smy_plt['套利证券名称'].apply(lambda x: x.split("-")[0])
    df_pair_matched_smy_plt['（卖出）证券名称'] = df_pair_matched_smy_plt['套利证券名称'].apply(lambda x: x.split("-")[1])
    df_rebal_heatmap = pd.pivot_table(df_pair_matched_smy_plt, index='（买入）证券名称', columns='（卖出）证券名称',
                                      values='套利PnL(扣除手续费)', aggfunc=np.sum)
    df_rebal_heatmap_val = pd.pivot_table(df_pair_matched_smy_plt, index='（买入）证券名称', columns='（卖出）证券名称',
                                          values='成交金额', aggfunc=np.sum)
    df_rebal_heatmap_perc = df_rebal_heatmap / df_rebal_heatmap_val

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(df_rebal_heatmap.values, cmap='Blues_r', aspect='auto')
    cmap_colors = im.cmap(im.norm(df_rebal_heatmap.values))
    ax.set_xticks(np.arange(len(df_rebal_heatmap.columns)))
    ax.set_yticks(np.arange(len(df_rebal_heatmap.index)))
    ax.set_xticklabels(df_rebal_heatmap.columns)
    ax.set_yticklabels(df_rebal_heatmap.index)
    ax.set_xlabel("卖出证券", fontsize=14)
    ax.set_ylabel("买入证券", fontsize=14)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(df_rebal_heatmap.index)):
        for j in range(len(df_rebal_heatmap.columns)):
            if not pd.isna(df_rebal_heatmap.values[i, j]):
                color_bg = cmap_colors[i][j]
                gray = 0.2989 * color_bg[0] + 0.5870 * color_bg[1] + 0.1140 * color_bg[2]
                if gray <= 0.5:
                    color_text = "white"
                else:
                    color_text = "black"
                ax.text(j, i - 0.1, "{:,.0f}".format(df_rebal_heatmap.values[i, j]), ha="center", va="center",
                        color=color_text, fontsize=15)
                ax.text(j, i + 0.25, "{:,.2f}".format(df_rebal_heatmap_perc.values[i, j] * 100) + "%", ha="center",
                        va="center", color=color_text, fontsize=10)
    ax.set_title('已平仓调仓相对盈亏(扣除手续费) ' + start_date + " 至 " + end_date)
    fig.tight_layout()
    plt.show()


def plot_rebal_unmatched_smy(df_pair_unmatched_smy):
    """
    df_pair_unmatched,df_pair_matched_smy = tk.rebalancing_match_algo(df_rebalance,commission = 0.002,drop_amt_below=0)

    plot df_pair_matched_smy
    
    cols: 
        [u'（买入）交割日期', u'（买入）证券代码', u'（买入）证券名称', u'（买入）成交价格', u'（买入）成交数量',
           u'（买入）成交金额', u'（卖出）交割日期', u'（卖出）证券代码', u'（卖出）证券名称', u'（卖出）成交价格',
           u'（卖出）成交数量', u'（卖出）成交金额', u'（买入）相对涨幅', u'（卖出）相对涨幅', u'净相对涨幅',
           u'净相对涨幅(扣除手续费)', u'调仓相对盈亏(扣除手续费)']
    """
    df_rebal_plot = pd.pivot_table(df_pair_unmatched_smy, index='（买入）交割日期', values=['（买入）成交金额', '调仓相对盈亏(扣除手续费)'],
                                   aggfunc=np.sum)
    df_rebal_plot['累计调仓相对盈亏(扣除手续费)'] = df_rebal_plot['调仓相对盈亏(扣除手续费)'].cumsum()
    df_rebal_plot["调仓平均盈利% "] = df_rebal_plot['调仓相对盈亏(扣除手续费)'] / df_rebal_plot['（买入）成交金额'] * 100.0
    start_date = min(df_rebal_plot.index).strftime("%Y-%m-%d")
    end_date = max(df_rebal_plot.index).strftime("%Y-%m-%d")

    # plot uncovered pnl by timeseries
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(df_rebal_plot.index, df_rebal_plot['累计调仓相对盈亏(扣除手续费)'], marker="o", label="累计调仓相对盈亏(扣除手续费) ")
    ax2.bar(df_rebal_plot.index, df_rebal_plot['调仓平均盈利% '], color="green", alpha=0.6, label='调仓平均盈利% ')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('盈利(元)')
    ax2.set_ylabel('Return %')
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax1.grid(ls='--', alpha=0.6)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.35, -0.2))
    ax2.legend(loc='lower center', bbox_to_anchor=(0.6, -0.2))
    plt.title("Uncovered Rebalance PnL Summary " + start_date + " to " + end_date)
    plt.show()

    # plot uncovered pnl by security
    df_rebal_heatmap = pd.pivot_table(df_pair_unmatched_smy, index='（买入）证券名称', columns='（卖出）证券名称',
                                      values='调仓相对盈亏(扣除手续费)', aggfunc=np.sum)
    df_rebal_heatmap_val = pd.pivot_table(df_pair_unmatched_smy, index='（买入）证券名称', columns='（卖出）证券名称',
                                          values='（买入）成交金额', aggfunc=np.sum)
    df_rebal_heatmap_perc = df_rebal_heatmap / df_rebal_heatmap_val

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(df_rebal_heatmap.values, cmap='Blues_r', aspect='auto')
    cmap_colors = im.cmap(im.norm(df_rebal_heatmap.values))
    ax.set_xticks(np.arange(len(df_rebal_heatmap.columns)))
    ax.set_yticks(np.arange(len(df_rebal_heatmap.index)))
    ax.set_xticklabels(df_rebal_heatmap.columns)
    ax.set_yticklabels(df_rebal_heatmap.index)
    ax.set_xlabel("卖出证券", fontsize=14)
    ax.set_ylabel("买入证券", fontsize=14)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(df_rebal_heatmap.index)):
        for j in range(len(df_rebal_heatmap.columns)):
            if not pd.isna(df_rebal_heatmap.values[i, j]):
                color_bg = cmap_colors[i][j]
                gray = 0.2989 * color_bg[0] + 0.5870 * color_bg[1] + 0.1140 * color_bg[2]
                if gray <= 0.5:
                    color_text = "white"
                else:
                    color_text = "black"
                ax.text(j, i - 0.1, "{:,.0f}".format(df_rebal_heatmap.values[i, j]), ha="center", va="center",
                        color=color_text, fontsize=15)
                ax.text(j, i + 0.25, "{:,.2f}".format(df_rebal_heatmap_perc.values[i, j] * 100) + "%", ha="center",
                        va="center", color=color_text, fontsize=10)
    ax.set_title('未平仓调仓相对盈亏(扣除手续费) ' + start_date + " 至 " + end_date)
    fig.tight_layout()
    plt.show()


#####################################################################################
# Plot trading summary (Plotly)
#####################################################################################

def TradeSmyPlot_commission(df_allrecord):
    """总佣金统计"""
    df_commission = pd.pivot_table(df_allrecord[df_allrecord["总佣金"] != 0], index="交割日期",
                                   values=["总佣金", "印花税", "过户费", "结算费"], aggfunc=np.sum)
    df_commission["佣金"] = df_commission["总佣金"] - df_commission[["印花税", "过户费", "结算费"]].sum(axis=1)
    temp = pd.pivot_table(df_allrecord[df_allrecord["业务类型"].isin(["证券买入", "证券卖出"])], index="交割日期", columns="业务类型",
                          values="成交金额", aggfunc=np.sum)
    temp.columns = [x + "金额" for x in temp.columns]
    df_commission = pd.concat([temp, df_commission], axis=1)
    df_commission = df_commission.fillna(0)
    df_commission["印花税费率"] = df_commission["印花税"] / df_commission["证券卖出金额"]
    df_commission["佣金费率"] = df_commission["佣金"] / (df_commission["证券卖出金额"] + df_commission["证券买入金额"])
    df_commission["过户费费率"] = df_commission["过户费"] / (df_commission["证券卖出金额"] + df_commission["证券买入金额"])
    for c in ["总佣金", "佣金", "印花税", "过户费", "结算费"]:
        df_commission["累计" + c] = df_commission[c].cumsum()
    # 佣金图表 plotly
    x = df_commission.index
    trace0 = []
    trace0.append(dict(x=x,
                       y=df_commission["累计结算费"] + df_commission["累计过户费"] + df_commission["累计印花税"] + df_commission[
                           "累计佣金"], name='累计结算费', mode='lines', fill='tonexty'))
    trace0.append(dict(x=x, y=df_commission["累计过户费"] + df_commission["累计印花税"] + df_commission["累计佣金"], name='累计过户费',
                       mode='lines', fill='tonexty'))
    trace0.append(
        dict(x=x, y=df_commission["累计印花税"] + df_commission["累计佣金"], name='累计印花税', mode='lines', fill='tonexty'))
    trace0.append(dict(x=x, y=df_commission["累计佣金"], name='累计佣金', mode='lines', fill='tozeroy'))
    trace1 = go.Bar(x=x, y=df_commission["总佣金"], name='当日总佣金',
                    marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=1.5), ), opacity=0.8)
    trace2 = []
    trace2.append(go.Scatter(x=x, y=df_commission["佣金费率"], name='佣金费率', mode='lines+markers'))
    trace2.append(go.Scatter(x=x, y=df_commission["印花税费率"], name='印花税费率', mode='lines+markers'))
    trace2.append(go.Scatter(x=x, y=df_commission["过户费费率"], name='过户费费率', mode='lines+markers'))
    fig = tools.make_subplots(rows=4, cols=1, specs=[[{'rowspan': 2}], [{}], [{}], [{}]], shared_xaxes=True,
                              subplot_titles=('累计手续费统计', '', '当日手续费统计', '费率统计'))
    for trace in trace0:
        fig.append_trace(trace, 1, 1)
    fig.append_trace(trace1, 3, 1)
    for trace in trace2:
        fig.append_trace(trace, 4, 1)
    fig['layout'].update(
        title="证券交易佣金统计" + df_commission.index[0].strftime("%Y/%m/%d") + " - " + df_commission.index[-1].strftime(
            "%Y/%m/%d"))
    py.plot(fig, filename=util_basics.PROJECT_ROOT_PATH + "/股票/个人研究/收益计算/证券交易佣金统计.html")


def TradeSmyPlot_cashjournal(df_allrecord):
    """资金转入转出"""
    df_cashjournel = pd.pivot_table(df_allrecord[df_allrecord["业务类型"].isin(["银行转证券", "证券转银行"])], index="交割日期",
                                    values=["发生金额"], aggfunc=np.sum)
    df_cashjournel.columns = ["银证转账"]
    df_cashjournel["累计银证转账"] = df_cashjournel["银证转账"].cumsum()
    # 资金转入转出图表 plotly
    x = df_cashjournel.index
    trace0 = go.Scatter(x=x, y=df_cashjournel["累计银证转账"], name='累计银证转账', mode='lines+markers', fill='tonexty')
    trace1 = go.Bar(x=x, y=df_cashjournel["银证转账"], name='银证转账',
                    marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=1.5), ), opacity=0.8)
    fig = tools.make_subplots(rows=3, cols=1, specs=[[{'rowspan': 2}], [{}], [{}]], shared_xaxes=True,
                              subplot_titles=('累计出入金统计', '', '当日出入金统计'))
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 3, 1)
    fig['layout'].update(
        title="证券交易入金统计" + df_cashjournel.index[0].strftime("%Y/%m/%d") + " - " + df_cashjournel.index[-1].strftime(
            "%Y/%m/%d"))
    py.plot(fig, filename=util_basics.PROJECT_ROOT_PATH + "/股票/个人研究/收益计算/证券交易出入金统计.html")


#####################################################################################
# Trading record adjustment
#####################################################################################


def dataprepare_adjIPO(df_allrecord, df_newstock):
    """
    新股 stockrecord 调节
    """
    df_newstock_pxdict = {x: y for x, y in zip(df_newstock["证券代码"], df_newstock["成交价格"])}
    sec_IPOlist = list(set(df_allrecord.loc[df_allrecord["业务类型"].isin(["新股入帐"]), "证券代码"].values))
    print("（根据本地新股信息）调整" + str(len(sec_IPOlist)) + "只新股买入价格，override为证券买入...")
    for k in sec_IPOlist:
        v = df_newstock_pxdict[k] if k in list(df_newstock_pxdict.keys()) else 0
        df_allrecord.loc[(df_allrecord["业务类型"].isin(["新股入帐"])) & (df_allrecord["证券代码"] == k), "成交价格"] = v
        df_allrecord.loc[(df_allrecord["业务类型"].isin(["新股入帐"])) & (df_allrecord["证券代码"] == k), "成交金额"] = \
        df_allrecord.loc[(df_allrecord["业务类型"].isin(["新股入帐"])) & (df_allrecord["证券代码"] == k), "成交价格"] * \
        df_allrecord.loc[(df_allrecord["业务类型"].isin(["新股入帐"])) & (df_allrecord["证券代码"] == k), "成交数量"]
        df_allrecord.loc[(df_allrecord["业务类型"].isin(["新股入帐"])) & (df_allrecord["证券代码"] == k), "发生金额"] = \
        df_allrecord.loc[(df_allrecord["业务类型"].isin(["新股入帐"])) & (df_allrecord["证券代码"] == k), "成交金额"]
        df_allrecord.loc[(df_allrecord["业务类型"].isin(["新股入帐"])) & (df_allrecord["证券代码"] == k), "业务类型"] = "证券买入"
        if v == 0:
            print("[warning] 新股 " + k + " 买入无本地价格 - default成本为0")
    return df_allrecord


def dataprepare_adjconvts(df_allrecord, df_cvbond):
    """
    可转债 stockrecord 调节 (用LIFO match转换债券)
    """
    print("（根据本地可转债转股信息）调整" + str(len(df_cvbond)) + "只可转债转股，override为可转债卖出+证券买入...")
    for curr_date in sorted(list(set(df_allrecord["交割日期"]))):
        df_curr_date = df_allrecord[df_allrecord["交割日期"] == curr_date]
        cb_dict = df_curr_date.loc[df_curr_date["业务类型"].isin(["债券转股回售转出"]), ["证券代码", "成交数量"]].to_dict()
        for k, v in list(cb_dict["证券代码"].items()):
            # 债券转股回售转出 -> 证券卖出
            temp = df_allrecord[
                (df_allrecord.index <= df_curr_date.index[-1]) & df_allrecord["业务类型"].isin(["证券买入"]) & (
                            df_allrecord["证券代码"] == v)].copy()
            factor = util_basics.waterfall_allocation(list(temp["成交数量"])[::-1], cb_dict["成交数量"][k])[::-1] / temp[
                "成交数量"]
            for c in ['成交数量', '成交金额', '手续费', '净佣金', '印花税', '过户费', '结算费', '其他费', '发生金额', '证券数量', '总佣金']:
                temp[c] *= factor
            df_allrecord.loc[k, "成交金额"] = temp["成交金额"].sum()
            df_allrecord.loc[k, "成交价格"] = float(df_allrecord.loc[k, "成交金额"]) / df_allrecord.loc[k, "成交数量"]
            df_allrecord.loc[k, ['手续费', '净佣金', '印花税', '过户费', '结算费', '其他费', '发生金额', '总佣金']] = -temp[
                ['手续费', '净佣金', '印花税', '过户费', '结算费', '其他费', '发生金额', '总佣金']].sum(axis=0)
            df_allrecord.loc[k, "业务类型"] = "证券卖出"

            # 转股入账 -> 证券买入
            cash_zglk = df_curr_date.loc[
                df_curr_date["证券代码"] == df_cvbond.loc[df_cvbond["转债代码"] == v, "转股零款证券代码"].values[0], "成交金额"]
            cash_zglk = cash_zglk.values[0] if len(cash_zglk) > 0 else 0
            df_buyoverride = df_curr_date[
                (df_curr_date["证券代码"] == df_cvbond.loc[df_cvbond["转债代码"] == v, "转股代码"].values[0]) & (
                            df_curr_date["业务类型"] == "转股入帐")]
            k_buy = df_buyoverride.index.values[0]
            df_allrecord.loc[k_buy, "成交金额"] = temp["成交金额"].sum() - cash_zglk
            df_allrecord.loc[k_buy, "成交价格"] = float(df_allrecord.loc[k_buy, "成交金额"]) / df_allrecord.loc[
                k_buy, "成交数量"]
            df_allrecord.loc[k_buy, ['手续费', '净佣金', '印花税', '过户费', '结算费', '其他费', '发生金额', '总佣金']] = temp[
                ['手续费', '净佣金', '印花税', '过户费', '结算费', '其他费', '发生金额', '总佣金']].sum(axis=0)
            df_allrecord.loc[k_buy, "业务类型"] = "证券买入"
    return df_allrecord


def trading_record_dvdadj(df_sec):
    """
    处理红利入帐 
    删除红利、调整成本
    
    可进一步优化: FIFO 按证券数量从头向后match，考虑持有的时间价值
    """
    buy_idx = 0
    sell_idx = 0

    if len(df_sec) == 0:
        return pd.DataFrame(), pd.DataFrame()
    sec_code = df_sec["证券代码"].values[0]
    df_sec_buy = df_sec[df_sec["业务类型"].isin(["证券买入"])]
    first_buy_idx = df_sec_buy.index[buy_idx]
    df_sec_sell = df_sec[df_sec["业务类型"].isin(["证券卖出"])]
    if len(df_sec_sell) > 0:
        first_sell_idx = df_sec_sell.index[sell_idx]
        temp = df_sec.loc[first_buy_idx:first_sell_idx]
    else:
        temp = df_sec.loc[first_buy_idx:]

    dvd_idx_list = list(temp[temp["业务类型"] == "红利入账"].index)
    if len(dvd_idx_list) > 0:
        dvd_adj_amt = temp.loc[dvd_idx_list, "成交金额"].sum()
        temp_buy = temp[temp["业务类型"] == "证券买入"].copy()
        temp_buy["dvd_adj"] = temp_buy["成交数量"] * float(dvd_adj_amt) / temp_buy["成交数量"].sum()
        dvd_adj_dict = temp_buy["dvd_adj"].to_dict()
        for k, v in list(dvd_adj_dict.items()):
            df_sec.loc[k, "成交金额"] -= v
            df_sec.loc[k, "成交价格"] = df_sec.loc[k, "成交金额"] / df_sec.loc[k, "成交数量"]
        print("[" + sec_code + "] 调整红利入帐前买入成本，共计" + str(len(dvd_idx_list)) + "笔分红, " + str(
            len(dvd_adj_dict)) + "笔买入")
        df_sec = df_sec[~df_sec.index.isin(dvd_idx_list)]

    return df_sec


def trading_record_rightadj(df_sec):
    """
    处理红股入帐
    删除红股、调整成本
    """
    buy_idx = 0
    sell_idx = 0

    if len(df_sec) == 0:
        return pd.DataFrame(), pd.DataFrame()
    sec_code = df_sec["证券代码"].values[0]
    df_sec_buy = df_sec[df_sec["业务类型"] == "证券买入"]
    first_buy_idx = df_sec_buy.index[buy_idx]
    df_sec_sell = df_sec[df_sec["业务类型"] == "证券卖出"]
    if len(df_sec_sell) > 0:
        first_sell_idx = df_sec_sell.index[sell_idx]
        temp = df_sec.loc[first_buy_idx:first_sell_idx]
    else:
        temp = df_sec.loc[first_buy_idx:]

    right_idx_list = list(temp[temp["业务类型"] == "红股入账"].index)
    reverseright_idx_list = list(temp[temp["业务类型"] == "股份转出"].index)

    if len(right_idx_list) > 0:
        right_adj_qty = temp.loc[right_idx_list, "成交数量"].sum()
        temp_buy = temp[temp["业务类型"] == "证券买入"].copy()
        temp_buy["qty_adj"] = temp_buy["成交数量"] * float(right_adj_qty) / temp_buy["成交数量"].sum()
        qty_adj_dict = temp_buy["qty_adj"].to_dict()
        for k, v in list(qty_adj_dict.items()):
            df_sec.loc[k, "成交数量"] += v
            df_sec.loc[k, "成交价格"] = df_sec.loc[k, "成交金额"] / df_sec.loc[k, "成交数量"]
        print("[" + sec_code + "] 调整红股入帐前买入成本/数量，共计" + str(len(right_idx_list)) + "笔红股, " + str(
            len(qty_adj_dict)) + "笔买入")
        df_sec = df_sec[~df_sec.index.isin(right_idx_list)]

    if len(reverseright_idx_list) > 0:
        right_adj_qty = temp.loc[reverseright_idx_list, "成交数量"].sum()
        temp_buy = temp[temp["业务类型"] == "证券买入"].copy()
        temp_buy["qty_adj"] = temp_buy["成交数量"] * float(right_adj_qty) / temp_buy["成交数量"].sum()
        qty_adj_dict = temp_buy["qty_adj"].to_dict()
        for k, v in list(qty_adj_dict.items()):
            df_sec.loc[k, "成交数量"] -= v
            df_sec.loc[k, "成交价格"] = df_sec.loc[k, "成交金额"] / df_sec.loc[k, "成交数量"]
        print("[" + sec_code + "] 调整股份合并前买入成本/数量，共计" + str(len(reverseright_idx_list)) + "笔合并, " + str(
            len(qty_adj_dict)) + "笔买入")
        df_sec = df_sec[~df_sec.index.isin(reverseright_idx_list)]

    return df_sec


def trading_record_match(df_sec, accounting_type="FIFO"):
    """
    针对每个股票，根据FIFO或LIFO match买入卖出记录，计算盈亏
    
    FIFO/LIFO trading record match
    input: sec lvl trading data
    output: matched buy sell record/buy only record
            sec lvl trading data
    data need to be cleaned / XD / XR / DR 
    """
    if accounting_type == "FIFO":
        buy_idx = 0
        sell_idx = 0
    elif accounting_type == "LIFO":
        buy_idx = -1
        sell_idx = 0
    else:
        raise Exception("Accounting Type Error, can only be FIFO/LIFO")

    if len(df_sec) == 0:
        return pd.DataFrame(), pd.DataFrame()

    sec_code = df_sec["证券代码"].values[0]
    df_sec_buy = df_sec[df_sec["业务类型"] == "证券买入"]
    df_sec_sell = df_sec[df_sec["业务类型"] == "证券卖出"]
    if len(df_sec_sell) > 0:
        sell_date = df_sec_sell["交割日期"].values[sell_idx]
        if accounting_type == "FIFO":
            buy_date = df_sec_buy["交割日期"].values[buy_idx]
        elif accounting_type == "LIFO":
            # get last buy before sell
            df_sec_buy = df_sec_buy[df_sec_buy["交割日期"] <= sell_date]
            buy_date = df_sec_buy["交割日期"].values[buy_idx]

        if buy_date <= sell_date:
            buy_qty = df_sec_buy["成交数量"].values[buy_idx]
            sell_qty = df_sec_sell["成交数量"].values[sell_idx]
            match_qty = min(buy_qty, sell_qty)
            match_record_buy = pd.DataFrame(df_sec_buy[['交割日期', '成交价格', '成交数量', '成交金额', '总佣金']].iloc[buy_idx, :]).T

            match_record_buy[['成交数量', '成交金额', '总佣金']] *= float(match_qty) / float(buy_qty)
            match_record_buy.columns = ['买入日期', '买入价格', '买入数量', '买入金额', '买入佣金']
            match_record_buy.index = [sec_code]
            match_record_sell = pd.DataFrame(
                df_sec_sell[['交割日期', '成交价格', '成交数量', '成交金额', '总佣金']].iloc[sell_idx, :]).T
            match_record_sell[['成交数量', '成交金额', '总佣金']] *= float(match_qty) / float(sell_qty)
            match_record_sell.columns = ['卖出日期', '卖出价格', '卖出数量', '卖出金额', '卖出佣金']
            match_record_sell.index = [sec_code]
            match_record = pd.concat([match_record_buy, match_record_sell], axis=1)
            match_record["PnL"] = match_record["卖出金额"] - match_record["买入金额"]
            match_record["PnL(扣除手续费)"] = match_record["PnL"] - match_record["买入佣金"] - match_record["卖出佣金"]
            # sell adjust
            df_sec.loc[df_sec_sell.index[sell_idx], '成交数量'] -= match_record_sell.loc[sec_code, '卖出数量']
            df_sec.loc[df_sec_sell.index[sell_idx], '成交金额'] -= match_record_sell.loc[sec_code, '卖出金额']
            df_sec.loc[df_sec_sell.index[sell_idx], '总佣金'] -= match_record_sell.loc[sec_code, '卖出佣金']
            if df_sec.loc[df_sec_sell.index[sell_idx], '成交数量'] < 0.00001:
                df_sec = df_sec.drop(df_sec_sell.index[sell_idx])
        else:
            # 无卖出，未平仓多头
            match_record_buy = pd.DataFrame(df_sec_buy[['交割日期', '成交价格', '成交数量', '成交金额', '总佣金']].iloc[buy_idx, :]).T
            match_record_buy.columns = ['买入日期', '买入价格', '买入数量', '买入金额', '买入佣金']
            match_record_buy.index = [sec_code]
            match_record = match_record_buy
    else:
        # 无卖出，未平仓多头
        match_record_buy = pd.DataFrame(df_sec_buy[['交割日期', '成交价格', '成交数量', '成交金额', '总佣金']].iloc[buy_idx, :]).T
        match_record_buy.columns = ['买入日期', '买入价格', '买入数量', '买入金额', '买入佣金']
        match_record_buy.index = [sec_code]
        match_record = match_record_buy

    # buy adjust
    df_sec.loc[df_sec_buy.index[buy_idx], '成交数量'] -= match_record_buy.loc[sec_code, '买入数量']
    df_sec.loc[df_sec_buy.index[buy_idx], '成交金额'] -= match_record_buy.loc[sec_code, '买入金额']
    df_sec.loc[df_sec_buy.index[buy_idx], '总佣金'] -= match_record_buy.loc[sec_code, '买入佣金']
    if df_sec.loc[df_sec_buy.index[buy_idx], '成交数量'] < 0.00001:
        df_sec = df_sec.drop(df_sec_buy.index[buy_idx])

    return df_sec, match_record


def match_sec_trading_record(df_allrecord, sec, accounting_type, plot=False):
    """
    match单只股票交易记录
    
    accounting_type: FIFO LIFO
    
    调用 trading_record_match(df_sec,accounting_type) 
    """
    # sec = "600383"
    df_match_res_sec = pd.DataFrame()
    df_err_sec = pd.DataFrame()

    df_sec = df_allrecord[df_allrecord["证券代码"] == sec].copy()
    df_sec.index = list(range(len(df_sec)))

    ########### 处理红利差异扣税/抵扣红利
    dvd_ind_list = df_sec[df_sec["业务类型"] == "红利入账"].index
    dvd_adj_map = {}
    for j in range(len(dvd_ind_list)):
        if j != len(dvd_ind_list) - 1:
            temp = df_sec.loc[dvd_ind_list[j]:dvd_ind_list[j + 1]]
        else:
            temp = df_sec.loc[dvd_ind_list[j]:]
        dvd_adj_map[dvd_ind_list[j]] = list(temp[temp["业务类型"] == "股息红利差异扣税"].index)
        dvd_tax_ratio = abs(float(temp.loc[temp["业务类型"] == "股息红利差异扣税", "发生金额"].sum())) / df_sec.loc[
            dvd_ind_list[j], "发生金额"].sum()
        df_sec.loc[dvd_ind_list[j], "成交金额"] += temp.loc[temp["业务类型"] == "股息红利差异扣税", "发生金额"].sum()
        df_sec.loc[dvd_ind_list[j], "发生金额"] += temp.loc[temp["业务类型"] == "股息红利差异扣税", "发生金额"].sum()
        print("[" + sec + "] 调整红利入帐第" + str(j + 1) + "次，共计" + str(
            len(dvd_adj_map[dvd_ind_list[j]])) + "次差异扣税, 补交税比例" + str(round(dvd_tax_ratio * 100.0, 2)) + "%")
    if len(dvd_adj_map) > 0:
        df_sec = df_sec[~df_sec.index.isin(reduce(lambda x, y: x + y, list(dvd_adj_map.values())))]
    else:
        print("[" + sec + "] 无红利入帐记录")

    ########### 调整 红利入账/证券买入 顺序,放到当日最后
    for idx in dvd_ind_list:
        temp = df_sec[(df_sec["交割日期"] == df_sec.loc[idx, "交割日期"]) & (df_sec["业务类型"] == "证券买入")].index
        if len(temp) > 0:
            idx_exchange = temp[-1]
            df_sec.loc[idx_exchange], df_sec.loc[idx] = df_sec.loc[idx].copy(), df_sec.loc[idx_exchange].copy()
            print("[" + sec + "] 调整红利入帐顺序")
    ###########
    i = 1
    match_record_all = pd.DataFrame()
    while len(df_sec[df_sec["业务类型"].isin(["证券买入", "证券卖出"])]) > 0:
        try:
            # FIFO/LIFO adj
            df_sec = trading_record_dvdadj(df_sec)
            df_sec = trading_record_rightadj(df_sec)

            df_sec, match_record = trading_record_match(df_sec, accounting_type)
            match_record_all = pd.concat([match_record_all, match_record], axis=0)
            i += 1
        except Exception as e:
            df_err_sec = pd.concat([df_err_sec, df_sec], axis=0)
            print("[" + sec + "] Error: " + str(e))
            break
    print("Security [" + sec + "] " + str(i) + " match records finished.")

    if "PnL" in match_record_all.columns:
        match_record_all = match_record_all.sort_values(by=["卖出日期", "买入日期"], ascending=True)

    # stats
    if len(match_record_all) > 0:
        if "PnL" not in match_record_all.columns:
            for c in ['卖出日期', '卖出价格', '卖出数量', '卖出金额', '卖出佣金', "PnL", "PnL(扣除手续费)"]:
                match_record_all[c] = float("nan")
        match_record_all["PnL统计量"] = ["盈利" if x > 0 else "持平" if x == 0 else "亏损" if x < 0 else x for x in match_record_all["PnL"]]
        match_record_all["累计PnL"] = match_record_all["PnL"].cumsum()
        match_record_all["累计PnL(扣除手续费)"] = match_record_all["PnL(扣除手续费)"].cumsum()
        match_record_all["累计手续费"] = match_record_all["累计PnL"] - match_record_all["累计PnL(扣除手续费)"]
        df_match_res_sec = pd.concat([df_match_res_sec, match_record_all], axis=0)

    # 平仓盈亏图
    if plot:
        if len(df_match_res_sec[df_match_res_sec["PnL"] == df_match_res_sec["PnL"]]) > 0:
            closepos_pospnl_ratio = len(match_record_all[match_record_all["PnL统计量"] == "盈利"]) / float(
                len(match_record_all))
            plt.figure(figsize=(10, 3))
            for c in ["累计PnL", "累计PnL(扣除手续费)", "累计手续费"]:
                plt.plot(match_record_all["卖出日期"], match_record_all[c], label=c)
            plt.title(sec + " 平仓盈亏 | 平仓盈利率:" + str(round(closepos_pospnl_ratio * 100.0, 2)) + "%")
            plt.legend()
            plt.grid(ls='--', alpha=0.6)
            plt.show()
        else:
            print(sec + " 无平仓盈亏！")

    return df_match_res_sec, df_err_sec


def match_all_trading_record(df_allrecord, accounting_type="FIFO"):
    """
    对调整后的交易记录统计平仓盈亏
    """
    #### 买卖记录
    df_buysell = pd.pivot_table(df_allrecord[df_allrecord["业务类型"].isin(["证券买入", "证券卖出"])],
                                index=["交割日期", "证券代码", "证券名称"], columns="业务类型",
                                values=["成交数量", "成交金额", "总佣金", "发生金额"], aggfunc=np.sum)
    df_buysell = df_buysell.fillna(0)
    df_buysell.columns = [x[1] + x[0] for x in df_buysell.columns]
    df_buysell.reset_index(inplace=True)

    ################### main loop
    # accounting_type="FIFO"
    df_match_res = pd.DataFrame()
    df_err_res = pd.DataFrame()
    sec_list = list(set(df_buysell["证券代码"].values))
    for sec in sec_list:
        df_match_res_sec, df_err_sec = match_sec_trading_record(df_allrecord, sec, accounting_type, plot=False)
        df_match_res = pd.concat([df_match_res, df_match_res_sec], axis=0)
        df_err_res = pd.concat([df_err_res, df_err_sec], axis=0)

    if len(df_err_res) > 0:
        print("=============分配算法Error【证券来源不明】============")
        print(df_err_res)

    return df_match_res, df_err_res


def TradeSmyPlot_closepospnl(df_match_res, accounting_type):
    """
    平仓盈亏统计
    accounting_type = "FIFO"/"LIFO"  for label purpose
    """
    # 平仓盈亏
    df_close_pos = df_match_res[df_match_res["PnL"] == df_match_res["PnL"]]
    closepos_pospnl_ratio = len(df_close_pos[df_close_pos["PnL统计量"] == "盈利"]) / float(len(df_close_pos))
    df_close_pos_date = pd.pivot_table(df_close_pos, index="买入日期", values=["PnL", "PnL(扣除手续费)"], aggfunc=np.sum)
    df_close_pos_date = df_close_pos_date.sort_index(ascending=True)
    df_close_pos_date["PnL统计量"] = ["盈利" if x > 0 else "持平" if x == 0 else "亏损" if x < 0 else x for x in df_close_pos_date["PnL"]]
    df_close_pos_date["累计PnL"] = df_close_pos_date["PnL"].cumsum()
    df_close_pos_date["累计PnL(扣除手续费)"] = df_close_pos_date["PnL(扣除手续费)"].cumsum()
    df_close_pos_date["累计手续费"] = df_close_pos_date["累计PnL"] - df_close_pos_date["累计PnL(扣除手续费)"]
    print("[" + df_close_pos_date.index[0].strftime("%Y/%m/%d") + "-" + df_close_pos_date.index[-1].strftime(
        "%Y/%m/%d") + "] 总平仓交易笔数: " + str(len(df_close_pos)) + " --- 盈亏比例：" + str(
        round(closepos_pospnl_ratio * 100.0, 2)) + "%")
    # 新股盈亏
    df_newstock = get_newIPO_trades()
    df_close_pos_newstock = df_close_pos[df_close_pos.index.isin(df_newstock["证券代码"])]
    df_close_pos_date_newstock = pd.pivot_table(df_close_pos_newstock, index="买入日期", values="PnL(扣除手续费)",
                                                aggfunc=np.sum)
    df_close_pos_date_newstock.columns = ["新股PnL(扣除手续费)"]
    df_close_pos_date = df_close_pos_date.merge(df_close_pos_date_newstock, left_index=True, right_index=True,
                                                how="left")
    df_close_pos_date["新股PnL(扣除手续费)"] = df_close_pos_date["新股PnL(扣除手续费)"].fillna(0)
    df_close_pos_date["新股累计PnL(扣除手续费)"] = df_close_pos_date["新股PnL(扣除手续费)"].cumsum()
    df_close_pos_date["累计PnL(扣除手续费/新股盈利)"] = df_close_pos_date["累计PnL(扣除手续费)"] - df_close_pos_date["新股累计PnL(扣除手续费)"]

    ############### 总平仓盈亏图 plotly
    x = df_close_pos_date.index
    trace0 = []
    trace0.append(dict(x=x, y=df_close_pos_date["累计手续费"], name='累计手续费', mode='lines+markers', fill='tonexty'))
    trace0.append(
        dict(x=x, y=df_close_pos_date["累计PnL(扣除手续费)"], name='累计PnL(扣除手续费)', mode='lines+markers', fill='tonexty'))
    trace0.append(dict(x=x, y=df_close_pos_date["累计PnL(扣除手续费/新股盈利)"], name='累计PnL(扣除手续费/新股盈利)', mode='lines+markers',
                       fill='tonexty'))
    trace0.append(dict(x=x, y=df_close_pos_date["累计PnL"], name='累计PnL', mode='lines+markers', fill='tonexty'))
    fig = tools.make_subplots(rows=4, cols=1, specs=[[{'rowspan': 3}], [{}], [{}], [{}]], shared_xaxes=True,
                              subplot_titles=('', '', '', '新股收益/累计新股收益'))
    for trace in trace0:
        fig.append_trace(trace, 1, 1)
    trace1 = go.Scatter(x=x, y=df_close_pos_date["新股累计PnL(扣除手续费)"], name='新股累计PnL(扣除手续费)', mode='lines+markers')
    trace2 = go.Bar(x=x, y=df_close_pos_date["新股PnL(扣除手续费)"], name='新股PnL(扣除手续费)',
                    marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=1.5), ), opacity=0.8)
    fig.append_trace(trace1, 4, 1)
    fig.append_trace(trace2, 4, 1)
    fig['layout'].update(
        title="(" + accounting_type + ") 证券交易平仓盈亏统计" + df_close_pos_date.index[0].strftime("%Y/%m/%d") + " - " +
              df_close_pos_date.index[-1].strftime("%Y/%m/%d"))
    py.plot(fig, filename=util_basics.PROJECT_ROOT_PATH + "/股票/个人研究/收益计算/证券交易平仓盈亏统计" + accounting_type + ".html")


def currpos_smy(df_match_res):
    """目前持仓统计"""
    df_curr_pos = df_match_res[(df_match_res["PnL"] != df_match_res["PnL"]) & (df_match_res["买入数量"] >= 1)].copy()
    df_curr_pos["证券代码"] = df_curr_pos.index
    df_curr_pos_smy = pd.pivot_table(df_curr_pos, index="证券代码", values=['买入金额', '买入佣金', '买入数量'], aggfunc=np.sum)
    df_curr_pos_smy['买入价格'] = df_curr_pos_smy['买入金额'] / df_curr_pos_smy['买入数量']
    df_curr_pos_smy.reset_index(inplace=True)
    currpos_seclist = list(df_curr_pos_smy["证券代码"])
    currpos_pxdict = {}
    for k in currpos_seclist:
        temp = util_tushare.ts.get_realtime_quotes(k)
        currpos_pxdict[k] = float(temp["price"][0]) if temp is not None else float("nan")
        print(k + " 最新价格: " + str(currpos_pxdict[k]))
    df_curr_pos_smy["当前价格"] = [currpos_pxdict[x] if x in list(currpos_pxdict.keys()) else float("nan") for x in df_curr_pos_smy["证券代码"]]
    df_curr_pos_smy["当前价格"] = list(map(lambda x, y: x if x == x else y, df_curr_pos_smy["当前价格"], df_curr_pos_smy["买入价格"]))
    df_curr_pos_smy["浮动盈亏"] = (df_curr_pos_smy["当前价格"] - df_curr_pos_smy["买入价格"]) * df_curr_pos_smy["买入数量"] - \
                               df_curr_pos_smy["买入佣金"]
    return df_curr_pos_smy


def overall_pnlsmy(df_match_res):
    """
    统计平仓与持仓盈亏
    """
    ################## 平仓盈亏统计
    df_closed_pos = df_match_res[(df_match_res["PnL"] == df_match_res["PnL"])].copy()
    realized_pnl = df_closed_pos['PnL(扣除手续费)'].sum()
    ################## 持仓盈亏统计
    df_curr_pos_smy = currpos_smy(df_match_res)
    unrealized_pnl = df_curr_pos_smy["浮动盈亏"].sum()
    df_smy = pd.DataFrame([realized_pnl, unrealized_pnl], columns=[0], index=["平仓盈亏", "持仓盈亏"]).T
    df_smy["总盈亏"] = df_smy["平仓盈亏"] + df_smy["持仓盈亏"]
    return df_smy


def get_historypos_all(df_allrecord):
    """
    获取以往历史持仓
    包含cash
    
    cash 在新股入账几日前就会扣除
    """
    dict_pos_smy = {}
    pre_date = None
    for idx in df_allrecord.index:
        curr_date = df_allrecord.loc[idx, "交割日期"]
        if curr_date != pre_date:
            if pre_date is None:
                dict_pos_smy[curr_date] = {}
            else:
                dict_pos_smy[curr_date] = dict_pos_smy[pre_date].copy()
                dict_pos_smy[pre_date]["cash"] = df_allrecord[df_allrecord["交割日期"] == pre_date].iloc[-1]["剩余金额"]

        if df_allrecord.loc[idx, "业务类型"] in ["证券买入", "红股入账", "新股入帐"]:
            if df_allrecord.loc[idx, "证券代码"] in list(dict_pos_smy[curr_date].keys()):
                dict_pos_smy[curr_date][df_allrecord.loc[idx, "证券代码"]] += df_allrecord.loc[idx, "成交数量"]
            else:
                dict_pos_smy[curr_date][df_allrecord.loc[idx, "证券代码"]] = df_allrecord.loc[idx, "成交数量"]
        elif df_allrecord.loc[idx, "业务类型"] in ["证券卖出", "股份转出"]:
            if df_allrecord.loc[idx, "证券代码"] in list(dict_pos_smy[curr_date].keys()):
                dict_pos_smy[curr_date][df_allrecord.loc[idx, "证券代码"]] -= df_allrecord.loc[idx, "成交数量"]
            else:
                print("Error! " + df_allrecord.loc[idx, "证券代码"] + " sell before buy")
        pre_date = curr_date
        dict_pos_smy[curr_date] = {x: y for x, y in list(dict_pos_smy[curr_date].items()) if y != 0 or x == "cash"}

    dict_pos_smy[curr_date]["cash"] = df_allrecord.iloc[-1]["剩余金额"]
    dict_secname = {x[0]: x[1] for x in df_allrecord[["证券代码", "证券名称"]].values if x[0] == x[0]}

    # concolidate to df
    df_poshistory = pd.DataFrame()
    for k, v in list(dict_pos_smy.items()):
        pass
        temp = pd.DataFrame(v, index=["qty"]).T
        temp.index.name = "seccode"
        temp.reset_index(inplace=True)
        temp["secname"] = [dict_secname[x] if x != "cash" else "现金" for x in temp["seccode"]]
        temp["date"] = k
        temp = temp[["date", "seccode", "secname", "qty"]]
        df_poshistory = pd.concat([df_poshistory, temp], axis=0)
    df_poshistory = df_poshistory.sort_values(by="date", ascending=True)
    return df_poshistory


def get_historypos_date(curr_date, df_poshistory=None, download_data=False):
    """
    获取某一天历史持仓
    """
    if df_poshistory is None:
        df_allrecord = get_stocktraderecord()
        df_poshistory = get_historypos_all(df_allrecord)

    active_date = df_poshistory[df_poshistory["date"] <= curr_date].iloc[-1].loc["date"]
    res = df_poshistory[df_poshistory["date"] == active_date].copy()
    sec_list = [x for x in res["seccode"] if x != "cash"]
    # add px and mv
    px_dict = {"cash": 1.0}
    for seccode in sec_list:
        pass
        if download_data:
            # download data from tushare
            util_tushare.tushare_get_history(seccode)
        try:
            temp = pd.read_csv(util_basics.PROJECT_DATA_PATH + "/tushare/history/" + seccode + ".csv")
            temp.index = pd.DatetimeIndex(list(map(str, temp["trade_date"])))
            px_dict[seccode] = temp.loc[active_date, "Close_Raw"]
        except:
            px_dict[seccode] = float("nan")
    res["Price"] = [px_dict[x] for x in res["seccode"]]
    res["MV"] = res["Price"] * res["qty"]
    return res


def compare_perf(start_date, end_date, highlight_seclist=[], download_data=False, benchmark_sec=None):
    """
    compare account perf and all holdings perfermance
    need download all holdings history data if not in local
    
    start_date = "20200101"
    end_date = "20200926"
    highlight_seclist=["600383"]
    benchmark_sec = "600383"
    download_data = False
    """
    df_allrecord = get_stocktraderecord(autoadj=True)
    df_poshistory = get_historypos_all(df_allrecord)
    df_poshistory_rng = df_poshistory[(df_poshistory["date"] >= start_date) & (df_poshistory["date"] <= end_date)]
    sec_list = sorted([x for x in set(df_poshistory_rng["seccode"]) if x != "cash"])
    sec_hist_tbl = pd.DataFrame()
    sec_hist_tbl_adj = pd.DataFrame()
    for seccode in sec_list:
        pass
        if download_data:
            # download data from tushare
            util_tushare.tushare_get_history(seccode)
        try:
            temp = pd.read_csv(util_basics.PROJECT_DATA_PATH + "/tushare/history/" + seccode + ".csv")
            temp.index = pd.DatetimeIndex(list(map(str, temp["trade_date"])))
            temp2 = temp[["Close_Raw"]]
            temp2.columns = [seccode]
            sec_hist_tbl = pd.concat([sec_hist_tbl, temp2[(temp2.index >= start_date) & (temp2.index <= end_date)]],
                                     axis=1)
            temp2 = temp[["close"]]
            temp2.columns = [seccode]
            sec_hist_tbl_adj = pd.concat(
                [sec_hist_tbl_adj, temp2[(temp2.index >= start_date) & (temp2.index <= end_date)]], axis=1)
        except:
            pass
    trade_calender = sorted(list(sec_hist_tbl.index))
    sec_hist_tbl = sec_hist_tbl.sort_index(ascending=True)
    sec_hist_tbl_adj = sec_hist_tbl_adj.sort_index(ascending=True)

    account_balance = {}
    for curr_date in trade_calender:
        df_curr_pos = get_historypos_date(curr_date, df_poshistory).copy()
        df_curr_pos["Close_Raw"] = float("nan")
        for seccode in df_curr_pos["seccode"]:
            if seccode in sec_hist_tbl.columns:
                df_curr_pos.loc[df_curr_pos["seccode"] == seccode, "Close_Raw"] = sec_hist_tbl.loc[curr_date, seccode]
            if seccode == "cash":
                df_curr_pos.loc[df_curr_pos["seccode"] == seccode, "Close_Raw"] = 1
        df_curr_pos["MV"] = df_curr_pos["Close_Raw"] * df_curr_pos["qty"]
        account_balance[curr_date] = df_curr_pos["MV"].sum()
    account_balance = pd.DataFrame(account_balance, index=["MV"]).T
    # adjust for cash in/out 
    df_cashjournel = pd.pivot_table(df_allrecord[df_allrecord["业务类型"].isin(["银行转证券", "证券转银行"])], index="交割日期",
                                    values=["发生金额"], aggfunc=np.sum)
    df_cashjournel.columns = ["cashinout"]
    account_balance = account_balance.merge(df_cashjournel, left_index=True, right_index=True, how="left")
    account_balance["MV(T-1)"] = account_balance["MV"].shift(1)
    account_balance["ret1d"] = list(map(lambda x, y, z: x / y - 1 if z != z else (x - z) / y - 1, account_balance["MV"],
                                   account_balance["MV(T-1)"], account_balance["cashinout"]))
    account_balance["ret1d"] = account_balance["ret1d"].fillna(0)
    account_balance["retcum"] = (account_balance["ret1d"] + 1).cumprod()

    sec_hist_tbl_rescale = sec_hist_tbl_adj.fillna(method="bfill")
    sec_hist_tbl_rescale = sec_hist_tbl_rescale / sec_hist_tbl_rescale.iloc[0, :]
    sec_hist_tbl_ret = sec_hist_tbl_rescale / sec_hist_tbl_rescale.shift(1) - 1
    sec_hist_tbl_ret = sec_hist_tbl_ret.fillna(0)
    dict_secname = {x[0]: x[1] for x in df_allrecord[["证券代码", "证券名称"]].values if x[0] == x[0]}
    if True:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.figure(figsize=(10, 5))
        plt.plot(account_balance.index, account_balance["retcum"], marker="o", label="Account")
        for c in sec_hist_tbl_rescale.columns:
            if (c in highlight_seclist) or (dict_secname[c] in highlight_seclist):
                plt.plot(sec_hist_tbl_rescale.index, sec_hist_tbl_rescale[c], alpha=0.7, linewidth=4.0,
                         label=dict_secname[c])
            else:
                plt.plot(sec_hist_tbl_rescale.index, sec_hist_tbl_rescale[c], alpha=0.3, label=dict_secname[c])
        ymin, ymax = plt.gca().get_ylim()
        plt.ylim(ymin, min(ymax, account_balance["retcum"].max() * 1.25))
        plt.title("Performance Summary " + start_date + " to " + end_date)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(ls='--', alpha=0.6)
        plt.show()
        if len(highlight_seclist) > 0:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.figure(figsize=(10, 3))
            for c in sec_hist_tbl_rescale.columns:
                if (c in highlight_seclist) or (dict_secname[c] in highlight_seclist):
                    plt.plot(sec_hist_tbl_rescale.index, account_balance["retcum"] - sec_hist_tbl_rescale[c],
                             marker="o", alpha=0.7, linewidth=1.0, label="Account-" + dict_secname[c])
            plt.title("Alpha Summary " + start_date + " to " + end_date)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(ls='--', alpha=0.6)
            plt.show()

            # get stats

    def get_perf_smy(ret_series, ret_bchmk_series=None):
        """get perf smy"""
        # import pyfolio.timeseries as pfts
        if ret_bchmk_series is None:
            ret_bchmk_series = ret_series
        temp = util_portfolio.perf_stats(ret_series, factor_returns=ret_bchmk_series)
        temp["Date_Start"] = ret_series.index.min().strftime("%Y/%m/%d")
        temp["Date_End"] = ret_series.index.max().strftime("%Y/%m/%d")
        temp["Days"] = (ret_series.index.max() - ret_series.index.min()).days
        perf_smy = pd.DataFrame(temp, columns=["values"])
        return perf_smy

    if benchmark_sec in sec_hist_tbl_ret.columns:
        res = get_perf_smy(account_balance["ret1d"], sec_hist_tbl_ret[benchmark_sec])
    else:
        res = get_perf_smy(account_balance["ret1d"])
    res.columns = ["Account"]
    for c in sec_hist_tbl_ret.columns:
        pass
        temp = get_perf_smy(sec_hist_tbl_ret[c])
        temp.columns = [dict_secname[c]]
        res = pd.concat([res, temp], axis=1)
    res = res.T.sort_values(by="Cumulative returns", ascending=False)
    return res

##################### test
# df_allrecord = get_stocktraderecord(autoadj = True)
# TradeSmyPlot_commission(df_allrecord)
# TradeSmyPlot_cashjournal(df_allrecord)
#
# # test one security
# df_match_res,df_err_res = match_sec_trading_record(df_allrecord,"000002","LIFO",plot=True)
# 
# df_match_res,df_err_res = match_all_trading_record(df_allrecord,"FIFO")
# TradeSmyPlot_closepospnl(df_match_res,"FIFO")
# df = currpos_smy(df_match_res)
# overall_pnlsmy(df_match_res)
#
# df_poshistory = get_historypos_all(df_allrecord)
# df_poshistory_date = get_historypos_date(df_poshistory,"20190817")
