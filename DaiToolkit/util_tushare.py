# -*- coding: utf-8 -*-

import datetime

import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts

from DaiToolkit import util_basics
from DaiToolkit import util_database

PROJECT_DATA_TUSHARE_PATH = util_basics.PROJECT_DATA_PATH + "/tushare"
# ts.set_token(util_readfile.read_yaml(util_basics.PROJECT_CODE_PATH + "/DaiToolkit/login.yaml")["tushare"]['token'])
pro = ts.pro_api()

def tushare_getallsec_basics():
    """
    全市场基本数据
    auto save to db
    
    这里的市盈率是动态市盈率

    stock_basic
    输入参数
    ts_code str N 股票代码
    list_status str N 上市状态： L上市 D退市 P暂停上市，默认L
    exchange str N 交易所 SSE上交所 SZSE深交所 HKEX港交所(未上线)
    is_hs str N 是否沪深港通标的，N否 H沪股通 S深股通

    输出参数
    ts_code str TS代码
    symbol str 股票代码
    name str 股票名称
    area str 所在地域
    industry str 所属行业
    fullname str 股票全称
    enname str 英文全称
    market str 市场类型 （主板/中小板/创业板/科创板/CDR）
    exchange str 交易所代码
    curr_type str 交易货币
    list_status str 上市状态： L上市 D退市 P暂停上市
    list_date str 上市日期
    delist_date str 退市日期
    is_hs str 是否沪深港通标的，N否 H沪股通 S深股通

    """
    print("Downloading all sec basics from tushare...")
    df = pro.stock_basic(exchange='', list_status='L',
                         fields='ts_code,symbol,name,fullname,enname,industry,area,industry,market,exchange,curr_type,list_status,list_date,delist_date,is_hs')
    colmap = {'ts_code': 'TS代码', 'symbol': '代码', 'name': '名称', 'fullname': '股票全称', 'enname': '英文全称',
              'industry': '所属行业', 'area': '地区', 'market': '市场类型', 'exchange': '交易所代码', 'curr_type': '交易货币',
              'list_status': '上市状态', 'list_date': '上市日期', 'delist_date': '退市日期', 'is_hs': '沪深港通'}
    df.columns = [colmap[x] for x in df.columns]
    today_date = datetime.datetime.now().strftime("%Y%m%d")
    df["更新时间"] = int(today_date)

    util_database.db_table_update(df, "allsecbasics_ashares", if_exists='replace')
    util_database.db_excute(
        "ALTER TABLE `auto_dai`.`allsecbasics_ashares` CHANGE COLUMN `代码` `代码` VARCHAR(20) NOT NULL ,ADD PRIMARY KEY (`代码`);;")

    print("A股股票列表更新完成，数据库更新日期[" + today_date + "]，总计记录" + str(len(df)) + "条")



def tusharelocal_get_allsecnames():
    """
    get latest sec dict 代码：名称
    
    mysql db
    """

    df_secbasics = util_database.db_query("SELECT * FROM allsecbasics_ashares")
    df_secbasics['名称'] = [x.strip().replace(" ", "") for x in df_secbasics['名称']]
    sec_dict = {x: util_basics.string_to_unicode(y) for x, y in zip(df_secbasics['代码'], df_secbasics['名称'])}
    return sec_dict


def get_nearest_eps_yrqtr(fully_released=True):
    """ return [(2017,4),(2018,1)] or [(2018,2)]
    """
    today_date = datetime.datetime.now().strftime("%Y%m%d")
    if fully_released:
        if today_date[-4:] > "0430" and today_date[-4:] <= "0831":
            return [(int(today_date[:4]) - 1, 4), (int(today_date[:4]), 1)]
        elif today_date[-4:] > "0831" and today_date[-4:] <= "1031":
            return [(int(today_date[:4]), 2)]
        elif today_date[-4:] > "1031":
            return [(int(today_date[:4]), 3)]
        elif today_date[-4:] <= "0430":
            return [(int(today_date[:4]) - 1, 3)]
    else:
        if today_date[-4:] < "0401":
            return [(int(today_date[:4]) - 1, 4)]
        elif today_date[-4:] < "0701":
            return [(int(today_date[:4]) - 1, 4), (int(today_date[:4]), 1)]
        elif today_date[-4:] < "1001":
            return [(int(today_date[:4]), 2)]
        else:
            return [(int(today_date[:4]), 3)]


def tushare_getqtr_profitdata(yr_qtr):
    """ yr_qtr = (1990,2) 
    
    code,代码
    name,名称
    roe,净资产收益率(%)
    net_profit_ratio,净利率(%)
    gross_profit_rate,毛利率(%)
    net_profits,净利润(万元)
    esp,每股收益
    business_income,营业收入(百万元)
    bips,每股主营业务收入(元)
    """

    qtr_flag = str(yr_qtr[0]) + "Q" + str(yr_qtr[1])
    try:
        df = pd.read_csv(
            PROJECT_DATA_TUSHARE_PATH + "/qtrly_profitdata/" + str(yr_qtr[0]) + "_Q" + str(yr_qtr[1]) + ".csv")
        df["update"] = int(datetime.datetime.now().strftime("%Y%m%d"))
    except:
        print("[" + str(yr_qtr[0]) + " Q" + str(yr_qtr[1]) + "] data downloading...")
        df = ts.get_profit_data(yr_qtr[0], yr_qtr[1])
        # unicode converting
        df['code'] = [x if x != x else "0" * (6 - len(str(int(x)))) + str(int(x)) for x in df['code']]
        df["date"] = qtr_flag
        df["update"] = int(datetime.datetime.now().strftime("%Y%m%d"))

    if not util_database.db_table_isexist("quarterlyearnings_ashares"):
        util_database.db_table_update(df, "quarterlyearnings_ashares")
    else:
        util_database.db_excute('DELETE FROM quarterlyearnings_ashares WHERE date="' + qtr_flag + '"')
        util_database.db_table_update(df, "quarterlyearnings_ashares", if_exists='append')
        print("Override data: " + qtr_flag)

    record_minqtr = util_database.db_query("select min(date) from quarterlyearnings_ashares").values[0][0]
    record_maxqtr = util_database.db_query("select max(date) from quarterlyearnings_ashares").values[0][0]
    print("季度盈利导入数据库，储存历史数据 [" + str(record_minqtr) + " - " + str(record_maxqtr) + "]")


def tushare_get_nearestqtr_profitdata(fully_released=True):
    """
    根据qtr下载eps数据
    
    根据数据库当前数据和最新数据的间隔，自动补足下载
    """
    record_maxqtr = util_database.db_query("select max(date) from quarterlyearnings_ashares").values[0][0]
    record_maxqtr = record_maxqtr.split("Q")
    record_maxqtr = (int(record_maxqtr[0]), int(record_maxqtr[1]))
    curr_qtr = get_nearest_eps_yrqtr(fully_released)[-1]
    qtr_list = []
    for y in range(record_maxqtr[0], curr_qtr[0] + 1):
        for q in range(1, 5):
            if y == record_maxqtr[0] and y < curr_qtr[0]:
                if q >= record_maxqtr[1]:
                    qtr_list.append((y, q))
            elif y == curr_qtr[0]:
                if y > record_maxqtr[0]:
                    if q <= curr_qtr[1]:
                        qtr_list.append((y, q))
                else:
                    if q <= curr_qtr[1] and q >= record_maxqtr[1]:
                        qtr_list.append((y, q))
            else:
                qtr_list.append((y, q))
    for yr_qtr in qtr_list:
        tushare_getqtr_profitdata(yr_qtr)


def tusharelocal_get_secprofitdata(seccode, start, end):
    """
    from ticker code: start and end year & quarter 
    ex: tusharelocal_get_secprofitdata("600655",start=(1995,2),end=(2018,4))
    """
    start_str = str(start[0]) + "Q" + str(start[1])
    end_str = str(end[0]) + "Q" + str(end[1])
    res = util_database.db_query(
        'select * from quarterlyearnings_ashares where date>="' + start_str + '" and date<="' + end_str + '" and code="' + seccode + '"')
    res = res.sort_values(by="date")

    return res


def get_eps_yrqtr(today_date, annual_type="TTM"):
    """
    get start and end year and quarter with given curr date
    annual_type: STATIC, DYNAMIC, TTM
    return qtr start/end for eps calculation
    """
    # today_date = "20180601"
    if annual_type == "TTM":
        year = today_date[:4]
        if today_date <= year + "0430":
            end = (int(year) - 1, 3)
            start = (int(year) - 2, 3)
        elif today_date <= year + "0831":
            end = (int(year), 1)
            start = (int(year) - 1, 1)
        elif today_date <= year + "1031":
            end = (int(year), 2)
            start = (int(year) - 1, 2)
        elif today_date <= year + "1231":
            end = (int(year), 3)
            start = (int(year) - 1, 3)
    elif annual_type == "STATIC":
        year = today_date[:4]
        start = (int(year) - 1, 4)
        end = (int(year) - 1, 4)
    elif annual_type == "DYNAMIC":
        year = today_date[:4]
        if today_date <= year + "0430":
            end = (int(year) - 1, 3)
            start = (int(year) - 1, 2)
        elif today_date <= year + "0831":
            end = (int(year), 1)
            start = (int(year), 1)
        elif today_date <= year + "1031":
            end = (int(year), 2)
            start = (int(year), 1)
        elif today_date <= year + "1231":
            end = (int(year), 3)
            start = (int(year), 2)
    return start, end


def tusharelocal_get_annual_EPS(seccode, today_date, annual_type="TTM"):
    """
    calculate last 1 year period eps sum, STATIC, DYNAMIC, TTM
    date_str: yyyymmdd

    ex: tusharelocal_get_annual_EPS('601601','20200501',annual_type="TTM")
    """
    if annual_type == "TTM":
        start, end = get_eps_yrqtr(today_date, annual_type="TTM")
        df = tusharelocal_get_secprofitdata(seccode, start, end)
        df = df.fillna(float('nan'))
        if len(df) < 4:
            return float("nan")
        elif len(df) == 4 and list(df["date"])[-1][-1] != "4":
            return float("nan")
        # caculation
        if list(df["date"])[-1][-1] == "1":
            res = list(df["eps"])[-1] + list(df["eps"])[3] - list(df["eps"])[0]
        elif list(df["date"])[-1][-1] == "2":
            res = list(df["eps"])[-1] + list(df["eps"])[2] - list(df["eps"])[0]
        elif list(df["date"])[-1][-1] == "3":
            res = list(df["eps"])[-1] + list(df["eps"])[1] - list(df["eps"])[0]
        elif list(df["date"])[-1][-1] == "4":
            res = list(df["eps"])[-1]
        return res
    elif annual_type == "STATIC":
        start, end = get_eps_yrqtr(today_date, annual_type="STATIC")
        df = tusharelocal_get_secprofitdata(seccode, start, end)
        df = df.fillna(float('nan'))
        if len(df) > 0:
            res = list(df["eps"])[-1]
        else:
            res = float("nan")
        return res
    elif annual_type == "DYNAMIC":
        start, end = get_eps_yrqtr(today_date, annual_type="DYNAMIC")
        df = tusharelocal_get_secprofitdata(seccode, start, end)
        df = df.fillna(float('nan'))
        if len(df) == 2:
            res = list(df["eps"])[1] - list(df["eps"])[0]
        elif len(df) == 1 and list(df["date"])[-1][-1] == "1":
            res = list(df["eps"])[-1]
        else:
            res = float("nan")
        return res * 4.0


def tushare_getPE(seccode, start, end=None, plot=True):
    """
    seccode = "601601"
    start='19900101'
    end='20210303'
    return df

    warning: this get PE time series func has several deficits
    1. earnings change date not accurate
    2. recent earnings might missing
    3. price not dvd adjusted
    
    only for approxi usage!
    """
    print("tushare_getPE: 下载不复权数据...")
    end = end if end != None else datetime.datetime.now().strftime("%Y%m%d")
    try:
        # 不复权
        df = ts.pro_bar(ts_code=seccode + ".SH" if seccode[0] == "6" else seccode + ".SZ", adj=None, start_date=start,
                        end_date=end)
        df.index = df["trade_date"]
        df = pd.DataFrame(df["close"])
        df.columns = ["Close_Raw"]
    except Exception as e:
        print("tushare_getPE: "+str(e))
        return pd.DataFrame()

    ########
    def mapdate(date):
        date_str = date.replace("-", "").replace("/", "")
        if date_str[4:] <= "0430":
            return str(int(date_str[:4]) - 1) + "1101"
        elif date_str[4:] <= "0831":
            return str(int(date_str[:4])) + "0501"
        elif date_str[4:] <= "1031":
            return str(int(date_str[:4])) + "0901"
        else:
            return str(int(date_str[:4])) + "1101"

    df["map_date"] = [mapdate(ind) for ind in df.index]

    dict_ttmeps = {x: tusharelocal_get_annual_EPS(seccode, x, annual_type="TTM") for x in set(df["map_date"])}
    dict_staticeps = {x: tusharelocal_get_annual_EPS(seccode, x, annual_type="STATIC") for x in set(df["map_date"])}
    dict_dynamiceps = {x: tusharelocal_get_annual_EPS(seccode, x, annual_type="DYNAMIC") for x in set(df["map_date"])}
    print("tushare_getPE: 不复权数据下载完毕! 开始计算PE时间序列...")
    df["EPS_TTM"] = [float(dict_ttmeps[ind]) for ind in df["map_date"]]
    df["PE_TTM"] = df["Close_Raw"] / df["EPS_TTM"]
    df["EPS_STATIC"] = [float(dict_staticeps[ind]) for ind in df["map_date"]]
    df["PE_STATIC"] = df["Close_Raw"] / df["EPS_STATIC"]
    df["EPS_DYNAMIC"] = [float(dict_dynamiceps[ind]) for ind in df["map_date"]]
    df["PE_DYNAMIC"] = df["Close_Raw"] / df["EPS_DYNAMIC"]

    del df["map_date"]
    df.index = pd.DatetimeIndex(df.index)
    if plot:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
        axes[0].plot(df["Close_Raw"].dropna(how="any"), label="Raw Price")
        axes[1].plot(df["PE_TTM"].dropna(how="any"), color="darkgreen", label="PE_TTM", alpha=0.5)
        axes[1].plot(df["PE_STATIC"].dropna(how="any"), color="darkblue", label="PE_STATIC", alpha=0.5)
        axes[0].grid(ls="--", alpha=0.8)
        axes[1].grid(ls="--", alpha=0.8)
        axes[0].set_title("SecCode [" + seccode + "]", fontsize=16, fontweight="bold")
        axes[0].legend(fontsize=12)
        axes[1].legend(fontsize=12)
        plt.show()

    return df


def tushare_get_history(seccode):
    """
    下载历史数据：高开低收量（前复权），raw收盘价（不复权），PE/TTM，调整因子
    seccode="601009"
    """
    print("----- [" + seccode + "] start downloading history -----")
    if "." in seccode:
        ts_code = seccode
        raw_code = seccode.split('.')[0]
    else:
        ts_code = seccode + ".SH" if seccode[0] == "6" else seccode + ".SZ"
        raw_code = seccode

    try:
        df_sec_factor = pro.adj_factor(ts_code=ts_code, trade_date='')
        df_sec_factor.index = pd.DatetimeIndex(df_sec_factor["trade_date"])
    except:
        df_sec_factor = pd.DataFrame(columns=["adj_factor"])

    try:
        df_PE = tushare_getPE(raw_code, '19890101', end=None, plot=False)
        print("开始下载前复权数据...")
        df_sec_qfq = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date='19890101',
                                end_date=datetime.datetime.now().strftime("%Y%m%d"))
        df_sec_qfq.index = pd.DatetimeIndex(df_sec_qfq["trade_date"])

        df_res = df_sec_qfq.merge(df_PE, left_index=True, right_index=True, how="left")
        df_res = df_res.merge(df_sec_factor[["adj_factor"]], left_index=True, right_index=True, how="left")
        df_res.to_csv(PROJECT_DATA_TUSHARE_PATH + "/history/" + raw_code + ".csv", index=False)
    except Exception as e:
        raise Exception("下载前复权错误:"+str(e))
    print("----- [" + seccode + "] finished! -----")
    return df_res


def tusharelocal_get_history(seccode):
    df = pd.read_csv(PROJECT_DATA_TUSHARE_PATH + "/history/" + seccode + ".csv")
    idx = df["trade_date"].apply(lambda x: str(x)[:4] + '/' + str(x)[4:6] + '/' + str(x)[6:])
    df.index = pd.DatetimeIndex(idx)
    return df


################################ test
if __name__ =="__main__":
    pass
    # tushare_getallsec_basics()
    # tushare_get_history(seccode="600383")