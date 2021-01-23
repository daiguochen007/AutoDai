# -*- coding: utf-8 -*-
import math

import numpy as np
import pandas as pd


###########################################################
# make history num format
###########################################################

def numformat_hist_df(sec_df, date_header="TradingDate", ignore_headers=["SecName"]):
    """
    numformat_hist_df(sec_df,date_header="TradingDate",ignore_headers=["SecName"])
    
    transfer dataframe from database to correct format data frame
    1. TradingDate to date format
    2. SecName remain text
    3. assume all other flds should convert to float
       nan indicators (None and N.A.) should set as float("nan")
    4. return df follow date ascending order
    """
    nan_indicator = ("N.A.", "None", "")
    headers = set(sec_df.columns) - set(ignore_headers + [date_header])
    if date_header:
        # different date type
        if len(sec_df) > 0:
            if "-" in sec_df[date_header][0]:
                sec_df[date_header] = pd.to_datetime(sec_df[date_header], format='%Y-%m-%d')
            elif "/" in sec_df[date_header][0]:
                sec_df[date_header] = pd.to_datetime(sec_df[date_header], format='%Y/%m/%d')
            else:
                sec_df[date_header] = pd.to_datetime(sec_df[date_header], format='%Y%m%d')
        sec_df = sec_df.sort_values(by=date_header, ascending=True)

    for h in headers:
        try:
            sec_df[h] = [float(str(x)) if str(x) not in nan_indicator else float("nan") for x in sec_df[h]]
        except ValueError:
            vset = set()
            for v in sec_df[h]:
                try:
                    float(str(v))
                except ValueError:
                    vset.add(str(v))
            print("Value Error in column [" + h + "]! More nan_indicator needed: " + str(vset))
    return sec_df


###########################################################
#  make dividend table
###########################################################

def get_divtbl(div_str, type_chg=True):
    """
    get_divtbl(div_str,type_chg = True)
    
    get div history table from bbg downloaded dividend string
    """
    div_tbl = div_str.split(";")[4:-1]
    div_tbl = div_tbl[1::2]
    div_tbl = np.reshape(div_tbl, (len(div_tbl) / 7, 7))
    clns = ["Declared Date", "Ex-Date", "Record Date", "Payable Date", "Dividend Amount", "Dividend Frequency",
            "Dividend Type"]
    div_tbl = pd.DataFrame(div_tbl, columns=clns)
    # --- change data type
    if type_chg:
        data_na_list = ["N.A.", " "]
        for c in clns[:4]:
            div_tbl.loc[:, c] = [pd.to_datetime(x) if len(x) > 1 and x not in data_na_list else float("nan") for x in div_tbl[c]]
        div_tbl.loc[:, clns[4]] = [float(x) if x not in data_na_list else float("nan") for x in div_tbl[clns[4]]]
    return div_tbl


###########################################################
#  get cash dividend adjusted price
###########################################################

def get_adjustedprice(sec_ts, div_tbl):
    """
    get_adjustedprice(sec_ts,div_tbl)
    
    get adjusted price from raw price(bloomberg default settings) and dividend table(EQY_DVD_HIST_GROSS)
    adjusted for cash dividend and dividend spinoff
        version 1 : O(n) takes 3.872s per ticker
        version 2 : only modify serval returns, takes 0.041s per ticker
    cash_dvd_type: all cash dvd types from bloomberg (LU_EQY_DVD_CASH_TYP_NEXT)
                   "Misc" excluded
    sec_ts should be nan omitted
    """
    sec_ts = sec_ts.dropna(how="any")
    cash_dvd_type = ["1st Interim", "2nd Interim", "3rd Interim", "4th Interim", "5th Interim",
                     "Accumulation", "Cancelled", "Capital Gains",
                     "Daily Accrual", "Deferred", "Discontinued", "Distribution", "Estimated", "Final", "Income",
                     "Interest on Capital", "Interim", "Liquidation",
                     "Long Term Cap Gain", "Memorial", "No DVD Period", "Omitted", "Partnership Dist",
                     "Pfd Rights Redemption", "Pro Rata",
                     "Proceeds from sale of Rights", "Proceeds from sale of Shares", "Proceeds from sale of Warrants",
                     "Regular Cash", "Return Prem.", "Return of Capital", "Rights Redemption", "Short Term Cap Gain",
                     "Special Cash", "Stock Reform-Cash"]

    # -------------------------------- modify dividend for spin off ----
    for k in range(len(div_tbl)):
        if div_tbl["Dividend Type"][k] == "Spinoff":
            div_tbl.loc[list(map(lambda x, y: x and y, div_tbl.index > k,
                            div_tbl["Dividend Type"].isin(cash_dvd_type))), "Dividend Amount"] *= \
                div_tbl["Dividend Amount"][k]

    # -------------------------------- adjust cash dvd ----
    sec_ret_ts = sec_ts / sec_ts.shift(1)
    div_tbl_cashdvd = div_tbl.loc[div_tbl["Dividend Type"].isin(cash_dvd_type), ["Ex-Date", "Dividend Amount"]]

    for date in sorted(set(div_tbl_cashdvd["Ex-Date"])):
        if date >= sec_ret_ts.index[0] and date <= sec_ret_ts.index[-1]:
            div_adj = sum(div_tbl_cashdvd.loc[div_tbl_cashdvd["Ex-Date"] == date, "Dividend Amount"].dropna(how="any"))
            # if div_adj!= nan
            if div_adj == div_adj:
                # trade date >= div date
                i = sum([0 if x else 1 for x in sec_ts.index >= date])
                # sec_ts.index[i]
                sec_ret_ts[i] = sec_ts[i] / (sec_ts.shift(1)[i] - div_adj)
    sec_ret_ts[0] = 1

    adj_price = sec_ret_ts.cumprod()
    adj_price = adj_price / adj_price[-1] * sec_ts[-1]
    return adj_price


###########################################################
#  best worst VAR case check func
###########################################################

def VARcase(series, qtl=[99, 99.5, 99.7], days=[5, 10], case="WB", start=None, end=None, is_ret=False):
    """
    VARcase(series,qtl=[99,99.5,99.7],days=[5,10],case="WB",start=None,end=None,is_ret=False)
    
    Get hist VAR for one security price series
    args:
        series: pd time series for price or return
        ---------------------------------
        qtl: quantile used for caculation 
        days: days of returns used for caculation      
        case: "W" for worst case "B" for best case "WB" for both
        start: start date in str format
        end: end date in str format
        is_ret: True if input return series, False for price series 
             if true, days param not useful, n day return series as input
        
    use quantile for history data with linear interpolation, other choices: lower, higher, nearest, midpoint
    """
    # date arrangement
    if start:
        start = max(series.index[0], pd.to_datetime(start))
    else:
        start = series.index[0]
    if end:
        end = min(series.index[-1], pd.to_datetime(end))
    else:
        end = series.index[-1]
    # get date
    series = series[np.logical_and(series.index >= start, series.index <= end)]

    # result dict
    try:
        # dataframe
        res = {"Name": series.columns[0]}
    except:
        # pandas series
        res = {"Name": series.name}

    for d in days:
        if is_ret:
            series_ret = series.dropna(how="any")
        else:
            # simple return
            series_ret = pd.Series(series / series.shift(d).values - 1, index=series.index).dropna(how="any")
            # map(math.log,(sec_ts/sec_ts.shift(5)).values)

        for c in case:
            if c == "W":
                w_qtl = [(1 - x / 100.0) for x in qtl]
                for q in w_qtl:  # q = 0.01
                    res[str(d) + "d-" + c + "-" + str(100 - q * 100) + "%"] = series_ret.quantile(q,
                                                                                                  interpolation='linear')
            if c == "B":
                b_qtl = [x / 100.0 for x in qtl]
                for q in b_qtl:
                    res[str(d) + "d-" + c + "-" + str(q * 100) + "%"] = series_ret.quantile(q, interpolation='linear')

    res["DateStart"] = start.strftime('%Y/%m/%d')
    res["DateEnd"] = end.strftime('%Y/%m/%d')
    res["Days"] = len(series)
    return res


###########################################################
#  Expected shortfall case check func
###########################################################

def CVARcase(series, qtl=[99, 99.5, 99.7], days=[5, 10], case="WB", start=None, end=None, is_ret=False):
    """
    CVARcase(series,qtl=[99,99.5,99.7],days=[5,10],case="WB",start=None,end=None)
    
    Get hist CVAR for one security price series
    args:
        series: pd time series for price
        ---------------------------------
        qtl: quantile used for caculation 
        days: days of returns used for caculation      
        case: "W" for worst case "B" for best case "WB" for both
        start: start date in str format
        end: end date in str format
    """
    # date arrangement
    if start:
        start = max(series.index[0], pd.to_datetime(start))
    else:
        start = series.index[0]
    if end:
        end = min(series.index[-1], pd.to_datetime(end))
    else:
        end = series.index[-1]
    # get date
    series = series[np.logical_and(series.index >= start, series.index <= end)]

    # result dict
    try:
        # dataframe
        res = {"Name": series.columns[0]}
    except:
        # pandas series
        res = {"Name": series.name}

    for d in days:
        if is_ret:
            series_ret = series.dropna(how="any")
        else:
            # simple return
            series_ret = pd.Series(series / series.shift(d).values - 1, index=series.index).dropna(how="any")
        # map(math.log,(sec_ts/sec_ts.shift(5)).values)
        for c in case:
            if c == "W":
                w_qtl = [(1 - x / 100.0) for x in qtl]
                for q in w_qtl:  # q = 1%, 0.5%, 0.3%
                    var = series_ret.quantile(q, interpolation='linear')
                    res[str(d) + "d-" + c + "-" + str(100 - q * 100) + "%"] = series_ret[series_ret <= var].mean()
            if c == "B":
                b_qtl = [x / 100.0 for x in qtl]
                for q in b_qtl:
                    var = series_ret.quantile(q, interpolation='linear')
                    res[str(d) + "d-" + c + "-" + str(q * 100) + "%"] = series_ret[series_ret >= var].mean()

    res["DateStart"] = start.strftime('%Y/%m/%d')
    res["DateEnd"] = end.strftime('%Y/%m/%d')
    res["Days"] = len(series)
    return res


###########################################################
#  Maximum Drawdown case check func
###########################################################

def MDDcase(series, days=[5, 10], case="WB", start=None, end=None, is_ret=False):
    """
    MDDcase(series,days=[5,10],case="WB",start=None,end=None)
    
    Get hist MaxDrawDown for one security price series
    args:
        series: pd time series for price
        ---------------------------------
        days: days of returns used for caculation      
        case: "W" for worst case "B" for best case "WB" for both
        start: start date in str format
        end: end date in str format
        is_ret: True will make days descriptive
    """
    # date arrangement
    if start:
        start = max(series.index[0], pd.to_datetime(start))
    else:
        start = series.index[0]
    if end:
        end = min(series.index[-1], pd.to_datetime(end))
    else:
        end = series.index[-1]
    # get date
    series = series[np.logical_and(series.index >= start, series.index <= end)]

    # result dict
    try:
        # dataframe
        res = {"Name": series.columns[0]}
    except:
        # pandas series
        res = {"Name": series.name}

    if is_ret:
        for d in days:
            for c in case:
                if c == "W":
                    # simple return
                    series_ret = series.dropna(how="any")
                    mdd = min(series_ret)
                    mdd_date = series_ret.index[np.argmin(series_ret.values)]
                    res[str(d) + "d-" + c + "-MDD"] = mdd
                    res[str(d) + "d-" + c + "-MDD-date"] = mdd_date
                if c == "B":
                    # simple return
                    series_ret = series.dropna(how="any")
                    mdd = max(series_ret)
                    mdd_date = series_ret.index[np.argmax(series_ret.values)]
                    res[str(d) + "d-" + c + "-MDD"] = mdd
                    res[str(d) + "d-" + c + "-MDD-date"] = mdd_date
    else:
        for d in days:
            if d < len(series):
                for c in case:
                    if c == "W":
                        # simple return
                        series_ret = pd.Series(series / series.shift(1).values - 1, index=series.index).dropna(
                            how="any")
                        mdd = min(series_ret)
                        mdd_date = np.argmin(series_ret)
                        for dnum in range(2, d + 1):
                            series_ret = pd.Series(series / series.shift(dnum).values - 1, index=series.index).dropna(
                                how="any")
                            if min(series_ret) <= mdd:
                                mdd = min(series_ret)
                                mdd_date = np.argmin(series_ret)
                        res[str(d) + "d-" + c + "-MDD"] = mdd
                        res[str(d) + "d-" + c + "-MDD-date"] = mdd_date
                    if c == "B":
                        # simple return
                        series_ret = pd.Series(series / series.shift(1).values - 1, index=series.index).dropna(
                            how="any")
                        mdd = max(series_ret)
                        mdd_date = np.argmax(series_ret)
                        for dnum in range(2, d + 1):
                            series_ret = pd.Series(series / series.shift(dnum).values - 1, index=series.index).dropna(
                                how="any")
                            if max(series_ret) >= mdd:
                                mdd = max(series_ret)
                                mdd_date = np.argmax(series_ret)
                        res[str(d) + "d-" + c + "-MDD"] = mdd
                        res[str(d) + "d-" + c + "-MDD-date"] = mdd_date
            else:
                for c in case:
                    res[str(d) + "d-" + c + "-MDD"] = float("nan")
                    res[str(d) + "d-" + c + "-MDD-date"] = float("nan")

    res["DateStart"] = start.strftime('%Y/%m/%d')
    res["DateEnd"] = end.strftime('%Y/%m/%d')
    res["Days"] = len(series)
    return res


###########################################################
#  Backfill data func
###########################################################

def backfill(sec_ts, sec_name, tkr_tbl, calendar, discontinued_list=["MEXCOMP Index", "KOSTAR Index"],
             special_start={"ERIE US Equity": "1995/10/02", "MADX Index": "1993/02/18", "SPI Index": "1988/01/04"}):
    """
    backfill(sec_ts,sec_name,tkr_tbl,calendar,discontinued_list=["MEXCOMP Index","KOSTAR Index"],
             special_start={"ERIE US Equity":"1995/10/02","MADX Index":"1993/02/18","SPI Index":"1988/01/04"})
    
    backfill data with given calendar (pd timeseries-in / pd timeseries-out)
        sec_ts: time series
        sec_name: security name
        tkr_tbl: sec_master (Group/MARKET_STATUS used) 
        calendar: business days used for the ticker
        discontinued_list: anything discontinued by bloomberg
        special_start: map, key=sec_name, value=start date
        
    tkr_tbl field used: "DVD_HIST_ALL"
                        "MARKET_STATUS"
                        "Group"
                        "DEFAULTED"
    
    all groups override by discontinued_list and special_start
        "Single Name Equity": start from start, end to depends market_status
                       "ETF": start from start, end to depends market_status
                     "Index": start from start, end to depends market_status
          "Single Name Bond": start from start, end to depends defaulted
          "Convertible Bond": start from start, end to depends defaulted
          
    """
    from itertools import groupby
    if len(sec_ts) != 0:
        # modify sec_ts as dividend adjusted
        div_str = tkr_tbl.loc[tkr_tbl["SecName"] == sec_name, "DVD_HIST_ALL"].values[0]
        if div_str:
            div_tbl = get_divtbl(div_str)
            sec_ts = get_adjustedprice(sec_ts, div_tbl)
            print(sec_name + " --- dvd adjusted")

        # use business date as benchmark
        sec_ts = pd.concat([sec_ts, calendar], axis=1).iloc[:, :1]
        group = tkr_tbl.loc[tkr_tbl["SecName"] == sec_name, "Group"].values[0]

        # ----process by different groups
        if group in ["Single Name Equity", "ETF", "Index"]:
            market_status = tkr_tbl.loc[tkr_tbl["SecName"] == sec_name, "MARKET_STATUS"].values[0]
            defaulted = float("nan")
        elif group in ["Single Name Bond", "Convertible Bond"]:
            market_status = float("nan")
            defaulted = tkr_tbl.loc[tkr_tbl["SecName"] == sec_name, "DEFAULTED"].values[0]
        else:
            print("Unspecified group!!! Time series backfilled from start date to calendar end!")
            market_status = float("nan")
            defaulted = float("nan")

        if sec_name in list(special_start.keys()):
            sec_ts = sec_ts.loc[sec_ts.index >= special_start[sec_name], :]

        # if special end(defaulted) data start to end else start to now
        if market_status in ["DLST", "ACQU", "PEND", "UNLS", "PRIV",
                             "LIQU"] or sec_name in discontinued_list or defaulted == "Y":
            sec_ts = sec_ts.loc[sec_ts.first_valid_index():sec_ts.last_valid_index(), :]
        else:
            sec_ts = sec_ts.loc[sec_ts.first_valid_index():, :]
            # penny stock modification, keep data all after $5
            if group == "Single Name Equity":
                if sec_ts.loc[:sec_ts.last_valid_index(), :].iloc[-1, 0] >= 5:
                    end_datapoints = \
                        [sum(1 for i in g) for k, g in groupby(sec_ts[sec_ts.columns[0]].fillna(method="ffill") >= 5) if
                         k == True][-1]
                    sec_ts = sec_ts.iloc[-1 * end_datapoints:, :]
                else:
                    # drop everything
                    return pd.Series(float("nan"), index=[0], name=sec_ts.columns[0])
        sec_ts = sec_ts.fillna(method="ffill")

    return sec_ts


def backfill_equity(sec_ts, sec_name, market_status, calendar, discontinued_list=[], special_start={}):
    """
    backfill_equity(sec_ts,sec_name,market_status,calendar,discontinued_list=[],special_start={})
    
    backfill data with given calendar (pd timeseries-in / pd timeseries-out)
        sec_ts: time series
        sec_name: security name
        calendar: business days used for the ticker
        discontinued_list: anything discontinued by bloomberg
        special_start: map, key=sec_name, value=start date
    
    override by discontinued_list and special_start, otherwise start from start, end to depends market_status
    """
    from itertools import groupby
    if len(sec_ts) != 0:
        # assume dividend adjusted already
        # use business date as benchmark
        sec_ts = pd.concat([sec_ts, calendar], axis=1).iloc[:, :1]

        if sec_name in list(special_start.keys()):
            sec_ts = sec_ts.loc[sec_ts.index >= special_start[sec_name], :]

        # if special end(defaulted) data start to end else start to now
        if market_status in ["DLST", "ACQU", "PEND", "UNLS", "PRIV", "LIQU"] or sec_name in discontinued_list:
            sec_ts = sec_ts.loc[sec_ts.first_valid_index():sec_ts.last_valid_index(), :]
            if sec_name in discontinued_list:
                description = "Not Active(Discontinued)"
            else:
                description = "Not Active(" + market_status + ")"
        else:
            # market status active
            sec_ts = sec_ts.loc[sec_ts.first_valid_index():, :]
            # test >= $5 
            if all((sec_ts.dropna(how="any") >= 5).values.flatten()):
                description = "Good"
            else:
                description = "has_penny_period"
                # penny stock modification, make sure all data after $5
            if sec_ts.loc[:sec_ts.last_valid_index(), :].iloc[-1, 0] >= 5:
                end_datapoints = \
                    [sum(1 for i in g) for k, g in groupby(sec_ts[sec_ts.columns[0]].fillna(method="ffill") >= 5) if
                     k == True][-1]
                sec_ts = sec_ts.iloc[-1 * end_datapoints:, :]
            else:
                # drop everything
                description = "end_with_price<5"
                return pd.Series([], name=sec_ts.columns[0]), description

        sec_ts = sec_ts.fillna(method="ffill")
    return sec_ts, description


###########################################################
#  nan count func
###########################################################
from itertools import groupby


def ts_nan_summary(sec_ts):
    """
    ts_nan_summary(sec_ts)
    
    send in dataframe/timeseries with date index and float values, nan included
    return dict with start date, end date, days, all nan days, all nan days%, max nan days, max nan start, max nan end  
    """
    sec_ts = pd.DataFrame(sec_ts)
    sec_ts["flag"] = [1 if x != x else 0 for x in sec_ts[sec_ts.columns[0]]]

    # summary table for nan and valid
    valid_nan_tbl = [{"if_nan": k, "max_consecutive_nan_days": sum(1 for i in g)} for k, g in
                     groupby(sec_ts["flag"])]  # if k==1]
    valid_nan_tbl = pd.DataFrame(valid_nan_tbl)
    valid_nan_tbl["nan_start"] = [0] + list(valid_nan_tbl["max_consecutive_nan_days"])[:-1]
    valid_nan_tbl["nan_start"] = valid_nan_tbl["nan_start"].cumsum()
    valid_nan_tbl["nan_end"] = valid_nan_tbl["nan_start"] + valid_nan_tbl["max_consecutive_nan_days"] - 1
    valid_nan_tbl["nan_start"] = [sec_ts.index[x] for x in valid_nan_tbl["nan_start"]]
    valid_nan_tbl["nan_end"] = [sec_ts.index[x] for x in valid_nan_tbl["nan_end"]]
    valid_nan_tbl = valid_nan_tbl.loc[valid_nan_tbl["if_nan"] == 1, :]

    res = {"start_date": sec_ts.index[0], "end_date": sec_ts.index[-1], "days": len(sec_ts)}

    if len(valid_nan_tbl) > 0:
        max_nan_days = max(valid_nan_tbl["max_consecutive_nan_days"])
        max_ind = list(valid_nan_tbl["max_consecutive_nan_days"]).index(max_nan_days)
        res.update(dict(valid_nan_tbl.iloc[max_ind, 1:]))
        res["sum_nan_days"] = sum(valid_nan_tbl["max_consecutive_nan_days"])
        res["sum_nan_days%"] = float(res["sum_nan_days"]) / res["days"]
    else:
        res["sum_nan_days"] = 0
        res["sum_nan_days%"] = 0
        res["max_consecutive_nan_days"] = 0

    return res


###########################################################
#  bloomberg trading cost calculation func
###########################################################

def bbg_tradingcost(volatility, size_adv, ba_spread, part=0.2):
    """
    bbg_tradingcost(volatility,size_adv,ba_spread,part=0.2)
    
    trading cost calculation tool using bloomberg TCA model
    tradingcost = instant impact + temporarily impact + permanent impact
    
    params:
        volatility: average 30 day volatility
        part: participation rate
        size_adv: size(# shares)/ADV(average 30 day volume)
        ba_spread: AVERAGE_BID_ASK_SPREAD_% (5D default)
    
    num params:
        get from bloomberg paper 
    """
    #    volatility=0.3328
    #    part = 0.2
    #    size_adv = 1.03
    #    ba_spread = 0.0933
    lambda1 = -0.25 + 3 * (max(min(0.3, part), 0.05) - 0.05)
    cost = (lambda1 * ba_spread / 100.0 + 0.023 * volatility * math.pow(part, 0.76) * math.pow(size_adv / part,
                                                                                               0.19) + (
                    0.03 - 0.0017) * volatility * math.pow(size_adv, (0.81 - 0.08))) * 10000
    return cost
