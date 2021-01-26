# -*- coding: utf-8 -*-
"""
和讯期货数据/日线
Created on Tue Mar 20 19:51:20 2018

@author: Dai
"""

import json
import time
import urllib.request
from datetime import datetime

import pandas as pd


def hxRecords(instrument, timeFrame=1, size=1, includeLastBar=True, to_df=True):
    """
    从和讯获取期货数据
    timeFrame=[1,5,15,30,60,'D','W']
    size for how many data 
    """
    instList = [{
        "xchg": "SHFE",
        "inst": ["fu", "ru", "wr"]
    }, {
        "xchg": "SHFE2",
        "inst": ["ag", "au"]
    }, {
        "xchg": "SHFE3",
        "inst": ["al", "bu", "cu", "hc", "ni", "pb", "rb", "sn", "zn"]
    }, {
        "xchg": "CZCE",
        "inst": ["cf", "fg", "lr", "ma", "oi", "pm", "ri", "rm", "rs", "sf", "sm", "sr", "ta", "wh", "zc"]
    }, {
        "xchg": "DCE",
        "inst": ["a", "b", "bb", "c", "cs", "fb", "i", "j", "jd", "jm", "l", "m", "p", "pp", "v", "y"]
    }]

    pInst = instrument.lower()
    if pInst[-4] != '1':
        pInst = pInst[:-3] + '1' + pInst[-3:]
    xchg = None
    for i in instList:
        if pInst[:-4] in i['inst']:
            xchg = i['xchg']
    if xchg is None:
        print ("获取K线时发生错误: 找不到合约")
        return None
    tfs = [1, 5, 15, 30, 60, 'D', 'W']
    tf = None
    for i in range(len(tfs)):
        if timeFrame == tfs[i]:
            tf = i
    if tf is None:
        print("获取K线时发生错误: K线周期不正确")
        return None
    now = time.localtime()
    timestr = str(now.tm_year + 1) + str(12) + str(31) + '000000'
    resp = 'http://webftcn.hermes.hexun.com/shf/kline?code=' + xchg + pInst + '&start=' + timestr + '&number=-' + str(
        size) + '&type=' + str(tf)
    try:
        resp = urllib.request.urlopen(resp)
        resp = resp.read()[1:-2]
        resp = json.loads(resp)['Data']
    except:
        print('获取K线时发生错误: 不完整的JSON数据')
        return None
    re = []
    pw = float(resp[4])
    for i in resp[0]:
        res = dict(Time=time.mktime(time.strptime(str(i[0]), '%Y%m%d%H%M%S')) * 1000, Open=i[2] / pw, High=i[4] / pw
                   , Low=i[5] / pw, Close=i[3] / pw, Volume=i[6])
        re.append(res)
    if to_df:
        re = pd.DataFrame(re)
        col = []
        for i in re.columns:
            if i is 'Time':
                i = 'Date'
            col.append(i.lower())
        re.columns = col
        re['date'] = [datetime.utcfromtimestamp(t / 1000.0) for t in re['date']]
    return re


# test download
tkr_list = []
for i in [14, 15, 16, 17]:
    for j in ["01", "05", "09"]:
        tkr_list.append("y" + str(i) + j)

data = pd.DataFrame()
for tkr in tkr_list + ["y1801"]:
    try:
        ind_ts = hxRecords(tkr, timeFrame="D", size=100000, includeLastBar=True, to_df=True)
        ind_ts["SecName"] = tkr
        #
        ind_ts.index = ind_ts["date"]
        ind_ts = ind_ts["close"]
        ind_ts.name = tkr
        data = pd.concat([data, ind_ts], axis=1)
        # ind_ts.to_csv("C:/Users/Dai/Desktop/investment/soilbean_oil_daily/"+tkr+".csv",index=False)
        print(tkr + " finished!")
    except:
        print(tkr + " error!")
