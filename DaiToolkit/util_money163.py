# -*- coding: UTF-8 -*-
"""
网易财经抓取

include func:

    money163_10K_rawdata(seccode)
    money163_10K_format_zcfzb(df)
    money163_10K_format_lrb(df)
    money163_10K_format_xjllb(df)
    money163_10K_dupont_analysis(df_res)
    money163_10K_zcfzb_analysis(df_res)
    money163_10K_lrb_analysis(df_res)
    money163_10K_xjllb_analysis(df_res)
    money163_10K_data_ana(seccode)
    
"""
import csv
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
import xlsxwriter

from DaiToolkit import util_basics
from DaiToolkit import util_excel
from DaiToolkit import util_sinafinance


def suojin(num=1):
    return " " * 4 * num


def money163_10K_rawdata(seccode):
    """
    从网易财经抓取财报历史数据
    seccode = "000002" 万科
    """
    res = {}
    for tbl in ["zcfzb", "lrb", "xjllb"]:
        url = "http://quotes.money.163.com/service/" + tbl + "_" + seccode + ".html"
        response = urllib.request.urlopen(url)
        cr = csv.reader(line.decode("gbk") for line in response)
        df = {}
        i = 0
        for row in cr:
            df[i] = row
            i += 1
        del df[i - 1]

        df = pd.DataFrame(df).T
        df.columns = df.iloc[0, :]
        df = df.iloc[1:, :-1]
        df.index = list(range(len(df)))
        # replace --
        df = df.replace("--", 0)
        for c in list(df.columns)[1:]:
            df[c] = [0 if str(x).strip() == '--' else float(x) if x == x else x for x in df[c]]
        res[tbl] = df
        print("get [" + seccode + "_" + tbl + "] finished!")
    return res


def money163_10K_format_zcfzb(df):
    """
    format zcfzb
    """
    tenK_structure = {"asset": {"liquid asset": [0, list(df["报告日期"]).index("流动资产合计(万元)")],
                                "illiquid asset": [list(df["报告日期"]).index("流动资产合计(万元)") + 1,
                                                   list(df["报告日期"]).index("非流动资产合计(万元)")]},
                      "liability": {"liquid liability": [list(df["报告日期"]).index("资产总计(万元)") + 1,
                                                         list(df["报告日期"]).index("流动负债合计(万元)")],
                                    "illiquid liability": [list(df["报告日期"]).index("流动负债合计(万元)") + 1,
                                                           list(df["报告日期"]).index("非流动负债合计(万元)")]},
                      "equity": {"parent equity": [list(df["报告日期"]).index("负债合计(万元)") + 1,
                                                   list(df["报告日期"]).index("归属于母公司股东权益合计(万元)")],
                                 "minor equity": [list(df["报告日期"]).index("归属于母公司股东权益合计(万元)") + 1,
                                                  list(df["报告日期"]).index("少数股东权益(万元)")]}}
    liq_ass = df.iloc[tenK_structure["asset"]["liquid asset"][0]:(tenK_structure["asset"]["liquid asset"][1] + 1),
              :].copy()
    illiq_ass = df.iloc[tenK_structure["asset"]["illiquid asset"][0]:(tenK_structure["asset"]["illiquid asset"][1] + 1),
                :].copy()
    sum_ass = df.iloc[list(df["报告日期"]).index("资产总计(万元)"):list(df["报告日期"]).index("资产总计(万元)") + 1, :].copy()

    liq_lia = df.iloc[tenK_structure["liability"]["liquid liability"][0]:(
            tenK_structure["liability"]["liquid liability"][1] + 1), :].copy()
    illiq_lia = df.iloc[tenK_structure["liability"]["illiquid liability"][0]:(
            tenK_structure["liability"]["illiquid liability"][1] + 1), :].copy()
    sum_lia = df.iloc[list(df["报告日期"]).index("负债合计(万元)"):list(df["报告日期"]).index("负债合计(万元)") + 1, :].copy()

    par_equ = df.iloc[tenK_structure["equity"]["parent equity"][0]:(tenK_structure["equity"]["parent equity"][1] + 1),
              :].copy()
    min_equ = df.iloc[tenK_structure["equity"]["minor equity"][0]:(tenK_structure["equity"]["minor equity"][1] + 1),
              :].copy()
    sum_equ = df.iloc[list(df["报告日期"]).index("所有者权益(或股东权益)合计(万元)"):list(df["报告日期"]).index("所有者权益(或股东权益)合计(万元)") + 1,
              :].copy()

    sum_lia_equ = df.iloc[
                  list(df["报告日期"]).index("负债和所有者权益(或股东权益)总计(万元)"):list(df["报告日期"]).index("负债和所有者权益(或股东权益)总计(万元)") + 1,
                  :].copy()

    for temp in [liq_ass, illiq_ass, liq_lia, illiq_lia, par_equ, min_equ]:
        temp["报告日期"] = [suojin(2) + x for x in temp["报告日期"]]

    def rowdf(mystr):
        return pd.DataFrame([mystr] + [float("nan")] * (len(df.columns) - 1), index=df.columns, columns=[0]).T

    zcfzb = pd.concat([rowdf("资产"), rowdf("    流动资产"), liq_ass, rowdf("    非流动资产"), illiq_ass, sum_ass,
                       rowdf("负债"), rowdf("    流动负债"), liq_lia, rowdf("    非流动负债"), illiq_lia, sum_lia,
                       rowdf("股东权益"), rowdf("    母公司权益"), par_equ, rowdf("    少数股东权益"), min_equ, sum_equ, sum_lia_equ],
                      axis=0)
    zcfzb.index = list(range(len(zcfzb)))
    return zcfzb


def money163_10K_format_lrb(df):
    """
    format lrb
    """
    tenK_structure = {"minus": [list(df["报告日期"]).index("营业总成本(万元)"), list(df["报告日期"]).index("资产减值损失(万元)")],
                      "EPS": list(df["报告日期"]).index("基本每股收益")}

    def rowdf(mystr):
        return pd.DataFrame([mystr] + [float("nan")] * (len(df.columns) - 1), index=df.columns, columns=[0]).T

    lrb = pd.concat([df.iloc[:tenK_structure["minus"][0], :], rowdf("减："),
                     df.iloc[tenK_structure["minus"][0]:tenK_structure["minus"][1] + 1, :], rowdf("加："),
                     df.iloc[tenK_structure["minus"][1] + 1:tenK_structure["EPS"], :], rowdf("每股收益"),
                     df.iloc[tenK_structure["EPS"]:]], axis=0)
    lrb.index = list(range(len(lrb)))

    # spacing
    def spacing(start_str, end_str):
        temp = [list(lrb["报告日期"]).index(start_str), list(lrb["报告日期"]).index(end_str)]
        lrb.iloc[temp[0]:temp[1] + 1, 0] = [suojin(1) + x for x in lrb.iloc[temp[0]:temp[1] + 1, 0]]

    spacing("营业收入(万元)", "其他业务收入(万元)")
    spacing("营业总成本(万元)", "资产减值损失(万元)")
    spacing(suojin(1) + "营业成本(万元)", suojin(1) + "资产减值损失(万元)")
    spacing("公允价值变动收益(万元)", "其他业务利润(万元)")
    spacing(suojin(1) + "对联营企业和合营企业的投资收益(万元)", suojin(1) + "对联营企业和合营企业的投资收益(万元)")
    spacing("营业外收入(万元)", "非流动资产处置损失(万元)")
    spacing("所得税费用(万元)", "未确认投资损失(万元)")

    temp = [list(lrb["报告日期"]).index("净利润(万元)") + 1, list(lrb["报告日期"]).index("每股收益") - 1]
    lrb.iloc[temp[0]:temp[1] + 1, 0] = [suojin(1) + x for x in lrb.iloc[temp[0]:temp[1] + 1, 0]]
    lrb.iloc[list(lrb["报告日期"]).index("基本每股收益"):, 0] = [suojin(1) + x for x in
                                                       lrb.iloc[list(lrb["报告日期"]).index("基本每股收益"):, 0]]
    return lrb


def money163_10K_format_xjllb(df):
    """
    format xjllb
    """
    df.columns = [x.strip() for x in df.columns]
    df["报告日期"] = [x.strip() for x in df["报告日期"]]
    tenK_structure = {"cf operating": [0, list(df["报告日期"]).index("经营活动产生的现金流量净额(万元)")],
                      "cf investing": [list(df["报告日期"]).index("收回投资所收到的现金(万元)"),
                                       list(df["报告日期"]).index("投资活动产生的现金流量净额(万元)")],
                      "cf financing": [list(df["报告日期"]).index("吸收投资收到的现金(万元)"),
                                       list(df["报告日期"]).index("筹资活动产生的现金流量净额(万元)")],
                      "cf FX chg": list(df["报告日期"]).index("汇率变动对现金及现金等价物的影响(万元)"),
                      "cf net chg": [list(df["报告日期"]).index("现金及现金等价物净增加额(万元)"),
                                     list(df["报告日期"]).index("期末现金及现金等价物余额(万元)")],
                      "cf from profit": [list(df["报告日期"]).index("净利润(万元)"), list(df["报告日期"]).index("经营活动产生现金流量净额(万元)")],
                      "cf from invfin": [list(df["报告日期"]).index("债务转为资本(万元)"), list(df["报告日期"]).index("融资租入固定资产(万元)")],
                      "cf net chg2": [list(df["报告日期"]).index("现金的期末余额(万元)"),
                                      list(df["报告日期"]).index("现金及现金等价物的净增加额(万元)")]
                      }

    def rowdf(mystr):
        return pd.DataFrame([mystr] + [float("nan")] * (len(df.columns) - 1), index=df.columns, columns=[0]).T

    xjllb = pd.concat([rowdf("一、经营活动产生的现金流量"), df.iloc[:tenK_structure["cf operating"][1] + 1, :],
                       rowdf("二、投资活动产生的现金流量"),
                       df.iloc[tenK_structure["cf investing"][0]:tenK_structure["cf investing"][1] + 1, :],
                       rowdf("三、筹资活动产生的现金流量"),
                       df.iloc[tenK_structure["cf financing"][0]:tenK_structure["cf financing"][1] + 1, :],
                       rowdf("四、汇率变动对现金及现金等价物的影响"),
                       df.iloc[tenK_structure["cf FX chg"]:tenK_structure["cf FX chg"] + 1, :],
                       rowdf("五、现金及现金等价物净增加额"),
                       df.iloc[tenK_structure["cf net chg"][0]:tenK_structure["cf net chg"][1] + 1, :],
                       rowdf("1、将净利润调节为经营活动的现金流量"),
                       df.iloc[tenK_structure["cf from profit"][0]:tenK_structure["cf from profit"][1] + 1, :],
                       rowdf("2、不涉及现金收支的重大投资和筹资活动"),
                       df.iloc[tenK_structure["cf from invfin"][0]:tenK_structure["cf from invfin"][1] + 1, :],
                       rowdf("3、现金及现金等价物净变动"),
                       df.iloc[tenK_structure["cf net chg2"][0]:tenK_structure["cf net chg2"][1] + 1, :]], axis=0)

    xjllb.index = list(range(len(xjllb)))

    # spacing
    def spacing(start_str, end_str, n=0):
        temp = [list(xjllb["报告日期"]).index(" " * n + start_str), list(xjllb["报告日期"]).index(" " * n + end_str)]
        xjllb.iloc[temp[0]:temp[1] + 1, 0] = [suojin(1) + x for x in xjllb.iloc[temp[0]:temp[1] + 1, 0]]

    spacing("销售商品、提供劳务收到的现金(万元)", "经营活动产生的现金流量净额(万元)")
    spacing("销售商品、提供劳务收到的现金(万元)", "收到的其他与经营活动有关的现金(万元)", 4)
    spacing("购买商品、接受劳务支付的现金(万元)", "支付的其他与经营活动有关的现金(万元)", 4)

    spacing("收回投资所收到的现金(万元)", "投资活动产生的现金流量净额(万元)")
    spacing("收回投资所收到的现金(万元)", "减少质押和定期存款所收到的现金(万元)", 4)
    spacing("购建固定资产、无形资产和其他长期资产所支付的现金(万元)", "增加质押和定期存款所支付的现金(万元)", 4)

    spacing("吸收投资收到的现金(万元)", "筹资活动产生的现金流量净额(万元)")
    spacing("吸收投资收到的现金(万元)", "收到其他与筹资活动有关的现金(万元)", 4)
    spacing("偿还债务支付的现金(万元)", "支付其他与筹资活动有关的现金(万元)", 4)

    spacing("汇率变动对现金及现金等价物的影响(万元)", "汇率变动对现金及现金等价物的影响(万元)")

    spacing("净利润(万元)", "经营活动产生现金流量净额(万元)")
    spacing("债务转为资本(万元)", "融资租入固定资产(万元)")
    spacing("现金的期末余额(万元)", "现金及现金等价物的净增加额(万元)")

    return xjllb


def money163_10K_dupont_analysis(df_res):
    """
    calculate dupont analysis from 10K/10Q
    """
    df_res_ = df_res.copy()
    for k, v in list(df_res_.items()):
        df_res_[k] = df_res[k].copy()
        df_res_[k]["key"] = [util_basics.string_to_unicode(x.strip()) for x in df_res_[k][df_res_[k].columns[0]]]

    df_dupont = pd.concat([df_res_['lrb'][df_res_['lrb']["key"].values == "净利润(万元)"],
                           df_res_['lrb'][df_res_['lrb']["key"].values == "营业总收入(万元)"]], axis=0)
    temp = df_res_['lrb'][df_res_['lrb']["key"].values == "营业总收入(万元)"].copy()
    temp["key"] = "营业总收入(万元) "
    df_dupont = pd.concat([df_dupont, temp], axis=0)
    df_dupont = pd.concat([df_dupont, df_res_['zcfzb'][df_res_['zcfzb']["key"].values == "资产总计(万元)"]], axis=0)
    temp = df_res_['zcfzb'][df_res_['zcfzb']["key"].values == "资产总计(万元)"].copy()
    temp["key"] = "资产总计(万元) "
    df_dupont = pd.concat([df_dupont, temp], axis=0)
    df_dupont = pd.concat([df_dupont, df_res_['zcfzb'][df_res_['zcfzb']["key"].values == "负债合计(万元)"]], axis=0)
    df_dupont = pd.concat([df_dupont, df_res_['lrb'][df_res_['lrb']["key"].values == "归属于母公司所有者的净利润(万元)"]], axis=0)
    df_dupont = pd.concat([df_dupont, df_res_['zcfzb'][df_res_['zcfzb']["key"].values == "少数股东权益(万元)"]], axis=0)

    df_dupont.columns = [util_basics.string_to_unicode(x.strip()) for x in df_dupont.columns]
    df_dupont.index = df_dupont["key"]
    del df_dupont["key"]
    del df_dupont["报告日期"]
    df_dupont = df_dupont.T

    df_dupont["营业净利润率"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), df_dupont["净利润(万元)"],
                                   df_dupont["营业总收入(万元)"]))
    df_dupont["总资产周转率"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), df_dupont["营业总收入(万元)"],
                                   df_dupont["资产总计(万元)"]))
    df_dupont["资产负债率"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), df_dupont["负债合计(万元)"],
                                  df_dupont["资产总计(万元)"]))

    df_dupont["归属母公司净利润占比"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), df_dupont["归属于母公司所有者的净利润(万元)"],
                                       df_dupont["净利润(万元)"]))
    df_dupont["归属母公司权益占比"] = list(map(lambda x, y, z: 1.0 - x / (y - z) if y - z != 0 else float("nan"),
                                      df_dupont["少数股东权益(万元)"], df_dupont["资产总计(万元)"], df_dupont["负债合计(万元)"]))

    df_dupont["总资产净利润率"] = df_dupont["营业净利润率"] * df_dupont["总资产周转率"]
    df_dupont["权益乘数"] = [1.0 / (1.0 - x) if x != 1 else float("nan") for x in df_dupont["资产负债率"]]

    df_dupont["净资产收益率"] = df_dupont["总资产净利润率"] * df_dupont["权益乘数"]
    df_dupont["净资产收益率（母公司）"] = df_dupont["净资产收益率"] * df_dupont["归属母公司净利润占比"] / df_dupont["归属母公司权益占比"]

    df_dupont = df_dupont[["净资产收益率（母公司）", '净资产收益率',
                           '总资产净利润率', '营业净利润率', '净利润(万元)', '营业总收入(万元)',
                           '总资产周转率', '营业总收入(万元) ', '资产总计(万元)',
                           '归属母公司净利润占比',
                           "归属母公司权益占比",
                           '权益乘数', '资产负债率', '负债合计(万元)', '资产总计(万元) ']]

    df_dupont.columns = ["净资产收益率（母公司）", '净资产收益率',
                         suojin(1) + '总资产净利润率', suojin(2) + '营业净利润率', suojin(3) + '净利润(万元)', suojin(3) + '营业总收入(万元)',
                         suojin(2) + '总资产周转率', suojin(3) + '营业总收入(万元) ', suojin(3) + '资产总计(万元)',
                         suojin(1) + '归属母公司净利润占比',
                         suojin(1) + '归属母公司权益占比',
                         suojin(1) + '权益乘数', suojin(2) + '资产负债率', suojin(3) + '负债合计(万元)', suojin(3) + '资产总计(万元) ']
    df_dupont = df_dupont.T
    temp = sorted(df_dupont.columns)[::-1]
    df_dupont["报告日期"] = df_dupont.index
    df_dupont = df_dupont[["报告日期"] + temp]
    return df_dupont


def money163_10K_zcfzb_analysis(df_res):
    """
    calculate zcfzb analysis from 10K/10Q
    """
    df_res_ = {}
    for k, v in list(df_res.items()):
        if k in ['zcfzb', 'lrb', 'xjllb']:
            df_res_[k] = df_res[k].copy()
            df_res_[k].index = df_res_[k][r"报告日期"]
            df_res_[k].index = [util_basics.string_to_unicode(x.strip()) for x in df_res_[k].index]

    df_zcfzb_ana = pd.DataFrame()

    i = 1
    temp = pd.concat([df_res_['zcfzb'].loc[["流动资产合计(万元)"]], df_res_['zcfzb'].loc[["流动负债合计(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["流动比率"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["流动资产合计(万元)"],
                                temp.loc["流动负债合计(万元)"]))
    temp = temp.loc[["流动比率", "流动资产合计(万元)", "流动负债合计(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_zcfzb_ana = pd.concat([df_zcfzb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['zcfzb'].loc[["存货(万元)"]], df_res_['lrb'].loc[["营业成本(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["营业成本(万元,TTM)"] = temp.loc["营业成本(万元)"].rolling(4).sum()
    temp.loc["存货周转天数(TTM)"] = list(map(lambda x, y: 360.0 * x / y if y != 0 else float("nan"), temp.loc["存货(万元)"],
                                       temp.loc["营业成本(万元,TTM)"]))
    temp = temp.loc[["存货周转天数(TTM)", "存货(万元)", "营业成本(万元,TTM)", "营业成本(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_zcfzb_ana = pd.concat([df_zcfzb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['zcfzb'].loc[["应收账款(万元)"]], df_res_['lrb'].loc[["营业收入(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["营业收入(万元,TTM)"] = temp.loc["营业收入(万元)"].rolling(4).sum()
    temp.loc["应收账款周转天数(TTM)"] = list(map(lambda x, y: 360.0 * x / y if y != 0 else float("nan"), temp.loc["应收账款(万元)"],
                                         temp.loc["营业收入(万元,TTM)"]))
    temp = temp.loc[["应收账款周转天数(TTM)", "应收账款(万元)", "营业收入(万元,TTM)", "营业收入(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_zcfzb_ana = pd.concat([df_zcfzb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['zcfzb'].loc[["应付账款(万元)"]], df_res_['lrb'].loc[["营业成本(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["营业成本(万元,TTM)"] = temp.loc["营业成本(万元)"].rolling(4).sum()
    temp.loc["应付账款周转天数(TTM)"] = list(map(lambda x, y: 360.0 * x / y if y != 0 else float("nan"), temp.loc["应付账款(万元)"],
                                         temp.loc["营业成本(万元,TTM)"]))
    temp = temp.loc[["应付账款周转天数(TTM)", "应付账款(万元)", "营业成本(万元,TTM)", "营业成本(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_zcfzb_ana = pd.concat([df_zcfzb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['zcfzb'].loc[["资产总计(万元)"]], df_res_['lrb'].loc[["营业收入(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["营业收入(万元,TTM)"] = temp.loc["营业收入(万元)"].rolling(4).sum()
    temp.loc["总资产周转天数(TTM)"] = list(map(lambda x, y: 360.0 * x / y if y != 0 else float("nan"), temp.loc["资产总计(万元)"],
                                        temp.loc["营业收入(万元,TTM)"]))
    temp = temp.loc[["总资产周转天数(TTM)", "资产总计(万元)", "营业收入(万元,TTM)", "营业收入(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_zcfzb_ana = pd.concat([df_zcfzb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['zcfzb'].loc[["货币资金(万元)"]], df_res_['zcfzb'].loc[["应收账款(万元)"]],
                      df_res_['zcfzb'].loc[["流动负债合计(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["速动比率"] = list(map(lambda x, y, z: (x + y) / z if z != 0 else float("nan"), temp.loc["货币资金(万元)"],
                                temp.loc["应收账款(万元)"], temp.loc["流动负债合计(万元)"]))
    temp = temp.loc[["速动比率", "货币资金(万元)", "应收账款(万元,TTM)", "流动负债合计(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_zcfzb_ana = pd.concat([df_zcfzb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['zcfzb'].loc[["短期借款(万元)"]],
                      df_res_['zcfzb'].loc[["向中央银行借款(万元)"]],
                      df_res_['zcfzb'].loc[["吸收存款及同业存放(万元)"]],
                      df_res_['zcfzb'].loc[["拆入资金(万元)"]],
                      df_res_['zcfzb'].loc[["交易性金融负债(万元)"]],
                      df_res_['zcfzb'].loc[["衍生金融负债(万元)"]],
                      df_res_['zcfzb'].loc[["应付短期债券(万元)"]],
                      df_res_['zcfzb'].loc[["一年内到期的非流动负债(万元)"]],
                      df_res_['zcfzb'].loc[["长期借款(万元)"]],
                      df_res_['zcfzb'].loc[["应付债券(万元)"]],
                      df_res_['zcfzb'].loc[["长期应付款(万元)"]],
                      df_res_['zcfzb'].loc[["专项应付款(万元)"]],
                      df_res_['zcfzb'].loc[["负债合计(万元)"]],
                      df_res_['zcfzb'].loc[["资产总计(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["总有息负债(万元)"] = temp.loc[["短期借款(万元)", "向中央银行借款(万元)", "吸收存款及同业存放(万元)", "拆入资金(万元)",
                                      "交易性金融负债(万元)", "衍生金融负债(万元)", "应付短期债券(万元)", "一年内到期的非流动负债(万元)",
                                      "长期借款(万元)", "应付债券(万元)", "长期应付款(万元)", "专项应付款(万元)"]].sum(axis=0)
    temp.loc["有息负债率"] = list(map(lambda x, y: float(x) / y if y != 0 else float("nan"), temp.loc["总有息负债(万元)"],
                                 temp.loc["资产总计(万元)"]))
    temp.loc["总负债率"] = list(map(lambda x, y: float(x) / y if y != 0 else float("nan"), temp.loc["负债合计(万元)"],
                                temp.loc["资产总计(万元)"]))
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_zcfzb_ana = pd.concat([df_zcfzb_ana, temp], axis=0)

    df_zcfzb_ana = df_zcfzb_ana[sorted(df_zcfzb_ana.columns)[::-1]]
    df_zcfzb_ana.index = list(range(len(df_zcfzb_ana)))
    df_zcfzb_ana.columns = [util_basics.string_to_unicode(x) for x in df_zcfzb_ana.columns]

    ### asset tbl
    temp = df_res_['zcfzb'].iloc[
           list(df_res_['zcfzb'].index).index("流动资产") + 1:list(df_res_['zcfzb'].index).index("流动资产合计(万元)")]
    temp = temp.iloc[:, [1]].copy()
    temp.columns = ["value"]
    temp["type"] = "流动资产(" + str(int(sum(temp["value"]) / 10000.0)) + "亿)"
    temp["item"] = list(map(lambda x, y: x[:-4] + "(" + str(int(y / 10000.0)) + "亿)", temp.index, temp["value"]))
    temp = temp.sort_values(by="value", ascending=False)
    tbl_asset = temp.copy()
    temp = df_res_['zcfzb'].iloc[
           list(df_res_['zcfzb'].index).index("非流动资产") + 1:list(df_res_['zcfzb'].index).index("非流动资产合计(万元)")]
    temp = temp.iloc[:, [1]].copy()
    temp.columns = ["value"]
    temp["type"] = "非流动资产(" + str(int(sum(temp["value"]) / 10000.0)) + "亿)"
    temp["item"] = list(map(lambda x, y: x[:-4] + "(" + str(int(y / 10000.0)) + "亿)", temp.index, temp["value"]))
    temp = temp.sort_values(by="value", ascending=False)
    tbl_asset = pd.concat([tbl_asset, temp], axis=0)
    tbl_asset["perc"] = tbl_asset["value"] / sum(tbl_asset["value"])
    tbl_asset.index = list(range(len(tbl_asset)))
    tbl_asset = tbl_asset[tbl_asset['value'] != 0]
    tbl_asset = tbl_asset[['type', 'item', 'value', "perc"]]

    ### liability tbl
    temp = df_res_['zcfzb'].iloc[
           list(df_res_['zcfzb'].index).index("流动负债") + 1:list(df_res_['zcfzb'].index).index("流动负债合计(万元)")]
    temp = temp.iloc[:, [1]].copy()
    temp.columns = ["value"]
    temp["type"] = "流动负债(" + str(int(sum(temp["value"]) / 10000.0)) + "亿)"
    temp["item"] = list(map(lambda x, y: x[:-4] + "(" + str(int(y / 10000.0)) + "亿)", temp.index, temp["value"]))
    temp = temp.sort_values(by="value", ascending=False)
    tbl_liability = temp.copy()
    temp = df_res_['zcfzb'].iloc[
           list(df_res_['zcfzb'].index).index("非流动负债") + 1:list(df_res_['zcfzb'].index).index("非流动负债合计(万元)")]
    temp = temp.iloc[:, [1]].copy()
    temp.columns = ["value"]
    temp["type"] = "非流动负债(" + str(int(sum(temp["value"]) / 10000.0)) + "亿)"
    temp["item"] = list(map(lambda x, y: x[:-4] + "(" + str(int(y / 10000.0)) + "亿)", temp.index, temp["value"]))
    temp = temp.sort_values(by="value", ascending=False)
    tbl_liability = pd.concat([tbl_liability, temp], axis=0)
    tbl_liability["perc"] = tbl_liability["value"] / sum(tbl_liability["value"])
    tbl_liability.index = list(range(len(tbl_liability)))
    tbl_liability = tbl_liability[tbl_liability['value'] != 0]
    tbl_liability = tbl_liability[['type', 'item', 'value', 'perc']]

    return {"tbl_ts": df_zcfzb_ana, "tbl_asset": tbl_asset, "tbl_liability": tbl_liability}


def money163_10K_lrb_analysis(df_res):
    """
    calculate zcfzb analysis from 10K/10Q
    """
    df_res_ = {}
    for k, v in list(df_res.items()):
        if k in ['zcfzb', 'lrb', 'xjllb']:
            df_res_[k] = df_res[k].copy()
            df_res_[k].index = df_res_[k][r"报告日期"]
            df_res_[k].index = [util_basics.string_to_unicode(x.strip()) for x in df_res_[k].index]

    df_lrb_ana = pd.DataFrame()

    i = 1
    temp = pd.concat([df_res_['lrb'].loc[["营业成本(万元)"]], df_res_['lrb'].loc[["营业收入(万元)"]],
                      df_res_['lrb'].loc[["营业总收入(万元)"]], df_res_['lrb'].loc[["净利润(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["营业毛利率"] = list(map(lambda x, y: 1 - x / y if y != 0 else float("nan"), temp.loc["营业成本(万元)"],
                                 temp.loc["营业收入(万元)"]))
    temp.loc["净利润率"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["净利润(万元)"],
                                temp.loc["营业总收入(万元)"]))
    temp = temp.loc[["营业毛利率", "营业成本(万元)", "营业收入(万元)", "净利润率", "净利润(万元)", "营业总收入(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_lrb_ana = pd.concat([df_lrb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['lrb'].loc[["营业总成本(万元)"]], df_res_['lrb'].loc[["销售费用(万元)"]],
                      df_res_['lrb'].loc[["管理费用(万元)"]], df_res_['lrb'].loc[["财务费用(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["销售费用(%)"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["销售费用(万元)"],
                                   temp.loc["营业总成本(万元)"]))
    temp.loc["管理费用(%)"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["管理费用(万元)"],
                                   temp.loc["营业总成本(万元)"]))
    temp.loc["财务费用(%)"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["财务费用(万元)"],
                                   temp.loc["营业总成本(万元)"]))
    temp = temp.loc[["营业总成本(万元)", "销售费用(万元)", "管理费用(万元)", "财务费用(万元)", "销售费用(%)", "管理费用(%)", "财务费用(%)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_lrb_ana = pd.concat([df_lrb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['lrb'].loc[["营业总收入(万元)"]], df_res_['lrb'].loc[["营业外收入(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["营业外收入/营业总收入"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["营业外收入(万元)"],
                                       temp.loc["营业总收入(万元)"]))
    temp = temp.loc[["营业外收入/营业总收入", "营业外收入(万元)", "营业总收入(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_lrb_ana = pd.concat([df_lrb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['lrb'].loc[["营业外支出(万元)"]], df_res_['lrb'].loc[["净利润(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["营业外支出/净利润"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["营业外支出(万元)"],
                                     temp.loc["净利润(万元)"]))
    temp = temp.loc[["营业外支出/净利润", "营业外支出(万元)", "净利润(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_lrb_ana = pd.concat([df_lrb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['lrb'].loc[["投资收益(万元)"]], df_res_['lrb'].loc[["净利润(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["投资收益/净利润"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["投资收益(万元)"],
                                    temp.loc["净利润(万元)"]))
    temp = temp.loc[["投资收益/净利润", "投资收益(万元)", "净利润(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_lrb_ana = pd.concat([df_lrb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['lrb'].loc[["利润总额(万元)"]], df_res_['lrb'].loc[["所得税费用(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["税率"] = list(
        map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["所得税费用(万元)"], temp.loc["利润总额(万元)"]))
    temp = temp.loc[["税率", "所得税费用(万元)", "利润总额(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_lrb_ana = pd.concat([df_lrb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['zcfzb'].loc[["递延所得税资产(万元)"]], df_res_['zcfzb'].loc[["递延所得税负债(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["递延所得税净资产"] = list(map(lambda x, y: x - y, temp.loc["递延所得税资产(万元)"], temp.loc["递延所得税负债(万元)"]))
    temp = temp.loc[["递延所得税净资产", "递延所得税资产(万元)", "递延所得税负债(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_lrb_ana = pd.concat([df_lrb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['lrb'].loc[["营业收入(万元)"]], df_res_['zcfzb'].loc[["应收账款(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["季度营业收入(万元)"] = temp.loc["营业收入(万元)"] - temp.loc["营业收入(万元)"].shift(1)
    temp.loc["季度营业收入(万元)"] = list(map(lambda x, y, z: x if str(z)[5:7] != "03" else y, temp.loc["季度营业收入(万元)"],
                                      temp.loc["营业收入(万元)"], temp.columns))
    temp.loc["应收账款QoQ(万元)"] = temp.loc["应收账款(万元)"] - temp.loc["应收账款(万元)"].shift(1)
    temp = temp.loc[["营业收入(万元)", "季度营业收入(万元)", "应收账款(万元)", "应收账款QoQ(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_lrb_ana = pd.concat([df_lrb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['lrb'].loc[["营业利润(万元)", "财务费用(万元)", '利润总额(万元)', '所得税费用(万元)']],
                      df_res_['zcfzb'].loc[["短期借款(万元)", '长期借款(万元)', '应付债券(万元)', '应付票据(万元)',
                                            '一年内到期的非流动负债(万元)', '应付短期债券(万元)', '所有者权益(或股东权益)合计(万元)']]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["EBIT*(1-Tax)(万元)"] = [(x[0] + x[1]) * min(1 - x[2] / x[3], 1) for x in
                                    temp.loc[["营业利润(万元)", "财务费用(万元)", "所得税费用(万元)", "利润总额(万元)"]].T.values]
    temp.loc["投入资本(万元)"] = temp.loc[["短期借款(万元)", '长期借款(万元)', '应付债券(万元)', '应付票据(万元)',
                                     '一年内到期的非流动负债(万元)', '应付短期债券(万元)', '所有者权益(或股东权益)合计(万元)']].sum(axis=0)
    temp.loc["ROIC"] = list(map(lambda x, y: x / y if y != 0 else float('nan'), temp.loc["EBIT*(1-Tax)(万元)"],
                                temp.loc["投入资本(万元)"]))
    temp = temp.loc[["EBIT*(1-Tax)(万元)", "投入资本(万元)", "ROIC"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_lrb_ana = pd.concat([df_lrb_ana, temp], axis=0)

    df_lrb_ana = df_lrb_ana[sorted(df_lrb_ana.columns)[::-1]]
    df_lrb_ana.index = list(range(len(df_lrb_ana)))
    df_lrb_ana.columns = list(map(util_basics.string_to_unicode, df_lrb_ana.columns))

    return df_lrb_ana


def money163_10K_xjllb_analysis(df_res):
    """
    calculate xjllb analysis from 10K/10Q
    """
    df_res_ = {}
    for k, v in list(df_res.items()):
        if k in ['zcfzb', 'lrb', 'xjllb']:
            df_res_[k] = df_res[k].copy()
            df_res_[k].index = df_res_[k][r"报告日期"]
            df_res_[k].index = [util_basics.string_to_unicode(x.strip()) for x in df_res_[k].index]

    df_xjllb_ana = pd.DataFrame()

    i = 1
    temp = pd.concat([df_res_['xjllb'].loc[["经营活动产生的现金流量净额(万元)"]], df_res_['xjllb'].loc[["投资活动产生的现金流量净额(万元)"]],
                      df_res_['xjllb'].loc[["筹资活动产生的现金流量净额(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_xjllb_ana = pd.concat([df_xjllb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['xjllb'].loc[["经营活动产生的现金流量净额(万元)"]], df_res_['lrb'].loc[["净利润(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["经营现金流/净利润"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["经营活动产生的现金流量净额(万元)"],
                                     temp.loc["净利润(万元)"]))
    temp = temp.loc[["经营现金流/净利润", "经营活动产生的现金流量净额(万元)", "净利润(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_xjllb_ana = pd.concat([df_xjllb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['xjllb'].loc[["投资活动现金流入小计(万元)"]], df_res_['xjllb'].loc[["投资活动现金流出小计(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["投资流入/投资流出"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["投资活动现金流入小计(万元)"],
                                     temp.loc["投资活动现金流出小计(万元)"]))
    temp = temp.loc[["投资流入/投资流出", "投资活动现金流入小计(万元)", "投资活动现金流出小计(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_xjllb_ana = pd.concat([df_xjllb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['xjllb'].loc[["筹资活动现金流入小计(万元)"]], df_res_['xjllb'].loc[["筹资活动现金流出小计(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["筹资流入/筹资流出"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["筹资活动现金流入小计(万元)"],
                                     temp.loc["筹资活动现金流出小计(万元)"]))
    temp = temp.loc[["筹资流入/筹资流出", "筹资活动现金流入小计(万元)", "筹资活动现金流出小计(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_xjllb_ana = pd.concat([df_xjllb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['xjllb'].loc[["销售商品、提供劳务收到的现金(万元)"]], df_res_['lrb'].loc[["营业收入(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["收现比"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["销售商品、提供劳务收到的现金(万元)"],
                               temp.loc["营业收入(万元)"]))
    temp = temp.loc[["收现比", "销售商品、提供劳务收到的现金(万元)", "营业收入(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_xjllb_ana = pd.concat([df_xjllb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['zcfzb'].loc[["货币资金(万元)"]], df_res_['zcfzb'].loc[["资产总计(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.loc["现金/总资产"] = list(map(lambda x, y: x / y if y != 0 else float("nan"), temp.loc["货币资金(万元)"],
                                  temp.loc["资产总计(万元)"]))
    temp = temp.loc[["现金/总资产", "货币资金(万元)", "资产总计(万元)"]]
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_xjllb_ana = pd.concat([df_xjllb_ana, temp], axis=0)

    i += 1
    temp = pd.concat([df_res_['xjllb'].loc[["期末现金及现金等价物余额(万元)"]],
                      df_res_['xjllb'].loc[["分配股利、利润或偿付利息所支付的现金(万元)"]]], axis=0)
    temp[r"报告日期"] = 0
    temp.index = [str(i) + '. ' + x for x in temp.index]
    temp[r"报告日期"] = temp.index
    df_xjllb_ana = pd.concat([df_xjllb_ana, temp], axis=0)

    df_xjllb_ana = df_xjllb_ana[sorted(df_xjllb_ana.columns)[::-1]]
    df_xjllb_ana.index = list(range(len(df_xjllb_ana)))
    df_xjllb_ana.columns = [util_basics.string_to_unicode(x) for x in df_xjllb_ana.columns]

    return df_xjllb_ana


def money163_10K_to_yearly(df):
    """
    filter data to yearly
    """
    df_res = df[[df.columns[0]] + [x for x in df.columns if '12-31' in x]].copy()
    df_res.columns = [df.columns[0]] + [x[:4] for x in df.columns if '12-31' in x]
    return df_res


def dvd_analysis(seccode, res):
    """
    res = money163_10K_rawdata(seccode)
    res includes all dfs after analysis
    """
    df_dvd = util_sinafinance.get_stockdvd_by_year(seccode)
    df_dvd['现金分红'] = df_dvd['现金分红'] / 10000.0
    df_dvd.columns = ['现金分红(万元)']

    res_ = {}
    for k, v in list(res.items()):
        if k in ['zcfzb', 'lrb', 'xjllb']:
            res_[k] = res[k].copy()
            res_[k].index = res_[k][r"报告日期"]
            res_[k].index = [util_basics.string_to_unicode(x.strip()) for x in res_[k].index]
            res_[k] = money163_10K_to_yearly(res_[k])
            del res_[k][r"报告日期"]
        elif k in ['lrbfx', 'dbfx']:
            res_[k] = res[k].copy()
            res_[k].index = res_[k]["报告日期"]
            res_[k].index = [x.split('.')[-1].strip() for x in res_[k].index]
            res_[k] = money163_10K_to_yearly(res_[k])
            del res_[k]["报告日期"]

    df_res = pd.concat([df_dvd, res_['lrb'].loc[['归属于母公司所有者的净利润(万元)']].T], axis=1, sort=True)
    df_res = df_res.sort_index(ascending=False)
    df_res = df_res.dropna(how='any')
    df_res['分红比例'] = df_res['现金分红(万元)'] / df_res['归属于母公司所有者的净利润(万元)']
    df_res['利润增长率'] = df_res['归属于母公司所有者的净利润(万元)'] / df_res['归属于母公司所有者的净利润(万元)'].shift(-1) - 1
    df_res['利润增长率3y'] = (df_res['归属于母公司所有者的净利润(万元)'] / df_res['归属于母公司所有者的净利润(万元)'].shift(-3)) ** (1 / 3.0) - 1.0
    df_res['利润增长率5y'] = (df_res['归属于母公司所有者的净利润(万元)'] / df_res['归属于母公司所有者的净利润(万元)'].shift(-5)) ** (1 / 5.0) - 1.0
    df_res = pd.concat([df_res, res_['dbfx'].loc[['净资产收益率（母公司）']].T], axis=1, sort=True)
    df_res = pd.concat([df_res, res_['lrbfx'].loc[['ROIC']].T], axis=1, sort=True)
    df_res = df_res.sort_index(ascending=False)

    return df_res


def money163_10K_data_ana(seccode):
    """
    抓取网易财经财报数据
    合并新浪财经分红数据
    
    return df:
        
        zcfzb   资产负债表
        lrb     利润表
        xjllb   现金流量表
        
        zcfzbfx 资产负债表分析
        lrbfx   利润表分析
        xjllbfx 现金流量表分析
        
        dbfx 杜邦分析
        
    """
    res = money163_10K_rawdata(seccode)
    res["zcfzb"] = money163_10K_format_zcfzb(res["zcfzb"])
    res["lrb"] = money163_10K_format_lrb(res["lrb"])
    res["xjllb"] = money163_10K_format_xjllb(res["xjllb"])
    res["dbfx"] = money163_10K_dupont_analysis(res)
    print("[" + seccode + "] Dupont analysis finished!")
    res["zcfzbfx"] = money163_10K_zcfzb_analysis(res)
    print("[" + seccode + "] zcfzb analysis finished!")
    res["lrbfx"] = money163_10K_lrb_analysis(res)
    print("[" + seccode + "] lrb analysis finished!")
    res["xjllbfx"] = money163_10K_xjllb_analysis(res)
    print("[" + seccode + "] xjllb analysis finished!")
    res["dvdfx"] = dvd_analysis(seccode, res)
    print("[" + seccode + "] dvd analysis finished!")

    # convert all to unicode
    for k, v in list(res.items()):
        try:
            res[k].columns = list(map(util_basics.string_to_unicode, res[k].columns))
            res[k]['报告日期'] = list(map(util_basics.string_to_unicode, res[k]['报告日期']))
        except:
            pass
    return res


#############################################################
# Summarize / Excel
#############################################################
def download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='杜邦分析',
                                          ncols=49, series_name="", line="2", insert_cell="A1", num_format="0.00%",
                                          color="black"):
    """
    color: 'blue', 'red', 'green', '#FF9900'
    """
    if type(line) != list:
        line = [line]
    if type(color) != list:
        color = [color]

    chart1 = workbook.add_chart({'type': 'line', })
    for l, c in zip(line, color):
        chart1.add_series({
            'name': '=' + sheet_name + '!$A$' + l,
            'categories': '=' + sheet_name + '!$B$1:$' + util_excel.excel_colnum_str(ncols) + '$1',
            'values': '=' + sheet_name + '!$B$' + l + ':$' + util_excel.excel_colnum_str(ncols) + '$' + l,
            'marker': {'type': 'square', 'size': 5, 'border': {'color': c}, 'fill': {'color': 'black'}},
            'line': {'color': c}
        })
    chart1.set_title({'name': secname + '-' + series_name})
    chart1.set_x_axis({'reverse': True})
    chart1.set_y_axis({'num_format': num_format})
    chart1.set_legend({'position': 'bottom'})
    # chart1.set_style(1)
    worksheet.insert_chart(insert_cell, chart1,
                           {'x_offset': 0, 'y_offset': 0, 'x_scale': 1.46, 'y_scale': 1.5})  # A1-K22


def download_finfdtml_excel(seccode, secname, dir_path):
    """ get fdmtl data online, perf analysis ad save """
    df_res = money163_10K_data_ana(seccode)

    fname = secname + seccode + ".xlsx"
    fname = fname.replace("*", "^")
    excel_path = os.path.abspath(dir_path + "/" + fname)

    # excel output
    workbook = xlsxwriter.Workbook(excel_path)

    worksheet = workbook.add_worksheet("资产负债表")
    util_excel.xlsxwriter_dftoExcel(df_res["zcfzb"], worksheet, index=False, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("利润表")
    util_excel.xlsxwriter_dftoExcel(df_res["lrb"], worksheet, index=False, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("现金流量表")
    util_excel.xlsxwriter_dftoExcel(df_res["xjllb"], worksheet, index=False, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("杜邦分析")
    util_excel.xlsxwriter_dftoExcel(df_res["dbfx"], worksheet, index=False, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("资产负债表分析")
    util_excel.xlsxwriter_dftoExcel(df_res["zcfzbfx"]["tbl_ts"], worksheet, index=False, header=True, startcol=0,
                                    startrow=0)
    util_excel.xlsxwriter_dftoExcel(df_res["zcfzbfx"]["tbl_asset"], worksheet, index=False, header=True, startcol=0,
                                    startrow=len(df_res["zcfzbfx"]["tbl_ts"]) + 2)
    util_excel.xlsxwriter_dftoExcel(df_res["zcfzbfx"]["tbl_liability"], worksheet, index=False, header=True, startcol=5,
                                    startrow=len(df_res["zcfzbfx"]["tbl_ts"]) + 2)
    worksheet = workbook.add_worksheet("利润表分析")
    util_excel.xlsxwriter_dftoExcel(df_res["lrbfx"], worksheet, index=False, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("现金流量表分析")
    util_excel.xlsxwriter_dftoExcel(df_res["xjllbfx"], worksheet, index=False, header=True, startcol=0, startrow=0)
    worksheet = workbook.add_worksheet("分红分析")
    util_excel.xlsxwriter_dftoExcel(df_res["dvdfx"], worksheet, index=True, header=True, startcol=0, startrow=0)

    worksheet = workbook.add_worksheet("图表_杜邦分析")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='杜邦分析',
                                          ncols=len(df_res["dbfx"].columns), series_name="净资产收益率ROE", line=["2", "3"],
                                          insert_cell="A23", num_format="0.00%", color=["#4ed865", "#3faec6"])
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='杜邦分析',
                                          ncols=len(df_res["dbfx"].columns), series_name="总资产收益率ROA", line="4",
                                          insert_cell="L1", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='杜邦分析',
                                          ncols=len(df_res["dbfx"].columns), series_name="总资产周转率", line="8",
                                          insert_cell="W1", color="#f2bf48")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='杜邦分析',
                                          ncols=len(df_res["dbfx"].columns), series_name="权益乘数", line="13",
                                          insert_cell="L45", num_format="0.00", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='杜邦分析',
                                          ncols=len(df_res["dbfx"].columns), series_name="归母净利润占比", line="11",
                                          insert_cell="L23", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='杜邦分析',
                                          ncols=len(df_res["dbfx"].columns), series_name="母公司权益占比", line="12",
                                          insert_cell="L67", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='杜邦分析',
                                          ncols=len(df_res["dbfx"].columns), series_name="营业净利润率", line="5",
                                          insert_cell="W23", color="#f2bf48")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='杜邦分析',
                                          ncols=len(df_res["dbfx"].columns), series_name="资产负债率", line="14",
                                          insert_cell="W45", color="#e55454")

    worksheet = workbook.add_worksheet("图表_资产负债表分析")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='资产负债表分析',
                                          ncols=len(df_res["zcfzbfx"]["tbl_ts"].columns), series_name="流动比率", line="2",
                                          insert_cell="A1", num_format="0.00", color="#4ed865")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='资产负债表分析',
                                          ncols=len(df_res["zcfzbfx"]["tbl_ts"].columns), series_name="速动比率",
                                          line="21",
                                          insert_cell="A23", num_format="0.00", color="#4ed865")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='资产负债表分析',
                                          ncols=len(df_res["zcfzbfx"]["tbl_ts"].columns), series_name="存货周转天数(TTM)",
                                          line="5",
                                          insert_cell="L1", num_format="0", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='资产负债表分析',
                                          ncols=len(df_res["zcfzbfx"]["tbl_ts"].columns), series_name="总资产周转天数(TTM)",
                                          line="17",
                                          insert_cell="L23", num_format="0", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='资产负债表分析',
                                          ncols=len(df_res["zcfzbfx"]["tbl_ts"].columns), series_name="应收账款周转天数(TTM)",
                                          line="9",
                                          insert_cell="W1", num_format="0", color="#f2bf48")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='资产负债表分析',
                                          ncols=len(df_res["zcfzbfx"]["tbl_ts"].columns), series_name="应付账款周转天数(TTM)",
                                          line="13",
                                          insert_cell="W23", num_format="0", color="#f2bf48")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='资产负债表分析',
                                          ncols=len(df_res["zcfzbfx"]["tbl_ts"].columns), series_name="有息负债率 vs 总负债率",
                                          line=["40", "41"],
                                          insert_cell="A45", num_format="0.00%", color=["#4ed865", "#cc862a"])

    worksheet = workbook.add_worksheet("图表_利润表分析")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='利润表分析',
                                          ncols=len(df_res["lrbfx"].columns), series_name="利润率", line=["2", "5"],
                                          insert_cell="A1", num_format="0.00", color=["#4ed865", "#3faec6"])
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='利润表分析',
                                          ncols=len(df_res["lrbfx"].columns), series_name="三费占总成本比例",
                                          line=["12", "13", "14"],
                                          insert_cell="L1", num_format="0.00%", color=["#4ed865", "#cc862a", "#c12c63"])
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='利润表分析',
                                          ncols=len(df_res["lrbfx"].columns), series_name="营业外支出/净利润", line="18",
                                          insert_cell="W1", num_format="0.00%", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='利润表分析',
                                          ncols=len(df_res["lrbfx"].columns), series_name="营业外收入/营业总收入", line="15",
                                          insert_cell="W23", num_format="0.00%", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='利润表分析',
                                          ncols=len(df_res["lrbfx"].columns), series_name="投资收益/净利润", line="21",
                                          insert_cell="L23", num_format="0.00%", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='利润表分析',
                                          ncols=len(df_res["lrbfx"].columns), series_name="税率", line="24",
                                          insert_cell="A23", num_format="0.00%", color="#e55454")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='利润表分析',
                                          ncols=len(df_res["lrbfx"].columns), series_name="递延所得税资产&负债",
                                          line=["27", "28", "29"],
                                          insert_cell="A45", num_format="#,##0",
                                          color=["#4ed865", "#cc862a", "#c12c63"])
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='利润表分析',
                                          ncols=len(df_res["lrbfx"].columns), series_name="营业收入 vs 应收账款变动（季度）",
                                          line=["31", "33"],
                                          insert_cell="L45", num_format="#,##0", color=["#4ed865", "#cc862a"])
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='利润表分析',
                                          ncols=len(df_res["lrbfx"].columns), series_name="ROIC", line=["36"],
                                          insert_cell="W45", num_format="0.00%", color=["#4ed865"])

    worksheet = workbook.add_worksheet("图表_现金流量表分析")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='现金流量表分析',
                                          ncols=len(df_res["xjllbfx"].columns), series_name="现金流量净额",
                                          line=["2", "3", "4"],
                                          insert_cell="A1", num_format="#,##0", color=["#4ed865", "#cc862a", "#c12c63"])
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='现金流量表分析',
                                          ncols=len(df_res["xjllbfx"].columns), series_name="经营现金流/净利润", line="5",
                                          insert_cell="A23", num_format="0.00%", color="#4ed865")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='现金流量表分析',
                                          ncols=len(df_res["xjllbfx"].columns), series_name="投资流入/投资流出", line="8",
                                          insert_cell="L1", num_format="0.00%", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='现金流量表分析',
                                          ncols=len(df_res["xjllbfx"].columns), series_name="筹资流入/筹资流出", line="11",
                                          insert_cell="L23", num_format="0.00%", color="#4193db")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='现金流量表分析',
                                          ncols=len(df_res["xjllbfx"].columns), series_name="收现比", line="14",
                                          insert_cell="W1", num_format="0.00%", color="#e55454")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='现金流量表分析',
                                          ncols=len(df_res["xjllbfx"].columns), series_name="现金/总资产", line="17",
                                          insert_cell="W23", num_format="0.00%", color="#e55454")
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='现金流量表分析',
                                          ncols=len(df_res["xjllbfx"].columns), series_name="经营现金流 vs 净利润",
                                          line=["2", "7"],
                                          insert_cell="A45", num_format="#,##0", color=["#4ed865", "#cc862a"])
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='现金流量表分析',
                                          ncols=len(df_res["xjllbfx"].columns), series_name="销售收入现金 vs 营业收入",
                                          line=["15", "16"],
                                          insert_cell="L45", num_format="#,##0", color=["#4ed865", "#cc862a"])
    download_finfdtml_excel_add_linegraph(workbook, worksheet, secname, sheet_name='现金流量表分析',
                                          ncols=len(df_res["xjllbfx"].columns), series_name="现金余额 vs 利息分红支出",
                                          line=["20", "21"],
                                          insert_cell="W45", num_format="#,##0", color=["#4ed865", "#cc862a"])
    workbook.close()

    # format
    import win32com.client as win32
    excel = win32.DispatchEx('Excel.Application')
    excel.Visible = False
    wb = excel.Workbooks.Open(excel_path, ReadOnly='False')

    for worksheet in wb.Sheets:
        if worksheet.Name == "资产负债表":
            util_excel.excel_tbl_bscfmt(worksheet, df_res["zcfzb"])
            worksheet.Range("B:ZZ").NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()
            window = wb.Windows(1)
            window.SplitRow = 1
            window.FreezePanes = True

        if worksheet.Name == "利润表":
            util_excel.excel_tbl_bscfmt(worksheet, df_res["lrb"])
            worksheet.Range("B:ZZ").NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Range(str(len(df_res["lrb"])+1)+":"+str(len(df_res["lrb"])+1)).NumberFormat = "#,##0.00;[红色](#,##0.00)"
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()
            window = wb.Windows(1)
            window.SplitRow = 1
            window.FreezePanes = True

        if worksheet.Name == "现金流量表":
            util_excel.excel_tbl_bscfmt(worksheet, df_res["xjllb"])
            worksheet.Range("B:ZZ").NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()
            window = wb.Windows(1)
            window.SplitRow = 1
            window.FreezePanes = True

        if worksheet.Name == "杜邦分析":
            util_excel.excel_tbl_bscfmt(worksheet, df_res["dbfx"])
            worksheet.Range("B:ZZ").NumberFormat = "0.00%;[红色](0.00%)"
            worksheet.Range("6:7,9:10,15:16").NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Range("13:13").NumberFormat = "0.00;[红色](#,##0.00)"
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()

        if worksheet.Name == "资产负债表分析":
            util_excel.excel_tbl_bscfmt(worksheet, df_res["zcfzbfx"]["tbl_ts"])
            util_excel.excel_tbl_bscfmt(worksheet, df_res["zcfzbfx"]["tbl_asset"], index=False, startcol=0,
                                        startrow=len(df_res["zcfzbfx"]["tbl_ts"]) + 2)
            util_excel.excel_tbl_bscfmt(worksheet, df_res["zcfzbfx"]["tbl_liability"], index=False, startcol=5,
                                        startrow=len(df_res["zcfzbfx"]["tbl_ts"]) + 2)
            worksheet.Range("B2:ZZ" + str(len(df_res["zcfzbfx"]["tbl_ts"])+1)).NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Range("40:41").NumberFormat = "0.00%;[红色](0.00%)"
            worksheet.Range("2:2,21:21").NumberFormat = "0.00;[红色](#,##0.00)"

            worksheet.Range("C"+str(len(df_res["zcfzbfx"]["tbl_ts"]) + 3)+":C"+str(len(df_res["zcfzbfx"]["tbl_ts"]) + 100)).NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Range("H"+str(len(df_res["zcfzbfx"]["tbl_ts"]) + 3)+":H"+str(len(df_res["zcfzbfx"]["tbl_ts"]) + 100)).NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Range("D"+str(len(df_res["zcfzbfx"]["tbl_ts"]) + 3)+":D"+str(len(df_res["zcfzbfx"]["tbl_ts"]) + 100)).NumberFormat = "0.00%;[红色](0.00%)"
            worksheet.Range("I"+str(len(df_res["zcfzbfx"]["tbl_ts"]) + 3)+":I"+str(len(df_res["zcfzbfx"]["tbl_ts"]) + 100)).NumberFormat = "0.00%;[红色](0.00%)"
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()

        if worksheet.Name == "利润表分析":
            util_excel.excel_tbl_bscfmt(worksheet, df_res["lrbfx"])
            worksheet.Range("B:ZZ").NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Range("2:2,5:5,12:15,18:18,21:21,24:24,36:36").NumberFormat = "0.00%;[红色](0.00%)"
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()

        if worksheet.Name == "现金流量表分析":
            util_excel.excel_tbl_bscfmt(worksheet, df_res["xjllbfx"])
            worksheet.Range("B:ZZ").NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Range("5:5,8:8,11:11,14:14,17:17").NumberFormat = "0.00%;[红色](0.00%)"
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()

        if worksheet.Name == "分红分析":
            util_excel.excel_tbl_bscfmt(worksheet, df_res["dvdfx"], index=True)
            worksheet.Range("B:C").NumberFormat = "#,##0;[红色](#,##0)"
            worksheet.Range("D:I").NumberFormat = "0.00%;[红色](0.00%)"
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 90
            worksheet.Columns.AutoFit()

        if worksheet.Name == "图表_杜邦分析":
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 70

        if worksheet.Name == "图表_资产负债表分析":
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 70

        if worksheet.Name == "图表_利润表分析":
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 70

        if worksheet.Name == "图表_现金流量表分析":
            worksheet.Activate()
            excel.ActiveWindow.Zoom = 70

    wb.Worksheets("图表_杜邦分析").Activate()
    wb.Close(True)


# %%
################################# main
if __name__ == "__main__":
    pass

    seccode = '601988'
    secname = "中国银行"
    dir_path = "/product/Dai_FinAna"
    download_finfdtml_excel(seccode, secname, dir_path)

    # res = money163_10K_data_ana(seccode)
