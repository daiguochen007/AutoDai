# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import requests


def sinafinance_stockdvd_history(stock="000541", date="1994-12-24", indicator="分红"):
    """
    # idea from akshare
    
    新浪财经-发行与分配-分红配股详情
    https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/300670.phtml
    
    :param indicator: choice of {"分红", "配股"}
    :type indicator: str
    :param stock: 股票代码
    :type stock: str
    :param date: 分红配股的具体日期, e.g., "1994-12-24"
    :type date: str
    :return: 指定 indicator, stock, date 的数据
    :rtype: pandas.DataFrame
    """
    if indicator == "分红":
        url = "http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/" + stock + ".phtml"
        r = requests.get(url)
        temp_df = pd.read_html(r.text)[12]
        # temp_df.columns = [item[2] for item in temp_df.columns.tolist()]
        temp_df.columns = ['公告日期', '每10股送股(股)', '每10股转增(股)', '每10股派息(税前)(元)', '进度', '除权除息日', '股权登记日', '红股上市日',
                           '查看详细']
        del temp_df['查看详细']
        if date:
            url = "https://vip.stock.finance.sina.com.cn/corp/view/vISSUE_ShareBonusDetail.php"
            params = {
                "stockid": stock,
                "type": "1",
                "end_date": date,
            }
            r = requests.get(url, params=params)
            temp_df = pd.read_html(r.text)[12]
            temp_df.columns = ["item", "value"]
            return temp_df
        else:
            return temp_df
    else:
        url = "http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/" + stock + ".phtml"
        r = requests.get(url)
        temp_df = pd.read_html(r.text)[13]
        temp_df.columns = [item[1] for item in temp_df.columns.tolist()]
        del temp_df['查看详细']
        if date:
            url = "https://vip.stock.finance.sina.com.cn/corp/view/vISSUE_ShareBonusDetail.php"
            params = {
                "stockid": stock,
                "type": "2",
                "end_date": date,
            }
            r = requests.get(url, params=params)
            temp_df = pd.read_html(r.text)[12]
            temp_df.columns = ["item", "value"]
            return temp_df
        else:
            return temp_df


def sinafinance_capitalstructure_history(stock="000002"):
    """
    新浪财经-股本结构
    http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/000002.phtml
    """
    url = "http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/" + stock + ".phtml"
    r = requests.get(url)
    print('Reorg capital structure history df for [' + stock + ']...')
    res = pd.DataFrame()
    for temp_df in pd.read_html(r.text)[12:]:
        temp_df.index = [x.strip('·') for x in temp_df.iloc[:, 0].values]
        temp_df = temp_df.iloc[:, 1:]
        res = pd.concat([res, temp_df.T], axis=0)

    for c in res.columns:
        res[c] = [x if type(x) != str else x if "万股" not in x else float(x.strip("万股").strip()) * 10000.0 for x in res[c]]
    res.index = list(range(len(res)))
    return res


def get_stockdvd_by_year(stock='000002'):
    """
    股票每年分红总金额
    """
    df_dvd = sinafinance_stockdvd_history(stock, date="", indicator="分红")
    df_capital = sinafinance_capitalstructure_history(stock)
    df_capital = df_capital.sort_values(by='变动日期', ascending=False)

    def get_total_shares(date, df_capital):
        """
        date = "2020-07-30"
        """
        pass
        for dtchg, shs in df_capital[['变动日期', '总股本(历史记录)']].values:
            if date >= dtchg:
                return shs
        print("Warning: query date [" + date + "] before start date [" + dtchg + "]")
        return df_capital['总股本(历史记录)'].values[-1]

    df_dvd['year'] = [x[:4] for x in df_dvd['公告日期']]
    df_dvd['shs_total'] = [get_total_shares(x, df_capital) for x in df_dvd['公告日期']]
    df_dvd['cash_dvd'] = df_dvd['shs_total'] * df_dvd['每10股派息(税前)(元)'] / 10.0
    res = pd.pivot_table(df_dvd, index='year', values='cash_dvd', aggfunc=np.sum)
    res = res.sort_index(axis=0, ascending=False)
    res.columns = ["现金分红"]
    return res


if __name__ == "__main__":
    pass
    # df = sinafinance_stockdvd_history(stock="000002",date="",indicator=u"分红")
    # df = sinafinance_stockdvd_history(stock="000002",date="",indicator=u"配股")
    # df = sinafinance_capitalstructure_history(stock="000002")
    df = get_stockdvd_by_year(stock='000002')
