###############################################################################
# tushare test
###############################################################################

import DaiToolkit as tk
from DaiToolkit import ts

# ---------------------------------------------------------- 交易数据
# get one ticker hist(3 years daily to now)
df = ts.get_hist_data('600690')
# 复权数据 (默认前复权，全历史数据)
df = ts.get_h_data('600383', start='2014-07-05', end='2019-01-09', autype="qfq")
df = ts.get_h_data('600383', start='2014-07-05', end='2019-01-09', autype=None)
# 指数全历史数据
df = ts.get_h_data('399106', index=True)  # 深圳综合指数
# 实时行情
df = ts.get_today_all()
# 历史分笔行情
df = ts.get_tick_data('600690', date='2018-01-15')
df["price"].plot()
# 实时行情
df = ts.get_realtime_quotes('600690')
# 当日历史数据
df = ts.get_today_ticks('601333')
# 大盘所有指数
df = ts.get_index()
# 大单交易数据
df = ts.get_sina_dd('600690', date='2018-01-15', vol=500)

# --------------------------------------------------------- 投资参考数据
# 分配预案
df = ts.profit_data(2017, top=60)
# 业绩预告 yyyy, quarter
df = ts.forecast_data(2017, 3)
# 限售股解禁 yyyy, month
df = ts.xsg_data(2017, 5)
# 基金持股
df = ts.fund_holdings(2017, 4)
# 新股数据
df = ts.new_stocks()
# 融资融券数据
df = ts.sh_margins(start='2018-01-01', end='2018-01-15')
df = ts.sz_margins(start='2018-01-01', end='2018-01-15')
# 融资融券明细数据
df = ts.sh_margin_details(start='2017-07-01', end='2018-01-15', symbol='600690')
df = ts.sz_margin_details('2018-01-12')

# -------------------------------------------------------- 分类数据
# 行业分类
df = ts.get_industry_classified()
df = ts.get_industry_classified('sw')

# 地域分类
df = ts.get_area_classified()
# 中小板
df = ts.get_sme_classified()
# 创业板
df = ts.get_gem_classified()
# st 板块 
df = ts.get_st_classified()
# 沪深300
df = ts.get_hs300s()
# 上证50
df = ts.get_sz50s()
# 中证500 
df = ts.get_zz500s()
# 终止上市
df = ts.get_terminated()
# 暂停交易
df = ts.get_suspended()

# ----------------------------------------------------- 基本面数据
# 全市场基本数据
df = ts.get_stock_basics()
# 业绩报告主表
df = ts.get_report_data(2017, 3)
# 盈利能力
df = ts.get_profit_data(2017, 3)
# 运营能力
df = ts.get_operation_data(2017, 3)
# 成长能力
df = ts.get_growth_data(2014, 3)
# 偿债能力
df = ts.get_debtpaying_data(2014, 3)
# 现金流量
df = ts.get_cashflow_data(2014, 3)

# 财报
df = ts.get_balance_sheet("600383")
df = ts.get_profit_statement("600383")
df = ts.get_cash_flow("600383")

# ----------------------------------------------------- 宏观经济数据
df = ts.get_deposit_rate()
df = ts.get_loan_rate()
# 存款准备金率
df = ts.get_rrr()
# 货币供应
df = ts.get_money_supply()
# 货币供应年余额
df = ts.get_money_supply_bal()
# GDP
df = ts.get_gdp_year()
df = ts.get_gdp_quarter()
df = ts.get_gdp_for()
df = ts.get_gdp_pull()
df = ts.get_gdp_contrib()
# CPI/PPI
df = ts.get_cpi()
df = ts.get_ppi()

# ----------------------------------------------------- 新闻数据
# 实时新闻
df = ts.get_latest_news(top=5, show_content=True)
# 信息地雷
df = ts.get_notices('600690')
# 新浪股吧
df = ts.guba_sina()

# ------------------------------------------------------ 龙虎榜数据
# 龙虎榜列表
df = ts.top_list('2016-06-12')
# 个股上榜统计
df = ts.cap_tops()
# 营业部上榜统计
df = ts.broker_tops()
# 机构席位追踪
df = ts.inst_tops()
# 机构成交明细
df = ts.inst_detail()

# ------------------------------------------------------ 银行间同业拆放利率
# this year shibor curve data
df = ts.shibor_data(2018)
df = ts.shibor_quote_data(2017)
# this year lpr 贷款基础利率
df = ts.lpr_data(2018)

###############################################################################
# ------------------------------------------------------ my toolkit func
###############################################################################
# get nearest Qtr info of all secs
tk.tushare_get_nearestqtr_profitdata()
# get Qtr info of all secs
tk.tushare_getqtr_profitdata((2018, 4))
# download all sec basic info to local
tk.tushare_getallsec_basics()
# get all secs dict (code-name)
tk.tusharelocal_get_allsecnames()
# local EPS TTM
tk.tusharelocal_get_TTMEPS("600383", "20180807")
# pe ttm timeseries
df = tk.tushare_getPETTM("600383", "20100101", end=None, plot=True)
# local ts data w pe
sec_code = "000002"
df = tk.tusharelocal_get_history(sec_code)
import matplotlib.pyplot as plt
import pandas as pd

df.index = pd.DatetimeIndex([str(x) for x in df["trade_date"]])
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
axes[0].plot(df["close"], label="Price")
axes[1].plot(df["PE_TTM"], color="darkgreen", label="PE")
axes[0].grid(ls="--", alpha=0.8)
axes[1].grid(ls="--", alpha=0.8)
axes[1].set_ylim(0, 20)
axes[0].set_title("SecCode [" + sec_code + "]", fontsize=16, fontweight="bold")
axes[0].legend(fontsize=12)
axes[1].legend(fontsize=12)
plt.show()

###############################################################################
# tushare Pro
###############################################################################
import tushare as ts
pro = ts.pro_api()

# 获取交易日历信息
df = pro.trade_cal(exchange='SSE', start_date='20180901', end_date='20181001',
                   fields='exchange,cal_date,is_open,pretrade_date', is_open='0')
df = pro.trade_cal(exchange='SZSE', start_date='20180101', end_date='20181231')
# 上市公司基本信息
df = pro.stock_company(exchange='SZSE', fields='ts_code,chairman,manager,secretary,reg_capital,setup_date,province')
# 历史名称变更记录
df = pro.namechange(ts_code='600848.SH', fields='ts_code,name,start_date,end_date,change_reason')
# IPO新股列表
df = pro.new_share(start_date='20180901', end_date='20181018')

# ---------------------------------------------------------- 交易数据
# get one ticker hist
df = pro.daily(ts_code='600383.SH', start_date='20100705', end_date='20190109')
df = pro.query('daily', ts_code='000001.SZ', start_date='20180701', end_date='20180718')
# get one date
df = pro.daily(trade_date='20180810')

# 复权数据 (默认前复权，全历史数据)
# 前复权行情 qfq / hfq
df = ts.pro_bar(pro_api=pro, ts_code='600383.SH', adj='qfq', start_date='20180101', end_date='20181011')
# 提取000001全部复权因子
df = pro.adj_factor(ts_code='000001.SZ', trade_date='')

# 全部股票每日重要的基本面指标 300积分
df = pro.daily_basic(ts_code='', trade_date='20180726',
                     fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pe_ttm,pb')
# 股票每日停复牌信息
df = pro.suspend(ts_code='600848.SH', suspend_date='', resume_date='', fields='')

# --------------------------------------------------------- 投资参考数据
# 沪深港通资金流向
df = pro.moneyflow_hsgt(start_date='20180125', end_date='20180808')
# 融资融券每日交易汇总数据
df = pro.margin(trade_date='20180802')
# 沪深两市每日融资融券明细
df = pro.margin_detail(trade_date='20180802')
# 上市公司前十大股东
df = pro.top10_holders(ts_code='600000.SH', start_date='20170101', end_date='20171231')
# 上市公司前十大流通股东
df = pro.top10_floatholders(ts_code='600000.SH', start_date='20170101', end_date='20171231')
# 限售股解禁
df = pro.share_float(ann_date='20181220')

# -------------------------------------------------------- 分类数据
# 指数基本信息
df = pro.index_basic(market='SW')
# ----------------------------------------------------- 基本面数据
# 全市场基本数据
df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
# ----------------------------------------------------- 新闻数据
df = pro.cctv_news(date='20181211')

# ------------------------------------------------------ 银行间同业拆放利率
# this year shibor curve data
df = pro.shibor(start_date='20180101', end_date='20181101')
df = pro.shibor_quote(start_date='20180101', end_date='20181101')
# 贷款基础利率
df = pro.shibor_lpr(start_date='20180101', end_date='20181130')
# libor
df = pro.libor(curr_type='USD', start_date='20180101', end_date='20181130')
df = pro.hibor(start_date='20180101', end_date='20181130')
