import alphalens
import tushare as ts
import pandas as pd
pro = ts.pro_api()

df_code = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
df_code = df_code[:20]
df_code['sector_name'] = df_code['industry']
sector_name = df_code.sector_name.unique()
sector_id = range(len(sector_name))

sec_names = dict(zip(sector_id,sector_name))
sec_names_rev = dict(zip(sector_name,sector_id))
df_code['sector_id'] = df_code['sector_name'].map(sec_names_rev)

code_sec = dict(zip(df_code.ts_code,df_code.sector_id))

# 下载池子里所有股票数据 (上证指数 日期为基准)
code = '399300'
start_date = '2013-01-01'
end_date = '2021-06-10'
df = ts.get_k_data(code, start=start_date,end=end_date)
df = df.set_index('date')
df[code] = df.open
df = df[[code]]

for code in df_code.ts_code:
    df_t = ts.get_k_data(code.split('.')[0],start=start_date,end=end_date)
    df_t = df_t.set_index('date')
    df[code] = df_t.open
    print(code+' finished')

df.index = pd.to_datetime(df.index)
del df['399300']

# df = df.fillna(method='ffill')
# df = df.dropna(how='any')
pricing = df

#使用未来5天/过去5天的回报当作因子
lookahead_days = 5
predictive_factor = df.pct_change(lookahead_days)
#predictive_factor = predictive_factor.shift(-lookahead_days)
predictive_factor = predictive_factor.stack()
predictive_factor.index = predictive_factor.index.set_names(['date','asset'])


# Ingest and format data
factor_data = alphalens.utils.get_clean_factor_and_forward_returns(predictive_factor,
                                                                   pricing,
                                                                   quantiles=5,
                                                                   groupby=code_sec,
                                                                   groupby_labels=sec_names)
factor_data.head()

# Run analysis tearsheet
# alphalens.tears.create_full_tear_sheet(factor_data)

######## create_returns_tear_sheet
factor_returns = alphalens.performance.factor_returns(factor_data, demeaned=False, group_adjust=False)
alpha_beta = alphalens.performance.factor_alpha_beta(factor_data, factor_returns, demeaned=False, group_adjust=False)
mean_quant_ret_bydate, std_quant_daily = alphalens.performance.mean_return_by_quantile(factor_data,
                                                                                        by_date=True,
                                                                                        by_group=False,
                                                                                        demeaned=False,
                                                                                        group_adjust=False)
mean_quant_ret, std_quantile = alphalens.performance.mean_return_by_quantile(factor_data,
                                                                            by_group=False,
                                                                            demeaned=False,
                                                                            group_adjust=False)
mean_quant_rateret = mean_quant_ret.apply(alphalens.utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0])
mean_quant_ret_bydate, std_quant_daily = alphalens.performance.mean_return_by_quantile(factor_data,
                                                                                        by_date=True,
                                                                                        by_group=False,
                                                                                        demeaned=False,
                                                                                        group_adjust=False)
mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(alphalens.utils.rate_of_return,axis=0,
                                                        base_period=mean_quant_ret_bydate.columns[0])
compstd_quant_daily = std_quant_daily.apply(alphalens.utils.std_conversion, axis=0,
                                            base_period=std_quant_daily.columns[0])
mean_ret_spread_quant, std_spread_quant = alphalens.performance.compute_mean_returns_spread(
                                                                    mean_quant_rateret_bydate,
                                                                    factor_data["factor_quantile"].max(),
                                                                    factor_data["factor_quantile"].min(),
                                                                    std_err=compstd_quant_daily)
alphalens.plotting.plot_returns_table(alpha_beta, mean_quant_rateret, mean_ret_spread_quant)

# 分组平均回报
mean_return_by_q, std_err_by_q = alphalens.performance.mean_return_by_quantile(factor_data, by_group=False,
                                                                               demeaned=False,
                                                                               group_adjust=False)
mean_return_by_q.head()
alphalens.plotting.plot_quantile_returns_bar(mean_return_by_q)
# 分组未来回报分布
alphalens.plotting.plot_quantile_returns_violin(mean_quant_rateret_bydate, ylim_percentiles=(1, 99))

# 时间序列（高组-低组 的回报 的时间序列是否稳定）
mean_return_by_q_daily, std_err = alphalens.performance.mean_return_by_quantile(factor_data, by_date=True)
quant_return_spread, std_err_spread = alphalens.performance.compute_mean_returns_spread(mean_return_by_q_daily,
                                                                                        upper_quant=5,
                                                                                        lower_quant=1,
                                                                                        std_err=std_err)
alphalens.plotting.plot_mean_quantile_returns_spread_time_series(quant_return_spread, std_err_spread)

# 时间序列 累计
# title = "Factor Weighted Portfolio Cumulative Return (1D Period)"
# alphalens.plotting.plot_cumulative_returns(factor_returns["1D"], period="1D", title=title)
alphalens.plotting.plot_cumulative_returns_by_quantile(mean_quant_rateret_bydate['1D'], period='1D')

# 分类分组，平均回报
mean_return_quantile_group, mean_return_quantile_group_std_err = alphalens.performance.mean_return_by_quantile(
                                                                                                factor_data,
                                                                                                by_date=False,
                                                                                                by_group=True,
                                                                                                demeaned=False,
                                                                                                group_adjust=False)
mean_quant_rateret_group = mean_return_quantile_group.apply(alphalens.utils.rate_of_return, axis=0,
                                                            base_period=mean_return_quantile_group.columns[0])
alphalens.plotting.plot_quantile_returns_bar(mean_quant_rateret_group, by_group=True,ylim_percentiles=(5, 95))


# create_information_tear_sheet
ic = alphalens.performance.factor_information_coefficient(factor_data)
alphalens.plotting.plot_ic_ts(ic)
alphalens.plotting.plot_ic_hist(ic)
alphalens.plotting.plot_ic_qq(ic)

mean_monthly_ic = alphalens.performance.mean_information_coefficient(factor_data, by_time='M')
mean_monthly_ic.head()
alphalens.plotting.plot_monthly_ic_heatmap(mean_monthly_ic)

mean_group_ic = alphalens.performance.mean_information_coefficient(factor_data, group_adjust=False,
                                                                   by_group=True)
alphalens.plotting.plot_ic_by_group(mean_group_ic)

######### create_turnover_tear_sheet
turnover_periods = [1,5,10]
quantile_factor = factor_data["factor_quantile"]
quantile_turnover = {p: pd.concat([alphalens.performance.quantile_turnover(quantile_factor, q, p)
                                   for q in quantile_factor.sort_values().unique().tolist()],
                                  axis=1) for p in turnover_periods}

autocorrelation = pd.concat([alphalens.performance.factor_rank_autocorrelation(factor_data, period)
                             for period in turnover_periods],axis=1)

alphalens.plotting.plot_turnover_table(autocorrelation, quantile_turnover)

for period in turnover_periods:
    alphalens.plotting.plot_top_bottom_quantile_turnover(quantile_turnover[period], period=period)

for period in autocorrelation:
    alphalens.plotting.plot_factor_rank_auto_correlation(autocorrelation[period], period=period)