import akshare as ak

stock_sse_summary_df = ak.stock_sse_summary()
print(stock_sse_summary_df)

stock_zh_a_spot_df = ak.stock_zh_a_spot()
print(stock_zh_a_spot_df)

# AH日线
stock_zh_ah_daily_df = ak.stock_zh_ah_daily(symbol="00966", start_year="2018", end_year="2020")
print(stock_zh_ah_daily_df)

# A+H同时上市
stock_zh_ah_name_dict = ak.stock_zh_ah_name()
print(stock_zh_ah_name_dict)

# 美股日线
stock_us_daily_df = ak.stock_us_daily(symbol="AAPL", adjust="")
print(stock_us_daily_df)

#
stock_financial_analysis_indicator_df = ak.stock_financial_analysis_indicator(stock="600383")

