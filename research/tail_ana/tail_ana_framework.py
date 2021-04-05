
import pandas as pd
import DaiToolkit as tk

# Data
df_raw = tk.get_history_data("sh000016", source='akshare', market='INDEX')  # 上证50
df_raw = tk.get_history_data("sh000300", source='akshare', market='INDEX')  # 沪深300

df_raw = pd.read_excel('C:/Users/Dai/Desktop/investment/data/crypto/BTC.xlsx', 'BTC')  # BTC
df_raw.columns = [x.lower() for x in df_raw.columns]
df_raw.index = df_raw['date']

# basic stats
df_basic_stats = tk.timeseries_ret_distri_stats(ts_px=df_raw["close"], log_ret=False, plot=True, plot_max_freq=252)

df_basic_stats = tk.timeseries_ret_distri_stats_ndays(ts_px=df_raw["close"], ndays=[1, 5, 10, 20, 40, 60, 120], log_ret=False)


# ret distribution
#fit_params = tk.timeseries_fit_ret_distri(df_raw["close"], freq=5, dis_type="norm", plot=True, bins=300)

# tail stats
df_tail = tk.timeseries_tail_ana(df_raw["close"], ret_freq=18, tail_start=None, plot=True)

df_tail_stat = pd.DataFrame()
for i in range(1, 41, 1):
    df_tail_stat = pd.concat([df_tail_stat, tk.timeseries_tail_ana(df_raw["close"], ret_freq=i, tail_start=None, plot=False).T], axis=0)
    print('tail ana ' + str(i) + 'd finished')

# var stats
df_var_stat = tk.timeseries_var_ana(df_raw["close"], days=[1, 5, 10, 20, 40, 60, 120], var_level=[0.995, 0.99])

# option skew level
print("\nOption tail analysis")
spot_px = 59126
r = 0
T = 18 / 365
q = 0
k_list = [40000, 45000, 50000, 52000, 54000, 56000]
moneyness = [x/spot_px for x in k_list]
sigma_list = [1.061,0.931,0.786,0.745,0.701,0.689]
optpx_list = tk.gen_option_pxlist(k_list, sigma_list, option_type='p', s=spot_px, r=r, T=T, q=q)
tk.option_tail_analysis(optpx_list, k_list, option_type='p', s=spot_px, r=r, T=T, q=q, tail_alpha=3.7)