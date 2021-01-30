import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sta
import yfinance as yf
from sklearn import linear_model

import DaiToolkit as tk

#############################
# log log plot of distribution/tail
#############################
x = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200,
     250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000,
     1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
y_list = []
y_list.append(sta.norm.sf(x))
y_list.append(sta.expon.sf(x))
y_list.append(sta.lognorm.sf(x, 1))
y_list.append(sta.t.sf(x, 2))
y_list.append(sta.pareto.sf(x, 1.5))
y_list.append(sta.cauchy.sf(x))
y_list.append(sta.levy_stable.sf(x, 0.5, 1))
y_list.append(sta.levy_stable.sf(x, 0.2, 0))
y_list = [[z if z >= 10 ** -20 else 0 for z in y] for y in y_list]
label_list = ["norm", "exponential", "lognorm(1)", "student t(2)", "pareto(1.5)", "cauchy(levy 1,0,0,1)",
              "levy(0.5,1,0,1)", "levy stable(0.2,0,0,1)"]

plt.figure(figsize=(9, 5))
for y, lbl in zip(y_list, label_list):
    plt.plot(x, y, label=lbl)
plt.yscale('log')
plt.xscale('log')
plt.title("Tail Survival of Distributions")
plt.legend()

############### pdf
x = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10]
x = [-1 * m for m in x[::-1]] + x
y_list = []
y_list.append(sta.norm.pdf(x))
y_list.append(sta.expon.pdf(x))
y_list.append(sta.lognorm.pdf(x, 1))
y_list.append(sta.t.pdf(x, 2))
y_list.append(sta.pareto.pdf(x, 1.5))
y_list.append(sta.cauchy.pdf(x))
y_list.append(sta.levy_stable.pdf(x, 0.5, 1))
y_list.append(sta.levy_stable.pdf(x, 0.2, 0))
label_list = ["norm", "exponential", "lognorm(1)", "student t(2)", "pareto(1.5)", "cauchy(levy 1,0,0,1)",
              "levy(0.5,1,0,1)", "levy stable(0.2,0,0,1)"]

plt.figure(figsize=(9, 5))
for y, lbl in zip(y_list, label_list):
    plt.plot(x, y, label=lbl)
plt.title("PDF of distributions")
# plt.yscale('log')
plt.legend()


#############################
# Data
#############################
### yahoo finance download
spx = yf.Ticker("^GSPC")
# spx = yf.Ticker("^SP500TR")
# spx = yf.Ticker("000002.SS")
df_spx_raw = spx.history(period="max")
# df_spx_raw = spx.history(period="max",interval='1mo')
df_spx_raw["Close"].plot()

# df_spx_raw.loc[(df_spx_raw.index>="1925/01/01")&(df_spx_raw.index<="1955/01/01"),"Close"].plot()


###############################################
# tail study
###############################################
#####
def plot_tailalpha(tail_ret_list, tail_p_list, tail_start=-100, alpha_fit_param=[[-100, 0]], title=""):
    """ 
    tail_ret_list: returns 
    tail_p_list: survial probabilities
    tail_start (log): start ret of tail
    alpha_fit_param (log)= [[lower,upper],[[lower,upper],[[lower,upper]...]
    """

    def log_list(some_list):
        return [math.log(x) for x in some_list]

    spx_lefttail_list_log = log_list(tail_ret_list)
    p_lefttail_list_log = log_list(tail_p_list)

    spx_lefttail_lr_x0 = [m[0] for m in zip(spx_lefttail_list_log, p_lefttail_list_log) if m[0] >= tail_start]
    spx_lefttail_lr_y0 = [m[1] for m in zip(spx_lefttail_list_log, p_lefttail_list_log) if m[0] >= tail_start]

    plot_params = []
    i = 1
    for lb, ub in alpha_fit_param:
        spx_lefttail_lr_x1 = [m[0] for m in zip(spx_lefttail_list_log, p_lefttail_list_log) if
                              m[0] >= lb and m[0] <= ub]
        spx_lefttail_lr_y1 = [m[1] for m in zip(spx_lefttail_list_log, p_lefttail_list_log) if
                              m[0] >= lb and m[0] <= ub]
        lm = linear_model.LinearRegression()
        lm.fit([[m] for m in spx_lefttail_lr_x1], spx_lefttail_lr_y1)
        alpha1 = lm.coef_
        lr_y_pred1 = lm.predict([[m] for m in spx_lefttail_lr_x1])
        plot_params.append({"x": np.array(spx_lefttail_lr_x1), "y": lr_y_pred1, "alpha": alpha1, "#": i})
        i += 1

    plt.figure(figsize=(9, 5))
    plt.plot(spx_lefttail_lr_x0, spx_lefttail_lr_y0, ls=':', label="Return")
    for ptm in plot_params:
        plt.plot(ptm['x'], ptm['y'], ls='-', alpha=0.6,
                 label="tail fit alpha" + str(ptm["#"]) + " = " + str(round(ptm['alpha'], 2)))
    plt.legend()
    plt.title(title)


df_spx = df_spx_raw.copy()
ret_days = 10
ret_label = "ret" + str(ret_days) + 'd'
logret_label = "logret" + str(ret_days) + 'd'
df_spx[ret_label] = df_spx["Close"] / df_spx["Close"].shift(ret_days) - 1.0
df_spx[logret_label] = df_spx[ret_label].apply(lambda x: x if pd.isna(x) else math.log(x + 1.0))

ntop = 5
df_topret = pd.concat([df_spx.sort_values(by=logret_label).iloc[:ntop, :],
                       df_spx.sort_values(by=logret_label, ascending=False).iloc[:ntop, :]], axis=0)[
    [ret_label, logret_label]]
df_topret[ret_label] = df_topret[ret_label].apply(lambda x: "{:,.2f}".format(x * 100) + '%')
df_topret[logret_label] = df_topret[logret_label].apply(lambda x: "{:,.2f}".format(x * 100) + '%')

spx_righttail_list = sorted([m for m in df_spx[logret_label].values if m > 0])
spx_lefttail_list = sorted([-1.0 * m for m in df_spx[logret_label].values if m < 0])

p_righttail_list = [1.0 - float(m) / (len(spx_righttail_list) + 1) for m in range(1, len(spx_righttail_list) + 1)]
p_lefttail_list = [1.0 - float(m) / (len(spx_lefttail_list) + 1) for m in range(1, len(spx_lefttail_list) + 1)]

# spx_right_sigma = np.std(spx_righttail_list+[-1*m for m in spx_righttail_list])

# 1d
plot_tailalpha(spx_lefttail_list, p_lefttail_list, tail_start=-6,
               alpha_fit_param=[[-4, -2.5], [-2.5, -2], [-2, 0], [-3.8, 0]], title="Left Tail Survival of SPX 1d")
plot_tailalpha(spx_righttail_list, p_righttail_list, tail_start=-6, alpha_fit_param=[[-4, -2.5], [-2.5, -2.1], [-4, 0]],
               title="Right Tail Survival of SPX 1d")
# 5d
plot_tailalpha(spx_lefttail_list, p_lefttail_list, tail_start=-6, alpha_fit_param=[[-3.5, -2], [-2, 0], [-3, 0]],
               title="Left Tail Survival of SPX 5d")
plot_tailalpha(spx_righttail_list, p_righttail_list, tail_start=-6, alpha_fit_param=[[-3.5, -2], [-2, 0], [-3, 0]],
               title="Right Tail Survival of SPX 5d")
# 10d
plot_tailalpha(spx_lefttail_list, p_lefttail_list, tail_start=-6, alpha_fit_param=[[-3, -2], [-2, 0], [-1.5, 0]],
               title="Left Tail Survival of SPX 10d")
plot_tailalpha(spx_righttail_list, p_righttail_list, tail_start=-6, alpha_fit_param=[[-3, -2], [-2, 0], [-1.5, 0]],
               title="Right Tail Survival of SPX 10d")

### scale wont affect alpha
# spx_lefttail_list = [m*100.0 for m in spx_lefttail_list]
# plot_tailalpha(spx_lefttail_list, p_lefttail_list, tail_start=0, alpha_fit_param=[[0.5,100],[2.6,100]])

# shang zheng zhi shu
plot_tailalpha(spx_lefttail_list, p_lefttail_list, tail_start=-8, alpha_fit_param=[[-4, -3], [-2.9, -1]],
               title="Left Tail Survival of A shares Shanghai 1d")
plot_tailalpha(spx_righttail_list, p_righttail_list, tail_start=-8, alpha_fit_param=[[-4, -3], [-3, -1]],
               title="Right Tail Survival of A shares Shanghai 1d")
# 5d
plot_tailalpha(spx_lefttail_list, p_lefttail_list, tail_start=-8, alpha_fit_param=[[-3.5, -2], [-2, 0], [-3, 0]],
               title="Left Tail Survival of A shares Shanghai 5d")
plot_tailalpha(spx_righttail_list, p_righttail_list, tail_start=-8, alpha_fit_param=[[-3.5, -2], [-3, 0]],
               title="Right Tail Survival of A shares Shanghai 5d")
# 10d
plot_tailalpha(spx_lefttail_list, p_lefttail_list, tail_start=-8, alpha_fit_param=[[-3.5, -2], [-2, 0], [-3, 0]],
               title="Left Tail Survival of A shares Shanghai 10d")
plot_tailalpha(spx_righttail_list, p_righttail_list, tail_start=-8, alpha_fit_param=[[-3.5, -2], [-3, 0], [-2, 0]],
               title="Right Tail Survival of A shares Shanghai 10d")

#####################################################
# kuto analysis

df_spx = df_spx_raw.copy()
kuto_dict = {}
for ret_days in range(1, 1500, 1):
    df_spx["ret"] = df_spx["Close"] / df_spx["Close"].shift(ret_days) - 1.0
    df_spx['logret'] = df_spx["ret"].apply(lambda x: x if pd.isna(x) else math.log(x + 1.0))
    ret_list = [m for m in df_spx['logret'].values if not pd.isna(m)]
    kuto_dict[ret_days] = pd.Series(ret_list).kurt()
    print(ret_days)

# reshuffle
df_spx["ret"] = df_spx["Close"] / df_spx["Close"].shift(1) - 1.0
df_spx['logret'] = df_spx["ret"].apply(lambda x: x if pd.isna(x) else math.log(x + 1.0))
ret_list = [m for m in df_spx['logret'].values if not pd.isna(m)]
random.shuffle(ret_list)

kuto_dict_resf = {}
for ret_days in range(1, 1500, 1):
    ret_list_resf = [sum(ret_list[i:i + ret_days]) for i in range(len(ret_list) + 1 - ret_days)]
    kuto_dict_resf[ret_days] = pd.Series(ret_list_resf).kurt()
    print(ret_days)

plt.figure(figsize=(9, 5))
plt.plot(kuto_dict.keys(), kuto_dict.values(), ls=':', label='Origin')
plt.plot(kuto_dict_resf.keys(), kuto_dict_resf.values(), ls='-', label='Reshuffled', alpha=0.3)
# plt.xlim((0,100))
plt.legend()
plt.title("Kuto of A shares Shanghai")

#####################################################
# CVAR
#####################################################

df_spx = df_spx_raw.copy()
df_spx["ret"] = df_spx["Close"] / df_spx["Close"].shift(10) - 1.0
df_spx['logret'] = df_spx["ret"].apply(lambda x: x if pd.isna(x) else math.log(x + 1.0))

p_tail = len(df_spx[df_spx['ret'] <= -0.2]) / float(len(df_spx))
e_tail = df_spx.loc[df_spx['ret'] <= -0.2, "ret"].mean()

#####################################################
# option delta hedge of returns:
#####################################################

size = 5000
rate = 0.00
# pareto
return_list = [m + rate / 252.0 for m in sta.pareto.rvs(2.75, loc=-0.01, scale=0.01, size=size / 2)]
return_list += [-1.0 * m + rate / 252.0 for m in sta.pareto.rvs(2.75, loc=-0.01, scale=0.01, size=size / 2)]
random.shuffle(return_list)

# norm
# return_list = [m + rate/252.0 for m in sta.norm.rvs(loc=0, scale=0.01, size=size)]

return_ts = pd.Series([0] + list(return_list))
px_ts = (return_ts + 1).cumprod()
df_asset = pd.DataFrame([px_ts, return_ts], index=["px", "ret1d"]).T
df_asset['time_to_maturity'] = [m / 365.0 for m in range(len(df_asset))[::-1]]

plt.hist(return_ts, bins=100)
plt.title('ret distribution')
plt.show()
plt.plot(px_ts)
plt.title('price')
plt.show()
historical_vol = return_ts.std() * math.sqrt(252)
print("min ret 1d: " + str(return_ts.min()))
print("max ret 1d: " + str(return_ts.max()))
print("historical_vol: " + str(historical_vol))

df_asset['hist_vol90d'] = df_asset['ret1d'].rolling(90).std() * math.sqrt(252)
df_asset['hist_vol90d'] = df_asset['hist_vol90d'].fillna(historical_vol)

call_px0 = tk.option_bsm_price(option_type='c', sigma=historical_vol, s=1.0, k=1.0, r=rate, T=size / 365.0, q=0)
delta0 = tk.option_bsm('delta', "c", spot=1.0, strike=1.0, maturity_years=size / 365.0, vol=historical_vol, rate=rate)

df_asset['option_px'] = [
    tk.option_bsm('px', "c", spot=m[0], strike=1.0, maturity_years=m[1], vol=historical_vol, rate=rate) for m in
    df_asset[['px', 'time_to_maturity', 'hist_vol90d']].values]
df_asset['option_pnl'] = (df_asset['option_px'] - df_asset['option_px'].shift(1)) * -1.0
df_asset['option_pnl'] = df_asset['option_pnl'].fillna(0)

df_asset['delta_staticvol'] = [
    tk.option_bsm('delta', "c", spot=m[0], strike=1.0, maturity_years=m[1], vol=historical_vol, rate=rate) for m in
    df_asset[['px', 'time_to_maturity']].values]
df_asset['pnl_staticvol'] = (df_asset['delta_staticvol'] * df_asset['px']).shift(1) * df_asset['ret1d']
df_asset['pnl_staticvol'] = df_asset['pnl_staticvol'].fillna(0)
df_asset['pnl_static_hedge'] = df_asset['pnl_staticvol'] + df_asset['option_pnl']
df_asset['cumpnl_staticvol'] = df_asset['pnl_static_hedge'].cumsum() + call_px0

df_asset['delta_dym_vol'] = [
    tk.option_bsm('delta', "c", spot=m[0], strike=1.0, maturity_years=m[1], vol=m[2], rate=rate) for m in
    df_asset[['px', 'time_to_maturity', 'hist_vol90d']].values]
df_asset['pnl_dym_vol'] = (df_asset['delta_dym_vol'] * df_asset['px']).shift(1) * df_asset['ret1d']
df_asset['pnl_dym_vol'] = df_asset['pnl_dym_vol'].fillna(0)
df_asset['pnl_dym_hedge'] = df_asset['pnl_dym_vol'] + df_asset['option_pnl']
df_asset['cumpnl_dymvol'] = df_asset['pnl_dym_hedge'].cumsum() + call_px0

plt.figure(figsize=(11, 4))
plt.plot(df_asset['pnl_static_hedge'], label='static vol delta hedge')
plt.plot(df_asset['pnl_dym_hedge'], label='dym vol delta hedge')
plt.title('delta hedge pnl')
plt.legend()
plt.show()

plt.plot(df_asset['cumpnl_staticvol'], label='static vol delta hedge')
plt.plot(df_asset['cumpnl_dymvol'], label='dym vol delta hedge')
plt.title('delta hedge cum pnl')
plt.legend()
plt.show()

#####################################################
# VAR/Ret Analysis
#####################################################
df_spx = df_spx_raw.copy()
length = 2520 * 5
var_res = {}
for i in range(2520, length, 20):
    df_spx["ret"] = df_spx["Close"] / df_spx["Close"].shift(i) - 1.0
    df_spx["ret"] = df_spx["ret"].apply(lambda x: math.pow(x + 1.0, 252.0 / i) - 1.0)
    var_res[i] = {}
    var_res[i]["min"] = df_spx["ret"].min()
    var_res[i]["btm 1%"] = df_spx["ret"].quantile(0.01)
    var_res[i]["btm 5%"] = df_spx["ret"].quantile(0.05)
    var_res[i]["btm 10%"] = df_spx["ret"].quantile(0.10)
    var_res[i]["medium"] = df_spx["ret"].quantile(0.50)
    var_res[i]["top 10%"] = df_spx["ret"].quantile(0.9)
    var_res[i]["top 5%"] = df_spx["ret"].quantile(0.95)
    var_res[i]["top 1%"] = df_spx["ret"].quantile(0.99)
    var_res[i]["max"] = df_spx["ret"].max()
    print(str(i) + " days finished.")
var_res = pd.DataFrame(var_res).T

plt.figure(figsize=(11, 4))
for c in var_res.columns:
    plt.plot(var_res[c], label=c)
plt.hlines(0, 2520, length)
plt.ylim(-0.2, 0.2)
plt.title('SPX Annualized Ret Distribution')
plt.legend()
plt.grid(ls="--", alpha=0.4)
plt.show()

#####################################################
# vol auto correlation
#####################################################
###### spx
df_spx = df_spx_raw.copy()
df_spx["ret"] = df_spx["Close"] / df_spx["Close"].shift(1) - 1.0
df_spx['logret'] = df_spx["ret"].apply(lambda x: x if pd.isna(x) else math.log(x + 1.0))

###### fake ret
size = 25000
rate = 0.02
# pareto
return_list = [m + rate / 252.0 for m in sta.pareto.rvs(2.75, loc=-0.01, scale=0.01, size=size / 2)]
return_list += [-1.0 * m + rate / 252.0 for m in sta.pareto.rvs(2.75, loc=-0.01, scale=0.01, size=size / 2)]
random.shuffle(return_list)
return_ts = pd.Series([0] + list(return_list))
px_ts = (return_ts + 1).cumprod()
df_spx = pd.DataFrame([px_ts, return_ts], index=["Close", "ret"]).T
df_spx['Close'].plot()

autocorr = {}
for l in range(30):
    autocorr[l] = df_spx['ret'].autocorr(lag=l)

plt.bar(autocorr.keys(), autocorr.values())
plt.title('1d ret auto correlation')
print("Note: there is no auto corr in daily returns!")

############################# vol (overlap/seperate)
bucket_length = 750
vol_type = "overlap"
vol_type = "seperate"

if vol_type == "overlap":
    buckets = len(df_spx) + 1 - 2 * bucket_length
    vol_list = {}
    for i in range(buckets):
        vol_list[df_spx.index[i + bucket_length]] = {}
        vol_list[df_spx.index[i + bucket_length]]['pre_vol'] = df_spx['ret'][i:(i + bucket_length)].std() * math.sqrt(
            252)
        vol_list[df_spx.index[i + bucket_length]]['post_vol'] = df_spx['ret'][(i + bucket_length):(
                i + 2 * bucket_length)].std() * math.sqrt(252)
else:
    buckets = int(len(df_spx) / bucket_length) - 1
    vol_list = {}
    for i in range(buckets):
        vol_list[df_spx.index[(i + 1) * bucket_length]] = {}
        vol_list[df_spx.index[(i + 1) * bucket_length]]['pre_vol'] = df_spx['ret'][(i * bucket_length):(
                (i + 1) * bucket_length)].std() * math.sqrt(252)
        vol_list[df_spx.index[(i + 1) * bucket_length]]['post_vol'] = df_spx['ret'][((i + 1) * bucket_length):(
                (i + 2) * bucket_length)].std() * math.sqrt(252)

df_vol = pd.DataFrame(vol_list).T
print
"Note: there is auto corr in volatility!"
print
df_vol.corr()
plt.figure(figsize=(9, 5))
plt.scatter(df_vol['pre_vol'], df_vol['post_vol'], alpha=0.2)
plt.xlabel('pre volatility ' + str(bucket_length) + 'd')
plt.ylabel('post volatility ' + str(bucket_length) + 'd')
plt.title(str(bucket_length) + "d pre/post rolling vol")
plt.show()

print
"\nRolling corr of vol graphs:"
df_vol['rolling_corr'] = df_vol['post_vol'].rolling(bucket_length * 1).corr(df_vol['pre_vol'])
plt.figure(figsize=(12, 4))
plt.plot(df_vol['rolling_corr'])
plt.title(str(bucket_length) + "d pre/post vol rolling corr")
plt.show()

vol_ts = pd.Series(list(vol_list.values()))
autocorr = {}
for l in range(30):
    autocorr[l] = vol_ts.autocorr(lag=l)

plt.bar(list(autocorr.keys()), list(autocorr.values()))
plt.title(str(bucket_length) + 'd vol auto correlation')
plt.show()

##python 3
# from statsmodels.graphics.tsaplots import plot_pacf
# plot_pacf(vol_ts, lags=10, alpha=0.05)

##fake auto corr vol series
fake_vol_list = list(sta.lognorm.rvs(s=1, loc=0, scale=0.15, size=1))
for i in range(1000):
    fake_vol_list.append(0.9 * fake_vol_list[i] + sta.lognorm.rvs(s=1, loc=0, scale=0.005))

plt.plot(range(1001), fake_vol_list)
df_vol = pd.DataFrame([fake_vol_list], index=["post_vol"]).T
df_vol["pre_vol"] = df_vol["post_vol"].shift(1)
df_vol = df_vol.dropna(how="any")
df_vol['rolling_corr'] = df_vol['post_vol'].rolling(30).corr(df_vol['pre_vol'])

plt.figure(figsize=(12, 4))
plt.plot(df_vol['rolling_corr'])
plt.title(str(30) + "d pre/post vol rolling corr")

#####################################################
df_spx = df_spx_raw.copy()
df_spx = pd.DataFrame(df_spx["Close"].resample('Y').last())
df_spx["ret"] = df_spx["Close"] / df_spx["Close"].shift(1) - 1.0
df_spx = df_spx.dropna(how='any')
df_spx['ret bucket'] = df_spx['ret'].apply(
    lambda x: "<-15%" if x <= -0.15 else "-15% ~ 0%" if x <= 0 else "0 ~ +15%" if x <= 0.15 else "> +15%")
df_spx.reset_index(inplace=True)

# Tail Risk Analysis
#####################################################
plt.figure(figsize=(12, 4))
plt.hist(df_spx['ret'], bins=[m / 100.0 for m in range(-100, 101, 10)])
plt.title("SPX Yearly Return Histogram")
