import numpy as np
import pandas as pd

import DaiToolkit as tk


def port_avg_corr(df_port_secs):
    df_corr = (df_port_secs / df_port_secs.shift(1) - 1).corr()
    values = []
    for i in range(len(df_corr)):
        for j in range(i + 1, len(df_corr)):
            values.append(df_corr.iloc[i, j])
    return np.mean(values)

### GET SECURITY LIST
df_sec = pd.read_excel(tk.PROJECT_CODE_PATH + "/research/rebalance_ana/sec_pool.xlsx", "Sheet1")
df_sec["SecCode"] = ["0" * (6 - len(str(x))) + str(x) for x in df_sec["SecCode"]]
selected_group = ["上证50"]  #"农商行","城商行","国有大行","股份制银行","房地产","白色家电","上证50",'白酒'
sec_list = list(df_sec.loc[df_sec["SecGroup"].isin(selected_group), "SecCode"].values)

# Top return securities
sec_list = ["000333",'000651','600519','000002','000568','000538','600887','000858']

### GET DATA
start_weight = np.array([1 / len(sec_list)] * len(sec_list))
df_port_secs = pd.DataFrame()
for sec in sec_list:
    ts_px = tk.get_history_data(sec, source='local')[['close']]
    ts_px.columns = [sec]
    df_port_secs = df_port_secs.merge(ts_px, left_index=True, right_index=True, how='outer')

### CLEAN DATA
df_port_secs = df_port_secs.fillna(method="ffill")
df_port_secs = df_port_secs.dropna(how='any')
# df_port_secs = df_port_secs[df_port_secs.index>"2017-01-01"]
#(df_port_secs/df_port_secs.iloc[0,:]).plot()

### ANALYSIS
print("average corr: " + str(port_avg_corr(df_port_secs)))
res = {}
for fq in [5, 10, 21, 63, 126, 252]:
    df_stats = tk.timeseries_port_rebal_ana(df_port_secs, start_weight, rebal_freq=fq, rebal_enhance_ratio=0.05,
                                            rebal_enhance_hurdle=0, rebal_anchor="fixed weight",
                                            longterm_growth="implied", plot=False)
    res[fq] = df_stats.loc['enhance-anchor'].to_dict()
    print('rebal freq '+str(fq)+' finished.')
res = pd.DataFrame(res).T
df_stats = tk.timeseries_port_rebal_ana(df_port_secs, start_weight, rebal_freq=126, rebal_enhance_ratio=0.25,
                                            rebal_enhance_hurdle=0, rebal_anchor="fixed weight",
                                            longterm_growth="implied", plot=True)