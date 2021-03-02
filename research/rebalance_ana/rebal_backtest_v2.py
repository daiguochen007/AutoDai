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
selected_group = ["上证50"]  # ,u"农商行",u"城商行",u"国有大行",u"股份制银行",u"房地产",u"白色家电",u"上证50"
sec_list = list(df_sec.loc[df_sec["SecGroup"].isin(selected_group), "SecCode"].values)

### GET DATA
start_weight = np.array([1 / len(sec_list)] * len(sec_list))
df_port_secs = pd.DataFrame()
for sec in sec_list:
    ts_px = tk.get_history_data(sec, source='tushare')[['close']]
    ts_px.columns = [sec]
    df_port_secs = df_port_secs.merge(ts_px, left_index=True, right_index=True, how='outer')

### CLEAN DATA
df_port_secs = df_port_secs.fillna(method="ffill")
df_port_secs = df_port_secs.dropna(how='any')
# df_port_secs = df_port_secs[df_port_secs.index>"2017-01-01"]
df_port_secs.plot()

### ANALYSIS
print("average corr: " + str(port_avg_corr(df_port_secs)))
res = {}
for fq in [5, 10, 21, 63, 126, 252]:
    df_stats = tk.timeseries_port_rebal_ana(df_port_secs, start_weight, rebal_freq=fq, rebal_enhance_ratio=0.1,
                                            rebal_enhance_hurdle=0, rebal_anchor="no rebalance",
                                            longterm_growth="implied", plot=False)
    res[fq] = df_stats.loc['enhance-anchor'].to_dict()
    print('rebal freq '+str(fq)+' finished.')
res = pd.DataFrame(res).T
df_stats = tk.timeseries_port_rebal_ana(df_port_secs, start_weight, rebal_freq=res['Annual return'].argmax(), rebal_enhance_ratio=0.1,
                                            rebal_enhance_hurdle=0, rebal_anchor="no rebalance",
                                            longterm_growth="implied", plot=True)