# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from pykalman import KalmanFilter

# import fix_yahoo_finance

pair = ['AAPL', 'SPY']
# data = pdr.DataReader(pair, 'yahoo', '2010-1-1', '2014-8-1')["Close"].dropna(how="any")
data = pdr.get_data_yahoo(pair, '2010-1-1', '2014-8-1')["Adj Close"]

################ ------- package
obs_mat = np.vstack([data.SPY, np.ones(data.AAPL.shape)]).T[:, np.newaxis]

delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2)

kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                  initial_state_mean=np.zeros(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=1.0,
                  transition_covariance=trans_cov)

state_means, state_covs = kf.filter(data.AAPL.values)

kalman_resi = data[pair[0]] - (state_means[:, 0] * data[pair[1]])

std = np.std(kalman_resi - state_means[:, 1])


################ ------- own implementation

def get_kf(data1, data2, initial_state_mean, initial_state_covariance, transition_matrices, trans_cov, R_k):
    x_k_1 = np.transpose(initial_state_mean)
    P_k_1 = initial_state_covariance
    test_means = []
    test_covs = []

    for i in range(len(data1)):
        # predict
        x_k = np.dot(transition_matrices, x_k_1)
        P_k = np.dot(np.dot(transition_matrices, P_k_1), np.transpose(transition_matrices)) + trans_cov  # Qk
        # update
        H_k = np.array([data2.iloc[i], 1])
        y_k = data1.iloc[i] - np.dot(H_k, x_k)
        S_k = np.dot(np.dot(H_k, P_k), np.transpose(H_k)) + R_k

        K_k = np.dot(P_k, np.transpose(H_k)) * 1 / S_k
        x_k = x_k + np.transpose([K_k]) * y_k
        P_k = np.dot((np.eye(2) - np.dot(np.transpose([K_k]), [H_k])), P_k)

        test_means.append(x_k.reshape((2,)))
        test_covs.append(P_k)

        # para
        x_k_1 = x_k
        P_k_1 = P_k

    test_means = np.array(test_means)
    test_covs = np.array(test_covs)
    return test_means, test_covs


#
initial_state_mean = np.zeros(2),
initial_state_covariance = np.ones((2, 2))
transition_matrices = np.eye(2)

delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2)
R_k = 1

#
state_means, state_covs = get_kf(data.loc[:, pair[0]], data.loc[:, pair[1]],
                                 initial_state_mean, initial_state_covariance, transition_matrices, trans_cov, R_k)

kalman_resi = data[pair[0]] - (state_means[:, 0] * data[pair[1]])
std = np.std(kalman_resi - state_means[:, 1])

# -----------------------plot
plt.figure(figsize=(10, 4))
plt.plot(kalman_resi)
plt.plot(data.index, state_means[:, 1], "--", label="alpha")
plt.plot(data.index, state_means[:, 1] + 1.3 * std, "--", label='upper')
plt.plot(data.index, state_means[:, 1] - 1.3 * std, "--", label='lower')
plt.plot(data.index, state_means[:, 1] + 3 * std, "--", label='upper_stoploss')
plt.plot(data.index, state_means[:, 1] - 3 * std, "--", label='lower_stoploss')
plt.title("kalman residual (" + "~".join(pair) + ")")
plt.legend(loc=3)
plt.show()


# get and plot performance
def get_perf(ts):
    annual_ret = (ts[-1] / ts[0] - 1) * 252 / len(ts)
    annual_vol = np.std((ts / ts.shift(1) - 1).dropna(how="any")) * np.sqrt(252)
    sharpe = annual_ret / annual_vol
    dd = ts / ts[0] - (ts / ts[0]).rolling(len(ts), min_periods=1).max()
    mdd = dd.rolling(len(ts), min_periods=1).min()
    return pd.DataFrame([annual_ret, annual_vol, mdd[-1], sharpe], index=["ann_return", "ann_vol", "maxdd", "sharpe"])


def get_kf_perf(data1, data2, resi, means, trading_cost=0.2 / 100):
    pos = pd.Series(0, index=resi.index)
    for i in range(len(resi)):
        pos[i] = pos[i - 1]
        if resi[i - 1] > means[i - 1, 1] + 1.3 * std and resi[i] <= means[i, 1] + 1.3 * std and resi[i] > means[i, 1]:
            pos[i] = -1
        if resi[i - 1] < means[i - 1, 1] - 1.3 * std and resi[i] >= means[i, 1] - 1.3 * std and resi[i] < means[i, 1]:
            pos[i] = 1
        if pos[i - 1] == 1 and (resi[i] >= means[i, 1] or resi[i] <= means[i, 1] - 3 * std):
            pos[i] = 0
        if pos[i - 1] == -1 and (resi[i] <= means[i, 1] or resi[i] >= means[i, 1] + 3 * std):
            pos[i] = 0

    pnl = (pos.shift(1) * (resi - resi.shift(1))).cumsum()
    pnl_cost_adj = pnl.copy()

    # trading cost and commission fee
    for i in range(len(pnl)):
        if (pos[i - 1] == 0 and pos[i] != 0) or (pos[i - 1] != 0 and pos[i] == 0):
            pnl_cost_adj[i:] -= (data1.iloc[i] + means[i, 0] * data2.iloc[i]) * trading_cost
        elif pos[i] != 0:
            pnl_cost_adj[i:] -= (means[i, 0] - means[i - 1, 0]) * data.iloc[i, 1] * trading_cost

    start_value = max(data1) + max(means[:, 0] * data2)
    pnl = (pnl + start_value) / start_value
    pnl_cost_adj = (pnl_cost_adj + start_value) / start_value

    # plot pnl
    plt.figure(figsize=(10, 4))
    plt.plot(pnl, label="kf_pnl (0 trading cost)")
    plt.plot(pnl_cost_adj, label="kf_pnl (" + str(trading_cost * 100) + "% trading cost)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # kf hit ratio
    pnl_bytrade = pd.Series([])
    for i in range(len(pos) - 1):
        if (pos[i] == 0 and pos[i + 1] != 0) or i == 0:
            pnl_start = pnl[i + 1]
        if pos[i] != 0 and (pos[i + 1] == 0 or i == len(pos) - 2):
            pnl_bytrade = pnl_bytrade.append(pd.Series(pnl[i + 1] - pnl_start, index=[pos.index[i + 1]]))

    kf_hit_ratio = float(len([temp for temp in pnl_bytrade if temp >= 0])) / len(pnl_bytrade)
    print("kf hit ratio: " + str(kf_hit_ratio))

    perf_mat = pd.concat([get_perf(pnl.dropna(how="any")), get_perf(pnl_cost_adj.dropna(how="any"))], axis=1)
    perf_mat.columns = ["kf(0%)", "kf(" + str(trading_cost * 100) + "%)"]
    print(perf_mat)


get_kf_perf(data.loc[:, pair[0]], data.loc[:, pair[1]], kalman_resi, state_means, trading_cost=0.2 / 100)
