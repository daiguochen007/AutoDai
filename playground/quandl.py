# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:20:34 2018

@author: Dai
"""

import pandas as pd

import DaiToolkit as tk
import quandl

token = tk.read_yaml(tk.PROJECT_CODE_PATH + "/DaiToolkit/login.yaml")["quandl"]['token']

n = list(range(1, 21))
nms = ["CHRIS/CME_ED" + str(i) for i in n]
dfs = [quandl.get(nm, authtoken=token) for nm in nms]  # an array of pandas data frames

data = [data_set.Settle for data_set in dfs]  # pull column Settle
new_data = pd.concat(data, axis=1)  # combine cols together
new_data.columns = ["Settle_ED" + str(i) for i in n]
new_data.describe()

# size:
print("number of non-NA/null observations over each depth: \n" + str(
    new_data.count()))  # Return Series with number of non-NA/null observations over requested axis
print("size of the data: " + str(new_data.size))
print("dimension of the data:" + str(new_data.shape))

# data cleaning
df = new_data[["Settle_ED15", "Settle_ED16", "Settle_ED17", "Settle_ED18"]]
set_standard_min = 70
mis = df[(df <= set_standard_min).all(axis=1)]
good_data = new_data.drop([mis.index[0], mis.index[1]])
