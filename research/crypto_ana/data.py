import datetime
import os
import re
import tarfile

import pandas as pd

data_path_raw = r'C:\Users\Dai\Desktop\investment\data\crypto\deribit\raw/'
data_path_unzip = r'C:\Users\Dai\Desktop\investment\data\crypto\deribit\unzipped'
data_pkg_list = sorted(os.listdir(data_path_raw))


def untar(tar_path, target_path):
    """
    解压tar.gz文件
    :param tar_path: 压缩文件名
    :param target_path: 解压后的存放路径
    """
    with tarfile.open(tar_path, 'r') as f:
        f.extractall(target_path + '/' + tar_path.split('/')[-1].split('.')[0])
        print()


def transfer_timestamp(time_msec, to='datetime'):
    """

    :param time_msec: 1617465646613
    :return:
    """
    res = datetime.datetime.fromtimestamp(time_msec / 1e3)
    if to == 'datetime':
        return res
    elif to == 'str':
        return res.strftime('%Y-%m-%d %H:%M:%S ')


# # unzip
# for i,data_pkg in enumerate(data_pkg_list):
#     tar_path = data_path_raw + data_pkg
#     untar(tar_path, data_path_unzip)
#     print(str(i+1)+". "+data_pkg.split('.')[0]+' finished!')

fdr_list = sorted(os.listdir(data_path_unzip))
calendar = []

df_alldata = pd.DataFrame()
for fdr in fdr_list:
    # fdr = 'deribit_option_trade_BTC_20191023'
    curr_date = fdr.split('_')[-1]
    calendar.append(curr_date)
    csv_path = data_path_unzip + '/' + fdr + '/' + fdr + '.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[['s', 'p', 'q', 'a', 'iv', 't']]
        df.columns = ['contract_id', 'price', 'quantity', 'direction', 'iv', 'timestamp']
        df['timestamp_str'] = df['timestamp'].apply(lambda x: transfer_timestamp(x, to='str'))
        df['date'] = curr_date
        df_alldata = pd.concat([df_alldata, df], axis=0)
    else:
        csv_list = sorted(os.listdir(data_path_unzip + '/' + fdr + '/' + fdr))
        for csv_p in csv_list:
            pass
            df = pd.read_csv(data_path_unzip + '/' + fdr + '/' + fdr + '/' + csv_p)
            contract_info_str = csv_p.split('.')[1]
            match_obj = re.search(r'\d{1,2}[A-Za-z]{3}\d{2}', contract_info_str)
            contract_id = '-'.join([contract_info_str[:3], match_obj.group(), contract_info_str[match_obj.span()[1]:-1], contract_info_str[-1]])
            df['contract_id'] = contract_id
            df['iv'] = float('nan')
            df = df[['contract_id', 'p', 'q', 'a', 'iv', 't']]
            df.columns = ['contract_id', 'price', 'quantity', 'direction', 'iv', 'timestamp']
            df['timestamp_str'] = df['timestamp'].apply(lambda x: transfer_timestamp(x, to='str'))
            df['date'] = curr_date
            df_alldata = pd.concat([df_alldata, df], axis=0)
    print('[' + fdr + '] finished!')

df_alldata.to_csv('C:/Users/Dai/Desktop/investment/data/crypto/deribit/deribit_opt_history.csv', index=False)
