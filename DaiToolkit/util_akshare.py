import akshare as ak
import pandas as pd

from DaiToolkit import util_basics


def akshare_get_history(security_id, market='H', adjust='qfq'):
    """
    下载历史数据：高开低收量（前复权）
    security_id="00700"
    """
    print("----- [" + security_id + "] start downloading history -----")
    try:
        if market == 'H':
            df_raw = ak.stock_hk_daily(symbol=security_id, adjust=adjust)
        elif market == 'US':
            df_raw = ak.stock_us_daily(symbol=security_id, adjust=adjust)
        elif market == 'A':
            df_raw = ak.stock_zh_a_daily(symbol=security_id, adjust=adjust)
        else:
            raise Exception("market support 'A', 'H' and 'US'")
        df_raw['trade_date'] = df_raw.index
        df_raw['trade_date'] = df_raw['trade_date'].apply(lambda x: x.strftime('%Y%m%d'))
        df_raw.to_csv(util_basics.PROJECT_DATA_PATH + "/akshare/history/" + security_id + ".csv", index=False)
    except Exception as e:
        raise Exception("下载前复权错误:" + str(e))
    print("----- [" + security_id + "] finished! -----")
    return df_raw


def aksharelocal_get_history(security_id):
    df_raw = pd.read_csv(util_basics.PROJECT_DATA_PATH + "/akshare/history/" + security_id + ".csv")
    idx = df_raw["trade_date"].apply(lambda x: str(x)[:4] + '/' + str(x)[4:6] + '/' + str(x)[6:])
    df_raw.index = pd.DatetimeIndex(idx)
    return df_raw


if __name__ == "__main__":
    df_raw = akshare_get_history("00700", market='H', adjust='qfq')
    df_raw = akshare_get_history("AAPL", market='US', adjust='qfq')
    df_raw = akshare_get_history('sz000002', market='A', adjust='qfq')

    df_raw = aksharelocal_get_history("00700")
    df_raw['close'].plot()
