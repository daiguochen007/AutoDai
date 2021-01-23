import numpy as np
import pandas as pd

from . import util_basics


def df_tbl_breakdown(df_tbls, preheadrows=0, header=True, auto_clean_empty_cols=True, auto_clean_empty_rows=False):
    """
    break down df_tbls in same excel sheet
    seperate by blank rows
    tables can have header/pre header description
    """
    df_tbls.index = list(range(len(df_tbls.index)))
    df_tbls_list = []
    last_nan_row = 0
    for i in range(len(df_tbls) - 1):
        if all(df_tbls.iloc[i, :].isnull()) and not all(df_tbls.iloc[i + 1, :].isnull()):
            df_tbls_list.append(df_tbls.iloc[last_nan_row:i, :])
            last_nan_row = i + 1
        if i == len(df_tbls) - 2:
            df_tbls_list.append(df_tbls.iloc[last_nan_row:, :])

    for i in range(len(df_tbls_list)):
        if auto_clean_empty_cols:
            for c in df_tbls_list[i].columns:
                if all(df_tbls_list[i].loc[:, c].isnull()):
                    del df_tbls_list[i][c]
        if auto_clean_empty_rows:
            for r in df_tbls_list[i].index:
                if all(df_tbls_list[i].loc[r, :].isnull()):
                    df_tbls_list[i] = df_tbls_list[i].drop(r)

        df_tbls_list[i] = df_tbls_list[i].iloc[preheadrows:, :]
        if header:
            df_tbls_list[i].columns = df_tbls_list[i].iloc[0, :].values
            df_tbls_list[i] = df_tbls_list[i].iloc[1:, :]

    return df_tbls_list


def df_fill_cols(df, cols, fill_value=0):
    """
    return dataframe with full columns
    fill_value added if not exit 0/nan/value
    """
    df_copy = df.copy()
    for c in cols:
        if c not in df_copy.columns:
            df_copy[c] = [fill_value] * len(df_copy)
    return df_copy[cols]


def df_add_blank_rows(df, nrows):
    """
    add blank rows to a given dataframe if nrows smaller than required length
    """
    if len(df) >= nrows:
        return df
    else:
        return df.append(pd.DataFrame([[float("nan")] * len(df.columns)] * (nrows - len(df)), columns=df.columns))


def df_to_utf8(df):
    """
    df encode to chinese / for spyder gui copy
    no df return
    """
    for c in df.columns:
        df[c] = [x.encode('utf-8') if type(x) == str else x for x in df[c]]
    df.columns = [x.encode('utf-8') if type(x) == str else x for x in df.columns]
    df.index = [x.encode('utf-8') if type(x) == str else x for x in df.index]


def df_fmt_num_cols(df, columns, fmt_type="#,##"):
    """
    fmt_type = "#,##", "#,##(#,##)", "0.00%", "0.00(0.00%)", "0%", "0.00"
    """

    def round_to_int(x):
        return int(round(x, 0))

    for c in columns:
        if fmt_type == "#,##":
            df[c] = [("{:,}".format(round_to_int(x)) if not pd.isna(x) else "") if util_basics.is_number(x) else x for x in df[c]]
        elif fmt_type == "#,##(#,##)":
            df[c] = [("{:,}".format(round_to_int(x)) if x >= 0 else "({:,})".format(
                abs(round_to_int(x))) if x < 0 else "") if util_basics.is_number(x) else x for x in df[c]]
        elif fmt_type == "0.00%":
            df[c] = [(str(round(x * 100.0, 2)) + "%" if not pd.isna(x) else "") if util_basics.is_number(x) else x for x in df[c]]
        elif fmt_type == "0.00%(0.00%)":
            df[c] = [("{:.2f}%".format(float(x) * 100.0) if x >= 0 else "({:.2f}%)".format(
                abs(float(x) * 100.0)) if x < 0 else "") if util_basics.is_number(x) else x for x in df[c]]
        elif fmt_type == "0%":
            df[c] = [(str(round_to_int(x * 100.0)) + "%" if not pd.isna(x) else "") if util_basics.is_number(
                    x) else x for x in df[c]]
        elif fmt_type == "0.00":
            df[c] = [(str(round(x, 2)) if not pd.isna(x) else "") if util_basics.is_number(x) else x for x in df[c]]

    return df


def df_strip_cols(df, columns=[]):
    if len(columns) == 0:
        columns_strip = df.columns
    else:
        columns_strip = columns
    for c in columns_strip:
        if c in df.columns:
            if df[c].dtype in [np.float64, np.int64]:
                pass
            else:
                df[c] = [str(x).strip() for x in df[c]]
    return df


def df_to_float(df, errors='ignore'):
    """
    errors: {'ignore', 'coerce'} coerce will return nan
    """
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors=errors, downcast='float')
    return df


