import math

import pandas as pd
import scipy as sp

from DaiToolkit.util_basics import PROJECT_DATA_PATH
from DaiToolkit.util_excel import excel_quick_output


def DCF_twostage(v0, r, n1, g1, n2, g2):
    """
    :param v0: T0 cashflow
    :param r: discount rate
    :param n1: stage 1 growth years
    :param g1: stage 1 growth
    :param n2: stage 2 growth years
    :param g2: stage 2 growth
    :return:
    """
    jiggle = 1e-8
    if r == g1:
        g1 += jiggle
    if r == g2:
        g2 += jiggle
    stage1_value = v0 * (1 + g1) / (r - g1) * (1 - math.pow((1 + g1) / (1 + r), n1))
    stage2_value = v0 * math.pow((1 + g1) / (1 + r), n1) * (1 + g2) / (r - g2) * (1 - math.pow((1 + g2) / (1 + r), n2))
    value = stage1_value + stage2_value
    return value


def DCF_twostage_impgrowth(pe, r, n1, n2, g2):
    """
    pe,r,n1,n2,g2 = (22.145, 0.12, 10, 10, 0.04)
    """
    res = sp.optimize.fsolve(lambda g1: DCF_twostage(1, r, n1, g1, n2, g2) - pe, [0.05])
    return res[0]


def DCF_twostage_impyears(pe, r, g1, n2, g2):
    """
    pe,r,g1,n2,g2 = (22.145, 0.12, 0.162, 10, 0.04)
    """
    res = sp.optimize.fsolve(lambda n1: DCF_twostage(1, r, n1, g1, n2, g2) - pe, [5])
    return res[0]


def DCF_twostage_impreturn(pe, n1, g1, n2, g2):
    """
    pe,n1,g1,n2,g2 = (22.145, 10, 0.162, 10, 0.04)
    """
    res = sp.optimize.fsolve(lambda r: DCF_twostage(1, r, n1, g1, n2, g2) - pe, [0.1])
    return res[0]


def dcf_analysis(security_info_list, n1=10, r=0.12):
    """
    static assumption for stage 2

    :param security_info_list:[(security_id,curr_pe,predict_growth),()()()...]
    :param n1:
    :param r:
    :return:
    """
    n2 = 10
    g2 = 0.04
    df_dcf = []
    for security_id, curr_pe, g1 in security_info_list:
        res = {'Security': security_id, 'Curr PE': curr_pe, 'Pred Growth(Stage 1)': g1}
        res['Implied PE'] = DCF_twostage(v0=1, r=r, n1=n1, g1=g1, n2=n2, g2=g2)
        res['Margin of Safety'] = res['Implied PE'] / res['Curr PE'] - 1
        res['Implied Growth(Stage 1)'] = DCF_twostage_impgrowth(curr_pe, r=r, n1=n1, n2=n2, g2=g2)
        res['Implied Ann Return'] = DCF_twostage_impreturn(curr_pe, n1=n1, g1=g1, n2=n2, g2=g2)
        res['Implied Growth Years(Stage 1)'] = DCF_twostage_impyears(curr_pe, r=r, g1=g1, n2=n2, g2=g2)
        df_dcf.append(res)
    df_dcf = pd.DataFrame(df_dcf)
    df_dcf = df_dcf[['Security', 'Curr PE', 'Pred Growth(Stage 1)', 'Implied Ann Return', 'Implied PE',
                     'Margin of Safety', 'Implied Growth(Stage 1)', 'Implied Growth Years(Stage 1)']]
    return df_dcf.sort_values(by='Implied Ann Return', ascending=False)


if __name__ == "__main__":
    pe = DCF_twostage(v0=1, r=0.12, n1=10, g1=0.162, n2=10, g2=0.04)
    imp_g1 = DCF_twostage_impgrowth(pe, r=0.12, n1=10, n2=10, g2=0.04)
    imp_n1 = DCF_twostage_impyears(pe, r=0.12, g1=0.162, n2=10, g2=0.04)
    imp_r = DCF_twostage_impreturn(pe, n1=10, g1=0.162, n2=10, g2=0.04)

    df_gurufocus = pd.read_excel(PROJECT_DATA_PATH + '/gurufocus/20210310.xlsx', 'sheet')
    security_info_list = df_gurufocus[['公司', '市盈率', 'pred_growth']].dropna(how='any').values
    print(security_info_list)

    df_val_15y15p = dcf_analysis(security_info_list, n1=15, r=0.15)
    df_val_15y20p = dcf_analysis(security_info_list, n1=15, r=0.20)
    df_val_10y15p = dcf_analysis(security_info_list, n1=15, r=0.15)
    df_val_20y15p = dcf_analysis(security_info_list, n1=20, r=0.15)
    df_val_25y15p = dcf_analysis(security_info_list, n1=25, r=0.15)

    excel_quick_output(PROJECT_DATA_PATH + '/gurufocus/20210310_dcf_result.xlsx',
                       [df_val_15y15p, df_val_15y20p, df_val_10y15p, df_val_20y15p, df_val_25y15p],
                       ["15y15p", "15y20p", "10y15p", "20y15p", '25y15p'])
