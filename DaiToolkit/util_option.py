# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:24:29 2020

@author: Dai
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm


def option_bsm(stat, callput, spot, strike, maturity_years, vol, rate):
    """
    price and stat for European option
    
    GOOGL
    option_BSM('gamma','c',1487.17,1320.0,0.411,0.2467,0.015)
    
    stat: px, delta, gamma, vega, theta, rho
    
    delta : $ of $1 chg
    gamma : delta in $1 chg 
    vega  : $ of 1% vol chg
    theta : $ in 1d chg
    rho   : $ in 1% rate chg   
    """
    callput = callput.lower()
    stat = stat.lower()
    S = spot
    K = strike
    t = maturity_years
    sigma = vol
    r = rate

    d1 = (1 / (sigma * np.sqrt(t))) * (np.log(S / K) + (r + (sigma ** 2.0) / 2.0) * t)
    d2 = d1 - sigma * np.sqrt(t)

    if stat == 'gamma':
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(t))
        return gamma
    elif stat == 'vega':
        vega = norm.pdf(d1) * S * np.sqrt(t) / 100.0
        return vega
    elif stat == 'theta' and callput == 'c':
        call_theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) - r * K * math.exp(-r * t) * norm.cdf(d2)
        return call_theta / 365.0
    elif stat == 'theta' and callput == 'p':
        put_theta = - (S * norm.pdf(-d1) * sigma) / (2 * np.sqrt(t)) + r * K * math.exp(-r * t) * norm.cdf(-d2)
        return put_theta / 365.0
    elif stat == 'rho' and callput == 'c':
        call_rho = K * t * math.exp(-r * t) * norm.cdf(d2) / 100.0
        return call_rho
    elif stat == 'rho' and callput == 'p':
        put_rho = -K * t * math.exp(-r * t) * norm.cdf(-d2) / 100.0
        return put_rho
    elif stat == "delta" and callput == 'c':
        call_delta = norm.cdf(d1)
        return call_delta
    elif stat == "delta" and callput == 'p':
        put_delta = -norm.cdf(-d1)
        return put_delta
    elif stat == "px" and callput == 'c':
        call_px = norm.cdf(d1) * S - norm.cdf(d2) * K * math.exp(-r * t)
        return call_px
    elif stat == "px" and callput == 'p':
        call_px = norm.cdf(d1) * S - norm.cdf(d2) * K * math.exp(-r * t)
        put_px = K * math.exp(-r * t) + call_px - S
        return put_px
    else:
        raise Exception(
            "Input not accepted! Please use stat in [px, delta, gamma, vega, theta, rho] and callput [c, p]")


def option_bsm_price(option_type, sigma, s, k, r, T, q):
    """
    calculate the bsm price of European call and put options
    
    option_bsm_price('', sigma, s, k, r, T, q)
    """
    sigma = float(sigma)
    d1 = (np.log(s / k) + (r - q + sigma ** 2 * 0.5) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'c':
        price = np.exp(-r * T) * (s * np.exp((r - q) * T) * norm.cdf(d1) - k * norm.cdf(d2))
        return price
    elif option_type == 'p':
        price = np.exp(-r * T) * (k * norm.cdf(-d2) - s * np.exp((r - q) * T) * norm.cdf(-d1))
        return price
    else:
        print('No such option type %s' % option_type)


def option_implied_vol(option_type, option_price, s, k, r, T, q, precision=None, max_vol=500, min_vol=0.000001):
    '''
    apply bisection method to get the implied volatility by solving the BSM function

    precision: None (Auto set precision with given price input prec + 4 digit)
    max_vol: 50000%
    min_vol: 0.0001%
    '''
    upper_vol = max_vol
    lower_vol = min_vol
    vol_precision = 1e-10
    if precision is None:
        precision = 10 ** (-1 * (round(math.log10(1 / option_price)) + 4))

    while 1:
        mid_vol = (upper_vol + lower_vol) / 2.0
        price = option_bsm_price(option_type, mid_vol, s, k, r, T, q)
        if option_type == 'c':
            lower_price = option_bsm_price(option_type, lower_vol, s, k, r, T, q)
            if (lower_price - option_price) * (price - option_price) > 0:
                lower_vol = mid_vol
            else:
                upper_vol = mid_vol
            if abs(price - option_price) < precision:
                break
        elif option_type == 'p':
            upper_price = option_bsm_price(option_type, upper_vol, s, k, r, T, q)
            if (upper_price - option_price) * (price - option_price) > 0:
                upper_vol = mid_vol
            else:
                lower_vol = mid_vol
            if abs(price - option_price) < precision:
                break
        if abs(mid_vol - min_vol) < vol_precision or abs(mid_vol - max_vol) < vol_precision:
            print("Warning: option implied vol iteration hit boundary " + str(mid_vol))
            break

    return mid_vol


def gen_option_pxlist(k_list, sigma_list, option_type, s, r, T, q):
    """
    generate option px list from implied vol list (panel data)

    :param k_list:
    :param sigma_list:
    :return: BSM px list from k ~ implied vol list
    """
    opt_list = []
    for k, sigma in zip(k_list, sigma_list):
        opt_list.append(option_bsm_price(option_type=option_type, sigma=sigma, s=s, k=k, r=r, T=T, q=q))
    return opt_list


def option_tail_analysis(optpx_list, k_list, option_type, s, r, T, q, tail_alpha):
    """
    analyze option px ratio under powerlaw (mkt, fitted power law, historical power law)

    :param optpx_list:
    :param k_list:
    :param option_type:
    :param s:
    :param r:
    :param T:
    :param q:
    :return:
    """
    k_list_rescale = [x / s * 100 for x in k_list]
    sigma_list = [option_implied_vol(option_type, p, s, k, r, T, q) for p, k in zip(optpx_list, k_list)]

    # compare tail_alpha
    opt_ratio_list = [x / max(optpx_list) for x in optpx_list]
    if option_type == 'p':
        theo_ratio_list = [((100 - x) / (100 - max(k_list_rescale))) ** (1 - tail_alpha) for x in k_list_rescale]
    else:
        theo_ratio_list = [((100 - x) / (100 - min(k_list_rescale))) ** (1 - tail_alpha) for x in k_list_rescale]

    # fit mkt implied alpha
    def fit_err(k_list_rescale, opt_ratio_list, alpha):
        if option_type == 'p':
            theo_ratio_list = [((100 - x) / (100 - max(k_list_rescale))) ** (1 - alpha) for x in k_list_rescale]
        else:
            theo_ratio_list = [((100 - x) / (100 - min(k_list_rescale))) ** (1 - alpha) for x in k_list_rescale]
        return sum([(x - y) ** 2 for x, y in zip(theo_ratio_list, opt_ratio_list)])

    res = sp.optimize.minimize(lambda x: fit_err(k_list_rescale, opt_ratio_list, x), tail_alpha)
    implied_alpha = res['x'][0]
    if option_type == 'p':
        imp_ratio_list = [((100 - x) / (100 - max(k_list_rescale))) ** (1 - implied_alpha) for x in k_list_rescale]
    else:
        imp_ratio_list = [((100 - x) / (100 - min(k_list_rescale))) ** (1 - implied_alpha) for x in k_list_rescale]

    fig, ((ax1, ax2)) = plt.subplots(2, 1)
    fig.suptitle('Option Power Law Analysis')
    ax1.plot(k_list_rescale, sigma_list, ls='-', marker='o', alpha=0.5, markersize=2, label='skewed vol')
    ax1.grid(ls='--', alpha=0.4)
    ax1.legend()
    ax1.set_title("Option Implied Volatility Curve")

    ax2.plot(k_list, opt_ratio_list, ls='-', marker='o', alpha=0.5, markersize=2, label='market price')
    ax2.plot(k_list, imp_ratio_list, ls='-', marker='o', alpha=0.5, markersize=2, label='fitted (alpha=' + str(round(implied_alpha, 2)) + ')')
    ax2.plot(k_list, theo_ratio_list, ls='-', marker='o', alpha=0.5, markersize=2, label='theoretical (alpha=' + str(round(tail_alpha, 2)) + ')')
    ax2.grid(ls='--', alpha=0.4)
    ax2.legend()
    ax2.set_title("Option OTM Price Ratio")
    plt.show()



if __name__ == "__main__":
    # test
    print("BSM func 1 (SPXW)")
    for stat in ['px', 'delta', 'gamma', 'vega', 'theta', 'rho']:
        print(stat + " : " + str(option_bsm(stat, 'p', spot=3348.44, strike=3350, maturity_years=14 / 365.0, vol=0.2260, rate=0.008)))

    print("\nBSM func 2")
    px = option_bsm_price(option_type='p', sigma=0.226, s=3348.44, k=2000, r=0.008, T=14 / 365.0, q=0)
    print("Price 2: " + str(px))
    print("imp vol: " + str(option_implied_vol(option_type='p', option_price=px, s=3348.44, k=2000, r=0.008, T=14 / 365.0, q=0)))

    print("\nBSM func 2")
    option_implied_vol(option_type='p', option_price=px, s=3348.44, k=3350, r=0.008, T=14 / 365.0, q=0)

    print("\n苏银转债")
    for stat in ['px', 'delta', 'gamma', 'vega', 'theta', 'rho']:
        print(stat + " : " + str(
            option_bsm(stat, 'c', spot=5.39, strike=6.69, maturity_years=4.21, vol=0.15, rate=0.02)))

    print("\nGME")
    imp_vol = option_implied_vol("p", 0.78, 347.51, 3.0, 0.02, 0.8222, 0)
    print("Implied Vol: " + str(imp_vol))
    for stat in ['px', 'delta', 'gamma', 'vega', 'theta', 'rho']:
        print(stat + " : " + str(
            option_bsm(stat, 'p', spot=347.51, strike=3, maturity_years=0.8222, vol=imp_vol, rate=0.02)))

    print("\nOption tail analysis (sample)")
    spot_px = 100
    r = 0.02
    T = 30/365
    q = 0
    k_list = np.arange(80, 97.01, 0.25)
    sigma_list = [(100 - x) ** 2 * 0.035 / 100 + 0.16 for x in k_list]
    optpx_list = gen_option_pxlist(k_list, sigma_list, option_type='p', s=spot_px, r=r, T=T, q=q)
    option_tail_analysis(optpx_list, k_list, option_type='p', s=spot_px, r=r, T=T, q=q, tail_alpha=4.6)

    option_bsm('theta', 'p', spot=100, strike=90, maturity_years=30 / 365.0, vol=0.25, rate=0.03)
    option_bsm('theta', 'p', spot=100, strike=97, maturity_years=30 / 365.0, vol=0.18, rate=0.03)
    option_bsm('theta', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.18, rate=0.03)
    option_bsm('theta', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
    option_bsm('theta', 'p', spot=100, strike=90, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
    option_bsm('theta', 'p', spot=100, strike=90, maturity_years=180 / 365.0, vol=0.18, rate=0.03)
    
    option_bsm('vega', 'p', spot=100, strike=90, maturity_years=30 / 365.0, vol=0.25, rate=0.03)
    option_bsm('vega', 'p', spot=100, strike=97, maturity_years=30 / 365.0, vol=0.18, rate=0.03)
    option_bsm('vega', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.18, rate=0.03)
    option_bsm('vega', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
    option_bsm('vega', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
    option_bsm('vega', 'p', spot=100, strike=90, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
    option_bsm('vega', 'p', spot=100, strike=90, maturity_years=180 / 365.0, vol=0.18, rate=0.03)
