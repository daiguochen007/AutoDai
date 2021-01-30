# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 19:24:29 2020

@author: Dai
"""

import math

import numpy as np
from scipy.stats import norm


def option_bsm(stat, callput, spot, strike, maturity_years, vol, rate):
    """
    price and stat for Eurpean option
    
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


def option_implied_vol(option_type, option_price, s, k, r, T, q):
    '''
    apply bisection method to get the implied volatility by solving the BSM function
    '''
    precision = 0.000001
    upper_vol = 500.0
    lower_vol = 0.0001
    max_vol = 500.0
    iteration = 0

    while 1:
        iteration += 1
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
            if mid_vol > max_vol - 5:
                mid_vol = 0.000001
                break
        elif option_type == 'p':
            upper_price = option_bsm_price(option_type, upper_vol, s, k, r, T, q)

            if (upper_price - option_price) * (price - option_price) > 0:
                upper_vol = mid_vol
            else:
                lower_vol = mid_vol
            if abs(price - option_price) < precision:
                break
            if iteration > 50:
                break

    return mid_vol


## SPXW
if __name__ == "__main__":
    pass
    #    print("BSM func 1")
    #    for stat in ['px','delta','gamma','vega','theta','rho']:
    #        print stat +" : " +str(option_bsm(stat,'p',spot=3348.44,strike=3350,maturity_years=14/365.0,vol=0.2260,rate=0.008))
    #
    #    print("\nBSM func 2")
    #    px = option_bsm_price(option_type='p',sigma=0.226, s=3348.44, k=3350, r=0.008, T=14/365.0, q=0)
    #    print("Price 2: "+str(px))
    #    print("imp vol: "+str(option_implied_vol(option_type='p', option_price=px, s=3348.44, k=3350, r=0.008, T=14/365.0, q=0)))
    #
    #    print("\nBSM func 2")
    #    option_implied_vol(option_type='p', option_price=px, s=3348.44, k=3350, r=0.008, T=14/365.0, q=0)

    print("苏银转债")
    for stat in ['px', 'delta', 'gamma', 'vega', 'theta', 'rho']:
        print(stat + " : " + str(
            option_bsm(stat, 'c', spot=5.39, strike=6.69, maturity_years=4.21, vol=0.15, rate=0.02)))

    print("GME")
    imp_vol = option_implied_vol("p", 0.78, 347.51, 3.0, 0.02, 0.8222, 0)
    print("Implied Vol: "+str(imp_vol))
    for stat in ['px', 'delta', 'gamma', 'vega', 'theta', 'rho']:
        print(stat + " : " + str(
            option_bsm(stat, 'p', spot=347.51, strike=3, maturity_years=0.8222, vol=imp_vol, rate=0.02)))
