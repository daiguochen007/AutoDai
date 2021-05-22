import numpy as np
import pandas as pd

import DaiToolkit as tk

print("\nOption tail analysis (sample)")
spot_px = 100
r = 0.02
T = 30 / 365
q = 0
k_list = np.arange(80, 97.01, 0.25)
sigma_list = [(100 - x) ** 2 * 0.035 / 100 + 0.16 for x in k_list]
optpx_list = tk.gen_option_pxlist(k_list, sigma_list, option_type='p', s=spot_px, r=r, T=T, q=q)
tk.option_tail_analysis(optpx_list, k_list, option_type='p', s=spot_px, r=r, T=T, q=q, tail_alpha=4.6)

tk.option_bsm('theta', 'p', spot=100, strike=90, maturity_years=30 / 365.0, vol=0.25, rate=0.03)
tk.option_bsm('theta', 'p', spot=100, strike=97, maturity_years=30 / 365.0, vol=0.18, rate=0.03)
tk.option_bsm('theta', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.18, rate=0.03)
tk.option_bsm('theta', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
tk.option_bsm('theta', 'p', spot=100, strike=90, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
tk.option_bsm('theta', 'p', spot=100, strike=90, maturity_years=180 / 365.0, vol=0.18, rate=0.03)

tk.option_bsm('vega', 'p', spot=100, strike=90, maturity_years=30 / 365.0, vol=0.25, rate=0.03)
tk.option_bsm('vega', 'p', spot=100, strike=97, maturity_years=30 / 365.0, vol=0.18, rate=0.03)
tk.option_bsm('vega', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.18, rate=0.03)
tk.option_bsm('vega', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
tk.option_bsm('vega', 'p', spot=100, strike=97, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
tk.option_bsm('vega', 'p', spot=100, strike=90, maturity_years=180 / 365.0, vol=0.25, rate=0.03)
tk.option_bsm('vega', 'p', spot=100, strike=90, maturity_years=180 / 365.0, vol=0.18, rate=0.03)

maturity = 7
v_atm = tk.option_bsm('vega', 'p', spot=100, strike=97, maturity_years=maturity / 365.0, vol=0.2, rate=0.03)
v_otm = tk.option_bsm('vega', 'p', spot=100, strike=90, maturity_years=maturity / 365.0, vol=0.3, rate=0.03)
print('Vega 倍数:' + str(v_atm / v_otm))

put_px_atm = tk.option_bsm('px', 'p', spot=100, strike=97, maturity_years=maturity / 365.0, vol=0.2, rate=0.03)
put_px_otm = tk.option_bsm('px', 'p', spot=100, strike=90, maturity_years=maturity / 365.0, vol=0.3, rate=0.03)
put_px_atm2 = tk.option_bsm('px', 'p', spot=95, strike=97, maturity_years=(maturity - 1) / 365.0, vol=0.2, rate=0.03)
put_px_otm2 = tk.option_bsm('px', 'p', spot=95, strike=90, maturity_years=(maturity - 1) / 365.0, vol=0.2, rate=0.03)
print('ATM chg 倍数:' + str(put_px_atm2 / put_px_atm))
print('OTM chg 倍数:' + str(put_px_otm2 / put_px_otm))

# option near mature anti fragile structure
t_list = range(30, 0, -1)
ret_list = [0] + list(np.random.normal(loc=-1, scale=1, size=len(t_list) - 1))
spot_list = ((pd.Series(ret_list) / 100 + 1).cumprod() * 100).to_list()

long_otm_num = 5
short_otm_num = -1
long_otm_strike = 88
short_otm_strike = 94

res = {}
for t, spot in zip(t_list, spot_list):
    res[t] = {}
    res[t]['spot'] = spot
    res[t]['K= ' + str(short_otm_strike) + ' impvol'] = 0.2
    res[t]['K= ' + str(long_otm_strike) + ' impvol'] = 0.25
    for n in ['px', 'delta', 'gamma', 'vega', 'theta']:
        res[t]['K= ' + str(short_otm_strike) + ' ' + n] = tk.option_bsm(n, 'p', spot=spot, strike=short_otm_strike, maturity_years=t / 365.0,
                                                                         vol=res[t]['K= ' + str(short_otm_strike) + ' impvol'], rate=0.03)
        res[t]['K= ' + str(long_otm_strike) + ' ' + n] = tk.option_bsm(n, 'p', spot=spot, strike=long_otm_strike, maturity_years=t / 365.0,
                                                                        vol=res[t]['K= ' + str(long_otm_strike) + ' impvol'], rate=0.03)
res = pd.DataFrame(res).T
res = res[['spot',
           'K= ' + str(short_otm_strike) + ' px', 'K= ' + str(long_otm_strike) + ' px',
           'K= ' + str(short_otm_strike) + ' impvol', 'K= ' + str(long_otm_strike) + ' impvol',
           'K= ' + str(short_otm_strike) + ' delta', 'K= ' + str(long_otm_strike) + ' delta',
           'K= ' + str(short_otm_strike) + ' gamma', 'K= ' + str(long_otm_strike) + ' gamma',
           'K= ' + str(short_otm_strike) + ' vega', 'K= ' + str(long_otm_strike) + ' vega',
           'K= ' + str(short_otm_strike) + ' theta','K= ' + str(long_otm_strike) + ' theta']]

res['port mv'] = res['K= ' + str(long_otm_strike) + ' px'] * long_otm_num + res['K= ' + str(short_otm_strike) + ' px'] * short_otm_num
res['port delta'] = res['K= ' + str(long_otm_strike) + ' delta'] * long_otm_num + res['K= ' + str(short_otm_strike) + ' delta'] * short_otm_num
res['port gamma'] = res['K= ' + str(long_otm_strike) + ' gamma'] * long_otm_num + res['K= ' + str(short_otm_strike) + ' gamma'] * short_otm_num
res['port vega'] = res['K= ' + str(long_otm_strike) + ' vega'] * long_otm_num + res['K= ' + str(short_otm_strike) + ' vega'] * short_otm_num
res['port theta'] = res['K= ' + str(long_otm_strike) + ' theta'] * long_otm_num + res['K= ' + str(short_otm_strike) + ' theta'] * short_otm_num

res.to_clipboard()

# option expected impied vol
res = {}
for px in np.arange(0.05, 3.51, 0.05):
    res[px] = tk.option_implied_vol(option_type='p', option_price=px, s=100, k=90, r=0.03, T=30 / 365.0, q=0)
res = pd.DataFrame(res, index=['impvol']).T
res.to_clipboard()
