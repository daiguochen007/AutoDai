# -*- coding: utf-8 -*-

from functools import partial

import empyrical as ep
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
from sklearn import linear_model

APPROX_BDAYS_PER_MONTH = 21
DAILY = 'daily'


def var_cov_var_normal(P, c, mu=0, sigma=1):
    """
    Variance-covariance calculation of daily Value-at-Risk in a
    portfolio.

    Parameters
    ----------
    P : float
        Portfolio value.
    c : float
        Confidence level.
    mu : float, optional
        Mean.

    Returns
    -------
    float
        Variance-covariance.
    """

    alpha = sp.stats.norm.ppf(1 - c, mu, sigma)
    return P - P * (alpha + 1)


def max_drawdown(returns):
    """
    Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    float
        Maximum drawdown.

    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """

    return ep.max_drawdown(returns)


def annual_return(returns, period=DAILY):
    """
    Determines the mean annual growth rate of returns.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    float
        Annual Return as CAGR (Compounded Annual Growth Rate).
    """

    return ep.annual_return(returns, period=period)


def annual_volatility(returns, period=DAILY):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing volatility. Can be 'monthly' or 'weekly' or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    float
        Annual volatility.
    """

    return ep.annual_volatility(returns, period=period)


def calmar_ratio(returns, period=DAILY):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    float
        Calmar ratio (drawdown ratio) as float. Returns np.nan if there is no
        calmar ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
    """

    return ep.calmar_ratio(returns, period=period)


def omega_ratio(returns, annual_return_threshhold=0.0):
    """
    Determines the Omega ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    annual_return_threshold : float, optional
        Minimum acceptable return of the investor. Annual threshold over which
        returns are considered positive or negative. It is converted to a
        value appropriate for the period of the returns for this ratio.
        E.g. An annual minimum acceptable return of 100 translates to a daily
        minimum acceptable return of 0.01848.
            (1 + 100) ** (1. / 252) - 1 = 0.01848
        Daily returns must exceed this value to be considered positive. The
        daily return yields the desired annual return when compounded over
        the average number of business days in a year.
            (1 + 0.01848) ** 252 - 1 = 99.93
        - Defaults to 0.0


    Returns
    -------
    float
        Omega ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.
    """

    return ep.omega_ratio(returns,
                          required_return=annual_return_threshhold)


def sortino_ratio(returns, required_return=0, period=DAILY):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> np.array

        Annualized Sortino ratio.
    """

    return ep.sortino_ratio(returns, required_return=required_return)


def downside_risk(returns, required_return=0, period=DAILY):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    depends on input type
    series ==> float
    DataFrame ==> np.array

        Annualized downside deviation
    """

    return ep.downside_risk(returns,
                            required_return=required_return,
                            period=period)


def sharpe_ratio(returns, risk_free=0, period=DAILY):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    risk_free : int, float
        Constant risk-free return throughout the period.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Can be 'monthly', 'weekly', or 'daily'.
        - Defaults to 'daily'.

    Returns
    -------
    float
        Sharpe ratio.
    np.nan
        If insufficient length of returns or if if adjusted returns are 0.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
    """

    return ep.sharpe_ratio(returns, risk_free=risk_free, period=period)


def alpha_beta(returns, factor_returns):
    """
    Calculates both alpha and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.

    Returns
    -------
    float
        Alpha.
    float
        Beta.
    """

    return ep.alpha_beta(returns, factor_returns=factor_returns)


def alpha(returns, factor_returns):
    """
    Calculates annualized alpha.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.

    Returns
    -------
    float
        Alpha.
    """

    return ep.alpha(returns, factor_returns=factor_returns)


def beta(returns, factor_returns):
    """
    Calculates beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    factor_returns : pd.Series
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.

    Returns
    -------
    float
        Beta.
    """

    return ep.beta(returns, factor_returns)


def stability_of_timeseries(returns):
    """
    Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.

    Returns
    -------
    float
        R-squared.
    """

    return ep.stability_of_timeseries(returns)


def tail_ratio(returns):
    """
    Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.

    Returns
    -------
    float
        tail ratio
    """

    return ep.tail_ratio(returns)


def common_sense_ratio(returns):
    """
    Common sense ratio is the multiplication of the tail ratio and the
    Gain-to-Pain-Ratio -- sum(profits) / sum(losses).

    See http://bit.ly/1ORzGBk for more information on motivation of
    this metric.


    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    float
        common sense ratio
    """

    return ep.tail_ratio(returns) * \
           (1 + ep.annual_return(returns))


def normalize(returns, starting_value=1):
    """
    Normalizes a returns timeseries based on the first value.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    starting_value : float, optional
       The starting returns (default 1).

    Returns
    -------
    pd.Series
        Normalized returns.
    """

    return starting_value * (returns / returns.iloc[0])


def cum_returns(returns, starting_value=0):
    """
    Compute cumulative returns from simple returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    starting_value : float, optional
       The starting returns (default 1).

    Returns
    -------
    pandas.Series
        Series of cumulative returns.

    Notes
    -----
    For increased numerical accuracy, convert input to log returns
    where it is possible to sum instead of multiplying.
    """

    return ep.cum_returns(returns, starting_value=starting_value)


def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    returns : pd.Series
       Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~pyfolio.timeseries.cum_returns`.
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    pd.Series
        Aggregated returns.
    """

    return ep.aggregate_returns(returns, convert_to=convert_to)


def rolling_beta(returns, factor_returns,
                 rolling_window=APPROX_BDAYS_PER_MONTH * 6):
    """
    Determines the rolling beta of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series or pd.DataFrame
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - If DataFrame is passed, computes rolling beta for each column.
         - This is in the same style as returns.
    rolling_window : int, optional
        The size of the rolling window, in days, over which to compute
        beta (default 6 months).

    Returns
    -------
    pd.Series
        Rolling beta.

    Note
    -----
    See https://en.wikipedia.org/wiki/Beta_(finance) for more details.
    """

    if factor_returns.ndim > 1:
        # Apply column-wise
        return factor_returns.apply(partial(rolling_beta, returns),
                                    rolling_window=rolling_window)
    else:
        out = pd.Series(index=returns.index)
        for beg, end in zip(returns.index[0:-rolling_window],
                            returns.index[rolling_window:]):
            out.loc[end] = ep.beta(
                returns.loc[beg:end],
                factor_returns.loc[beg:end])

        return out


def rolling_regression(returns, factor_returns,
                       rolling_window=APPROX_BDAYS_PER_MONTH * 6,
                       nan_threshold=0.1):
    """
    Computes rolling factor betas using a multivariate linear regression
    (separate linear regressions is problematic because the factors may be
    confounded).

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.DataFrame
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - Computes rolling beta for each column.
         - This is in the same style as returns.
    rolling_window : int, optional
        The days window over which to compute the beta. Defaults to 6 months.
    nan_threshold : float, optional
        If there are more than this fraction of NaNs, the rolling regression
        for the given date will be skipped.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing rolling beta coefficients to SMB, HML and UMD
    """

    # We need to drop NaNs to regress
    ret_no_na = returns.dropna()

    columns = ['alpha'] + factor_returns.columns.tolist()
    rolling_risk = pd.DataFrame(columns=columns,
                                index=ret_no_na.index)

    rolling_risk.index.name = 'dt'

    for beg, end in zip(ret_no_na.index[:-rolling_window],
                        ret_no_na.index[rolling_window:]):
        returns_period = ret_no_na[beg:end]
        factor_returns_period = factor_returns.loc[returns_period.index]

        if np.all(factor_returns_period.isnull().mean()) < nan_threshold:
            factor_returns_period_dnan = factor_returns_period.dropna()
            reg = linear_model.LinearRegression(fit_intercept=True).fit(
                factor_returns_period_dnan,
                returns_period.loc[factor_returns_period_dnan.index])
            rolling_risk.loc[end, factor_returns.columns] = reg.coef_
            rolling_risk.loc[end, 'alpha'] = reg.intercept_

    return rolling_risk


def gross_lev(positions):
    """
    Calculates the gross leverage of a strategy.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in tears.create_full_tear_sheet.

    Returns
    -------
    pd.Series
        Gross leverage.
    """

    exposure = positions.drop('cash', axis=1).abs().sum(axis=1)
    return exposure / positions.sum(axis=1)


def value_at_risk(returns, period=None, sigma=2.0):
    """
    Get value at risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    period : str, optional
        Period over which to calculate VaR. Set to 'weekly',
        'monthly', or 'yearly', otherwise defaults to period of
        returns (typically daily).
    sigma : float, optional
        Standard deviations of VaR, default 2.
    """
    if period is not None:
        returns_agg = ep.aggregate_returns(returns, period)
    else:
        returns_agg = returns.copy()

    value_at_risk = returns_agg.mean() - sigma * returns_agg.std()
    return value_at_risk


SIMPLE_STAT_FUNCS = [
    ep.annual_return,
    ep.cum_returns_final,
    ep.annual_volatility,
    ep.sharpe_ratio,
    ep.calmar_ratio,
    ep.stability_of_timeseries,
    ep.max_drawdown,
    ep.omega_ratio,
    ep.sortino_ratio,
    stats.skew,
    stats.kurtosis,
    ep.tail_ratio,
    value_at_risk
]

FACTOR_STAT_FUNCS = [
    ep.alpha,
    ep.beta,
]

STAT_FUNC_NAMES = {
    'annual_return': 'Annual return',
    'cum_returns_final': 'Cumulative returns',
    'annual_volatility': 'Annual volatility',
    'sharpe_ratio': 'Sharpe ratio',
    'calmar_ratio': 'Calmar ratio',
    'stability_of_timeseries': 'Stability',
    'max_drawdown': 'Max drawdown',
    'omega_ratio': 'Omega ratio',
    'sortino_ratio': 'Sortino ratio',
    'skew': 'Skew',
    'kurtosis': 'Kurtosis',
    'tail_ratio': 'Tail ratio',
    'common_sense_ratio': 'Common sense ratio',
    'value_at_risk': 'Daily value at risk',
    'alpha': 'Alpha',
    'beta': 'Beta',
}


def perf_stats(returns, factor_returns=None):
    """
    Calculates various performance metrics of a strategy, for use in
    plotting.show_perf_stats.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
         - If None, do not compute alpha, beta, and information ratio.

    Returns
    -------
    pd.Series
        Performance metrics.
    """

    stats = pd.Series()
    for stat_func in SIMPLE_STAT_FUNCS:
        stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns)

    if factor_returns is not None:
        for stat_func in FACTOR_STAT_FUNCS:
            res = stat_func(returns, factor_returns)
            stats[STAT_FUNC_NAMES[stat_func.__name__]] = res

    return stats
