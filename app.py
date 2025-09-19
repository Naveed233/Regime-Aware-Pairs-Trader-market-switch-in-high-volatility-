"""
Regime-Aware Pairs Trader Streamlit Application

This app allows users to explore a simple mean‑reversion pair trading strategy
with and without a market‑stress switch.  Users can pick two securities
(defaulting to the energy ETFs XLE and XOP) and a date range, adjust the
parameters used to build the spread and the mean‑reversion rules, and
configure the regime filter.  The application pulls price data from Yahoo
Finance via the `yfinance` package, computes the hedge ratio and spread,
normalizes it into a z‑score, and then backtests two strategies:

  • Strategy A (Always‑On): trades the pair whenever the z‑score
    breaches a user‑selected entry threshold and exits when the z‑score
    reverts to zero.  An optional stop can close trades when the z‑score
    becomes too extreme.

  • Strategy B (Regime‑Aware): follows the same entry/exit rules but
    only trades when the market is deemed calm.  The calm state is
    determined either by a Markov‑switching model fitted to daily
    returns of a market index (default SPY) or by a simple VIX
    threshold.  Hysteresis thresholds control when the filter turns
    on and off.  Users can optionally scale positions to target a
    specific volatility.

Both strategies include transaction costs measured in basis points per
position change.  After each run the app displays key performance
statistics—cumulative return, annualized Sharpe ratio, maximum drawdown,
number of trades, and percentage of days invested—along with equity
curves and diagnostic charts.  A mini assessment summarises whether the
regime filter improved drawdowns and/or risk‑adjusted returns.

Note: To run this application you will need an internet connection and
the `yfinance` package installed.  You can install it with
``pip install yfinance``.  When fitting the Markov model the
`statsmodels` library is used; it is included in most Python
distributions for data science.  If the Markov model fails to
converge, the app will automatically fall back to a VIX rule.
"""

import datetime as _dt
from functools import lru_cache
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None

import statsmodels.api as sm  # type: ignore


def load_price_data(ticker: str, start: _dt.date, end: _dt.date) -> pd.DataFrame:
    """Download adjusted close prices for a single ticker.

    Parameters
    ----------
    ticker : str
        The ticker symbol to download (e.g., 'XLE').
    start : datetime.date
        Start date (inclusive).
    end : datetime.date
        End date (exclusive).

    Returns
    -------
    pandas.DataFrame
        DataFrame with a DatetimeIndex and an 'Adj Close' column.
    """
    if yf is None:
        raise RuntimeError(
            "The yfinance package is required for data download. Please install it with 'pip install yfinance'."
        )
    # Use yfinance download; auto adjusts dates to include end date
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}.")
    
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex columns - yfinance returns (price_type, ticker) format
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        # Look for adjusted close or close columns
        adj_close_cols = [col for col in df.columns if 'Adj Close' in col]
        close_cols = [col for col in df.columns if 'Close' in col and 'Adj Close' not in col]
        
        if adj_close_cols:
            price_col = adj_close_cols[0]
        elif close_cols:
            price_col = close_cols[0]
        else:
            raise ValueError(f"Downloaded data for {ticker} columns {df.columns.tolist()} does not contain Close data.")
    else:
        # Handle regular column names
        if 'Adj Close' in df.columns:
            price_col = 'Adj Close'
        elif 'Close' in df.columns:
            price_col = 'Close'
        else:
            raise ValueError(f"Downloaded data for {ticker} columns {df.columns.tolist()} does not contain 'Adj Close' or 'Close'.")
    
    return df[[price_col]].rename(columns={price_col: ticker})


@st.cache_data
def get_prices(tickers: Tuple[str, str, str], start: _dt.date, end: _dt.date) -> pd.DataFrame:
    """Fetch prices for the pair and the market index.

    Parameters
    ----------
    tickers : tuple of (str, str, str)
        Tuple containing the pair tickers (y, x) and the market proxy (e.g., ('XLE','XOP','SPY')).
    start, end : datetime.date
        Start and end dates.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns for each ticker's adjusted close price.
    """
    y_ticker, x_ticker, mkt_ticker = tickers
    df_list = []
    for t in {y_ticker, x_ticker, mkt_ticker}:
        df_list.append(load_price_data(t, start, end))
    data = pd.concat(df_list, axis=1).dropna()
    return data


def compute_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    lookback: Optional[int] = None,
    include_intercept: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """Compute rolling or static hedge ratios via linear regression.

    This function estimates the hedge ratio between the dependent series `y` and
    independent series `x`.  If a lookback window is specified, a rolling
    regression is performed; otherwise a single regression is fit to the
    full sample and its coefficients repeated.

    Parameters
    ----------
    y, x : pandas.Series
        Price series (must be aligned by index).
    lookback : int or None
        Window length in days for the rolling regression.  If None or <= 0,
        a single regression is performed on the entire dataset.
    include_intercept : bool
        Whether to include an intercept term in the regression.  The
        intercept captures the mean spread level.

    Returns
    -------
    hedge_ratio : pandas.Series
        Estimated hedge ratio (slope) at each time point.
    intercept : pandas.Series
        Estimated intercept at each time point (zero if include_intercept
        is False).
    """
    if lookback and lookback > 0:
        slopes = []
        intercepts = []
        for i in range(len(y)):
            # Determine the window slice
            start_idx = max(0, i - lookback + 1)
            end_idx = i + 1
            y_window = y.iloc[start_idx:end_idx]
            x_window = x.iloc[start_idx:end_idx]
            if include_intercept:
                X = sm.add_constant(x_window)
            else:
                X = x_window.values[:, None]
            try:
                model = sm.OLS(y_window.values, X).fit()
                params = model.params
                if include_intercept:
                    intercepts.append(params[0])
                    slopes.append(params[1])
                else:
                    intercepts.append(0.0)
                    slopes.append(params[0])
            except Exception:
                # Fallback to previous values on failure
                if slopes:
                    slopes.append(slopes[-1])
                    intercepts.append(intercepts[-1])
                else:
                    slopes.append(1.0)
                    intercepts.append(0.0)
        hr_series = pd.Series(slopes, index=y.index)
        ic_series = pd.Series(intercepts, index=y.index)
    else:
        # Static regression
        if include_intercept:
            X = sm.add_constant(x)
        else:
            X = x.values[:, None]
        model = sm.OLS(y.values, X).fit()
        params = model.params
        if include_intercept:
            ic = float(params[0])
            slope = float(params[1])
        else:
            ic = 0.0
            slope = float(params[0])
        hr_series = pd.Series(slope, index=y.index)
        ic_series = pd.Series(ic, index=y.index)
    return hr_series, ic_series


def compute_spread(y: pd.Series, x: pd.Series, hr: pd.Series, ic: pd.Series) -> pd.Series:
    """Compute the spread as y - hr*x - intercept.

    Parameters
    ----------
    y, x : pandas.Series
        Dependent and independent series.
    hr, ic : pandas.Series
        Hedge ratio and intercept series (can be constant).

    Returns
    -------
    pandas.Series
        The spread series.
    """
    return y - hr * x - ic


def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z‑score of a series.

    Parameters
    ----------
    series : pandas.Series
        Input series.
    window : int
        Rolling window length for mean and standard deviation.

    Returns
    -------
    pandas.Series
        Z‑score series.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std(ddof=0)
    z = (series - rolling_mean) / rolling_std
    return z


def fit_markov_regime(series: pd.Series) -> pd.DataFrame:
    """Fit a two‑state Markov‑switching AR model to a return series.

    Returns a DataFrame with the smoothed probabilities of each regime.

    Parameters
    ----------
    series : pandas.Series
        The return series to fit.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns corresponding to the probability of
        being in each regime.  Column names are integers (0 and 1).
    """
    # Remove NaNs
    series = series.dropna()
    if series.empty or len(series) < 50:
        raise ValueError("Return series is too short to fit Markov model.")
    try:
        model = sm.tsa.MarkovRegression(series, k_regimes=2, trend='c', switching_variance=True)
        res = model.fit(disp=False)
        probs = res.smoothed_marginal_probabilities
        # Align index to original series
        probs.index = series.index
        return probs
    except Exception as e:
        raise RuntimeError(f"Markov model failed: {e}")


def classify_calm_state(probs: pd.DataFrame, series: pd.Series) -> pd.Series:
    """Select which regime corresponds to a 'calm' state.

    Chooses the regime with the lower volatility of residuals.  Uses the
    probabilities to compute weighted volatility per regime.

    Parameters
    ----------
    probs : pandas.DataFrame
        Smoothed probabilities with columns 0 and 1.
    series : pandas.Series
        Original return series used to fit the model.

    Returns
    -------
    pandas.Series
        Series of probabilities of the calm regime.
    """
    # Compute regime‑specific variance as weighted average of squared returns
    regime_vars = {}
    for r in probs.columns:
        regime_vars[r] = np.sum(probs[r] * series ** 2) / np.sum(probs[r])
    calm_regime = min(regime_vars, key=regime_vars.get)
    return probs[calm_regime]


def compute_regime_signal(
    data: pd.DataFrame,
    mkt_ticker: str,
    filter_type: str = 'Markov',
    p_on: float = 0.6,
    p_off: float = 0.4,
    vix_threshold: float = 20.0,
) -> Tuple[pd.Series, str]:
    """Compute the market regime signal (ON = trade allowed).

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the market proxy price series and, if
        applicable, the VIX series (ticker '^VIX').  Must contain a
        column for `mkt_ticker`.
    mkt_ticker : str
        Ticker used as the market proxy (e.g. 'SPY').
    filter_type : {'Markov', 'VIX'}
        Which filter to use.  If 'Markov', a two‑state Markov model is
        fitted to the market returns; if fitting fails, the function
        falls back to the VIX threshold.  If 'VIX', the threshold
        directly defines the signal.
    p_on : float
        Probability threshold above which the filter turns ON (calm).
    p_off : float
        Probability threshold below which the filter turns OFF (stress).
    vix_threshold : float
        VIX level below which the regime is considered calm.

    Returns
    -------
    regime : pandas.Series of bool
        True when the regime filter allows trading, False otherwise.
    used_filter : str
        Description of which filter was used ('Markov' or 'VIX').
    """
    index = data.index
    if filter_type == 'Markov':
        try:
            returns = data[mkt_ticker].pct_change().dropna()
            probs = fit_markov_regime(returns)
            p_calm = classify_calm_state(probs, returns)
            # Align p_calm with full index (forward fill and initial NaNs to False)
            p_calm = p_calm.reindex(index).fillna(method='ffill').fillna(0.0)
            # Hysteresis: compute on/off signal
            signal = []
            current_on = False
            for prob in p_calm:
                if current_on:
                    # Only turn off if calm probability drops below p_off
                    current_on = prob >= p_off
                else:
                    # Turn on if calm probability rises above p_on
                    current_on = prob >= p_on
                signal.append(current_on)
            regime = pd.Series(signal, index=index)
            return regime, 'Markov'
        except Exception:
            # Fallback to VIX if Markov fails
            filter_type = 'VIX'
    # VIX based filter
    if '^VIX' not in data.columns:
        # Attempt to download VIX data only when needed
        if yf is None:
            raise RuntimeError(
                "VIX data is required for the VIX filter but yfinance is not installed."
            )
        vix = load_price_data('^VIX', data.index[0].date(), data.index[-1].date())
        data = data.join(vix, how='left')
    vix_series = data['^VIX'].ffill().reindex(index)
    regime = vix_series < vix_threshold
    return regime.astype(bool), 'VIX'


def backtest_pair(
    prices: pd.DataFrame,
    pair: Tuple[str, str],
    hedge_lookback: Optional[int],
    z_window: int,
    z_in: float,
    z_exit: float = 0.0,
    z_stop: Optional[float] = None,
    regime: Optional[pd.Series] = None,
    vol_target: Optional[float] = None,
    cost_bps: float = 10.0,
) -> Dict[str, any]:
    """Backtest the pair trading strategy with optional regime filter and volatility sizing.

    Parameters
    ----------
    prices : pandas.DataFrame
        DataFrame containing the price series for the pair (Y and X).  Must
        have columns named pair[0] and pair[1].
    pair : tuple of (str, str)
        Names of the dependent (Y) and independent (X) assets.
    hedge_lookback : int or None
        Window for the rolling hedge ratio.  Use None for a static ratio.
    z_window : int
        Rolling window used to compute the z‑score of the spread.
    z_in : float
        Absolute z‑score threshold to enter a trade (both long and short).
    z_exit : float
        Z‑score level to exit a trade (0 means exit when z crosses zero).
    z_stop : float or None
        Optional stop level.  If provided, positions are closed when the
        absolute z‑score exceeds this value.
    regime : pandas.Series of bool or None
        Series indicating when trading is allowed (True) or not (False).
        If None, the strategy is always allowed to trade.
    vol_target : float or None
        Annualized volatility target for position sizing (e.g., 0.1 for
        10% annualized).  If None, positions are unscaled (units of 1).
    cost_bps : float
        Transaction cost in basis points per change in position.

    Returns
    -------
    dict
        Dictionary containing returns, positions, trades, and summary metrics.
    """
    y_name, x_name = pair
    y = prices[y_name]
    x = prices[x_name]
    # Estimate hedge ratio and intercept
    hr, ic = compute_hedge_ratio(y, x, lookback=hedge_lookback)
    spread = compute_spread(y, x, hr, ic)
    z = compute_zscore(spread, window=z_window)
    # Spread return (long y, short x*hr)
    spread_return = y.pct_change().fillna(0) - hr * x.pct_change().fillna(0)
    # Initialise arrays
    pos = np.zeros(len(prices))  # Position: +1 long spread, -1 short, 0 flat
    weight = np.zeros(len(prices))  # Position weight after scaling
    pnl = np.zeros(len(prices))
    costs = np.zeros(len(prices))
    trades = []  # List of (entry_date, exit_date, direction)
    current_pos = 0.0
    current_weight = 0.0
    entry_date = None
    # Rolling volatility for sizing
    if vol_target:
        # Estimate daily volatility of spread return with a 20‑day rolling window
        vol_window = 20
        rolling_vol = spread_return.rolling(vol_window, min_periods=1).std(ddof=0)
    for i, date in enumerate(prices.index):
        allow_trade = True if regime is None else bool(regime.iloc[i])
        # Extract scalar z‑score for this date.  If the value is missing
        # (NaN), treat it as zero to avoid ambiguous truth values.
        z_raw = z.iloc[i]
        z_i = float(z_raw) if not pd.isna(z_raw) else 0.0
        # Determine desired position sign: +1 long, -1 short, 0 flat
        desired_sign = current_pos
        if allow_trade:
            if current_pos == 0:
                # Flat: open positions if z hits entry threshold
                if z_i <= -abs(z_in):
                    desired_sign = +1
                    entry_date = date
                elif z_i >= abs(z_in):
                    desired_sign = -1
                    entry_date = date
            else:
                # In a trade: check exit or stop
                if current_pos > 0 and z_i >= z_exit:
                    desired_sign = 0
                elif current_pos < 0 and z_i <= -z_exit:
                    desired_sign = 0
                # Stop if z gets too extreme
                if z_stop is not None and abs(z_i) >= z_stop:
                    desired_sign = 0
        else:
            # Regime is off: no new trades, close any open positions
            desired_sign = 0
        # Compute weight scaling
        desired_weight = desired_sign
        if vol_target and desired_sign != 0:
            # Target daily volatility = vol_target / sqrt(252)
            target_daily_vol = vol_target / np.sqrt(252)
            current_vol = rolling_vol.iloc[i]
            if current_vol > 0:
                desired_weight = desired_sign * (target_daily_vol / current_vol)
            else:
                desired_weight = desired_sign
        # Apply transaction costs on weight changes
        delta_weight = desired_weight - current_weight
        cost = abs(delta_weight) * cost_bps / 1e4
        costs[i] = cost
        current_weight = desired_weight
        current_pos = desired_sign
        pos[i] = current_sign = current_pos
        weight[i] = current_weight
        # Compute daily PnL: position weight * spread return minus cost
        pnl[i] = current_weight * spread_return.iloc[i] - cost
        # Record trades
        if i > 0:
            if pos[i-1] == 0 and current_pos != 0:
                trades.append((date, None, 'Long' if current_pos > 0 else 'Short'))
            elif pos[i-1] != 0 and current_pos == 0:
                # Close trade; fill exit_date for last unmatched trade
                # Find last open trade and close it
                for j in range(len(trades)-1, -1, -1):
                    if trades[j][1] is None:
                        trades[j] = (trades[j][0], date, trades[j][2])
                        break
    # Create results DataFrame
    results = pd.DataFrame({
        'SpreadReturn': spread_return,
        'Position': pos,
        'Weight': weight,
        'DailyPnL': pnl,
        'Cost': costs,
    }, index=prices.index)
    # Compute equity curve
    results['Equity'] = (1 + results['DailyPnL']).cumprod()
    # Summary metrics
    total_return = results['Equity'].iloc[-1] - 1
    daily_ret = results['DailyPnL']
    if daily_ret.std(ddof=0) > 0:
        sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std(ddof=0)
    else:
        sharpe = np.nan
    # Max drawdown
    equity_curve = results['Equity']
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1
    max_dd = dd.min()
    n_trades = sum(1 for t in trades if t[1] is not None)
    percent_active = np.mean(results['Position'] != 0)
    summary = {
        'Cumulative Return': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        '# Trades': n_trades,
        '% Days Active': percent_active,
    }
    return {
        'results': results,
        'summary': summary,
        'trades': trades,
    }


def mini_assessment(summary_a: Dict[str, float], summary_b: Dict[str, float]) -> str:
    """Generate a simple text assessment comparing two strategies.

    Parameters
    ----------
    summary_a, summary_b : dict
        Summary metric dictionaries for strategy A and B.

    Returns
    -------
    str
        Narrative highlighting differences between the strategies.
    """
    lines = []
    # Compare drawdown
    if summary_b['Max Drawdown'] > summary_a['Max Drawdown']:
        lines.append(
            f"The regime‑aware strategy increased the maximum drawdown from {summary_a['Max Drawdown']:.1%} to {summary_b['Max Drawdown']:.1%}."
        )
    elif summary_b['Max Drawdown'] < summary_a['Max Drawdown']:
        lines.append(
            f"The regime‑aware strategy reduced the maximum drawdown from {summary_a['Max Drawdown']:.1%} to {summary_b['Max Drawdown']:.1%}."
        )
    else:
        lines.append(
            f"Both strategies had the same maximum drawdown of {summary_a['Max Drawdown']:.1%}."
        )
    # Compare Sharpe
    if np.isnan(summary_b['Sharpe Ratio']) or np.isnan(summary_a['Sharpe Ratio']):
        pass
    elif summary_b['Sharpe Ratio'] > summary_a['Sharpe Ratio']:
        lines.append(
            f"Risk‑adjusted returns improved with the regime filter (Sharpe {summary_a['Sharpe Ratio']:.2f} → {summary_b['Sharpe Ratio']:.2f})."
        )
    elif summary_b['Sharpe Ratio'] < summary_a['Sharpe Ratio']:
        lines.append(
            f"Risk‑adjusted returns deteriorated with the regime filter (Sharpe {summary_a['Sharpe Ratio']:.2f} → {summary_b['Sharpe Ratio']:.2f})."
        )
    else:
        lines.append(
            f"Both strategies delivered the same Sharpe ratio ({summary_a['Sharpe Ratio']:.2f})."
        )
    # Compare active time
    if summary_b['% Days Active'] < summary_a['% Days Active']:
        lines.append(
            f"The regime filter reduced trading activity from {summary_a['% Days Active']:.1%} to {summary_b['% Days Active']:.1%}."
        )
    else:
        lines.append(
            f"The regime filter did not reduce trading activity (both {summary_a['% Days Active']:.1%})."
        )
    return ' '.join(lines)


def main() -> None:
    st.set_page_config(
        page_title="Regime‑Aware Pairs Trader",
        layout="wide",
    )
    st.title("Regime‑Aware Pairs Trading App")
    st.markdown(
        """
        Adjust the parameters below to explore how a simple mean‑reversion
        pair trade performs with and without a market‑stress switch.  The
        default example uses energy ETFs (XLE & XOP) over the 2017–2021
        period.  The stress switch is driven either by a Markov‑switching
        model on a market index (SPY) or by a VIX threshold.
        """
    )
    # Sidebar inputs
    st.sidebar.header("Data")
    default_start = _dt.date(2017, 1, 1)
    default_end = _dt.date(2021, 12, 31)
    y_ticker = st.sidebar.text_input("Dependent ticker (Y)", value="XLE")
    x_ticker = st.sidebar.text_input("Independent ticker (X)", value="XOP")
    mkt_ticker = st.sidebar.text_input("Market proxy ticker", value="SPY")
    start_date = st.sidebar.date_input("Start date", value=default_start)
    end_date = st.sidebar.date_input("End date", value=default_end)
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    # Strategy parameters
    st.sidebar.header("Spread & Z‑Score")
    lookback = st.sidebar.number_input("Hedge ratio lookback (days, 0 = static)", min_value=0, value=0, step=1)
    z_window = st.sidebar.number_input("Z‑score window (days)", min_value=1, value=30, step=1)
    z_in = st.sidebar.number_input("Entry threshold (|z|)", min_value=0.1, value=2.0, step=0.1)
    z_stop = st.sidebar.number_input("Stop threshold (|z|, 0 = none)", min_value=0.0, value=4.0, step=0.5)
    if z_stop <= 0.0:
        z_stop = None
    # Regime filter settings
    st.sidebar.header("Regime Filter")
    filter_type = st.sidebar.selectbox("Filter type", options=["Markov", "VIX"])
    p_on = st.sidebar.slider("Calm probability to turn ON", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    p_off = st.sidebar.slider("Calm probability to turn OFF", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    vix_threshold = st.sidebar.number_input("VIX threshold", min_value=0.0, value=20.0, step=1.0)
    # Volatility sizing and cost
    st.sidebar.header("Position Sizing & Costs")
    vol_target_on = st.sidebar.checkbox("Target volatility?", value=False)
    target_vol = None
    if vol_target_on:
        target_vol = st.sidebar.number_input("Annualized vol target", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    cost_bps = st.sidebar.number_input("Cost per transaction (bps)", min_value=0.0, value=10.0, step=1.0)
    # Load data
    try:
        prices = get_prices((y_ticker, x_ticker, mkt_ticker), start_date, end_date)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    # Compute regime signal
    regime_signal, used_filter = compute_regime_signal(
        prices[[mkt_ticker]].copy(), mkt_ticker,
        filter_type=filter_type,
        p_on=p_on, p_off=p_off,
        vix_threshold=vix_threshold,
    )
    # Add VIX if used
    if used_filter == 'VIX':
        # Ensure VIX data is in prices
        try:
            vix_df = load_price_data('^VIX', start_date, end_date)
            prices = prices.join(vix_df, how='left')
        except Exception:
            pass
    # Backtest always‑on
    res_a = backtest_pair(
        prices[[y_ticker, x_ticker]],
        pair=(y_ticker, x_ticker),
        hedge_lookback=lookback if lookback > 0 else None,
        z_window=z_window,
        z_in=z_in,
        z_exit=0.0,
        z_stop=z_stop,
        regime=None,
        vol_target=target_vol,
        cost_bps=cost_bps,
    )
    # Backtest regime aware
    res_b = backtest_pair(
        prices[[y_ticker, x_ticker]],
        pair=(y_ticker, x_ticker),
        hedge_lookback=lookback if lookback > 0 else None,
        z_window=z_window,
        z_in=z_in,
        z_exit=0.0,
        z_stop=z_stop,
        regime=regime_signal,
        vol_target=target_vol,
        cost_bps=cost_bps,
    )
    # Build summary table
    summary_df = pd.DataFrame({
        'Metric': ['Cumulative Return', 'Sharpe Ratio', 'Max Drawdown', '# Trades', '% Days Active'],
        'Always‑On': [
            res_a['summary']['Cumulative Return'],
            res_a['summary']['Sharpe Ratio'],
            res_a['summary']['Max Drawdown'],
            res_a['summary']['# Trades'],
            res_a['summary']['% Days Active'],
        ],
        'Regime‑Aware': [
            res_b['summary']['Cumulative Return'],
            res_b['summary']['Sharpe Ratio'],
            res_b['summary']['Max Drawdown'],
            res_b['summary']['# Trades'],
            res_b['summary']['% Days Active'],
        ],
    })
    # Display outputs
    st.subheader("Performance Summary")
    # Format numbers for display
    def fmt(val):
        if isinstance(val, float):
            return f"{val:.2%}" if abs(val) < 10 else f"{val:.2f}"
        else:
            return val
    fmt_df = summary_df.copy()
    for col in ['Always‑On', 'Regime‑Aware']:
        fmt_df[col] = fmt_df[col].apply(fmt)
    st.table(fmt_df)
    # Mini assessment
    st.subheader("Mini Assessment")
    assessment_text = mini_assessment(res_a['summary'], res_b['summary'])
    st.write(assessment_text)
    # Charts
    st.subheader("Equity Curves")
    try:
        import altair as alt  # type: ignore
        eq_df = pd.DataFrame({
            'Date': prices.index,
            'Always‑On': res_a['results']['Equity'],
            'Regime‑Aware': res_b['results']['Equity'],
        }).melt('Date', var_name='Strategy', value_name='Equity')
        equity_chart = alt.Chart(eq_df).mark_line().encode(
            x='Date:T',
            y=alt.Y('Equity:Q', title='Equity (growth of $1)'),
            color='Strategy:N'
        ).properties(width=800, height=400)
        st.altair_chart(equity_chart, use_container_width=True)
    except Exception:
        # Fallback to Streamlit's native line chart
        eq_df = pd.DataFrame({
            'Always‑On': res_a['results']['Equity'],
            'Regime‑Aware': res_b['results']['Equity'],
        }, index=prices.index)
        st.line_chart(eq_df)
    # Probability or VIX chart for regime filter
    st.subheader("Regime Signal")
    if used_filter == 'Markov':
        p_series = regime_signal.astype(float)
    else:
        p_series = prices['^VIX'] if '^VIX' in prices.columns else regime_signal.astype(float)
    reg_df = pd.DataFrame({
        'Signal': regime_signal.astype(int),
        'Value': p_series.reindex(prices.index).astype(float),
    }, index=prices.index)
    try:
        import altair as alt  # type: ignore
        reg_df_reset = reg_df.copy()
        reg_df_reset['Date'] = reg_df_reset.index
        base = alt.Chart(reg_df_reset).encode(x='Date:T')
        if used_filter == 'Markov':
            line = base.mark_line(color='steelblue').encode(y=alt.Y('Value:Q', title='Calm probability'))
        else:
            line = base.mark_line(color='steelblue').encode(y=alt.Y('Value:Q', title='VIX level'))
        bar = base.mark_bar(color='orange', opacity=0.3).encode(y=alt.Y('Signal:Q', title='Regime ON (1)'))
        st.altair_chart((line + bar).properties(width=800, height=300), use_container_width=True)
    except Exception:
        # Fallback to line charts for value and signal separately
        st.line_chart(reg_df[['Value', 'Signal']])
    # Show trade blotter
    st.subheader("Trade Blotter (Regime‑Aware)")
    if res_b['trades']:
        blotter = pd.DataFrame(res_b['trades'], columns=['Entry Date', 'Exit Date', 'Direction'])
        st.dataframe(blotter)
    else:
        st.write("No trades executed.")


if __name__ == "__main__":
    main()
