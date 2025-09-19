import datetime as _dt
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# ---- Optional deps: the app runs without statsmodels via VIX fallback
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    import statsmodels.api as sm  # type: ignore
except Exception:
    sm = None


# ---------------------- Data ----------------------
def load_price_data(ticker: str, start: _dt.date, end: _dt.date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is required. pip install yfinance")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} between {start} and {end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns.values]
        adj_close = [c for c in df.columns if "Adj Close" in c]
        close = [c for c in df.columns if c.startswith("Close") and "Adj Close" not in c]
        col = adj_close[0] if adj_close else close[0]
    else:
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[[col]].rename(columns={col: ticker})


@st.cache_data
def get_prices(tickers: Tuple[str, str, str], start: _dt.date, end: _dt.date) -> pd.DataFrame:
    y_t, x_t, m_t = tickers
    dfs = [load_price_data(t, start, end) for t in {y_t, x_t, m_t}]
    return pd.concat(dfs, axis=1).dropna()


# ---------------------- Math ----------------------
def compute_hedge_ratio(
    y: pd.Series, x: pd.Series, lookback: Optional[int] = None, include_intercept: bool = True
) -> Tuple[pd.Series, pd.Series]:
    if sm is None:
        # fall back to a simple ratio if statsmodels is unavailable
        slope = (y / x).median()
        return pd.Series(float(slope), index=y.index), pd.Series(0.0, index=y.index)

    if lookback and lookback > 0:
        slopes, intercepts = [], []
        for i in range(len(y)):
            s = max(0, i - lookback + 1)
            y_w, x_w = y.iloc[s : i + 1], x.iloc[s : i + 1]
            X = sm.add_constant(x_w) if include_intercept else x_w.values[:, None]
            try:
                res = sm.OLS(y_w.values, X).fit()
                p = res.params
                if include_intercept:
                    intercepts.append(p[0]); slopes.append(p[1])
                else:
                    intercepts.append(0.0); slopes.append(p[0])
            except Exception:
                slopes.append(slopes[-1] if slopes else 1.0)
                intercepts.append(intercepts[-1] if intercepts else 0.0)
        return pd.Series(slopes, index=y.index), pd.Series(intercepts, index=y.index)

    X = sm.add_constant(x) if include_intercept else x.values[:, None]
    res = sm.OLS(y.values, X).fit()
    params = res.params
    if include_intercept:
        return pd.Series(float(params[1]), index=y.index), pd.Series(float(params[0]), index=y.index)
    return pd.Series(float(params[0]), index=y.index), pd.Series(0.0, index=y.index)


def compute_spread(y: pd.Series, x: pd.Series, hr: pd.Series, ic: pd.Series) -> pd.Series:
    return y - hr * x - ic


def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std(ddof=0)
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_markov_regime(series: pd.Series) -> pd.DataFrame:
    if sm is None:
        raise RuntimeError("statsmodels not available")
    series = series.dropna()
    if len(series) < 50:
        raise ValueError("Return series too short for Markov fit")
    model = sm.tsa.MarkovRegression(series, k_regimes=2, trend="c", switching_variance=True)
    res = model.fit(disp=False)
    probs = res.smoothed_marginal_probabilities
    probs.index = series.index
    return probs


def classify_calm_state(probs: pd.DataFrame, series: pd.Series) -> pd.Series:
    # pick the state with lower variance
    var = {c: np.sum(probs[c] * series**2) / np.sum(probs[c]) for c in probs.columns}
    calm_state = min(var, key=var.get)
    return probs[calm_state]


def compute_regime_signal(
    data: pd.DataFrame,
    mkt_ticker: str,
    filter_type: str = "Markov",
    p_on: float = 0.6,
    p_off: float = 0.4,
    vix_threshold: float = 20.0,
) -> Tuple[pd.Series, str]:
    index = data.index
    if filter_type == "Markov" and sm is not None:
        try:
            r = data[mkt_ticker].pct_change().dropna()
            p = classify_calm_state(fit_markov_regime(r), r).reindex(index).ffill().fillna(0.0)
            sig, on = [], False
            for prob in p:
                on = prob >= (p_off if on else p_on)
                sig.append(on)
            return pd.Series(sig, index=index), "Markov"
        except Exception:
            filter_type = "VIX"

    if "^VIX" not in data.columns:
        vix = load_price_data("^VIX", index[0].date(), index[-1].date())
        data = data.join(vix, how="left")

    vix_series = data["^VIX"].ffill().reindex(index)
    return (vix_series < vix_threshold).astype(bool), "VIX"


# ---------------------- Backtest ----------------------
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
) -> Dict[str, Any]:

    y, x = prices[pair[0]], prices[pair[1]]
    hr, ic = compute_hedge_ratio(y, x, lookback=hedge_lookback)
    spread = compute_spread(y, x, hr, ic)
    z = compute_zscore(spread, window=z_window)
    spread_ret = y.pct_change().fillna(0.0) - hr * x.pct_change().fillna(0.0)

    n = len(prices)
    pos = np.zeros(n)
    w = np.zeros(n)
    pnl = np.zeros(n)
    costs = np.zeros(n)
    trades = []

    if vol_target:
        roll_vol = spread_ret.rolling(20, min_periods=1).std(ddof=0)  # daily vol estimate

    cur_pos, cur_w = 0.0, 0.0
    for i, date in enumerate(prices.index):
        allow = True if regime is None else bool(regime.iloc[i])
        zi = float(z.iloc[i]) if not pd.isna(z.iloc[i]) else 0.0

        want = cur_pos
        if allow:
            if cur_pos == 0:
                if zi <= -abs(z_in):
                    want = +1
                elif zi >= abs(z_in):
                    want = -1
            else:
                if (cur_pos > 0 and zi >= z_exit) or (cur_pos < 0 and zi <= -z_exit):
                    want = 0
                if z_stop is not None and abs(zi) >= z_stop:
                    want = 0
        else:
            want = 0

        want_w = want
        if vol_target and want != 0:
            target_d = vol_target / np.sqrt(252.0)
            cur_vol = float(roll_vol.iloc[i])
            want_w = want * (target_d / cur_vol if cur_vol > 0 else 1.0)

        delta = want_w - cur_w
        cost = abs(delta) * (cost_bps / 1e4)
        costs[i] = cost
        cur_w, cur_pos = want_w, want
        w[i], pos[i] = cur_w, cur_pos
        pnl[i] = cur_w * float(spread_ret.iloc[i]) - cost

        if i > 0:
            if pos[i - 1] == 0 and cur_pos != 0:
                trades.append((date, None, "Long" if cur_pos > 0 else "Short"))
            elif pos[i - 1] != 0 and cur_pos == 0:
                # close last open trade
                for j in range(len(trades) - 1, -1, -1):
                    if trades[j][1] is None:
                        trades[j] = (trades[j][0], date, trades[j][2]); break

    results = pd.DataFrame(
        {"SpreadReturn": spread_ret, "Position": pos, "Weight": w, "DailyPnL": pnl, "Cost": costs},
        index=prices.index,
    )
    results["Equity"] = (1 + results["DailyPnL"]).cumprod()

    ret = results["DailyPnL"]
    sharpe = np.sqrt(252.0) * ret.mean() / ret.std(ddof=0) if ret.std(ddof=0) > 0 else np.nan
    dd = results["Equity"] / results["Equity"].cummax() - 1.0

    summary = {
        "Cumulative Return": float(results["Equity"].iloc[-1] - 1.0),
        "Sharpe Ratio": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "Max Drawdown": float(dd.min()),
        "# Trades": int(sum(1 for t in trades if t[1] is not None)),
        "% Days Active": float(np.mean(results["Position"] != 0)),
    }
    return {"results": results, "summary": summary, "trades": trades}


def mini_assessment(a: Dict[str, float], b: Dict[str, float]) -> str:
    out = []
    if b["Max Drawdown"] < a["Max Drawdown"]:
        out.append(f"Regime-aware reduced max drawdown {a['Max Drawdown']:.1%} → {b['Max Drawdown']:.1%}.")
    else:
        out.append(f"Regime-aware increased max drawdown {a['Max Drawdown']:.1%} → {b['Max Drawdown']:.1%}.")
    if not np.isnan(a["Sharpe Ratio"]) and not np.isnan(b["Sharpe Ratio"]):
        if b["Sharpe Ratio"] > a["Sharpe Ratio"]:
            out.append(f"Sharpe improved {a['Sharpe Ratio']:.2f} → {b['Sharpe Ratio']:.2f}.")
        else:
            out.append(f"Sharpe deteriorated {a['Sharpe Ratio']:.2f} → {b['Sharpe Ratio']:.2f}.")
    if b["% Days Active"] < a["% Days Active"]:
        out.append(f"Trading time fell {a['% Days Active']:.1%} → {b['% Days Active']:.1%}.")
    return " ".join(out)


# ---------------------- UI ----------------------
def main() -> None:
    st.set_page_config(page_title="Regime-Aware Pairs Trader", layout="wide")
    st.title("Regime-Aware Pairs Trading App")
    st.markdown(
        "Adjust the parameters below to explore a pair trade with and without a regime switch."
    )

    # Sidebar
    st.sidebar.header("Data")
    y_ticker = st.sidebar.text_input("Dependent ticker (Y)", "XLE")
    x_ticker = st.sidebar.text_input("Independent ticker (X)", "XOP")
    mkt_ticker = st.sidebar.text_input("Market proxy ticker", "SPY")
    start_date = st.sidebar.date_input("Start date", _dt.date(2017, 1, 1))
    end_date = st.sidebar.date_input("End date", _dt.date(2021, 12, 31))
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    st.sidebar.header("Spread & Z-Score")
    lookback = st.sidebar.number_input("Hedge ratio lookback (days, 0 = static)", 0, value=0, step=1)
    z_window = st.sidebar.number_input("Z-score window (days)", 1, value=30, step=1)
    z_in = st.sidebar.number_input("Entry threshold (|z|)", 0.1, value=2.0, step=0.1)
    z_stop = st.sidebar.number_input("Stop threshold (|z|, 0 = none)", 0.0, value=4.0, step=0.5)
    z_stop = None if z_stop <= 0 else z_stop

    st.sidebar.header("Regime Filter")
    filter_type = st.sidebar.selectbox("Filter type", ["Markov", "VIX"])
    p_on = st.sidebar.slider("Calm probability to turn ON", 0.0, 1.0, 0.6, 0.05)
    p_off = st.sidebar.slider("Calm probability to turn OFF", 0.0, 1.0, 0.4, 0.05)
    vix_threshold = st.sidebar.number_input("VIX threshold", 0.0, value=20.0, step=1.0)

    st.sidebar.header("Position Sizing & Costs")
    target_vol = st.sidebar.number_input("Annualized vol target", 0.01, 1.0, 0.10, 0.01) if st.sidebar.checkbox("Target volatility?", False) else None
    cost_bps = st.sidebar.number_input("Cost per transaction (bps)", 0.0, value=10.0, step=1.0)

    # Optional simple parameter search
    st.sidebar.header("Search")
    run_search = st.sidebar.checkbox("Suggest regime thresholds", value=False)

    # Load
    try:
        prices = get_prices((y_ticker, x_ticker, mkt_ticker), start_date, end_date)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Regime
    regime_signal, used_filter = compute_regime_signal(
        prices[[mkt_ticker]].copy(),
        mkt_ticker,
        filter_type=filter_type,
        p_on=p_on,
        p_off=p_off,
        vix_threshold=vix_threshold,
    )

    # Backtests
    res_a = backtest_pair(
        prices[[y_ticker, x_ticker]],
        (y_ticker, x_ticker),
        hedge_lookback=None if lookback == 0 else lookback,
        z_window=z_window,
        z_in=z_in,
        z_stop=z_stop,
        regime=None,
        vol_target=target_vol,
        cost_bps=cost_bps,
    )

    res_b = backtest_pair(
        prices[[y_ticker, x_ticker]],
        (y_ticker, x_ticker),
        hedge_lookback=None if lookback == 0 else lookback,
        z_window=z_window,
        z_in=z_in,
        z_stop=z_stop,
        regime=regime_signal,
        vol_target=target_vol,
        cost_bps=cost_bps,
    )

    # Summary table with explicit strings so Streamlit doesn’t treat anything as Markdown
    st.subheader("Performance Summary")
    summary_df = pd.DataFrame(
        {
            "Metric": ["Cumulative Return", "Sharpe Ratio", "Max Drawdown", "# Trades", "% Days Active"],
            "Always-On": [
                f"{res_a['summary']['Cumulative Return']:.2%}",
                "NaN" if np.isnan(res_a['summary']['Sharpe Ratio']) else f"{res_a['summary']['Sharpe Ratio']:.2f}",
                f"{res_a['summary']['Max Drawdown']:.2%}",
                f"{res_a['summary']['# Trades']}",
                f"{res_a['summary']['% Days Active']:.2%}",
            ],
            "Regime-Aware": [
                f"{res_b['summary']['Cumulative Return']:.2%}",
                "NaN" if np.isnan(res_b['summary']['Sharpe Ratio']) else f"{res_b['summary']['Sharpe Ratio']:.2f}",
                f"{res_b['summary']['Max Drawdown']:.2%}",
                f"{res_b['summary']['# Trades']}",
                f"{res_b['summary']['% Days Active']:.2%}",
            ],
        }
    )
    st.table(summary_df)

    st.subheader("Mini Assessment")
    st.write(mini_assessment(res_a["summary"], res_b["summary"]))

    # Charts
    st.subheader("Equity Curves")
    try:
        import altair as alt  # type: ignore

        eq_df = pd.DataFrame(
            {
                "Date": prices.index,
                "Always-On": res_a["results"]["Equity"],
                "Regime-Aware": res_b["results"]["Equity"],
            }
        ).melt("Date", var_name="Strategy", value_name="Equity")
        chart = (
            alt.Chart(eq_df)
            .mark_line()
            .encode(x="Date:T", y=alt.Y("Equity:Q", title="Equity (growth of $1)"), color="Strategy:N")
            .properties(width=800, height=350)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.line_chart(
            pd.DataFrame(
                {"Always-On": res_a["results"]["Equity"], "Regime-Aware": res_b["results"]["Equity"]},
                index=prices.index,
            )
        )

    st.subheader("Regime Signal")
    if used_filter == "Markov":
        value_series = regime_signal.astype(float)
        ylabel = "Calm probability ON (1.0)"
    else:
        value_series = prices.get("^VIX", regime_signal.astype(float))
        ylabel = "VIX / Signal"
    reg_df = pd.DataFrame({"Signal": regime_signal.astype(int), "Value": value_series.reindex(prices.index).astype(float)}, index=prices.index)
    try:
        import altair as alt  # type: ignore

        r = reg_df.copy()
        r["Date"] = r.index
        base = alt.Chart(r).encode(x="Date:T")
        line = base.mark_line().encode(y=alt.Y("Value:Q", title="Calm probability / VIX"))
        bar = base.mark_bar(opacity=0.3).encode(y=alt.Y("Signal:Q", title="Regime ON (1)"))
        st.altair_chart((line + bar).properties(width=800, height=300), use_container_width=True)
    except Exception:
        st.line_chart(reg_df[["Value", "Signal"]])

    st.subheader("Trade Blotter (Regime-Aware)")
    if res_b["trades"]:
        blotter = pd.DataFrame(res_b["trades"], columns=["Entry Date", "Exit Date", "Direction"])
        st.dataframe(blotter)
    else:
        st.write("No trades executed.")

    # -------- simple search for good thresholds (optional) --------
    if run_search:
        st.markdown("**Searching thresholds…**")
        grid = []
        if filter_type == "Markov":
            for on in np.linspace(0.55, 0.85, 7):
                for off in np.linspace(0.35, 0.65, 7):
                    if off >= on:
                        continue  # enforce hysteresis
                    sig, _ = compute_regime_signal(prices[[mkt_ticker]], mkt_ticker, "Markov", on, off, vix_threshold)
                    r = backtest_pair(prices[[y_ticker, x_ticker]], (y_ticker, x_ticker),
                                      None if lookback == 0 else lookback, z_window, z_in, z_stop=z_stop,
                                      regime=sig, vol_target=target_vol, cost_bps=cost_bps)
                    grid.append((on, off, r["summary"]["Sharpe Ratio"], r["summary"]["Max Drawdown"]))
            df = pd.DataFrame(grid, columns=["p_on", "p_off", "Sharpe", "MaxDD"]).dropna()
            df = df.sort_values(["Sharpe", "MaxDD"], ascending=[False, True]).head(10)
            st.dataframe(df.reset_index(drop=True))
        else:
            grid = []
            for thr in range(15, 35, 1):
                sig, _ = compute_regime_signal(prices[[mkt_ticker]], mkt_ticker, "VIX", vix_threshold=thr)
                r = backtest_pair(prices[[y_ticker, x_ticker]], (y_ticker, x_ticker),
                                  None if lookback == 0 else lookback, z_window, z_in, z_stop=z_stop,
                                  regime=sig, vol_target=target_vol, cost_bps=cost_bps)
                grid.append((thr, r["summary"]["Sharpe Ratio"], r["summary"]["Max Drawdown"]))
            df = pd.DataFrame(grid, columns=["VIX_threshold", "Sharpe", "MaxDD"]).dropna()
            df = df.sort_values(["Sharpe", "MaxDD"], ascending=[False, True]).head(10)
            st.dataframe(df.reset_index(drop=True))


if __name__ == "__main__":
    main()
