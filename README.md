# Regime Aware Pairs Trader

This app explores a simple mean reversion pairs strategy with and without a market stress switch. Select two tickers, choose a date range, set spread and entry rules, then configure a regime filter. Prices are pulled from Yahoo Finance through `yfinance`. The app computes a hedge ratio and spread, normalizes it into a z score, and backtests two variants:

* **Strategy A, Always On**: trade whenever the z score crosses the entry threshold, exit at zero. An optional stop closes trades when the z score becomes too extreme.
* **Strategy B, Regime Aware**: the same entry and exit rules, but orders only when the market is calm. Calm is detected either by a two state Markov switching model fitted to daily returns of a market index (default SPY), or by a simple VIX threshold. Hysteresis thresholds control when the filter turns on and off. Optional position scaling targets a specific annualized volatility.

Both strategies include transaction costs in basis points per change in position. After each run the app reports cumulative return, annualized Sharpe ratio, maximum drawdown, number of trades, and percentage of days invested. It also shows equity curves, a regime signal chart, and a trade blotter. A short assessment summarizes whether the regime filter helped drawdowns or risk adjusted returns.

## Quick start

```bash
# 1. Python 3.11 is recommended for prebuilt wheels
python -m venv .venv
source .venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

**requirements.txt**

```
streamlit>=1.36
pandas>=2.2
numpy>=1.26
yfinance>=0.2.40
altair>=5.0
scipy>=1.11
statsmodels>=0.14.2
```

**Optional on Streamlit Cloud**

Add `runtime.txt` with

```
3.11
```

## Parameters

**Data**

* Dependent ticker (Y), default XLE
* Independent ticker (X), default XOP
* Market proxy, default SPY
* Start date and end date

**Spread and z score**

* Hedge ratio lookback in days, zero means static regression
* Z score rolling window
* Entry threshold, absolute z
* Stop threshold, absolute z, zero means disabled

**Regime filter**

* Type, Markov or VIX
* Calm probability to turn ON (Markov)
* Calm probability to turn OFF (Markov)
* VIX threshold, calm when VIX is below this level

**Position sizing and costs**

* Target volatility, annualized, optional
* Cost per transaction in basis points

## Markov fallback

If `statsmodels` is unavailable, or the Markov fit fails or the return series is too short, the app automatically switches to the VIX rule. A notice appears in the UI when this happens.

## Notes and tips

* For ETFs like XLE and XOP, a static hedge ratio often works well. Try lookback zero, or sixty days.
* Start with z window 30 to 60, entry 2.0, stop 4.0 to 5.0.
* For Markov, try ON 0.70 to 0.80, OFF 0.40 to 0.55.
* For VIX, try thresholds 22 to 28.
* Use the optional search panel to scan thresholds and surface combinations with higher Sharpe and lower drawdown for your chosen sample.

## Troubleshooting

* **Import error for statsmodels or scipy**: ensure the versions above, and Python 3.11. On Streamlit Cloud, add `runtime.txt` and redeploy so wheels install correctly.
* **No data returned**: check ticker symbols and date range.
* **CORS or network errors**: verify the machine has internet access to Yahoo endpoints.
* **Odd Sharpe formatting**: the table shows Sharpe as a plain number, returns and drawdowns as percentages by design.

## License

For educational use. No investment advice is provided.
