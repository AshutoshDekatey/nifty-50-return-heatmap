# NIFTY 50 Return Heatmap

A Streamlit dashboard that compares the arithmetic average daily returns of the current NIFTY 50 constituents using equal-sized, colour-coded tiles.

## What it shows

- Current NIFTY 50 constituents from Nifty Indices
- Adjusted daily prices downloaded through `yfinance`
- Trailing 1-month, 3-month, 6-month, or 1-year periods
- Average daily return shown on each tile
- Cumulative return, industry, and observation count on hover
- Symmetric red-to-green colouring centred on zero
- Batched downloads with automatic retries for tickers omitted by Yahoo Finance

## Run locally

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Return calculations

For adjusted closing prices \(P_t\), the daily simple return is:

```text
r_t = (P_t / P_(t-1)) - 1
```

The tile value is the arithmetic mean of those daily returns:

```text
average_daily_return = mean(r_t)
```

Cumulative return is retained as a secondary metric:

```text
cumulative_return = (last_adjusted_close / first_adjusted_close) - 1
```

## Refresh behaviour

Constituents and prices are cached for 24 hours. Ordinary Streamlit reruns—such as opening the metric table—reuse cached data. The **Refresh market data** button clears both caches and immediately downloads fresh data.

This hybrid approach avoids unnecessary API requests while ensuring the app does not remain stale indefinitely.

Yahoo Finance may occasionally return only part of a large multi-ticker request. The app downloads stocks in batches of ten, retries missing symbols in batches of five, and stops with a clear error instead of displaying a misleading one-stock heatmap when fewer than 40 stocks are available.

## Important limitation

The dashboard uses today's NIFTY 50 constituent list across the selected trailing period. It does not reconstruct historical index membership, so longer-period comparisons can contain survivorship bias.
