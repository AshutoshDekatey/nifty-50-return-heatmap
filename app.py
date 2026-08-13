from __future__ import annotations

from datetime import datetime
from html import escape
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf


NIFTY_50_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
PERIODS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
}
IST = ZoneInfo("Asia/Kolkata")


st.set_page_config(
    page_title="NIFTY 50 Return Heatmap",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def get_nifty_constituents() -> pd.DataFrame:
    """Download the current NIFTY 50 constituent list from Nifty Indices."""
    response = requests.get(
        NIFTY_50_CSV,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    constituents = pd.read_csv(StringIO(response.text))

    required = {"Company Name", "Industry", "Symbol"}
    missing = required.difference(constituents.columns)
    if missing:
        raise ValueError(f"Constituent file is missing columns: {sorted(missing)}")

    constituents = constituents[["Company Name", "Industry", "Symbol"]].copy()
    constituents["Ticker"] = constituents["Symbol"] + ".NS"
    return constituents


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def download_market_data(
    tickers: tuple[str, ...], period: str
) -> tuple[pd.DataFrame, datetime, tuple[str, ...]]:
    """Download adjusted closes in small batches and retry missing tickers."""

    def extract_close(raw: pd.DataFrame, requested: tuple[str, ...]) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                close = raw["Close"]
            elif "Close" in raw.columns.get_level_values(1):
                close = raw.xs("Close", axis=1, level=1)
            else:
                return pd.DataFrame()
        elif "Close" in raw.columns:
            close = raw["Close"]
        else:
            return pd.DataFrame()

        if isinstance(close, pd.Series):
            close = close.to_frame(name=requested[0])

        close.columns = [str(column).upper() for column in close.columns]
        if len(requested) == 1 and close.shape[1] == 1:
            close.columns = [requested[0]]
        return close.dropna(axis=1, how="all")

    def fetch_batch(batch: tuple[str, ...]) -> pd.DataFrame:
        raw = yf.download(
            list(batch),
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
            timeout=20,
        )
        return extract_close(raw, batch)

    frames: list[pd.DataFrame] = []
    for start in range(0, len(tickers), 10):
        batch = tickers[start : start + 10]
        batch_close = fetch_batch(batch)
        if not batch_close.empty:
            frames.append(batch_close)

    close = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    close = close.loc[:, ~close.columns.duplicated(keep="last")]

    available = set(close.columns)
    missing = tuple(ticker for ticker in tickers if ticker not in available)

    # Yahoo sometimes returns a partial batch without raising an exception.
    retry_frames: list[pd.DataFrame] = []
    for start in range(0, len(missing), 5):
        retry_batch = missing[start : start + 5]
        retry_close = fetch_batch(retry_batch)
        if not retry_close.empty:
            retry_frames.append(retry_close)

    if retry_frames:
        close = pd.concat([close, *retry_frames], axis=1)
        close = close.loc[:, ~close.columns.duplicated(keep="last")]

    close = close.reindex(columns=list(tickers)).dropna(axis=1, how="all")
    missing = tuple(ticker for ticker in tickers if ticker not in close.columns)

    if close.shape[1] < 40:
        raise ValueError(
            f"Yahoo Finance returned usable history for only {close.shape[1]} of "
            f"{len(tickers)} stocks. Use Refresh market data to retry."
        )

    close = close.sort_index().dropna(how="all")
    return close, datetime.now(IST), missing


def calculate_metrics(close: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    """Calculate arithmetic average daily and cumulative returns."""
    daily_returns = close.pct_change(fill_method=None)
    average_daily = daily_returns.mean(skipna=True) * 100

    cumulative = {}
    observations = {}
    for ticker in close.columns:
        prices = close[ticker].dropna()
        observations[ticker] = max(len(prices) - 1, 0)
        cumulative[ticker] = (
            (prices.iloc[-1] / prices.iloc[0] - 1) * 100 if len(prices) >= 2 else np.nan
        )

    metrics = constituents.copy()
    metrics["Average Daily Return"] = metrics["Ticker"].map(average_daily)
    metrics["Cumulative Return"] = metrics["Ticker"].map(cumulative)
    metrics["Observations"] = metrics["Ticker"].map(observations).fillna(0).astype(int)
    return metrics.dropna(subset=["Average Daily Return"]).sort_values(
        "Average Daily Return", ascending=False
    )


def return_colour(value: float, scale_limit: float) -> str:
    """Map a return to a symmetric red-neutral-green colour scale."""
    strength = float(np.clip(abs(value) / scale_limit, 0, 1))
    neutral = np.array([245, 247, 250])
    target = np.array([20, 125, 70]) if value >= 0 else np.array([180, 35, 45])
    rgb = (neutral * (1 - strength) + target * strength).astype(int)
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def text_colour(value: float, scale_limit: float) -> str:
    return "#ffffff" if abs(value) / scale_limit >= 0.58 else "#17202a"


def render_heatmap(metrics: pd.DataFrame) -> None:
    absolute_returns = metrics["Average Daily Return"].abs()
    scale_limit = max(float(absolute_returns.quantile(0.95)), 0.01)

    st.markdown(
        """
        <style>
        .stock-card {
            min-height: 112px;
            border-radius: 12px;
            padding: 18px 12px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 7px rgba(0,0,0,0.08);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }
        .stock-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 14px rgba(0,0,0,0.14);
        }
        .symbol { font-size: 1.05rem; font-weight: 750; letter-spacing: 0.02em; }
        .return { font-size: 1.35rem; font-weight: 750; margin-top: 5px; }
        .label { font-size: 0.72rem; opacity: 0.82; margin-top: 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render each tile independently. A single large HTML grid is unreliable in
    # Streamlit Markdown and can collapse after its first child on deployment.
    records = metrics.to_dict("records")
    for start in range(0, len(records), 5):
        columns = st.columns(5, gap="small")
        for column, row in zip(columns, records[start : start + 5]):
            avg_return = float(row["Average Daily Return"])
            cumulative_return = float(row["Cumulative Return"])
            observations = int(row["Observations"])
            company = escape(str(row["Company Name"]), quote=True)
            industry = escape(str(row["Industry"]), quote=True)
            symbol = escape(str(row["Symbol"]), quote=True)
            background = return_colour(avg_return, scale_limit)
            foreground = text_colour(avg_return, scale_limit)
            tooltip = (
                f"{company} | {industry} | Cumulative return: "
                f"{cumulative_return:+.2f}% | {observations} daily observations"
            )
            card = (
                f'<div class="stock-card" '
                f'style="background:{background};color:{foreground}" '
                f'title="{tooltip}">'
                f'<div class="symbol">{symbol}</div>'
                f'<div class="return">{avg_return:+.3f}%</div>'
                f'<div class="label">avg daily</div>'
                f'</div>'
            )
            column.markdown(card, unsafe_allow_html=True)


def render_system_animation() -> None:
    """Show an animated, conceptual view of the dashboard data pipeline."""
    components.html(
        """
        <!doctype html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <style>
          * { box-sizing: border-box; }
          body {
            margin: 0;
            padding: 8px 4px 4px;
            background: transparent;
            color: #17202a;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
              "Segoe UI", sans-serif;
          }
          .pipeline {
            display: grid;
            grid-template-columns: minmax(145px,1fr) 58px minmax(145px,1fr) 58px minmax(145px,1fr);
            align-items: center;
            gap: 4px;
            max-width: 1050px;
            margin: 0 auto;
          }
          .node {
            min-height: 122px;
            border: 1px solid #dfe5eb;
            border-radius: 16px;
            background: linear-gradient(145deg,#ffffff,#f4f7f9);
            padding: 18px 14px;
            text-align: center;
            box-shadow: 0 5px 18px rgba(31,45,61,.07);
            animation: breathe 3.6s ease-in-out infinite;
          }
          .node:nth-of-type(3) { animation-delay: .6s; }
          .node:nth-of-type(5) { animation-delay: 1.2s; }
          .icon { font-size: 27px; line-height: 1; }
          .title { margin-top: 10px; font-size: 14px; font-weight: 750; }
          .detail { margin-top: 6px; color: #647381; font-size: 11px; line-height: 1.35; }
          .flow {
            height: 4px;
            position: relative;
            border-radius: 4px;
            background: #dce4e9;
            overflow: visible;
          }
          .flow::after {
            content: "";
            position: absolute;
            top: -4px;
            left: -2px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #16834c;
            box-shadow: 0 0 0 5px rgba(22,131,76,.13);
            animation: travel 2.1s ease-in-out infinite;
          }
          .flow.second::after { animation-delay: 1.05s; }
          .subflow {
            display: grid;
            grid-template-columns: repeat(3,1fr);
            gap: 10px;
            max-width: 720px;
            margin: 18px auto 0;
          }
          .pill {
            border-radius: 999px;
            padding: 8px 10px;
            background: #edf3f0;
            color: #37554a;
            font-size: 11px;
            text-align: center;
          }
          .legend {
            margin-top: 15px;
            color: #70808d;
            text-align: center;
            font-size: 11px;
          }
          @keyframes travel {
            0% { left: -2px; opacity: 0; transform: scale(.7); }
            15% { opacity: 1; }
            85% { opacity: 1; }
            100% { left: calc(100% - 10px); opacity: 0; transform: scale(1); }
          }
          @keyframes breathe {
            0%,100% { transform: translateY(0); border-color: #dfe5eb; }
            50% { transform: translateY(-3px); border-color: #a8cbbb; }
          }
          @media (max-width: 720px) {
            .pipeline { grid-template-columns: 1fr; gap: 8px; }
            .flow { width: 4px; height: 34px; margin: 0 auto; }
            .flow::after { top: -2px; left: -4px; animation: travel-down 2.1s ease-in-out infinite; }
            .subflow { grid-template-columns: 1fr; }
            @keyframes travel-down {
              0% { top: -2px; opacity: 0; }
              15% { opacity: 1; }
              85% { opacity: 1; }
              100% { top: calc(100% - 10px); opacity: 0; }
            }
          }
          @media (prefers-reduced-motion: reduce) {
            .node, .flow::after { animation: none; }
          }
        </style>
        </head>
        <body>
          <div class="pipeline" role="img" aria-label="Animated NIFTY return dashboard data pipeline">
            <div class="node">
              <div class="icon">🏛️</div>
              <div class="title">1. Identify the universe</div>
              <div class="detail">Load the current NIFTY 50 company symbols from Nifty Indices.</div>
            </div>
            <div class="flow"></div>
            <div class="node">
              <div class="icon">📥</div>
              <div class="title">2. Retrieve market history</div>
              <div class="detail">Fetch adjusted daily closing prices from Yahoo Finance in resilient batches.</div>
            </div>
            <div class="flow second"></div>
            <div class="node">
              <div class="icon">🧮</div>
              <div class="title">3. Calculate and display</div>
              <div class="detail">Calculate daily returns, average them, rank stocks, and map values to colour.</div>
            </div>
          </div>
          <div class="subflow">
            <div class="pill">⚡ 24-hour Streamlit cache</div>
            <div class="pill">↻ Manual refresh clears cache</div>
            <div class="pill">🟩 Positive · 🟥 Negative</div>
          </div>
          <div class="legend">The moving dots represent data passing through the pipeline; this is a conceptual system visualization.</div>
        </body>
        </html>
        """,
        height=250,
        scrolling=False,
    )


st.title("NIFTY 50 Return Heatmap")
st.caption(
    "Each equal-sized tile shows the arithmetic mean of the stock's daily returns. "
    "Green is positive, red is negative, and the colour scale is centred on zero."
)

control_left, control_middle, control_right = st.columns([2, 2, 5])
with control_left:
    selected_label = st.selectbox("Trailing period", list(PERIODS), index=3)
with control_middle:
    st.write("")
    st.write("")
    refresh = st.button("Refresh market data", use_container_width=True)

if refresh:
    get_nifty_constituents.clear()
    download_market_data.clear()
    st.rerun()

try:
    with st.spinner("Loading NIFTY 50 market data..."):
        constituents = get_nifty_constituents()
        tickers = tuple(constituents["Ticker"])
        close_prices, fetched_at, missing_tickers = download_market_data(
            tickers, PERIODS[selected_label]
        )
        metrics = calculate_metrics(close_prices, constituents)

    if metrics.empty:
        st.warning("No stocks had enough price history to calculate returns.")
        st.stop()

    latest_market_date = pd.Timestamp(close_prices.index.max()).strftime("%d %b %Y")
    with control_right:
        st.caption(
            f"Market data through **{latest_market_date}**  ·  "
            f"Fetched **{fetched_at.strftime('%d %b %Y, %I:%M %p IST')}**  ·  "
            f"{len(metrics)} stocks"
        )

    if missing_tickers:
        missing_symbols = ", ".join(
            ticker.removesuffix(".NS") for ticker in missing_tickers
        )
        st.warning(
            f"Yahoo Finance did not return data for {len(missing_tickers)} stock(s): "
            f"{missing_symbols}. The remaining stocks are shown."
        )

    render_heatmap(metrics)

    st.divider()
    st.subheader("How this dashboard works")
    st.caption("A live view of the data pipeline behind the heatmap.")
    render_system_animation()

    with st.expander("Metric details"):
        st.markdown(
            "**Average daily return** is the arithmetic mean of daily percentage changes. "
            "**Cumulative return** measures the compounded change from the first available "
            "adjusted closing price to the last. Hover over a tile to see cumulative return, "
            "sector, and observation count."
        )
        st.dataframe(
            metrics[
                [
                    "Symbol",
                    "Company Name",
                    "Industry",
                    "Average Daily Return",
                    "Cumulative Return",
                    "Observations",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Average Daily Return": st.column_config.NumberColumn(format="%.3f%%"),
                "Cumulative Return": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    st.caption("Sources: Nifty Indices constituent list and Yahoo Finance adjusted price data via yfinance.")

except requests.RequestException as exc:
    st.error("Could not download the NIFTY 50 constituent list. Please try Refresh market data.")
    st.exception(exc)
except Exception as exc:
    st.error("The dashboard could not be loaded. Yahoo Finance may be temporarily unavailable.")
    st.exception(exc)
