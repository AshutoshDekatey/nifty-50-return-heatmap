from __future__ import annotations

from datetime import datetime
from html import escape
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
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

    cards = []
    for _, row in metrics.iterrows():
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
        cards.append(
            f"""
            <div class="stock-card" style="background:{background};color:{foreground}" title="{tooltip}">
                <div class="symbol">{symbol}</div>
                <div class="return">{avg_return:+.3f}%</div>
                <div class="label">avg daily</div>
            </div>
            """
        )

    st.markdown(
        """
        <style>
        .heatmap-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(120px, 1fr));
            gap: 12px;
            margin-top: 0.5rem;
        }
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
        @media (max-width: 900px) {
            .heatmap-grid { grid-template-columns: repeat(3, minmax(105px, 1fr)); }
        }
        @media (max-width: 520px) {
            .heatmap-grid { grid-template-columns: repeat(2, minmax(100px, 1fr)); gap: 8px; }
            .stock-card { min-height: 96px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="heatmap-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
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
