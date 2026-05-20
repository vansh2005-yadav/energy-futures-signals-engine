"""
data.py
─────────────────────────────────────────────
Downloads and cleans real energy market data
from Yahoo Finance using yfinance.
─────────────────────────────────────────────
"""

import pandas as pd
import yfinance as yf
import config


def download_prices() -> pd.DataFrame:
    """
    Download closing prices for all tickers in config.
    Returns a clean DataFrame — no NaNs, all tickers present.
    """
    print("\n" + "="*60)
    print("  ENERGY FUTURES — TERM STRUCTURE & MEAN REVERSION  V1")
    print("="*60)
    print(f"\n[1/5] Downloading data  "
          f"({config.START_DATE} → {config.END_DATE})...\n")

    data = {}

    for name, ticker in config.TICKERS.items():
        try:
            df = yf.download(
                ticker,
                start    = config.START_DATE,
                end      = config.END_DATE,
                progress = False,
                auto_adjust = True,
            )
            if not df.empty:
                series = df["Close"].squeeze()
                data[name] = series
                print(f"    ✓  {name:<12} ({ticker})  —  {len(df)} rows")
            else:
                print(f"    ✗  {name:<12} ({ticker})  —  no data returned")

        except Exception as e:
            print(f"    ✗  {name:<12} ({ticker})  —  error: {e}")

    if not data:
        raise RuntimeError("No data downloaded. Check your internet connection.")

    prices = pd.DataFrame(data).dropna()

    print(f"\n    ✓  Clean dataset: {len(prices)} trading days  "
          f"| {prices.index[0].date()} → {prices.index[-1].date()}")

    return prices


def compute_spread(prices: pd.DataFrame) -> pd.Series:
    """
    Compute the M1 - M3 calendar spread.
    Uses tickers defined in config (SPREAD_LEG1, SPREAD_LEG2).
    """
    leg1 = config.SPREAD_LEG1
    leg2 = config.SPREAD_LEG2

    if leg1 not in prices.columns or leg2 not in prices.columns:
        raise ValueError(
            f"Spread legs '{leg1}' or '{leg2}' not found in prices DataFrame."
        )

    spread = prices[leg1] - prices[leg2]
    spread.name = f"Spread_{leg1}_minus_{leg2}"
    return spread