"""
stats.py
─────────────────────────────────────────────
Statistical tests for mean reversion:

  1. ADF Test (Augmented Dickey-Fuller)
     — tests if the spread is stationary
     — p-value < 0.05  →  stationary = mean-reverting

  2. Hurst Exponent  (R/S analysis)
     — H < 0.5  →  mean-reverting
     — H = 0.5  →  random walk
     — H > 0.5  →  trending
─────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def run_adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test.
    Returns a dict with test stat, p-value, critical values,
    and a boolean 'stationary' flag.
    """
    result = adfuller(series.dropna(), autolag="AIC")

    return {
        "adf_stat"   : result[0],
        "p_value"    : result[1],
        "critical_1" : result[4]["1%"],
        "critical_5" : result[4]["5%"],
        "stationary" : result[1] < 0.05,
    }


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """
    Estimate the Hurst Exponent via R/S (rescaled range) analysis.

    Returns H in [0, 1]:
      H < 0.45  →  strong mean reversion
      0.45–0.55 →  random walk / no clear regime
      H > 0.55  →  trending
    """
    x    = series.dropna().values
    lags = range(2, min(max_lag, len(x) // 2))
    tau  = []

    for lag in lags:
        segments  = len(x) // lag
        if segments < 2:
            continue
        rs_values = []
        for i in range(segments):
            seg  = x[i * lag : (i + 1) * lag]
            mean = np.mean(seg)
            dev  = np.cumsum(seg - mean)
            R    = np.max(dev) - np.min(dev)
            S    = np.std(seg, ddof=1)
            if S > 0:
                rs_values.append(R / S)
        if rs_values:
            tau.append((lag, np.mean(rs_values)))

    if len(tau) < 2:
        return 0.5   # not enough data — assume random walk

    log_lags = np.log([t[0] for t in tau])
    log_rs   = np.log([t[1] for t in tau])
    H, _     = np.polyfit(log_lags, log_rs, 1)
    return H


def print_test_results(adf: dict, H: float) -> None:
    """Pretty-print the test results to console."""
    print("\n[3/5] Mean reversion tests on calendar spread...")

    print(f"\n    ── ADF Test ──────────────────────────────")
    print(f"    ADF Statistic  : {adf['adf_stat']:.4f}")
    print(f"    p-value        : {adf['p_value']:.4f}")
    print(f"    Critical (1%)  : {adf['critical_1']:.4f}")
    print(f"    Critical (5%)  : {adf['critical_5']:.4f}")
    print(f"    Stationary     : {'YES ✓' if adf['stationary'] else 'NO ✗'}")

    print(f"\n    ── Hurst Exponent ────────────────────────")
    print(f"    H = {H:.4f}")

    if H < 0.45:
        label = "MEAN-REVERTING  ✓  strong signal candidate"
    elif H < 0.55:
        label = "RANDOM WALK     ~  weak / uncertain signal"
    else:
        label = "TRENDING        ✗  avoid mean-reversion here"

    print(f"    Regime: {label}")