"""
term_structure.py
─────────────────────────────────────────────
Builds the futures term structure (forward curve)
from the 3-tenor ETF proxies and detects the
daily Contango / Backwardation regime.

Tenors used:
  M1  =  USO  (front-month)
  M3  =  DBO  (mid-term, optimum-yield roll)
  M6  =  BNO  (back-month, Brent proxy)
─────────────────────────────────────────────
"""

import pandas as pd


def build_term_structure(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the 3 tenor proxies to 100 at start date.
    Add a Contango flag: 1 = contango, 0 = backwardation.

    Contango     : M1 < M6  (futures curve slopes upward — normal)
    Backwardation: M1 > M6  (spot above futures — supply squeeze)
    """
    print("\n[2/5] Building term structure (forward curve)...")

    required = ["USO", "DBO", "BNO"]
    for col in required:
        if col not in prices.columns:
            raise ValueError(f"Column '{col}' missing from prices DataFrame.")

    # Normalise each tenor to 100 at first observation
    norm = prices[required] / prices[required].iloc[0] * 100

    term_structure = pd.DataFrame({
        "M1_front" : norm["USO"],
        "M3_mid"   : norm["DBO"],
        "M6_back"  : norm["BNO"],
    })

    # Contango flag
    term_structure["Contango"] = (
        term_structure["M1_front"] < term_structure["M6_back"]
    ).astype(int)

    # Summary
    pct_contango = term_structure["Contango"].mean() * 100
    print(f"    Contango     : {pct_contango:.1f}% of trading days")
    print(f"    Backwardation: {100 - pct_contango:.1f}% of trading days")

    return term_structure