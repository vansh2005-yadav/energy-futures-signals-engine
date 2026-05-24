"""
signals.py
─────────────────────────────────────────────
Two responsibilities:

  1. fit_ou_process()
     Fit Ornstein-Uhlenbeck parameters to the
     calendar spread via closed-form MLE.

     OU model:  dX = κ(μ − X)dt + σ dW
       κ  =  speed of mean reversion
       μ  =  long-run equilibrium level
       σ  =  diffusion / volatility
       half-life = ln(2) / κ  (trading days)

  2. generate_signals()
     Build Long / Short / Flat signals using the
     OU z-score:
       z = (X − μ) / σ_eq
       σ_eq = σ / sqrt(2κ)   ← equilibrium std

     Entry : |z| > ENTRY_THRESHOLD  (default ±1σ)
     Exit  : |z| < EXIT_THRESHOLD   (default ±0.25σ)
─────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import config


def fit_ou_process(spread: pd.Series) -> dict:
    """
    Fit OU parameters via closed-form exact MLE
    (Euler-Maruyama discrete approximation with dt=1 day).

    Returns dict: kappa, mu, sigma, half_life
    """
    print("\n[4/5] Fitting Ornstein-Uhlenbeck process (MLE)...")

    x  = spread.dropna().values
    dt = 1.0
    n  = len(x) - 1

    # Sufficient statistics
    sx  = np.sum(x[:-1])
    sy  = np.sum(x[1:])
    sxx = np.sum(x[:-1] ** 2)
    sxy = np.sum(x[:-1] * x[1:])
    syy = np.sum(x[1:]  ** 2)

    # MLE for mu and kappa
    denom = n * (sxx - sxy) - (sx**2 - sx * sy)
    if abs(denom) < 1e-12:
        raise ValueError("Degenerate spread — cannot fit OU process.")

    mu    = (sy * sxx - sx * sxy) / denom
    alpha = (sxy - mu * sx - mu * sy + n * mu**2) / \
            (sxx - 2 * mu * sx + n * mu**2)

    # Guard: alpha must be in (0, 1) for mean reversion
    alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
    kappa = -np.log(alpha) / dt
    kappa = max(kappa, 1e-6)

    # MLE for sigma
    sigma_sq = (
        syy
        - 2 * alpha * sxy
        + alpha**2 * sxx
        - 2 * mu * (1 - alpha) * (sy - alpha * sx)
        + n * mu**2 * (1 - alpha)**2
    ) / n
    sigma     = np.sqrt(max(sigma_sq, 1e-10))
    half_life = np.log(2) / kappa

    print(f"    κ  (mean-reversion speed) : {kappa:.6f}")
    print(f"    μ  (long-run mean)        : {mu:.4f}")
    print(f"    σ  (volatility)           : {sigma:.4f}")
    print(f"    Half-life                 : {half_life:.1f} trading days"
          f"  ({half_life / 21:.1f} months)")

    return {
        "kappa"    : kappa,
        "mu"       : mu,
        "sigma"    : sigma,
        "half_life": half_life,
    }


def generate_signals(spread: pd.Series, ou: dict) -> pd.DataFrame:
    """
    Generate trading signals from OU z-score.

    Signal values:
      +1  →  Long  (spread too low, expect reversion up)
      -1  →  Short (spread too high, expect reversion down)
       0  →  Flat  (no position)
    """
    print("\n[5/5] Generating trading signals...")

    kappa = ou["kappa"]
    mu    = ou["mu"]
    sigma = ou["sigma"]

    # Equilibrium std of the OU process
    sigma_eq = sigma / np.sqrt(2 * kappa)

    z_score  = (spread - mu) / sigma_eq

    # ── Signal state machine ──────────────────────────────
    signal   = pd.Series(0, index=spread.index, name="Signal")
    position = 0

    for i in range(len(z_score)):
        z = z_score.iloc[i]

        if position == 0:
            if z < -config.ENTRY_THRESHOLD:
                position = 1       # spread below mean → go long
            elif z > config.ENTRY_THRESHOLD:
                position = -1      # spread above mean → go short

        elif position == 1:
            if z > -config.EXIT_THRESHOLD:
                position = 0       # close long near mean

        elif position == -1:
            if z < config.EXIT_THRESHOLD:
                position = 0       # close short near mean

        signal.iloc[i] = position

    # ── Build results DataFrame ───────────────────────────
    df = pd.DataFrame({
        "Spread"      : spread,
        "Z_Score"     : z_score,
        "Signal"      : signal,
        "Mean"        : mu,
        "Upper_entry" : mu + config.ENTRY_THRESHOLD * sigma_eq,
        "Lower_entry" : mu - config.ENTRY_THRESHOLD * sigma_eq,
        "Upper_exit"  : mu + config.EXIT_THRESHOLD  * sigma_eq,
        "Lower_exit"  : mu - config.EXIT_THRESHOLD  * sigma_eq,
    })

    # ── P&L ──────────────────────────────────────────────
    df["Spread_Return"]   = spread.pct_change()
    df["Strategy_Return"] = df["Signal"].shift(1) * df["Spread_Return"]
    df["Cum_Strategy"]    = (1 + df["Strategy_Return"].fillna(0)).cumprod()
    df["Cum_BuyHold"]     = (1 + df["Spread_Return"].fillna(0)).cumprod()

    # ── Summary stats ─────────────────────────────────────
    rets   = df["Strategy_Return"].dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    total_ret = (df["Cum_Strategy"].iloc[-1] - 1) * 100

    print(f"\n    ── Signal Summary ──────────────────────────")
    print(f"    Entry threshold   : ±{config.ENTRY_THRESHOLD:.2f}σ")
    print(f"    Exit  threshold   : ±{config.EXIT_THRESHOLD:.2f}σ")
    print(f"    Long  days        : {(signal == 1).sum()}")
    print(f"    Short days        : {(signal == -1).sum()}")
    print(f"    Flat  days        : {(signal == 0).sum()}")
    print(f"    Annualised Sharpe : {sharpe:.3f}")
    print(f"    Total Return      : {total_ret:.1f}%")

    return df