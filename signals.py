"""
signals.py
─────────────────────────────────────────────
V2 UPGRADE over V1:

V1 had static OU — fitted once on full history.
V2 adds rolling OU — parameters re-estimated
every day on a sliding window (default 126 days).

This means:
  • κ, μ, σ adapt to changing market conditions
  • Signals reflect current regime, not stale params
  • Half-life is dynamic — you can see it evolve

Both static (V1-style) and rolling (V2) signals
are computed and compared in the output.
─────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import config


# ─────────────────────────────────────────────────────────
# SHARED: OU parameter estimation (MLE, closed-form)
# Same core maths as V1 — used by both static & rolling
# ─────────────────────────────────────────────────────────

def _fit_ou_mle(x: np.ndarray) -> tuple:
    """
    Closed-form exact MLE for discrete OU process (dt=1 day).
    Returns (kappa, mu, sigma) or None if degenerate.
    """
    n   = len(x) - 1
    if n < 20:
        return None

    sx  = np.sum(x[:-1])
    sy  = np.sum(x[1:])
    sxx = np.sum(x[:-1] ** 2)
    sxy = np.sum(x[:-1] * x[1:])
    syy = np.sum(x[1:]  ** 2)

    denom = n * (sxx - sxy) - (sx**2 - sx * sy)
    if abs(denom) < 1e-12:
        return None

    mu    = (sy * sxx - sx * sxy) / denom
    alpha = (sxy - mu*sx - mu*sy + n*mu**2) / \
            max(sxx - 2*mu*sx + n*mu**2, 1e-12)
    alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
    kappa = max(-np.log(alpha), 1e-6)

    sigma_sq = (
        syy - 2*alpha*sxy + alpha**2*sxx
        - 2*mu*(1-alpha)*(sy - alpha*sx)
        + n*mu**2*(1-alpha)**2
    ) / n
    sigma = np.sqrt(max(sigma_sq, 1e-10))

    return kappa, mu, sigma


# ─────────────────────────────────────────────────────────
# STATIC OU (same as V1 — kept for comparison)
# ─────────────────────────────────────────────────────────

def fit_ou_static(spread: pd.Series) -> dict:
    """
    Fit OU on full history (V1 approach).
    Used as a baseline for comparison.
    """
    result = _fit_ou_mle(spread.dropna().values)
    if result is None:
        raise ValueError("Cannot fit static OU — insufficient data.")

    kappa, mu, sigma = result
    half_life = np.log(2) / kappa

    return {
        "kappa"    : kappa,
        "mu"       : mu,
        "sigma"    : sigma,
        "half_life": half_life,
    }


# ─────────────────────────────────────────────────────────
# ROLLING OU (NEW in V2)
# ─────────────────────────────────────────────────────────

def fit_ou_rolling(spread: pd.Series,
                   window: int = None) -> pd.DataFrame:
    """
    Re-estimate OU parameters on a rolling window.
    Returns a DataFrame with daily kappa, mu, sigma, half_life,
    sigma_eq, and z_score.

    window defaults to config.ROLLING_WINDOW (126 days).
    """
    if window is None:
        window = config.ROLLING_WINDOW

    print(f"\n[6/7] Fitting rolling OU  (window = {window} days)...")

    x      = spread.values
    idx    = spread.index
    n      = len(x)

    kappas     = np.full(n, np.nan)
    mus        = np.full(n, np.nan)
    sigmas     = np.full(n, np.nan)
    half_lives = np.full(n, np.nan)
    sigma_eqs  = np.full(n, np.nan)
    z_scores   = np.full(n, np.nan)

    for i in range(window, n):
        window_data = x[i - window : i]
        result      = _fit_ou_mle(window_data)
        if result is None:
            continue

        kappa, mu, sigma = result
        half_life        = np.log(2) / kappa
        sigma_eq         = sigma / np.sqrt(2 * kappa)

        kappas[i]     = kappa
        mus[i]        = mu
        sigmas[i]     = sigma
        half_lives[i] = half_life
        sigma_eqs[i]  = sigma_eq
        z_scores[i]   = (x[i] - mu) / sigma_eq if sigma_eq > 1e-10 else 0.0

    rolling = pd.DataFrame({
        "kappa"     : kappas,
        "mu"        : mus,
        "sigma"     : sigmas,
        "half_life" : half_lives,
        "sigma_eq"  : sigma_eqs,
        "z_score"   : z_scores,
        "spread"    : x,
    }, index=idx)

    valid = rolling["kappa"].dropna()
    print(f"    Rolling κ  — mean: {valid.mean():.4f}  "
          f"min: {valid.min():.4f}  max: {valid.max():.4f}")
    print(f"    Rolling half-life — "
          f"mean: {rolling['half_life'].mean():.1f} days")

    return rolling


# ─────────────────────────────────────────────────────────
# SIGNAL GENERATION — uses rolling OU z-score (V2)
# ─────────────────────────────────────────────────────────

def generate_signals(spread: pd.Series,
                     rolling_ou: pd.DataFrame,
                     ou_static: dict) -> pd.DataFrame:
    """
    Generate Long / Short / Flat signals using the ROLLING
    OU z-score (V2 upgrade).

    Also computes a static-OU signal (V1 baseline) for comparison.

    Signal values:
      +1  →  Long  (spread too low — buy the spread)
      -1  →  Short (spread too high — sell the spread)
       0  →  Flat
    """
    n = len(spread)

    # ── Rolling signal (V2) ───────────────────────────────
    rolling_signal = pd.Series(0, index=spread.index, name="Rolling_Signal")
    position       = 0

    for i in range(n):
        z = rolling_ou["z_score"].iloc[i]
        if np.isnan(z):
            rolling_signal.iloc[i] = 0
            continue

        if position == 0:
            if z < -config.ENTRY_THRESHOLD:
                position = 1
            elif z > config.ENTRY_THRESHOLD:
                position = -1
        elif position == 1:
            if z > -config.EXIT_THRESHOLD:
                position = 0
        elif position == -1:
            if z < config.EXIT_THRESHOLD:
                position = 0

        rolling_signal.iloc[i] = position

    # ── Static signal (V1 baseline) ───────────────────────
    kappa_s    = ou_static["kappa"]
    mu_s       = ou_static["mu"]
    sigma_s    = ou_static["sigma"]
    sigma_eq_s = sigma_s / np.sqrt(2 * kappa_s)
    z_static   = (spread - mu_s) / sigma_eq_s

    static_signal = pd.Series(0, index=spread.index, name="Static_Signal")
    pos2          = 0
    for i in range(n):
        z = z_static.iloc[i]
        if pos2 == 0:
            if z < -config.ENTRY_THRESHOLD:
                pos2 = 1
            elif z > config.ENTRY_THRESHOLD:
                pos2 = -1
        elif pos2 == 1:
            if z > -config.EXIT_THRESHOLD:
                pos2 = 0
        elif pos2 == -1:
            if z < config.EXIT_THRESHOLD:
                pos2 = 0
        static_signal.iloc[i] = pos2

    # ── P&L ───────────────────────────────────────────────
    spread_ret = spread.pct_change()

    rolling_ret  = rolling_signal.shift(1) * spread_ret
    static_ret   = static_signal.shift(1)  * spread_ret

    cum_rolling  = (1 + rolling_ret.fillna(0)).cumprod()
    cum_static   = (1 + static_ret.fillna(0)).cumprod()
    cum_buyhold  = (1 + spread_ret.fillna(0)).cumprod()

    # Sharpe
    def sharpe(rets):
        r = rets.dropna()
        return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0

    sh_roll   = sharpe(rolling_ret)
    sh_static = sharpe(static_ret)

    print(f"\n    ── Signal Summary ──────────────────────────")
    print(f"    {'':30s}  {'Rolling (V2)':>12}  {'Static (V1)':>12}")
    print(f"    {'Annualised Sharpe':30s}  {sh_roll:>12.3f}  {sh_static:>12.3f}")
    print(f"    {'Total Return':30s}  "
          f"{(cum_rolling.iloc[-1]-1)*100:>11.1f}%  "
          f"{(cum_static.iloc[-1]-1)*100:>11.1f}%")
    print(f"    {'Long days':30s}  {(rolling_signal==1).sum():>12}  "
          f"{(static_signal==1).sum():>12}")
    print(f"    {'Short days':30s}  {(rolling_signal==-1).sum():>12}  "
          f"{(static_signal==-1).sum():>12}")

    df = pd.DataFrame({
        "Spread"         : spread,
        "Z_Rolling"      : rolling_ou["z_score"],
        "Z_Static"       : z_static,
        "Signal_Rolling" : rolling_signal,
        "Signal_Static"  : static_signal,
        "Rolling_Kappa"  : rolling_ou["kappa"],
        "Rolling_HalfLife": rolling_ou["half_life"],
        "Rolling_Mu"     : rolling_ou["mu"],
        "Rolling_SigmaEq": rolling_ou["sigma_eq"],
        "Upper_entry"    : rolling_ou["mu"] + config.ENTRY_THRESHOLD * rolling_ou["sigma_eq"],
        "Lower_entry"    : rolling_ou["mu"] - config.ENTRY_THRESHOLD * rolling_ou["sigma_eq"],
        "Cum_Rolling"    : cum_rolling,
        "Cum_Static"     : cum_static,
        "Cum_BuyHold"    : cum_buyhold,
    })

    return df