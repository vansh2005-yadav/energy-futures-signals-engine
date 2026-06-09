"""
kalman.py
─────────────────────────────────────────────
NEW in V2.

Two components:

1. Kalman Filter
   ─────────────
   Treats the log-price as an observable.
   Extracts two latent (hidden) state variables:
     • Spot price level  (smoothed)
     • Convenience yield (implied carry)

   The filter updates its belief about the hidden
   state every day as new prices arrive — it never
   looks ahead (causal / real-time safe).

   Why this matters for trading:
   Convenience yield rising  →  backwardation building
   Convenience yield falling →  contango building
   This is a leading indicator of term structure regime shifts.

2. Schwartz (1997) 1-Factor Model
   ────────────────────────────────
   Models the log spot price S as an OU process:
     d(ln S) = κ(μ − ln S)dt + σ dW

   We fit κ, μ, σ, λ to real futures price data
   using MLE (via scipy.optimize.minimize).

   Outputs theoretical forward prices for any tenor T:
     F(t,T) = exp(e^{-κT} ln S + (1−e^{-κT})μ̂ + ½σ²(1−e^{-2κT})/(2κ))
   where μ̂ = μ − λσ/κ  (risk-adjusted long-run mean)
─────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
import config


# ─────────────────────────────────────────────────────────
# PART 1 — KALMAN FILTER
# ─────────────────────────────────────────────────────────

def run_kalman_filter(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a 2-state Kalman filter to log WTI prices.

    State vector:  [log_spot, convenience_yield]
    Observation:   log(CL_front)

    Returns DataFrame with columns:
      kalman_spot       — filtered log-spot estimate
      kalman_cy         — filtered convenience yield estimate
      kalman_spot_price — exp(kalman_spot) in USD
    """
    print("\n[4/7] Running Kalman Filter (latent spot + convenience yield)...")

    log_price = np.log(prices["CL_front"].values)
    n         = len(log_price)
    dt        = 1 / 252   # daily in years

    # ── State transition matrix  F ────────────────────────
    # State: [log_spot(t), cy(t)]
    # log_spot evolves with drift adjusted by cy
    # cy has mild mean reversion
    kappa_cy = 2.0   # convenience yield mean reversion speed
    F = np.array([
        [1.0,  dt],
        [0.0,  1.0 - kappa_cy * dt],
    ])

    # ── Observation matrix  H ─────────────────────────────
    # We observe log_spot directly
    H = np.array([[1.0, 0.0]])

    # ── Noise covariances ─────────────────────────────────
    q1 = config.KF_PROCESS_NOISE          # spot process noise
    q2 = config.KF_PROCESS_NOISE * 0.1    # cy process noise
    Q  = np.array([[q1,  0.0],
                   [0.0, q2]])

    R = np.array([[config.KF_OBS_NOISE]])  # observation noise

    # ── Initialise ────────────────────────────────────────
    x  = np.array([log_price[0], 0.0])    # state mean
    P  = np.eye(2) * 1.0                  # state covariance

    spots = np.zeros(n)
    cys   = np.zeros(n)

    # ── Filter loop ───────────────────────────────────────
    for t in range(n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        z   = np.array([log_price[t]])
        y   = z - H @ x_pred                   # innovation
        S   = H @ P_pred @ H.T + R             # innovation cov
        K   = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

        x = x_pred + K.flatten() * y.flatten()[0]
        P = (np.eye(2) - K @ H) @ P_pred

        spots[t] = x[0]
        cys[t]   = x[1]

    result = pd.DataFrame({
        "kalman_log_spot"  : spots,
        "kalman_cy"        : cys,
        "kalman_spot_price": np.exp(spots),
    }, index=prices.index)

    cy_mean = result["kalman_cy"].mean()
    cy_std  = result["kalman_cy"].std()
    print(f"    Convenience yield — mean: {cy_mean:.4f}  std: {cy_std:.4f}")
    print(f"    Filtered spot range: "
          f"${result['kalman_spot_price'].min():.2f} — "
          f"${result['kalman_spot_price'].max():.2f}")

    return result


# ─────────────────────────────────────────────────────────
# PART 2 — SCHWARTZ 1-FACTOR MODEL
# ─────────────────────────────────────────────────────────

def _schwartz_forward(log_S: float, kappa: float, mu_hat: float,
                      sigma: float, T: float) -> float:
    """
    Schwartz (1997) 1-factor theoretical forward price.

    F(t, T) = exp(
        e^{-κT} * ln(S)
        + (1 - e^{-κT}) * μ̂
        + σ²(1 - e^{-2κT}) / (4κ)
    )
    where μ̂ = μ - λσ/κ  (risk-neutral long-run mean)
    """
    e_kT    = np.exp(-kappa * T)
    var_term = (sigma**2) * (1 - np.exp(-2 * kappa * T)) / (4 * kappa)
    return np.exp(e_kT * log_S + (1 - e_kT) * mu_hat + var_term)


def fit_schwartz_model(prices: pd.DataFrame,
                       kalman_result: pd.DataFrame) -> dict:
    """
    Fit Schwartz 1-factor model parameters to observed
    M1 / M3 / M6 forward curve.

    Uses the Kalman-filtered log spot as the state variable.
    Minimises sum of squared errors between theoretical
    and observed forward prices across 3 tenors.

    Returns dict: kappa, mu, sigma, lambda_, mu_hat, errors
    """
    print("\n[5/7] Fitting Schwartz 1-Factor Model...")

    # Use last 252 days (1 year) for calibration — more stable
    n_cal = min(252, len(prices))
    log_S = kalman_result["kalman_log_spot"].values[-n_cal:]

    # Observed "forward" prices at 3 tenors (in years)
    tenors = np.array([1/12, 3/12, 6/12])   # M1, M3, M6
    obs    = np.column_stack([
        prices["USO"].values[-n_cal:],
        prices["DBO"].values[-n_cal:],
        prices["BNO"].values[-n_cal:],
    ])

    def objective(params):
        kappa, mu, sigma, lam = params
        if kappa <= 0 or sigma <= 0:
            return 1e10
        mu_hat = mu - lam * sigma / kappa
        total_err = 0.0
        for i in range(n_cal):
            for j, T in enumerate(tenors):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    F_model = _schwartz_forward(log_S[i], kappa,
                                                mu_hat, sigma, T)
                total_err += (F_model - obs[i, j]) ** 2
        return total_err

    x0     = list(config.SCHWARTZ_INIT.values())
    bounds = [(0.01, 10), (0.5, 6), (0.01, 2), (-2, 2)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(objective, x0, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-8})

    kappa, mu, sigma, lam = res.x
    mu_hat = mu - lam * sigma / kappa

    print(f"    κ  (mean-reversion speed) : {kappa:.4f}")
    print(f"    μ  (long-run log-mean)    : {mu:.4f}  "
          f"(≈ ${np.exp(mu):.2f})")
    print(f"    σ  (volatility)           : {sigma:.4f}")
    print(f"    λ  (market price of risk) : {lam:.4f}")
    print(f"    μ̂  (risk-neutral mean)    : {mu_hat:.4f}")
    print(f"    Fit converged             : {res.success}")

    # Build model forward curve for TODAY (last observation)
    last_log_S = kalman_result["kalman_log_spot"].iloc[-1]
    model_curve = {
        T: _schwartz_forward(last_log_S, kappa, mu_hat, sigma, T)
        for T in np.linspace(1/12, 24/12, 24)
    }

    return {
        "kappa"       : kappa,
        "mu"          : mu,
        "sigma"       : sigma,
        "lambda_"     : lam,
        "mu_hat"      : mu_hat,
        "model_curve" : model_curve,
        "converged"   : res.success,
    }