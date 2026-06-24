"""
main.py
─────────────────────────────────────────────
ENERGY FUTURES TERM STRUCTURE
& MEAN REVERSION SIGNAL ENGINE
Version 2.0

Entry point — run this file.

V2 Project structure:
  config.py          ← all settings
  data.py            ← download + spread         (same as V1)
  term_structure.py  ← forward curve + regime    (same as V1)
  stats.py           ← ADF + Hurst tests         (same as V1)
  kalman.py          ← Kalman filter + Schwartz  (NEW in V2)
  signals.py         ← rolling OU + signals      (upgraded V2)
  plots.py           ← 10-panel dashboard        (upgraded V2)
  main.py            ← YOU ARE HERE

Usage:
  python main.py
─────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

import data
import term_structure
import stats
import kalman
import signals
import plots
import config


def main():

    # ── Step 1: Download real market data ─────────────────
    prices = data.download_prices()

    # ── Step 2: Build forward curve + regime ──────────────
    ts = term_structure.build_term_structure(prices)

    # ── Step 3: Compute calendar spread ───────────────────
    spread = data.compute_spread(prices)

    # ── Step 4: Mean reversion tests (ADF + Hurst) ────────
    adf = stats.run_adf_test(spread)
    H   = stats.hurst_exponent(spread)
    stats.print_test_results(adf, H)

    # ── Step 5: Kalman Filter ─────────────────────────────
    #    Extracts latent spot price + convenience yield
    kalman_result = kalman.run_kalman_filter(prices)

    # ── Step 6: Schwartz 1-Factor Model ───────────────────
    #    Fits κ, μ, σ, λ to real forward curve data
    schwartz_params = kalman.fit_schwartz_model(prices, kalman_result)

    # ── Step 7: Fit OU — static (V1 baseline) + rolling (V2)
    ou_static  = signals.fit_ou_static(spread)

    print(f"\n    ── Static OU (V1 baseline) ─────────────────")
    print(f"    κ         : {ou_static['kappa']:.6f}")
    print(f"    μ         : {ou_static['mu']:.4f}")
    print(f"    σ         : {ou_static['sigma']:.4f}")
    print(f"    Half-life : {ou_static['half_life']:.1f} trading days"
          f"  ({ou_static['half_life']/21:.1f} months)")

    rolling_ou = signals.fit_ou_rolling(spread)

    # ── Step 8: Generate signals ───────────────────────────
    sig_df = signals.generate_signals(spread, rolling_ou, ou_static)

    # ── Step 9: Plot 10-panel dashboard ───────────────────
    plots.plot_dashboard(
        prices          = prices,
        term_structure  = ts,
        signals_df      = sig_df,
        kalman_result   = kalman_result,
        schwartz_params = schwartz_params,
        ou_static       = ou_static,
        H               = H,
        adf             = adf,
    )

    print("\n" + "="*60)
    print("  ✓  V2 complete.")
    print(f"  ✓  Chart saved  →  {config.CHART_OUTPUT}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()