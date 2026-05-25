"""
main.py
─────────────────────────────────────────────
ENERGY FUTURES TERM STRUCTURE
& MEAN REVERSION SIGNAL ENGINE
Version 1.0

Entry point — run this file.

Project structure:
  config.py        ← all settings
  data.py          ← download + spread
  term_structure.py← forward curve + regime
  stats.py         ← ADF + Hurst tests
  signals.py       ← OU fit + signal engine
  plots.py         ← dashboard charts
  main.py          ← YOU ARE HERE

Usage:
  python main.py
─────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

import data
import term_structure
import stats
import signals
import plots


def main():

    # 1. Download real market data
    prices = data.download_prices()

    # 2. Build forward curve + contango/backwardation regime
    ts = term_structure.build_term_structure(prices)

    # 3. Compute M1 - M3 calendar spread
    spread = data.compute_spread(prices)

    # 4. Test spread for mean reversion
    adf = stats.run_adf_test(spread)
    H   = stats.hurst_exponent(spread)
    stats.print_test_results(adf, H)

    # 5. Fit Ornstein-Uhlenbeck process to the spread
    ou = signals.fit_ou_process(spread)

    # 6. Generate Long / Short / Flat signals
    sig_df = signals.generate_signals(spread, ou)

    # 7. Plot 7-panel dashboard + save PNG
    plots.plot_dashboard(
        prices         = prices,
        term_structure = ts,
        signals_df     = sig_df,
        ou_params      = ou,
        H              = H,
        adf            = adf,
    )

    print("\n  ✓  V1 complete.")
    print("  ✓  Chart saved →", __import__("config").CHART_OUTPUT)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()