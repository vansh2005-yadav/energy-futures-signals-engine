"""
plots.py
─────────────────────────────────────────────
V2 UPGRADE — 10-panel dark-theme dashboard.

New panels vs V1:
  Panel 4 : Kalman filtered spot vs raw price
  Panel 5 : Kalman convenience yield over time
  Panel 6 : Schwartz model forward curve (theoretical vs observed)
  Panel 7 : Rolling OU half-life over time
  Panel 8 : Rolling vs static z-score
  Panel 9 : Signal comparison (rolling V2 vs static V1)
  Panel 10: Cumulative P&L — rolling vs static vs buy & hold
─────────────────────────────────────────────
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import config


def _style(ax, title: str) -> None:
    """Apply dark theme to axes."""
    ax.set_facecolor(config.PANEL)
    ax.set_title(title, color=config.WHITE,
                 fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(colors=config.GREY, labelsize=7)
    ax.xaxis.label.set_color(config.GREY)
    ax.yaxis.label.set_color(config.GREY)
    for spine in ax.spines.values():
        spine.set_edgecolor(config.GREY)
        spine.set_linewidth(0.4)
    ax.grid(True, color=config.GREY, alpha=0.1, linewidth=0.4)


def plot_dashboard(
    prices          : pd.DataFrame,
    term_structure  : pd.DataFrame,
    signals_df      : pd.DataFrame,
    kalman_result   : pd.DataFrame,
    schwartz_params : dict,
    ou_static       : dict,
    H               : float,
    adf             : dict,
) -> None:

    fig = plt.figure(figsize=(20, 30))
    fig.patch.set_facecolor(config.DARK)
    gs  = GridSpec(6, 2, figure=fig, hspace=0.52, wspace=0.30)

    # ── Panel 1: WTI Crude price (full span) ──────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(prices.index, prices["CL_front"],
             color=config.GOLD, linewidth=1.0, label="WTI Crude (CL=F)")
    ax1.fill_between(prices.index, prices["CL_front"],
                     prices["CL_front"].min(),
                     alpha=0.07, color=config.GOLD)
    ax1.set_ylabel("Price  (USD)", color=config.GREY)
    ax1.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _style(ax1, "WTI Crude Oil — Front Month Futures  (Real CME Data via Yahoo Finance)")

    # ── Panel 2: Term structure ────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for col, color, label in [
        ("M1_front", config.BLUE,  "M1 USO"),
        ("M3_mid",   config.GREEN, "M3 DBO"),
        ("M6_back",  config.GOLD,  "M6 BNO"),
    ]:
        ax2.plot(term_structure.index, term_structure[col],
                 color=color, linewidth=0.85, label=label)
    ax2.set_ylabel("Normalised (100=start)", color=config.GREY)
    ax2.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax2, "Term Structure — Forward Curve (Normalised)")

    # ── Panel 3: Contango / Backwardation ─────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    bar_c = [config.GREEN if v else config.RED
             for v in term_structure["Contango"]]
    ax3.bar(term_structure.index, term_structure["Contango"],
            color=bar_c, width=1.5, alpha=0.8)
    ax3.set_ylabel("1=Contango  0=Backwardation", color=config.GREY)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax3, "Market Regime  |  Green=Contango  |  Red=Backwardation")

    # ── Panel 4: Kalman filtered spot vs raw price ────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(prices.index, prices["USO"],
             color=config.GREY, linewidth=0.7, alpha=0.6, label="USO (raw)")
    ax4.plot(kalman_result.index, kalman_result["kalman_spot_price"],
             color=config.PURP, linewidth=1.1, label="Kalman Filtered Spot")
    ax4.set_ylabel("Price  (USD)", color=config.GREY)
    ax4.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=7)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax4, "Kalman Filter — Latent Spot Price Extraction")

    # ── Panel 5: Kalman convenience yield ─────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    cy = kalman_result["kalman_cy"]
    ax5.plot(kalman_result.index, cy,
             color=config.TEAL, linewidth=0.85, label="Convenience Yield")
    ax5.axhline(0, color=config.WHITE, linewidth=0.5,
                linestyle=":", alpha=0.5)
    ax5.fill_between(kalman_result.index, cy, 0,
                     where=cy > 0, alpha=0.15, color=config.GREEN)
    ax5.fill_between(kalman_result.index, cy, 0,
                     where=cy < 0, alpha=0.15, color=config.RED)
    ax5.set_ylabel("Convenience Yield", color=config.GREY)
    ax5.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=7)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax5, "Kalman Convenience Yield  |  +ve = Backwardation signal")

    # ── Panel 6: Schwartz forward curve ───────────────────
    ax6 = fig.add_subplot(gs[3, 0])
    sc  = schwartz_params
    tenors_m  = list(sc["model_curve"].keys())
    fwd_model = list(sc["model_curve"].values())
    ax6.plot([t * 12 for t in tenors_m], fwd_model,
             color=config.PURP, linewidth=1.2,
             label=f"Schwartz Model  (κ={sc['kappa']:.2f}, σ={sc['sigma']:.2f})")

    # Observed 3 points (latest prices)
    obs_tenors = [1, 3, 6]
    obs_prices = [
        prices["USO"].iloc[-1],
        prices["DBO"].iloc[-1],
        prices["BNO"].iloc[-1],
    ]
    ax6.scatter(obs_tenors, obs_prices,
                color=config.GOLD, s=60, zorder=5, label="Observed (latest)")
    ax6.set_xlabel("Tenor (months)", color=config.GREY)
    ax6.set_ylabel("Forward Price  (USD)", color=config.GREY)
    ax6.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=7)
    _style(ax6, "Schwartz 1-Factor — Theoretical vs Observed Forward Curve")

    # ── Panel 7: Rolling OU half-life ─────────────────────
    ax7 = fig.add_subplot(gs[3, 1])
    hl = signals_df["Rolling_HalfLife"].dropna()
    ax7.plot(hl.index, hl,
             color=config.GOLD, linewidth=0.85, label="Rolling Half-Life")
    ax7.axhline(ou_static["half_life"], color=config.GREY,
                linewidth=0.8, linestyle="--",
                label=f"Static HL = {ou_static['half_life']:.1f}d")
    ax7.fill_between(hl.index, hl, ou_static["half_life"],
                     alpha=0.08, color=config.GOLD)
    ax7.set_ylabel("Half-Life  (trading days)", color=config.GREY)
    ax7.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=7)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax7, "Rolling OU Half-Life — How Fast the Spread Mean-Reverts")

    # ── Panel 8: Spread + dynamic OU bands ────────────────
    ax8 = fig.add_subplot(gs[4, 0])
    ax8.plot(signals_df.index, signals_df["Spread"],
             color=config.BLUE, linewidth=0.8, label="Spread")
    ax8.plot(signals_df.index, signals_df["Upper_entry"],
             color=config.RED,   linewidth=0.6, linestyle=":", alpha=0.9)
    ax8.plot(signals_df.index, signals_df["Lower_entry"],
             color=config.GREEN, linewidth=0.6, linestyle=":", alpha=0.9)
    ax8.plot(signals_df.index, signals_df["Rolling_Mu"],
             color=config.WHITE, linewidth=0.7, linestyle="--",
             alpha=0.6, label="Rolling μ")
    ax8.fill_between(signals_df.index, signals_df["Spread"],
                     where=signals_df["Signal_Rolling"] == 1,
                     alpha=0.13, color=config.GREEN)
    ax8.fill_between(signals_df.index, signals_df["Spread"],
                     where=signals_df["Signal_Rolling"] == -1,
                     alpha=0.13, color=config.RED)
    ax8.set_ylabel("Spread  (USD)", color=config.GREY)
    ax8.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=7)
    ax8.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax8, "Calendar Spread + Dynamic Rolling OU Bands  (V2 Signal)")

    # ── Panel 9: Z-score comparison ───────────────────────
    ax9 = fig.add_subplot(gs[4, 1])
    ax9.plot(signals_df.index, signals_df["Z_Rolling"],
             color=config.PURP, linewidth=0.8, alpha=0.9,
             label="Rolling Z (V2)")
    ax9.plot(signals_df.index, signals_df["Z_Static"],
             color=config.GREY, linewidth=0.7, alpha=0.6,
             label="Static Z  (V1)")
    ax9.axhline( config.ENTRY_THRESHOLD, color=config.RED,
                linewidth=0.7, linestyle="--", alpha=0.8)
    ax9.axhline(-config.ENTRY_THRESHOLD, color=config.GREEN,
                linewidth=0.7, linestyle="--", alpha=0.8)
    ax9.axhline(0, color=config.WHITE, linewidth=0.4,
                linestyle=":", alpha=0.4)
    ax9.set_ylabel("Z-Score  (σ)", color=config.GREY)
    ax9.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=7)
    ax9.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax9, "Z-Score: Rolling V2 vs Static V1 Comparison")

    # ── Panel 10: Cumulative P&L ───────────────────────────
    ax10 = fig.add_subplot(gs[5, :])
    ax10.plot(signals_df.index, signals_df["Cum_Rolling"],
              color=config.GREEN, linewidth=1.3,
              label="Rolling OU Strategy  (V2)")
    ax10.plot(signals_df.index, signals_df["Cum_Static"],
              color=config.PURP,  linewidth=1.0, linestyle="-.",
              label="Static OU Strategy  (V1 baseline)")
    ax10.plot(signals_df.index, signals_df["Cum_BuyHold"],
              color=config.GREY,  linewidth=0.8, linestyle="--",
              alpha=0.65, label="Buy & Hold  (Spread)")
    ax10.axhline(1.0, color=config.WHITE, linewidth=0.4,
                 linestyle=":", alpha=0.3)
    ax10.fill_between(signals_df.index, signals_df["Cum_Rolling"], 1,
                      where=signals_df["Cum_Rolling"] >= 1,
                      alpha=0.08, color=config.GREEN)
    ax10.fill_between(signals_df.index, signals_df["Cum_Rolling"], 1,
                      where=signals_df["Cum_Rolling"] < 1,
                      alpha=0.08, color=config.RED)
    ax10.set_ylabel("Cumulative Return  (1.0 = start)", color=config.GREY)
    ax10.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=9)
    ax10.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _style(ax10, "Cumulative P&L — Rolling V2  vs  Static V1  vs  Buy & Hold")

    fig.suptitle(
        "ENERGY FUTURES  ·  TERM STRUCTURE & MEAN REVERSION SIGNAL ENGINE  ·  V2"
        "  |  Kalman Filter  +  Schwartz 1-Factor  +  Rolling OU",
        color=config.WHITE, fontsize=11, fontweight="bold", y=0.999,
    )

    plt.savefig(config.CHART_OUTPUT, dpi=config.CHART_DPI,
                bbox_inches="tight", facecolor=config.DARK)
    print(f"\n    Chart saved  →  {config.CHART_OUTPUT}")
    plt.show()