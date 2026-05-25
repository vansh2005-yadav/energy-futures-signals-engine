"""
plots.py
─────────────────────────────────────────────
All visualisation logic.
Produces a 7-panel dark-theme dashboard and
saves it as a PNG file.

Panels:
  1. WTI Crude Oil price (full period)
  2. Normalised term structure (M1 / M3 / M6)
  3. Contango vs Backwardation regime
  4. Calendar spread + OU entry/exit bands
  5. OU Z-Score with signal zones
  6. Trading signal position (+1 / 0 / -1)
  7. Cumulative P&L — strategy vs buy & hold
─────────────────────────────────────────────
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import pandas as pd
import config


def style_ax(ax, title: str) -> None:
    """Apply dark theme styling to a single axes."""
    ax.set_facecolor(config.PANEL)
    ax.set_title(title, color=config.WHITE,
                 fontsize=10, fontweight="bold", pad=7)
    ax.tick_params(colors=config.GREY, labelsize=8)
    ax.xaxis.label.set_color(config.GREY)
    ax.yaxis.label.set_color(config.GREY)
    for spine in ax.spines.values():
        spine.set_edgecolor(config.GREY)
        spine.set_linewidth(0.4)
    ax.grid(True, color=config.GREY, alpha=0.12, linewidth=0.4)


def plot_dashboard(
    prices          : pd.DataFrame,
    term_structure  : pd.DataFrame,
    signals_df      : pd.DataFrame,
    ou_params       : dict,
    H               : float,
    adf             : dict,
) -> None:
    """Render and save the full 7-panel dashboard."""

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor(config.DARK)
    gs  = GridSpec(5, 2, figure=fig, hspace=0.48, wspace=0.28)

    # ── Panel 1: WTI Crude Oil price ──────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(prices.index, prices["CL_front"],
             color=config.GOLD, linewidth=1.1,
             label="WTI Crude (CL=F)")
    ax1.fill_between(prices.index, prices["CL_front"],
                     prices["CL_front"].min(),
                     alpha=0.07, color=config.GOLD)
    ax1.set_ylabel("Price  (USD)", color=config.GREY)
    ax1.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    style_ax(ax1, "WTI Crude Oil — Front Month Futures  (Real CME Data via Yahoo Finance)")

    # ── Panel 2: Normalised term structure ────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    tenor_map = {
        "M1_front": (config.BLUE,  "M1 — USO (front)"),
        "M3_mid"  : (config.GREEN, "M3 — DBO (mid)"),
        "M6_back" : (config.GOLD,  "M6 — BNO (back)"),
    }
    for col, (color, label) in tenor_map.items():
        ax2.plot(term_structure.index, term_structure[col],
                 color=color, linewidth=0.9, label=label)
    ax2.set_ylabel("Normalised Price  (100 = start)", color=config.GREY)
    ax2.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_ax(ax2, "Term Structure — Forward Curve (Normalised)")

    # ── Panel 3: Contango / Backwardation regime ──────────
    ax3 = fig.add_subplot(gs[1, 1])
    bar_colors = [
        config.GREEN if v else config.RED
        for v in term_structure["Contango"]
    ]
    ax3.bar(term_structure.index, term_structure["Contango"],
            color=bar_colors, width=1.5, alpha=0.75)
    ax3.set_ylabel("1 = Contango  |  0 = Backwardation", color=config.GREY)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_ax(ax3, "Market Regime  |  Green = Contango  |  Red = Backwardation")

    # ── Panel 4: Spread + OU bands ────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(signals_df.index, signals_df["Spread"],
             color=config.BLUE, linewidth=0.85,
             label="Calendar Spread (M1 − M3)")
    ax4.axhline(ou_params["mu"], color=config.WHITE, linewidth=1.0,
                linestyle="--",
                label=f"OU Mean  μ = {ou_params['mu']:.2f}")
    ax4.plot(signals_df.index, signals_df["Upper_entry"],
             color=config.RED,   linewidth=0.75, linestyle=":",
             label="+1σ  Short entry")
    ax4.plot(signals_df.index, signals_df["Lower_entry"],
             color=config.GREEN, linewidth=0.75, linestyle=":",
             label="−1σ  Long entry")
    ax4.fill_between(signals_df.index,
                     signals_df["Upper_exit"],
                     signals_df["Lower_exit"],
                     alpha=0.06, color=config.WHITE, label="Exit zone")

    # Shade active positions
    ax4.fill_between(signals_df.index, signals_df["Spread"],
                     where=signals_df["Signal"] == 1,
                     alpha=0.14, color=config.GREEN)
    ax4.fill_between(signals_df.index, signals_df["Spread"],
                     where=signals_df["Signal"] == -1,
                     alpha=0.14, color=config.RED)

    ax4.set_ylabel("Spread  (USD)", color=config.GREY)
    ax4.legend(facecolor=config.PANEL, labelcolor=config.WHITE,
               fontsize=8, ncol=3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    hl = ou_params["half_life"]
    style_ax(ax4,
             f"Calendar Spread + OU Bands  |  "
             f"Half-life = {hl:.1f} days  |  "
             f"Hurst H = {H:.3f}")

    # ── Panel 5: Z-Score ──────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(signals_df.index, signals_df["Z_Score"],
             color=config.GOLD, linewidth=0.8, label="Z-Score")
    ax5.axhline( 1.0, color=config.RED,   linewidth=0.8,
                linestyle="--", alpha=0.9)
    ax5.axhline(-1.0, color=config.GREEN, linewidth=0.8,
                linestyle="--", alpha=0.9)
    ax5.axhline( 0.0, color=config.WHITE, linewidth=0.5,
                linestyle=":",  alpha=0.4)
    ax5.fill_between(signals_df.index, signals_df["Z_Score"],
                     where=signals_df["Z_Score"] >  1,
                     alpha=0.18, color=config.RED,   label="Short zone")
    ax5.fill_between(signals_df.index, signals_df["Z_Score"],
                     where=signals_df["Z_Score"] < -1,
                     alpha=0.18, color=config.GREEN, label="Long zone")
    ax5.set_ylabel("Z-Score  (σ)", color=config.GREY)
    ax5.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=8)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_ax(ax5, "OU Z-Score  |  Entry at ±1σ  |  Exit at ±0.25σ")

    # ── Panel 6: Signal position ──────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    pos_color_map = {1: config.GREEN, -1: config.RED, 0: config.GREY}
    bar_c = [pos_color_map[s] for s in signals_df["Signal"]]
    ax6.bar(signals_df.index, signals_df["Signal"],
            color=bar_c, width=1.5, alpha=0.85)
    ax6.set_ylabel("Position", color=config.GREY)
    ax6.set_yticks([-1, 0, 1])
    ax6.set_yticklabels(["Short", "Flat", "Long"], color=config.GREY)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_ax(ax6, "Trading Signal  |  +1 Long  |  −1 Short  |  0 Flat")

    # ── Panel 7: Cumulative P&L ───────────────────────────
    ax7 = fig.add_subplot(gs[4, :])
    ax7.plot(signals_df.index, signals_df["Cum_Strategy"],
             color=config.GREEN, linewidth=1.2,
             label="OU Mean-Reversion Strategy")
    ax7.plot(signals_df.index, signals_df["Cum_BuyHold"],
             color=config.GREY,  linewidth=0.9, linestyle="--",
             label="Buy & Hold  (Spread)", alpha=0.7)
    ax7.axhline(1.0, color=config.WHITE, linewidth=0.5,
                linestyle=":", alpha=0.35)
    ax7.fill_between(signals_df.index, signals_df["Cum_Strategy"], 1,
                     where=signals_df["Cum_Strategy"] >= 1,
                     alpha=0.09, color=config.GREEN)
    ax7.fill_between(signals_df.index, signals_df["Cum_Strategy"], 1,
                     where=signals_df["Cum_Strategy"] < 1,
                     alpha=0.09, color=config.RED)
    ax7.set_ylabel("Cumulative Return  (1.0 = start)", color=config.GREY)
    ax7.legend(facecolor=config.PANEL, labelcolor=config.WHITE, fontsize=9)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    style_ax(ax7, "Cumulative P&L — Strategy vs Buy & Hold")

    # ── Super title ───────────────────────────────────────
    fig.suptitle(
        "ENERGY FUTURES  ·  TERM STRUCTURE & MEAN REVERSION SIGNAL ENGINE  ·  V1",
        color=config.WHITE, fontsize=13, fontweight="bold", y=0.998,
    )

    # ── Save ──────────────────────────────────────────────
    plt.savefig(config.CHART_OUTPUT, dpi=config.CHART_DPI,
                bbox_inches="tight", facecolor=config.DARK)
    print(f"\n    Chart saved  →  {config.CHART_OUTPUT}")
    plt.show()