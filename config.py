"""
config.py
────────────────────────────────────────────
All project-wide settings in one place.
Change values here — everything else adapts.
────────────────────────────────────────────
"""

# ── Data Settings ─────────────────────────
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

# Yahoo Finance tickers
TICKERS = {
    "CL_front" : "CL=F",   # WTI crude front-month futures
    "USO"      : "USO",    # M1 proxy  (front-month oil ETF)
    "DBO"      : "DBO",    # M3 proxy  (optimum-yield rolled)
    "BNO"      : "BNO",    # M6 proxy  (Brent oil ETF)
    "UNG"      : "UNG",    # Natural gas front ETF (bonus)
}

# Which two tickers form the calendar spread
SPREAD_LEG1 = "USO"   # front (M1)
SPREAD_LEG2 = "DBO"   # back  (M3)

# ── Signal Settings ───────────────────────
ENTRY_THRESHOLD = 1.0    # z-score to enter trade  (±1σ)
EXIT_THRESHOLD  = 0.25   # z-score to exit trade   (±0.25σ)

# ── Plot / Output Settings ────────────────
CHART_OUTPUT = "energy_futures_v1_dashboard.png"
CHART_DPI    = 150

# Dark theme colours
DARK  = "#0d1117"
PANEL = "#161b22"
GREEN = "#39d353"
RED   = "#f85149"
GOLD  = "#e3b341"
BLUE  = "#58a6ff"
GREY  = "#8b949e"
WHITE = "#f0f6fc"