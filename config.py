"""
config.py
─────────────────────────────────────────────
All project-wide settings in one place.
Version 2 adds:
  - Kalman filter noise parameters
  - Rolling OU window size
  - Schwartz model settings
─────────────────────────────────────────────
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
    "UNG"      : "UNG",    # Natural gas front ETF
}

# Calendar spread legs
SPREAD_LEG1 = "USO"   # front (M1)
SPREAD_LEG2 = "DBO"   # back  (M3)

# ── Signal Settings ───────────────────────
ENTRY_THRESHOLD = 1.0    # z-score to enter  (±1σ)
EXIT_THRESHOLD  = 0.25   # z-score to exit   (±0.25σ)

# ── V2: Rolling OU Window ─────────────────
# Number of trading days for rolling OU parameter estimation
# 126 = ~6 months, balances stability vs responsiveness
ROLLING_WINDOW = 126

# ── V2: Kalman Filter ─────────────────────
# Observation noise (how much we trust the price data)
# Lower = trust data more; Higher = smoother latent state
KF_OBS_NOISE    = 0.1

# Process noise (how fast latent state can change)
# Lower = slower-moving state; Higher = more reactive
KF_PROCESS_NOISE = 0.01

# ── V2: Schwartz 1-Factor Model ───────────────────────────
# Initial guesses for Schwartz parameter optimisation
# kappa = mean reversion speed, mu = long run mean,
# sigma = vol, lambda = market price of risk
SCHWARTZ_INIT = {
    "kappa"  : 0.5,
    "mu"     : 3.5,    # log-price long run mean (ln ~$33)
    "sigma"  : 0.3,
    "lambda" : 0.1,    # market price of risk
}

# ── Plot / Output Settings ────────────────
CHART_OUTPUT    = "energy_futures_v2_dashboard.png"
CHART_DPI       = 150

# Dark theme colours (same as V1)
DARK  = "#0d1117"
PANEL = "#161b22"
GREEN = "#39d353"
RED   = "#f85149"
GOLD  = "#e3b341"
BLUE  = "#58a6ff"
GREY  = "#8b949e"
WHITE = "#f0f6fc"
TEAL  = "#56d364"
PURP  = "#bc8cff"