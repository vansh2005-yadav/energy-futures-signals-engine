# Energy Futures — Term Structure & Mean Reversion Signal Engine
## Version 1.0 | Python 3.10.10

---

## Project Structure

```
energy_futures_v1/
│
├── main.py              ← RUN THIS — entry point
├── config.py            ← all settings (dates, tickers, thresholds)
├── data.py              ← download real data + compute spread
├── term_structure.py    ← forward curve + contango/backwardation
├── stats.py             ← ADF test + Hurst exponent
├── signals.py           ← OU process fitting + signal generation
├── plots.py             ← full 7-panel dashboard
├── requirements.txt     ← dependencies
└── README.md
```

---

## Setup & Run

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

---

## What Each File Does

| File | Purpose |
|------|---------|
| `config.py` | Single source of truth for all settings |
| `data.py` | Downloads CL=F, USO, DBO, BNO, UNG via yfinance |
| `term_structure.py` | Normalises 3 tenors, flags contango/backwardation |
| `stats.py` | ADF stationarity test + Hurst exponent (R/S method) |
| `signals.py` | Fits OU via MLE, builds z-score, generates signals |
| `plots.py` | 7-panel dark-theme dashboard, saves PNG |
| `main.py` | Calls all modules in order |

---


## Key Maths

**Ornstein-Uhlenbeck process:**
```
dX = κ(μ − X)dt + σ dW
```
- `κ` = speed of mean reversion (higher = faster snap back)
- `μ` = long-run equilibrium of the spread
- `σ` = spread volatility
- `Half-life = ln(2) / κ` (days to revert halfway to mean)

**Signal Z-score:**
```
z = (X − μ) / σ_eq      where  σ_eq = σ / sqrt(2κ)
```
- Enter Long  when z < −1.0
- Enter Short when z > +1.0
- Exit        when |z| < 0.25

---

## Roadmap

| Version | New features |
|---------|-------------|
| *V1* | OU MLE, ADF, Hurst, z-score signals ← you are here |
| *V2* | Kalman Filter, Schwartz 1-factor model, rolling OU window |
| *V3* | HMM regime switching, GARCH vol, Kelly position sizing |
| *V4* | Exotic derivatives pricing, Greeks, full backtester tearsheet |