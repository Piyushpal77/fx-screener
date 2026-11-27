# Multi-Factor FX & Precious Metals Terminal Screener

A quantitative **Position & Swing (`H4`, `D1`, `W1`)** Forex & Precious Metals Terminal Screener covering all **28 G8 Currency Pairs** (`USD`, `EUR`, `GBP`, `JPY`, `AUD`, `CAD`, `CHF`, `NZD`) plus **Gold (`XAU/USD`)** and **Silver (`XAG/USD`)**.

Built with **Textual** (interactive TUI dashboard) and **Rich** (one-shot CLI & CSV/JSON export).

---

## Features

1. **30-Instrument Universe**:
   - **7 G8 Majors**: `EUR/USD`, `GBP/USD`, `USD/JPY`, `USD/CHF`, `AUD/USD`, `USD/CAD`, `NZD/USD`
   - **21 G8 Minor Crosses**: All cross combinations of `EUR`, `GBP`, `AUD`, `NZD`, `CAD`, `CHF`, `JPY`
   - **2 Precious Metals**: `XAU/USD` (Gold) and `XAG/USD` (Silver)
2. **Weighted 4-Pillar Composite Score (`-100` to `+100`) — `Fx_Screener2` Early-Inflection Model**:
   - **Early Trend & Fast Breakout (45%)**: Fast `Donchian(10)` Entry + `Donchian(5)` Trailing Exit Channel, Fast/Slow Hull Moving Averages (`HMA 9`, `HMA 21`), `SuperTrend(10, 2.5)`, `EMA 20/50/200` stack, `DI+/DI-`, and FX-calibrated **Early Trend Efficiency Multiplier** (`35% DX(14) + 25% RAVI(7, 65) + 20% VHF(14) + 20% Inverted CHOP(14)`).
   - **Momentum Acceleration & Elliott Wave (30%)**: 2nd-Derivative `MACD Δ²H` (`Hist_t - 2*Hist_{t-1} + Hist_{t-2}`), `EWO(5, 35)` Elliott Wave Oscillator, Causal `ATR-ZigZag` Wave 1-2-3 structure, and Constance Brown `RSI(14)` regimes.
   - **Currency Strength (15%)**: True G8 relative strength matrix (`-100` to `+100`) computed across all 28 pairs (`Base Score - Quote Score`).
   - **TPO Value Area & Squeeze (10%)**: Steidlmayer TPO Market Profile (`VAH`, `POC`, `VAL` 70% value area) + **TTM Squeeze** (`Bollinger(20, 2.0)` inside `Keltner(20, 1.5x ATR)`).
3. **Fast Intraday-to-Swing Confluence & 4-Trigger ASAP Exit Engine**:
   - Combines `H4` (**45%**), `D1` (**40%**), and `W1` (**15%**), plus **`⚡ EARLY ENTRY` (`H1 + H4 + D1`)** to catch breakouts before `W1` lags.
   - **4-Trigger ASAP Exit Engine (`⚠ EXIT NOW`)**: Alerts immediately when an open trade hits (1) `H4 HMA(9/21)` reversal, (2) `Donchian(5)` trailing channel break after `+0.75R`, (3) Momentum Exhaustion climax at `>= +1.2R` (`MACD Δ²H` + `RAVI` contraction), or (4) Hard `SL` / `TP2`.
   - **`🧹 Clean Chart` Mode**: One-click button in the Web UI to hide all indicator/level overlays for a pure logarithmic candlestick view.
4. **ATR Trade Planner & Cross-Currency Position Sizer**:
   - Computes Entry, `1.5 × ATR` Stop Loss (in price and pips), and `1.5R`, `2.0R`, `3.0R` Take-Profit levels.
   - Converts quote-currency pip values back to `USD` across majors, `USD/XXX`, cross pairs (e.g. `GBP/JPY`, `EUR/AUD`), and metals (`XAU/USD`, `XAG/USD`) to output exact **Standard Lots** and **Units** for your account balance and risk % (`$10,000 @ 1.0%` default).
5. **Local SQLite OHLCV Cache & Pluggable Data Feeds**:
   - Free no-key default feed via `yfinance` (`1h` resampled to `H4`, `1d`, `1wk`) cached in `.fx_screener_cache.sqlite3`, plus pluggable `OANDA v20` REST API and deterministic offline `synthetic` provider.

---

## Quick Start — Side-by-Side V1 & V2 Launchers

Both versions live side-by-side in this workspace with independent packages, configs, caches, and ports so you can even run both simultaneously to compare signals live:

### 1. Launch **FX Screener 2.0** (`fx_screener2` — Port `8060`)
Double-click `run_screener2.bat` (or `FX Screener2\run_screener2.bat`) in Windows Explorer, or run:
```powershell
.\run_screener2.bat
```
- **Module**: `fx_screener2` | **Config**: `config_screener2.yaml` | **Default URL**: `http://127.0.0.1:8060`
- Includes **45/30/15/10** Early Trend & Momentum Acceleration weights, **Fast `Donchian(10/5)` + `SuperTrend`**, **`EWO(5,35)` & `MACD Δ²H`**, **`TPO VAH/POC/VAL`**, **4-Trigger ASAP Exit Engine (`⚠ EXIT NOW`)**, and **`🧹 Clean Chart` Mode**.

### 2. Launch **FX Screener 1.0 (Original `main`)** (`fx_screener` — Port `8050`)
Double-click `run_screener.bat` in Windows Explorer, or run:
```powershell
.\run_screener.bat
```
- **Module**: `fx_screener` | **Config**: `config.yaml` | **Default URL**: `http://127.0.0.1:8050`

### 3. Terminal UI (`Textual`) Mode
```powershell
.\run_screener2.bat --tui
```

### 2. Keyboard Shortcuts in the Terminal UI
| Key | Action |
| :--- | :--- |
| `1` | Show **All 30** instruments |
| `2` | Filter **MTF Confluence Only (`★ ALIGNED`)** |
| `3` | Filter **Strong Signals (`|Score| >= 60`)** |
| `4` | Filter **Volatility Squeeze / Breakout** |
| `5` | Filter **Watchlist (`★`)** |
| `/` | Search / Filter by Currency or Signal (e.g., `JPY`, `USD`, `XAU`, `BUY`) |
| `S` | Cycle Sort Column (`Composite Score`, `|Score|`, `D1`, `H4`, `DX(14)`, `D1%`) |
| `W` | Toggle Watchlist star (`★`) on selected pair (saved to `config.yaml`) |
| `A` | Toggle Trade Planner between `D1 ATR(14)` and `H4 ATR(14)` |
| `P` | Open **Risk & Position Sizing Modal** (edit Account Balance, Risk %, SL ATR Multiplier) |
| `R` | Force Refresh market data (bypass SQLite cache TTL) |
| `E` | Export current view to `screener_export.csv` and `screener_export.json` |
| `Q` | Quit |

---

## Headless One-Shot CLI & Export Mode

Print the G8 Currency Strength Matrix and Screener Table directly to the terminal and exit:

```powershell
# Full 30-instrument scan
python -m fx_screener --once

# Filter only MTF Confluence pairs and export to CSV
python -m fx_screener --once --filter confluence --export confluence_setups.csv

# Filter JPY pairs using H4 ATR and a $50,000 account at 0.5% risk
python -m fx_screener --once --currency JPY --atr-tf H4 --balance 50000 --risk 0.5

# Run offline with deterministic synthetic market data
python -m fx_screener --once --provider synthetic
```
