# Multi-Factor FX & Precious Metals Terminal Screener

A quantitative **Position & Swing (`H4`, `D1`, `W1`)** Forex & Precious Metals Terminal Screener covering all **28 G8 Currency Pairs** (`USD`, `EUR`, `GBP`, `JPY`, `AUD`, `CAD`, `CHF`, `NZD`) plus **Gold (`XAU/USD`)** and **Silver (`XAG/USD`)**.

Built with **Textual** (interactive TUI dashboard) and **Rich** (one-shot CLI & CSV/JSON export).

---

## Features

1. **30-Instrument Universe**:
   - **7 G8 Majors**: `EUR/USD`, `GBP/USD`, `USD/JPY`, `USD/CHF`, `AUD/USD`, `USD/CAD`, `NZD/USD`
   - **21 G8 Minor Crosses**: All cross combinations of `EUR`, `GBP`, `AUD`, `NZD`, `CAD`, `CHF`, `JPY`
   - **2 Precious Metals**: `XAU/USD` (Gold) and `XAG/USD` (Silver)
2. **Weighted 4-Pillar Composite Score (`-100` to `+100`)**:
   - **Trend (35%)**: `EMA 20`, `EMA 50`, `EMA 200` stack alignment, `EMA 20` slope, `DI+/DI-`, and `ADX(14)` conviction multiplier.
   - **Currency Strength (25%)**: True G8 relative strength matrix (`-100` to `+100`) computed across all 28 pairs, scoring each pair by `Base Score - Quote Score`.
   - **Momentum (25%)**: `RSI(14)` directional momentum zones + `MACD(12, 26, 9)` histogram impulse & signal crossover.
   - **Volatility & Structure (15%)**: `Bollinger Bands(20, 2)` bandwidth percentile rank (`SQUEEZE` / `BREAKOUT`) and `ATR(14)` expansion ratio.
3. **Swing-Centric Multi-Timeframe Confluence**:
   - Combines `D1` (**50%**), `H4` (**30%**), and `W1` (**20%**).
   - Flags `★ BULL ALIGNED` or `★ BEAR ALIGNED` when all three timeframes align directionally (`>= +20` or `<= -20`).
4. **ATR Trade Planner & Cross-Currency Position Sizer**:
   - Computes Entry, `1.5 × ATR` Stop Loss (in price and pips), and `1.5R`, `2.0R`, `3.0R` Take-Profit levels.
   - Converts quote-currency pip values back to `USD` across majors, `USD/XXX`, cross pairs (e.g. `GBP/JPY`, `EUR/AUD`), and metals (`XAU/USD`, `XAG/USD`) to output exact **Standard Lots** and **Units** for your account balance and risk % (`$10,000 @ 1.0%` default).
5. **Local SQLite OHLCV Cache & Pluggable Data Feeds**:
   - Free no-key default feed via `yfinance` (`1h` resampled to `H4`, `1d`, `1wk`) cached in `.fx_screener_cache.sqlite3`, plus pluggable `OANDA v20` REST API and deterministic offline `synthetic` provider.

---

## Quick Start

### 1. Launch the Interactive Web UI (Default)
Double-click `run_screener.bat` in Windows Explorer or run:
```powershell
.\run_screener.bat
```
This starts the FastAPI + TradingView Lightweight Charts UI at `http://127.0.0.1:8050` and opens it in your browser.
- **Click any FX pair or metal** in the screener table (or top divergence chips) to load its interactive candlestick chart.
- **Switch timeframes** from **W1 (Weekly)** → **D1 (Daily)** → **H4 (4-Hour)** → **H1 (Hourly)**.
- **Toggle overlays**: `EMA 20/50/200`, `Bollinger Bands (20, 2)`, `Entry / Stop-Loss / TP1 / TP2` price lines, and switch the subchart between `RSI (14)` and `MACD (12, 26, 9)`.
- **Click any G8 Currency card** (`USD`, `EUR`, `GBP`, `JPY`, etc.) to filter all pairs containing that currency.

### 2. Launch the Terminal UI (`Textual`) in PowerShell
```powershell
.\run_screener.bat --tui
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
| `S` | Cycle Sort Column (`Composite Score`, `|Score|`, `D1`, `H4`, `ADX`, `D1%`) |
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
