"""Base interface and synthetic generator for Market Data Providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
import pandas as pd

from fx_screener2.universe import ALL_PAIRS, UNIVERSE_MAP


TIMEFRAMES = ("H1", "H4", "D1", "W1")

# Realistic baseline reference prices for synthetic/offline fallback
BASELINE_PRICES: Dict[str, float] = {
    "EUR/USD": 1.0540,
    "GBP/USD": 1.2630,
    "USD/JPY": 154.20,
    "USD/CHF": 0.8840,
    "AUD/USD": 0.6510,
    "USD/CAD": 1.3980,
    "NZD/USD": 0.5860,
    "EUR/GBP": 0.8345,
    "EUR/JPY": 162.53,
    "EUR/CHF": 0.9317,
    "EUR/AUD": 1.6190,
    "EUR/CAD": 1.4735,
    "EUR/NZD": 1.7986,
    "GBP/JPY": 194.75,
    "GBP/CHF": 1.1165,
    "GBP/AUD": 1.9401,
    "GBP/CAD": 1.7657,
    "GBP/NZD": 2.1553,
    "AUD/JPY": 100.38,
    "AUD/NZD": 1.1109,
    "AUD/CAD": 0.9101,
    "AUD/CHF": 0.5755,
    "NZD/JPY": 90.36,
    "NZD/CAD": 0.8192,
    "NZD/CHF": 0.5180,
    "CAD/JPY": 110.30,
    "CAD/CHF": 0.6323,
    "CHF/JPY": 174.43,
    "USD/INR": 84.40,
    "XAU/USD": 2655.40,
    "XAG/USD": 31.15,
}


class MarketDataProvider(ABC):
    """Abstract interface for fetching multi-timeframe OHLCV bars."""

    name: str = "base"

    @abstractmethod
    def fetch_universe(
        self, pairs: List[str] | None = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Return {pair: {'H1': df, 'H4': df, 'D1': df, 'W1': df}} with columns [Open, High, Low, Close, Volume]."""

    def update_live_bars(
        self,
        existing_data: Dict[str, Dict[str, pd.DataFrame]],
        pairs: List[str] | None = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Incrementally merge the latest 1m/1h price ticks into existing H1, H4, D1, W1 DataFrames."""
        return self.fetch_universe(pairs)


class SyntheticProvider(MarketDataProvider):
    """Deterministic multi-factor synthetic OHLCV generator for offline mode & testing.

    Generates coherent G8 currency factor paths so currency strength, cross rates,
    trends, and Bollinger/ATR regimes behave realistically when offline.
    """

    name: str = "synthetic"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fetch_universe(
        self, pairs: List[str] | None = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        target_pairs = pairs or ALL_PAIRS
        rng = np.random.default_rng(self.seed)

        # Assign latent drift & volatility profiles to individual currencies
        # so G8 currency strength and cross-pair trends align naturally
        ccy_drift = {
            "USD": 0.00055,
            "GBP": 0.00038,
            "AUD": 0.00020,
            "CAD": 0.00010,
            "EUR": -0.00025,
            "CHF": -0.00015,
            "NZD": -0.00030,
            "JPY": -0.00060,
            "XAU": 0.00080,
            "XAG": 0.00065,
        }

        result: Dict[str, Dict[str, pd.DataFrame]] = {}
        now = pd.Timestamp.now("UTC").floor("4h")

        for idx, pair in enumerate(target_pairs):
            spec = UNIVERSE_MAP[pair]
            base_price = BASELINE_PRICES.get(pair, 1.0)
            net_drift = ccy_drift.get(spec.base, 0.0) - ccy_drift.get(spec.quote, 0.0)

            # Generate 2400 H1 bars (~100 trading days) for realistic H1 -> H4 -> D1 -> W1 hierarchy
            n_h1 = 2400
            pair_rng = np.random.default_rng(self.seed + idx * 17)

            # Regime-switching volatility so some pairs are in Bollinger Squeeze or Breakout
            vol_base = 0.0011 if spec.category != "Commodity" else 0.0020
            vol_profile = np.full(n_h1, vol_base)
            if idx % 5 == 0:
                vol_profile[-80:] = vol_base * 0.35
            elif idx % 4 == 0:
                vol_profile[-90:-18] = vol_base * 0.45
                vol_profile[-18:] = vol_base * 1.95

            shocks = pair_rng.normal(loc=net_drift * 0.12, scale=vol_profile)
            t = np.linspace(0, 18 * np.pi, n_h1)
            cycle = 0.0002 * np.sin(t + idx)
            returns = shocks + cycle
            if idx % 4 == 0:
                returns[-18:] += np.sign(net_drift or 1.0) * vol_base * 0.95

            close_series = base_price * np.exp(np.cumsum(returns) - np.sum(returns))
            open_series = np.roll(close_series, 1)
            open_series[0] = close_series[0] * (1.0 - returns[0])

            intrabar_range = np.abs(pair_rng.normal(0.0, vol_profile * 0.75, size=n_h1)) * close_series
            high_series = np.maximum(open_series, close_series) + intrabar_range
            low_series = np.minimum(open_series, close_series) - intrabar_range
            volume_series = pair_rng.integers(400, 4500, size=n_h1).astype(float)

            h1_index = pd.date_range(end=now, periods=n_h1, freq="1h", tz="UTC")
            df_h1 = pd.DataFrame(
                {
                    "Open": open_series,
                    "High": high_series,
                    "Low": low_series,
                    "Close": close_series,
                    "Volume": volume_series,
                },
                index=h1_index,
            )

            df_h4 = resample_ohlcv(df_h1, "4h")
            df_d1 = resample_ohlcv(df_h4, "1D")
            df_w1 = resample_ohlcv(df_d1, "1W")

            result[pair] = {
                "H1": df_h1.tail(320),
                "H4": df_h4.tail(260),
                "D1": df_d1.tail(260),
                "W1": df_w1.tail(160),
            }

        return result

    def update_live_bars(
        self,
        existing_data: Dict[str, Dict[str, pd.DataFrame]],
        pairs: List[str] | None = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Simulate a realistic 1-minute live price tick across all pairs when running in synthetic mode."""
        if not existing_data:
            return self.fetch_universe(pairs)
        target_pairs = pairs or list(existing_data.keys())
        self.seed += 1
        rng = np.random.default_rng(self.seed)
        updated: Dict[str, Dict[str, pd.DataFrame]] = {}
        for pair in target_pairs:
            tfs = existing_data.get(pair)
            if not tfs or "H1" not in tfs:
                continue
            spec = UNIVERSE_MAP[pair]
            vol = 0.00025 if spec.category != "Commodity" else 0.00045
            shock = float(rng.normal(0.0, vol))
            new_tfs: Dict[str, pd.DataFrame] = {}
            for tf, df in tfs.items():
                work = df.copy()
                if not work.empty:
                    last_c = float(work["Close"].iloc[-1])
                    new_c = last_c * (1.0 + shock)
                    work.iloc[-1, work.columns.get_loc("Close")] = new_c
                    work.iloc[-1, work.columns.get_loc("High")] = max(float(work["High"].iloc[-1]), new_c)
                    work.iloc[-1, work.columns.get_loc("Low")] = min(float(work["Low"].iloc[-1]), new_c)
                new_tfs[tf] = work
            updated[pair] = new_tfs
        return updated


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame cleanly to a higher timeframe (e.g. '4h', '1D', '1W')."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index, utc=True)
    elif work.index.tz is None:
        work.index = work.index.tz_localize("UTC")
    else:
        work.index = work.index.tz_convert("UTC")

    work = work.sort_index()
    agg = (
        work.resample(rule)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    return agg


def build_clean_daily_and_weekly_ohlcv(
    df_1h: pd.DataFrame,
    df_d1_raw: pd.DataFrame,
    df_w1_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repair Yahoo Finance's broken FX Daily/Weekly candles where Open == Close on every bar.

    1. Resamples true 24h Daily bars (`1D`) directly from the 60-day `1h` intraday bars
       (and drops tiny <2-hour Sunday pre-open stub bars).
    2. For older D1 history (>60 days ago) in `df_d1_raw`, sets each day's `Open` to the
       previous trading day's `Close` whenever Yahoo's `|Open - Close|` is <8% of `High - Low`,
       and ensures `High >= max(Open, Close)` and `Low <= min(Open, Close)`.
    3. Splices the historical D1 bars with the intraday-resampled D1 bars and builds W1 (`W-FRI`)
       so both D1 and W1 charts have proper full-bodied candlesticks.
    """
    # Step 1: Repair raw D1 historical bars (continuous 24h FX market: Open_t = Close_{t-1})
    d1_hist = pd.DataFrame()
    has_broken_opens = False
    if df_d1_raw is not None and not df_d1_raw.empty:
        d1_hist = df_d1_raw.copy().sort_index()
        d1_hist.index = d1_hist.index.floor("1D")
        d1_hist = d1_hist[~d1_hist.index.duplicated(keep="last")]

        if len(d1_hist) >= 2:
            prev_closes = d1_hist["Close"].shift(1)
            ranges = (d1_hist["High"] - d1_hist["Low"]).clip(lower=0.0)
            bodies = (d1_hist["Close"] - d1_hist["Open"]).abs()
            # Yahoo Finance FX bug: Open is identical (or within 1 pip) to Close on almost all bars
            broken_open_mask = (bodies <= (ranges * 0.08)) & prev_closes.notna()
            if broken_open_mask.mean() >= 0.40:
                has_broken_opens = True
                d1_hist = d1_hist[d1_hist.index.dayofweek < 5]
                prev_closes = d1_hist["Close"].shift(1)
                ranges = (d1_hist["High"] - d1_hist["Low"]).clip(lower=0.0)
                bodies = (d1_hist["Close"] - d1_hist["Open"]).abs()
                broken_open_mask = (bodies <= (ranges * 0.08)) & prev_closes.notna()
                d1_hist.loc[broken_open_mask, "Open"] = prev_closes[broken_open_mask]
                d1_hist["High"] = d1_hist[["Open", "High", "Close"]].max(axis=1)
                d1_hist["Low"] = d1_hist[["Open", "Low", "Close"]].min(axis=1)

    if not has_broken_opens and not d1_hist.empty:
        # D1 candles are already properly formed (e.g. OANDA or already repaired cache)
        df_w1_out = df_w1_raw if (df_w1_raw is not None and not df_w1_raw.empty) else resample_ohlcv(d1_hist, "W-FRI")
        return d1_hist, df_w1_out

    # Step 2: Resample recent ~60 days of D1 directly from true 1-hour bars
    d1_from_1h = pd.DataFrame()
    if df_1h is not None and not df_1h.empty:
        work_1h = df_1h.copy().sort_index()
        # Count hours per UTC day so 1-2 hour Sunday 22:00 UTC pre-open stubs don't create tiny fake daily bars
        bar_counts = work_1h["Close"].resample("1D").count()
        agg_1d = resample_ohlcv(work_1h, "1D")
        valid_days = bar_counts[bar_counts >= 3].index
        d1_from_1h = agg_1d.loc[agg_1d.index.intersection(valid_days)]
        d1_from_1h = d1_from_1h[d1_from_1h.index.dayofweek < 5]

    # Step 3: Splice older repaired D1 history with recent 1h-resampled D1 bars
    if not d1_hist.empty and not d1_from_1h.empty:
        cutoff = d1_from_1h.index.min()
        older = d1_hist[d1_hist.index < cutoff]
        df_d1 = pd.concat([older, d1_from_1h]).sort_index()
        df_d1 = df_d1[~df_d1.index.duplicated(keep="last")]
        # Smooth the seam bar's Open with the last historical Close if needed
        seam_idx = df_d1.index.get_loc(cutoff)
        if isinstance(seam_idx, int) and seam_idx > 0:
            prev_c = float(df_d1["Close"].iloc[seam_idx - 1])
            cur_o = float(df_d1["Open"].iloc[seam_idx])
            cur_c = float(df_d1["Close"].iloc[seam_idx])
            cur_rng = float(df_d1["High"].iloc[seam_idx] - df_d1["Low"].iloc[seam_idx])
            if abs(cur_c - cur_o) <= cur_rng * 0.08:
                df_d1.iloc[seam_idx, df_d1.columns.get_loc("Open")] = prev_c
                df_d1.iloc[seam_idx, df_d1.columns.get_loc("High")] = max(
                    float(df_d1["High"].iloc[seam_idx]), prev_c, cur_c
                )
                df_d1.iloc[seam_idx, df_d1.columns.get_loc("Low")] = min(
                    float(df_d1["Low"].iloc[seam_idx]), prev_c, cur_c
                )
    elif not d1_from_1h.empty:
        df_d1 = d1_from_1h
    else:
        df_d1 = d1_hist

    # Step 4: Build/repair W1 candles
    if not df_d1.empty:
        w1_from_d1 = resample_ohlcv(df_d1, "W-FRI")
    else:
        w1_from_d1 = pd.DataFrame()

    if df_w1_raw is not None and not df_w1_raw.empty:
        w1_hist = df_w1_raw.copy().sort_index()
        if len(w1_hist) >= 2:
            prev_w_closes = w1_hist["Close"].shift(1)
            w_ranges = (w1_hist["High"] - w1_hist["Low"]).clip(lower=0.0)
            w_bodies = (w1_hist["Close"] - w1_hist["Open"]).abs()
            broken_w = (w_bodies <= (w_ranges * 0.08)) & prev_w_closes.notna()
            w1_hist.loc[broken_w, "Open"] = prev_w_closes[broken_w]
            w1_hist["High"] = w1_hist[["Open", "High", "Close"]].max(axis=1)
            w1_hist["Low"] = w1_hist[["Open", "Low", "Close"]].min(axis=1)
        if not w1_from_d1.empty:
            w_cutoff = w1_from_d1.index.min()
            w_older = w1_hist[w1_hist.index < w_cutoff]
            df_w1 = pd.concat([w_older, w1_from_d1]).sort_index()
            df_w1 = df_w1[~df_w1.index.duplicated(keep="last")]
        else:
            df_w1 = w1_hist
    else:
        df_w1 = w1_from_d1

    return df_d1, df_w1
