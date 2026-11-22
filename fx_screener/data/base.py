"""Base interface and synthetic generator for Market Data Providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
import pandas as pd

from fx_screener.universe import ALL_PAIRS, UNIVERSE_MAP


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
        """Return {pair: {'H4': df, 'D1': df, 'W1': df}} with columns [Open, High, Low, Close, Volume]."""


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
