"""Pluggable OANDA v20 REST API Market Data Provider."""

from __future__ import annotations

from typing import Dict, List
import pandas as pd
import requests

from fx_screener2.data.base import MarketDataProvider
from fx_screener2.universe import ALL_PAIRS, UNIVERSE_MAP


class OandaProvider(MarketDataProvider):
    """Fetches H4, D1, and W1 candles directly from OANDA v20 REST API."""

    name: str = "oanda"

    GRANULARITY_MAP = {
        "H1": "H1",
        "H4": "H4",
        "D1": "D",
        "W1": "W",
    }

    def __init__(self, api_key: str, environment: str = "practice") -> None:
        self.api_key = api_key
        if environment.lower() == "live":
            self.base_url = "https://api-fxtrade.oanda.com/v3"
        else:
            self.base_url = "https://api-fxpractice.oanda.com/v3"

    def _fetch_candles(self, instrument: str, granularity: str, count: int = 250) -> pd.DataFrame:
        url = f"{self.base_url}/instruments/{instrument}/candles"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        rows = []
        for c in payload.get("candles", []):
            mid = c.get("mid", {})
            rows.append(
                {
                    "ts": pd.to_datetime(c["time"], utc=True),
                    "Open": float(mid["o"]),
                    "High": float(mid["h"]),
                    "Low": float(mid["l"]),
                    "Close": float(mid["c"]),
                    "Volume": float(c.get("volume", 0.0)),
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("ts").sort_index()

    def fetch_universe(
        self, pairs: List[str] | None = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        if not self.api_key:
            raise ValueError("OANDA API key is required for OandaProvider.")

        target_pairs = pairs or ALL_PAIRS
        result: Dict[str, Dict[str, pd.DataFrame]] = {}
        for pair in target_pairs:
            spec = UNIVERSE_MAP[pair]
            tfs: Dict[str, pd.DataFrame] = {}
            for tf, gran in self.GRANULARITY_MAP.items():
                df = self._fetch_candles(spec.oanda_instrument, gran, count=250)
                if not df.empty:
                    tfs[tf] = df
            if len(tfs) >= 3:
                result[pair] = tfs
        return result
