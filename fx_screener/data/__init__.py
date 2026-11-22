"""Dataproviders and cache orchestrator."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import pandas as pd

from fx_screener.config import AppConfig
from fx_screener.data.base import MarketDataProvider, SyntheticProvider
from fx_screener.data.cache import MarketDataCache
from fx_screener.data.oanda_provider import OandaProvider
from fx_screener.data.yfinance_provider import YFinanceProvider
from fx_screener.universe import ALL_PAIRS

logger = logging.getLogger(__name__)


def get_provider(config: AppConfig) -> MarketDataProvider:
    active = config.provider.active.lower()
    if active == "oanda" and config.provider.oanda_api_key:
        return OandaProvider(
            api_key=config.provider.oanda_api_key,
            environment=config.provider.oanda_environment,
        )
    if active == "synthetic":
        return SyntheticProvider()
    return YFinanceProvider()


def load_or_fetch_universe(
    config: AppConfig,
    force_refresh: bool = False,
    pairs: List[str] | None = None,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], str]:
    """Load universe OHLCV data from SQLite cache or fetch from active provider.

    Falls back gracefully to cached data or synthetic data if offline.
    Returns (universe_data, source_description).
    """
    target_pairs = pairs or ALL_PAIRS
    cache = MarketDataCache(ttl_seconds=config.provider.cache_ttl_seconds)

    if not force_refresh and cache.is_fresh(target_pairs):
        cached = cache.load_universe(target_pairs)
        if len(cached) >= len(target_pairs):
            _, prov = cache.last_updated_info()
            return cached, f"cache ({prov})"

    provider = get_provider(config)
    try:
        fresh = provider.fetch_universe(target_pairs)
        # If partial fetch, fill any missing pairs from synthetic fallback so all 30 render
        if len(fresh) < len(target_pairs):
            missing = [p for p in target_pairs if p not in fresh]
            if missing:
                synth = SyntheticProvider().fetch_universe(missing)
                fresh.update(synth)
        cache.save_universe(fresh, provider_name=provider.name)
        return fresh, provider.name
    except Exception as exc:
        logger.warning("Primary provider '%s' failed (%s). Checking cache/fallback.", provider.name, exc)
        cached = cache.load_universe(target_pairs)
        if cached and len(cached) == len(target_pairs):
            _, prov = cache.last_updated_info()
            return cached, f"stale-cache ({prov})"

        synth_data = SyntheticProvider().fetch_universe(target_pairs)
        cache.save_universe(synth_data, provider_name="synthetic-fallback")
        return synth_data, "synthetic-fallback"


__all__ = [
    "MarketDataProvider",
    "SyntheticProvider",
    "YFinanceProvider",
    "OandaProvider",
    "MarketDataCache",
    "get_provider",
    "load_or_fetch_universe",
]
