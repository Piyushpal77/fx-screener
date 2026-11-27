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
        # If partial fetch, prefer stale SQLite cache for missing pairs before synthetic fallback
        synthetic_pairs: List[str] = []
        if len(fresh) < len(target_pairs):
            missing = [p for p in target_pairs if p not in fresh]
            if missing:
                stale_cached = cache.load_universe(missing)
                for p, tfs in stale_cached.items():
                    fresh[p] = tfs
                still_missing = [p for p in missing if p not in fresh]
                if still_missing:
                    synth = SyntheticProvider().fetch_universe(still_missing)
                    for p, tfs in synth.items():
                        for df in tfs.values():
                            df.attrs["is_synthetic"] = True
                        fresh[p] = tfs
                        synthetic_pairs.append(p)
        # Only cache non-synthetic pairs under the live provider name
        to_cache = {p: tfs for p, tfs in fresh.items() if p not in synthetic_pairs}
        if to_cache:
            cache.save_universe(to_cache, provider_name=provider.name)
        return fresh, provider.name
    except Exception as exc:
        logger.warning("Primary provider '%s' failed (%s). Checking cache/fallback.", provider.name, exc)
        cached = cache.load_universe(target_pairs)
        if cached and len(cached) == len(target_pairs):
            _, prov = cache.last_updated_info()
            return cached, f"stale-cache ({prov})"

        synth_data = SyntheticProvider().fetch_universe(target_pairs)
        for tfs in synth_data.values():
            for df in tfs.values():
                df.attrs["is_synthetic"] = True
        return synth_data, "synthetic-fallback"


def update_live_universe(
    config: AppConfig,
    existing_data: Dict[str, Dict[str, pd.DataFrame]],
    pairs: List[str] | None = None,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], str]:
    """Incrementally fetch 1-minute live price updates and merge into existing OHLCV bars."""
    target_pairs = pairs or ALL_PAIRS
    if not existing_data:
        return load_or_fetch_universe(config, force_refresh=True, pairs=target_pairs)

    provider = get_provider(config)
    cache = MarketDataCache(ttl_seconds=config.provider.cache_ttl_seconds)
    try:
        updated = provider.update_live_bars(existing_data, target_pairs)
        to_cache = {
            p: tfs
            for p, tfs in updated.items()
            if not any(df.attrs.get("is_synthetic", False) for df in tfs.values())
        }
        if to_cache:
            cache.save_universe(to_cache, provider_name=provider.name)
        return updated, f"{provider.name} (live 1m)"
    except Exception as exc:
        logger.warning("Live 1m update failed (%s); retaining current bars.", exc)
        return existing_data, f"{provider.name} (cached)"


__all__ = [
    "MarketDataProvider",
    "SyntheticProvider",
    "YFinanceProvider",
    "OandaProvider",
    "MarketDataCache",
    "get_provider",
    "load_or_fetch_universe",
    "update_live_universe",
]
