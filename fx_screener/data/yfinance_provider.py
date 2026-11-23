"""Yahoo Finance (yfinance) batch OHLCV provider with 1h -> H4 resampling."""

from __future__ import annotations

import logging
from typing import Dict, List
import pandas as pd

from fx_screener.data.base import (
    MarketDataProvider,
    build_clean_daily_and_weekly_ohlcv,
    resample_ohlcv,
)
from fx_screener.universe import ALL_PAIRS, UNIVERSE_MAP

logger = logging.getLogger(__name__)


class YFinanceProvider(MarketDataProvider):
    """Fetches H4 (resampled from 1h), D1, and W1 OHLCV bars from Yahoo Finance."""

    name: str = "yfinance"

    def _extract_ticker_df(self, raw: pd.DataFrame, ticker: str, single_ticker: bool) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        try:
            if single_ticker and not isinstance(raw.columns, pd.MultiIndex):
                df = raw.copy()
            elif isinstance(raw.columns, pd.MultiIndex):
                level0 = raw.columns.get_level_values(0)
                level1 = raw.columns.get_level_values(1)
                if ticker in level0:
                    df = raw[ticker].copy()
                elif ticker in level1:
                    df = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    return pd.DataFrame()
            else:
                df = raw.copy()
        except Exception:
            return pd.DataFrame()

        cols = {c.capitalize(): c for c in df.columns}
        required = ["Open", "High", "Low", "Close"]
        if not all(r in cols for r in required):
            return pd.DataFrame()

        clean = pd.DataFrame(index=df.index)
        for r in required:
            clean[r] = pd.to_numeric(df[cols[r]], errors="coerce")
        if "Volume" in cols:
            clean["Volume"] = pd.to_numeric(df[cols["Volume"]], errors="coerce").fillna(0.0)
        else:
            clean["Volume"] = 0.0

        clean = clean.dropna(subset=["Open", "High", "Low", "Close"])
        if clean.empty:
            return clean

        if not isinstance(clean.index, pd.DatetimeIndex):
            clean.index = pd.to_datetime(clean.index, utc=True)
        elif clean.index.tz is None:
            clean.index = clean.index.tz_localize("UTC")
        else:
            clean.index = clean.index.tz_convert("UTC")

        return clean.sort_index()

    def fetch_universe(
        self, pairs: List[str] | None = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        import yfinance as yf

        target_pairs = pairs or ALL_PAIRS
        specs = [UNIVERSE_MAP[p] for p in target_pairs]
        tickers = list({s.yf_ticker for s in specs})
        single = len(tickers) == 1

        # 1. Batch download 1h data (60 days gives ~360 H4 bars, ideal for EMA 200)
        raw_1h = yf.download(
            tickers=tickers,
            period="60d",
            interval="1h",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        # 2. Batch download 1d data (2 years gives ~520 D1 bars)
        raw_1d = yf.download(
            tickers=tickers,
            period="2y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        # 3. Batch download 1wk data (5 years gives ~260 W1 bars)
        raw_1wk = yf.download(
            tickers=tickers,
            period="5y",
            interval="1wk",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        result: Dict[str, Dict[str, pd.DataFrame]] = {}
        for spec in specs:
            df_1h = self._extract_ticker_df(raw_1h, spec.yf_ticker, single)
            df_d1 = self._extract_ticker_df(raw_1d, spec.yf_ticker, single)
            df_w1 = self._extract_ticker_df(raw_1wk, spec.yf_ticker, single)

            # Fallback ticker check for Gold/Silver if primary ticker missing
            if (df_d1.empty or df_1h.empty) and spec.yf_fallback != spec.yf_ticker:
                fb = spec.yf_fallback
                if df_1h.empty:
                    fb_1h = yf.download(fb, period="60d", interval="1h", progress=False)
                    df_1h = self._extract_ticker_df(fb_1h, fb, True)
                if df_d1.empty:
                    fb_1d = yf.download(fb, period="2y", interval="1d", progress=False)
                    df_d1 = self._extract_ticker_df(fb_1d, fb, True)
                if df_w1.empty:
                    fb_1wk = yf.download(fb, period="5y", interval="1wk", progress=False)
                    df_w1 = self._extract_ticker_df(fb_1wk, fb, True)

            df_h4 = resample_ohlcv(df_1h, "4h") if not df_1h.empty else pd.DataFrame()
            df_d1, df_w1 = build_clean_daily_and_weekly_ohlcv(df_1h, df_d1, df_w1)

            if not df_h4.empty and not df_d1.empty and not df_w1.empty:
                result[spec.pair] = {
                    "H1": df_1h.tail(320),
                    "H4": df_h4.tail(260),
                    "D1": df_d1.tail(260),
                    "W1": df_w1.tail(220),
                }

        return result
