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

    def update_live_bars(
        self,
        existing_data: Dict[str, Dict[str, pd.DataFrame]],
        pairs: List[str] | None = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fast 1-minute live price updater.

        Downloads recent 1h + 1m intraday bars (period='5d'/'1d') for all 31 tickers in a single fast batch,
        merges the live candles into the existing H1 history, and updates the live H4, D1, and W1 candles.
        """
        import yfinance as yf

        if not existing_data:
            return self.fetch_universe(pairs)

        target_pairs = pairs or list(existing_data.keys()) or ALL_PAIRS
        specs = [UNIVERSE_MAP[p] for p in target_pairs if p in UNIVERSE_MAP]
        tickers = list({s.yf_ticker for s in specs})
        single = len(tickers) == 1

        # 1. Fast batch download of recent 5-day 1h bars + 1-day 1m bars for up-to-the-minute spot price
        raw_1h = yf.download(
            tickers=tickers,
            period="5d",
            interval="1h",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        raw_1m = yf.download(
            tickers=tickers,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        updated: Dict[str, Dict[str, pd.DataFrame]] = {
            p: {tf: df.copy() for tf, df in tfs.items()}
            for p, tfs in existing_data.items()
        }

        for spec in specs:
            pair = spec.pair
            if pair not in updated or "H1" not in updated[pair]:
                continue

            recent_1h = self._extract_ticker_df(raw_1h, spec.yf_ticker, single)
            recent_1m = self._extract_ticker_df(raw_1m, spec.yf_ticker, single)

            # Fallback ticker check for metals if needed
            if recent_1h.empty and recent_1m.empty and spec.yf_fallback != spec.yf_ticker:
                fb = spec.yf_fallback
                fb_1m = yf.download(fb, period="1d", interval="1m", progress=False)
                recent_1m = self._extract_ticker_df(fb_1m, fb, True)

            old_h1 = updated[pair]["H1"]
            if not recent_1h.empty:
                merged_h1 = pd.concat([old_h1, recent_1h]).sort_index()
                merged_h1 = merged_h1[~merged_h1.index.duplicated(keep="last")]
            else:
                merged_h1 = old_h1.copy()

            # Incorporate latest 1-minute live bars into the current H1 candle
            if not recent_1m.empty and not merged_h1.empty:
                h1_from_1m = resample_ohlcv(recent_1m, "1h")
                if not h1_from_1m.empty:
                    last_1m_hour = h1_from_1m.index[-1]
                    m_bar = h1_from_1m.iloc[-1]
                    if last_1m_hour in merged_h1.index:
                        loc = merged_h1.index.get_loc(last_1m_hour)
                        if isinstance(loc, int):
                            merged_h1.iloc[loc, merged_h1.columns.get_loc("Close")] = float(m_bar["Close"])
                            merged_h1.iloc[loc, merged_h1.columns.get_loc("High")] = max(
                                float(merged_h1.iloc[loc]["High"]), float(m_bar["High"])
                            )
                            merged_h1.iloc[loc, merged_h1.columns.get_loc("Low")] = min(
                                float(merged_h1.iloc[loc]["Low"]), float(m_bar["Low"])
                            )
                    elif last_1m_hour > merged_h1.index[-1]:
                        merged_h1 = pd.concat([merged_h1, h1_from_1m.tail(1)]).sort_index()
                        merged_h1 = merged_h1[~merged_h1.index.duplicated(keep="last")]

            if merged_h1.empty:
                continue

            # Recompute H4 from merged H1
            recent_h4 = resample_ohlcv(merged_h1, "4h")
            old_h4 = updated[pair].get("H4", pd.DataFrame())
            if not old_h4.empty and not recent_h4.empty:
                merged_h4 = pd.concat([old_h4[old_h4.index < recent_h4.index.min()], recent_h4]).sort_index()
                merged_h4 = merged_h4[~merged_h4.index.duplicated(keep="last")]
            else:
                merged_h4 = recent_h4

            # Update latest D1 and W1 candle with the live price from merged_h1
            live_ts = merged_h1.index[-1]
            live_close = float(merged_h1["Close"].iloc[-1])
            live_high = float(merged_h1["High"].iloc[-1])
            live_low = float(merged_h1["Low"].iloc[-1])

            old_d1 = updated[pair].get("D1", pd.DataFrame()).copy()
            if not old_d1.empty:
                live_day = live_ts.floor("1D")
                last_d1_day = old_d1.index[-1].floor("1D")
                if live_day == last_d1_day or live_ts.dayofweek >= 5:
                    old_d1.iloc[-1, old_d1.columns.get_loc("Close")] = live_close
                    old_d1.iloc[-1, old_d1.columns.get_loc("High")] = max(float(old_d1["High"].iloc[-1]), live_high)
                    old_d1.iloc[-1, old_d1.columns.get_loc("Low")] = min(float(old_d1["Low"].iloc[-1]), live_low)
                elif live_day > last_d1_day and live_ts.dayofweek < 5:
                    today_h1 = merged_h1[merged_h1.index >= live_day]
                    if len(today_h1) >= 1:
                        prev_d1_close = float(old_d1["Close"].iloc[-1])
                        new_d1_row = pd.DataFrame(
                            [
                                {
                                    "Open": prev_d1_close,
                                    "High": max(prev_d1_close, float(today_h1["High"].max())),
                                    "Low": min(prev_d1_close, float(today_h1["Low"].min())),
                                    "Close": live_close,
                                    "Volume": float(today_h1["Volume"].sum()),
                                }
                            ],
                            index=[live_day],
                        )
                        old_d1 = pd.concat([old_d1, new_d1_row]).sort_index()

            old_w1 = updated[pair].get("W1", pd.DataFrame()).copy()
            if not old_w1.empty:
                old_w1.iloc[-1, old_w1.columns.get_loc("Close")] = live_close
                old_w1.iloc[-1, old_w1.columns.get_loc("High")] = max(float(old_w1["High"].iloc[-1]), live_high)
                old_w1.iloc[-1, old_w1.columns.get_loc("Low")] = min(float(old_w1["Low"].iloc[-1]), live_low)

            updated[pair] = {
                "H1": merged_h1.tail(320),
                "H4": merged_h4.tail(260),
                "D1": old_d1.tail(260),
                "W1": old_w1.tail(220),
            }

        return updated
