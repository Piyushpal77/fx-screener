"""FastAPI Web Server and Interactive Charting API for the FX & Precious Metals Screener."""

from __future__ import annotations

from dataclasses import asdict
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fx_screener2.config import AppConfig
from fx_screener2.data import load_or_fetch_universe, update_live_universe
from fx_screener2.engine.indicators import (
    bollinger_bands,
    compute_indicators,
    compute_tpo_profile,
    donchian_channels,
    elliott_wave_oscillator,
    ema,
    hma,
    macd,
    rsi,
    supertrend,
)
from fx_screener2.engine.scorer import (
    PairAnalysis,
    ScreenerResult,
    analyze_universe,
    score_momentum_pillar,
    score_trend_pillar,
    score_volatility_pillar,
)
from fx_screener2.engine.strength import get_pair_strength_pillar_score
from fx_screener2.engine.tracker import SignalTracker, TrackedSignal
from fx_screener2.universe import UNIVERSE_MAP


STATIC_DIR = Path(__file__).resolve().parent / "static"


class RiskUpdatePayload(BaseModel):
    account_balance_usd: float
    risk_percent: float
    atr_timeframe: str
    sl_atr_multiplier: float


class WatchlistTogglePayload(BaseModel):
    pair: str


class ScreenerWebState:
    def __init__(
        self,
        config: AppConfig | None = None,
        tracker: SignalTracker | None = None,
        start_worker: bool = True,
    ) -> None:
        self.lock = threading.Lock()
        self.config = config or AppConfig.load()
        self.tracker = tracker or SignalTracker()
        self.universe_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.result: ScreenerResult | None = None
        self.tracked_signals: List[TrackedSignal] = []
        self.last_updated: str = "-"
        self.last_refresh_epoch: float = 0.0
        self._stop_event = threading.Event()
        self.refresh(force=False)
        if start_worker and self.config.provider.active.lower() != "synthetic":
            self._worker_thread = threading.Thread(
                target=self._live_minute_worker, daemon=True, name="fx-live-1m-worker"
            )
            self._worker_thread.start()
            self._quote_thread = threading.Thread(
                target=self._live_quote_stream_worker, daemon=True, name="fx-live-quote-streamer"
            )
            self._quote_thread.start()

    def _live_quote_stream_worker(self) -> None:
        """Fast 15-second live quote streamer that updates spot prices in-memory between 1-minute full bar recalculations."""
        from fx_screener2.data import get_provider

        while not self._stop_event.wait(15.0):
            try:
                provider = get_provider(self.config)
                fetch_fn = getattr(provider, "fetch_live_spot_quotes", None)
                if not callable(fetch_fn) or not self.universe_data:
                    continue
                quotes = fetch_fn()
                if not quotes:
                    continue
                with self.lock:
                    for pair, spot_price in quotes.items():
                        tfs = self.universe_data.get(pair)
                        if not tfs:
                            continue
                        for tf_df in tfs.values():
                            if tf_df is not None and not tf_df.empty:
                                tf_df.iloc[-1, tf_df.columns.get_loc("Close")] = float(spot_price)
                                tf_df.iloc[-1, tf_df.columns.get_loc("High")] = max(
                                    float(tf_df["High"].iloc[-1]), float(spot_price)
                                )
                                tf_df.iloc[-1, tf_df.columns.get_loc("Low")] = min(
                                    float(tf_df["Low"].iloc[-1]), float(spot_price)
                                )
                    if self.result:
                        for p in self.result.pairs:
                            sp = quotes.get(p.spec.pair)
                            if sp is not None and sp > 0:
                                prev_d1 = p.d1.indicators.prev_close
                                p.price = float(sp)
                                if prev_d1 > 0:
                                    p.change_d1_pct = (float(sp) - prev_d1) / prev_d1 * 100.0
                        for sig in self.tracked_signals:
                            sp = quotes.get(sig.pair)
                            if sp is not None and sp > 0 and sig.status in ("ACTIVE", "TP1_HIT"):
                                spec = UNIVERSE_MAP.get(sig.pair)
                                dec = spec.decimals if spec else 5
                                sig.current_h4_price = round(float(sp), dec)
                    self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

    def _live_minute_worker(self) -> None:
        """Background daemon that updates live FX & Metal prices and recalculates scores every 60s."""
        while not self._stop_event.wait(60.0):
            try:
                self.refresh_live_tick()
            except Exception:
                pass

    def refresh_live_tick(self) -> None:
        """Incrementally fetch 1-minute live price updates and recalculate all 31 instruments."""
        new_data, source = update_live_universe(self.config, self.universe_data)
        new_result = analyze_universe(new_data, self.config, source=source)
        new_tracked = self.tracker.sync_and_evaluate(
            new_result.pairs, new_data, source=source
        )
        with self.lock:
            self.universe_data = new_data
            self.result = new_result
            self.tracked_signals = new_tracked
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_refresh_epoch = datetime.now().timestamp()

    def refresh(self, force: bool = False) -> None:
        with self.lock:
            self.universe_data, source = load_or_fetch_universe(
                self.config, force_refresh=force
            )
            self.result = analyze_universe(
                self.universe_data, self.config, source=source
            )
            self.tracked_signals = self.tracker.sync_and_evaluate(
                self.result.pairs, self.universe_data, source=source
            )
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_refresh_epoch = datetime.now().timestamp()

    def ensure_fresh(self) -> None:
        ttl = max(30, int(self.config.provider.auto_refresh_seconds or 60))
        now_ts = datetime.now().timestamp()
        if getattr(self, "last_refresh_epoch", 0.0) + ttl <= now_ts:
            self.refresh_live_tick()

    def recompute(self) -> None:
        with self.lock:
            if not self.universe_data or not self.result:
                return
            self.result = analyze_universe(
                self.universe_data, self.config, source=self.result.source
            )
            self.tracked_signals = self.tracker.sync_and_evaluate(
                self.result.pairs, self.universe_data, source=self.result.source
            )


def _serialize_timeframe_score(tf_score: Any) -> Dict[str, Any]:
    ind = tf_score.indicators
    return {
        "timeframe": tf_score.timeframe,
        "trend_score": round(tf_score.trend_score, 1),
        "strength_score": round(tf_score.strength_score, 1),
        "momentum_score": round(tf_score.momentum_score, 1),
        "volatility_score": round(tf_score.volatility_score, 1),
        "total_score": round(tf_score.total_score, 1),
        "bias": tf_score.bias,
        "indicators": {
            "close": ind.close,
            "change_pct": round(ind.change_pct, 2),
            "ema20": ind.ema20,
            "ema50": ind.ema50,
            "ema200": ind.ema200,
            "hma9": ind.hma9,
            "hma21": ind.hma21,
            "hma_status": ind.hma_status,
            "hma_bull_cross": ind.hma_bull_cross,
            "hma_bear_cross": ind.hma_bear_cross,
            "dx14": round(ind.dx14, 1),
            "ravi": round(ind.ravi, 2),
            "vhf14": round(ind.vhf14, 2),
            "chop14": round(ind.chop14, 1),
            "adx14": round(ind.adx14, 1),
            "plus_di": round(ind.plus_di, 1),
            "minus_di": round(ind.minus_di, 1),
            "rsi14": round(ind.rsi14, 1),
            "macd_hist": ind.macd_hist,
            "macd_accel_d2": ind.macd_accel_d2,
            "ewo": round(ind.ewo, 2),
            "wave_bias": ind.wave_bias,
            "donchian10_high": ind.donchian10_high,
            "donchian10_low": ind.donchian10_low,
            "donchian5_high": ind.donchian5_high,
            "donchian5_low": ind.donchian5_low,
            "supertrend_line": ind.supertrend_line,
            "supertrend_dir": ind.supertrend_dir,
            "tpo_poc": ind.tpo_poc,
            "tpo_vah": ind.tpo_vah,
            "tpo_val": ind.tpo_val,
            "ttm_squeeze_on": ind.ttm_squeeze_on,
            "ttm_squeeze_fired": ind.ttm_squeeze_fired,
            "atr14": ind.atr14,
            "atr_pct": round(ind.atr_pct, 2),
            "bb_width_pct_rank": round(ind.bb_width_pct_rank, 1),
            "vol_regime": ind.vol_regime,
        },
    }


def _serialize_pair(p: PairAnalysis) -> Dict[str, Any]:
    dec = p.spec.decimals
    tp = p.trade_plan
    return {
        "pair": p.spec.pair,
        "base": p.spec.base,
        "quote": p.spec.quote,
        "category": p.spec.category,
        "decimals": dec,
        "pip_size": p.spec.pip_size,
        "price": round(p.price, dec),
        "change_d1_pct": round(p.change_d1_pct, 2),
        "raw_composite_score": round(p.raw_composite_score, 1),
        "hma_h4_boost": round(p.hma_h4_boost, 1),
        "hma_trigger": p.hma_trigger,
        "is_fresh_hma_entry": p.is_fresh_hma_entry,
        "is_early_entry": p.is_early_entry,
        "top_pick_rank": p.top_pick_rank,
        "top_pick_label": p.top_pick_label,
        "asap_exit_flag": p.asap_exit_flag,
        "asap_exit_reason": p.asap_exit_reason,
        "donchian5_trail_stop": round(p.donchian5_trail_stop, dec),
        "composite_score": round(p.composite_score, 1),
        "signal": p.signal,
        "confluence": p.confluence,
        "confluence_label": p.confluence_label,
        "vol_regime": p.vol_regime,
        "is_watchlist": p.is_watchlist,
        "is_alert": p.is_alert,
        "pillars": {
            "trend": round(p.pillar_trend, 1),
            "strength": round(p.pillar_strength, 1),
            "momentum": round(p.pillar_momentum, 1),
            "volatility": round(p.pillar_volatility, 1),
        },
        "h1": _serialize_timeframe_score(p.h1) if p.h1 else None,
        "h4": _serialize_timeframe_score(p.h4),
        "d1": _serialize_timeframe_score(p.d1),
        "w1": _serialize_timeframe_score(p.w1),
        "sparkline_d1": [round(v, dec) for v in p.d1_sparkline_closes],
        "sparkline_h4": [round(v, dec) for v in p.h4_sparkline_closes],
        "trade_plan": {
            "direction": tp.direction,
            "atr_timeframe": tp.atr_timeframe,
            "stop_mode": tp.stop_mode,
            "entry_price": round(tp.entry_price, dec),
            "hma21_h4_price": round(tp.hma21_h4_price, dec),
            "atr_value": round(tp.atr_value, dec),
            "atr_pips": round(tp.atr_pips, 1),
            "stop_loss_price": round(tp.stop_loss_price, dec),
            "stop_loss_pips": round(tp.stop_loss_pips, 1),
            "tp1_price": round(tp.tp1_price, dec),
            "tp1_pips": round(tp.tp1_pips, 1),
            "tp1_r": tp.tp1_r,
            "tp2_price": round(tp.tp2_price, dec),
            "tp2_pips": round(tp.tp2_pips, 1),
            "tp2_r": tp.tp2_r,
            "tp3_price": round(tp.tp3_price, dec),
            "tp3_pips": round(tp.tp3_pips, 1),
            "tp3_r": tp.tp3_r,
            "account_balance_usd": tp.account_balance_usd,
            "risk_percent": tp.risk_percent,
            "risk_amount_usd": round(tp.risk_amount_usd, 2),
            "standard_lots": round(tp.standard_lots, 2),
            "mini_lots": round(tp.mini_lots, 2),
            "micro_lots": round(tp.micro_lots, 1),
            "units": tp.units,
        },
    }


def create_web_app(
    config: AppConfig | None = None,
    tracker: SignalTracker | None = None,
) -> FastAPI:
    app = FastAPI(
        title="FX & Precious Metals Multi-Factor Screener UI",
        version="1.1.0",
    )
    state = ScreenerWebState(config=config, tracker=tracker)
    app.state.screener = state

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/api/screener")
    def get_screener() -> Dict[str, Any]:
        state.ensure_fresh()
        res = state.result
        if res is None:
            raise HTTPException(status_code=503, detail="Screener state not initialized")

        return {
            "source": res.source,
            "last_updated": state.last_updated,
            "auto_refresh_seconds": state.config.provider.auto_refresh_seconds,
            "risk_config": {
                "account_balance_usd": state.config.risk.account_balance_usd,
                "risk_percent": state.config.risk.risk_percent,
                "atr_timeframe": state.config.risk.atr_timeframe,
                "sl_atr_multiplier": state.config.risk.sl_atr_multiplier,
            },
            "currency_strength": {
                "ranked": [
                    {
                        "rank": e.rank,
                        "currency": e.currency,
                        "composite_score": round(e.composite_score, 1),
                        "h4_score": round(e.h4_score, 1),
                        "d1_score": round(e.d1_score, 1),
                        "w1_score": round(e.w1_score, 1),
                    }
                    for e in res.strength_matrix.ranked
                ],
                "top_pairings": [
                    {"pair": p, "direction": d, "spread": round(s, 1)}
                    for p, d, s in res.strength_matrix.top_pairings
                ],
            },
            "pairs": [_serialize_pair(p) for p in res.pairs],
            "alerts": res.alert_pairs,
            "tracker": {
                "summary": state.tracker.summary_stats(state.tracked_signals),
                "signals": [asdict(s) for s in state.tracked_signals],
            },
        }

    @app.get("/api/quotes")
    def get_live_quotes() -> Dict[str, Any]:
        """Lightweight streaming endpoint returning live spot prices, D1% changes, and current candle OHLC for the active pair."""
        res = state.result
        if res is None:
            raise HTTPException(status_code=503, detail="Screener state not initialized")
        return {
            "last_updated": state.last_updated,
            "quotes": {
                p.spec.pair: {
                    "price": round(p.price, p.spec.decimals),
                    "change_d1_pct": round(p.change_d1_pct, 2),
                }
                for p in res.pairs
            },
        }

    @app.get("/api/tracker/csv")
    def download_tracker_csv() -> FileResponse:
        csv_file = state.tracker.export_csv(state.tracked_signals)
        return FileResponse(
            str(csv_file),
            media_type="text/csv",
            filename="strong_signals_tracker_v2.csv",
        )

    @app.get("/api/chart")
    def get_chart(
        pair: str = Query(..., description="Instrument pair e.g. EUR/USD"),
        tf: str = Query("D1", description="Timeframe: W1, D1, H4, or H1"),
    ) -> Dict[str, Any]:
        pair_key = pair.strip().upper()
        tf_key = tf.strip().upper()
        if pair_key not in UNIVERSE_MAP:
            raise HTTPException(status_code=404, detail=f"Unknown pair: {pair_key}")
        if tf_key not in ("W1", "D1", "H4", "H1"):
            raise HTTPException(status_code=400, detail=f"Invalid timeframe: {tf_key}")

        pair_tfs = state.universe_data.get(pair_key, {})
        df = pair_tfs.get(tf_key)
        # Fallback if H1 wasn't in an older cache file
        if (df is None or df.empty) and tf_key == "H1":
            state.refresh(force=True)
            df = state.universe_data.get(pair_key, {}).get(tf_key)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No {tf_key} data for {pair_key}")

        spec = UNIVERSE_MAP[pair_key]
        dec = spec.decimals
        work = df.copy().sort_index()
        work = work[~work.index.duplicated(keep="last")]
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
        close_s = work["Close"].astype(float)

        ema20_s = ema(close_s, 20)
        ema50_s = ema(close_s, 50)
        ema200_s = ema(close_s, min(200, max(20, len(close_s))))
        hma9_s = hma(close_s, 9)
        hma21_s = hma(close_s, 21)
        dc10_u, dc10_l, _ = donchian_channels(work, 10)
        dc5_u, dc5_l, _ = donchian_channels(work, 5)
        st_line_s, st_dir_s = supertrend(work, 10, 2.5)
        bb_u, bb_m, bb_l, _ = bollinger_bands(close_s, 20, 2.0)
        rsi_s = rsi(close_s, 14)
        macd_l, macd_sig, macd_h = macd(close_s, 12, 26, 9)
        ewo_s = elliott_wave_oscillator(work, 5, 35)
        tpo_poc, tpo_vah, tpo_val = compute_tpo_profile(work, lookback=60, num_bins=28)

        candles: List[Dict[str, Any]] = []
        ema20_pts: List[Dict[str, Any]] = []
        ema50_pts: List[Dict[str, Any]] = []
        ema200_pts: List[Dict[str, Any]] = []
        hma9_pts: List[Dict[str, Any]] = []
        hma21_pts: List[Dict[str, Any]] = []
        dc10_upper_pts: List[Dict[str, Any]] = []
        dc10_lower_pts: List[Dict[str, Any]] = []
        dc5_upper_pts: List[Dict[str, Any]] = []
        dc5_lower_pts: List[Dict[str, Any]] = []
        supertrend_pts: List[Dict[str, Any]] = []
        bb_upper_pts: List[Dict[str, Any]] = []
        bb_lower_pts: List[Dict[str, Any]] = []
        rsi_pts: List[Dict[str, Any]] = []
        macd_hist_pts: List[Dict[str, Any]] = []
        macd_line_pts: List[Dict[str, Any]] = []
        macd_sig_pts: List[Dict[str, Any]] = []
        ewo_pts: List[Dict[str, Any]] = []

        seen_ts: set[int] = set()
        for idx, row in work.iterrows():
            ts = int(pd.Timestamp(idx).timestamp())
            if ts in seen_ts:
                continue
            seen_ts.add(ts)
            o = round(float(row["Open"]), dec)
            h = round(float(row["High"]), dec)
            l = round(float(row["Low"]), dec)
            c = round(float(row["Close"]), dec)
            v = float(row.get("Volume", 0.0))

            candles.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
            e20 = float(ema20_s.loc[idx])
            e50 = float(ema50_s.loc[idx])
            e200 = float(ema200_s.loc[idx])
            h9 = float(hma9_s.loc[idx])
            h21 = float(hma21_s.loc[idx])
            if np.isfinite(e20):
                ema20_pts.append({"time": ts, "value": round(e20, dec)})
            if np.isfinite(e50):
                ema50_pts.append({"time": ts, "value": round(e50, dec)})
            if np.isfinite(e200):
                ema200_pts.append({"time": ts, "value": round(e200, dec)})
            if np.isfinite(h9):
                hma9_pts.append({"time": ts, "value": round(h9, dec)})
            if np.isfinite(h21):
                hma21_pts.append({"time": ts, "value": round(h21, dec)})

            d10u = float(dc10_u.loc[idx])
            d10l = float(dc10_l.loc[idx])
            d5u = float(dc5_u.loc[idx])
            d5l = float(dc5_l.loc[idx])
            stl = float(st_line_s.loc[idx])
            std = int(st_dir_s.loc[idx])
            if np.isfinite(d10u) and np.isfinite(d10l):
                dc10_upper_pts.append({"time": ts, "value": round(d10u, dec)})
                dc10_lower_pts.append({"time": ts, "value": round(d10l, dec)})
            if np.isfinite(d5u) and np.isfinite(d5l):
                dc5_upper_pts.append({"time": ts, "value": round(d5u, dec)})
                dc5_lower_pts.append({"time": ts, "value": round(d5l, dec)})
            if np.isfinite(stl):
                supertrend_pts.append(
                    {
                        "time": ts,
                        "value": round(stl, dec),
                        "color": "#10b981" if std >= 0 else "#ef4444",
                    }
                )

            bu = float(bb_u.loc[idx])
            bl = float(bb_l.loc[idx])
            if np.isfinite(bu) and np.isfinite(bl):
                bb_upper_pts.append({"time": ts, "value": round(bu, dec)})
                bb_lower_pts.append({"time": ts, "value": round(bl, dec)})
            else:
                bb_upper_pts.append({"time": ts})
                bb_lower_pts.append({"time": ts})

            rv = float(rsi_s.loc[idx])
            if np.isfinite(rv):
                rsi_pts.append({"time": ts, "value": round(rv, 2)})
            else:
                rsi_pts.append({"time": ts})

            mh = float(macd_h.loc[idx])
            ml = float(macd_l.loc[idx])
            ms = float(macd_sig.loc[idx])
            if np.isfinite(mh) and np.isfinite(ml) and np.isfinite(ms):
                macd_hist_pts.append(
                    {
                        "time": ts,
                        "value": round(mh, dec + 1),
                        "color": "rgba(16, 185, 129, 0.75)" if mh >= 0 else "rgba(239, 68, 68, 0.75)",
                    }
                )
                macd_line_pts.append({"time": ts, "value": round(ml, dec + 1)})
                macd_sig_pts.append({"time": ts, "value": round(ms, dec + 1)})
            else:
                macd_hist_pts.append({"time": ts})
                macd_line_pts.append({"time": ts})
                macd_sig_pts.append({"time": ts})

            ew_val = float(ewo_s.loc[idx])
            if np.isfinite(ew_val):
                ewo_pts.append(
                    {
                        "time": ts,
                        "value": round(ew_val, 3),
                        "color": "rgba(16, 185, 129, 0.80)" if ew_val >= 0 else "rgba(239, 68, 68, 0.80)",
                    }
                )
            else:
                ewo_pts.append({"time": ts})

        # Compute on-the-fly timeframe snapshot & pillar breakdown for H1/H4/D1/W1
        snap = compute_indicators(work)
        t_score = score_trend_pillar(snap)
        ref_tf = "H4" if tf_key == "H1" else tf_key
        s_score = (
            get_pair_strength_pillar_score(pair_key, ref_tf, snap, state.result.strength_matrix)
            if state.result
            else 0.0
        )
        m_score = score_momentum_pillar(snap)
        v_score = score_volatility_pillar(snap, t_score)
        pw = state.config.pillar_weights
        tf_total = float(
            np.clip(
                pw.trend * t_score + pw.strength * s_score + pw.momentum * m_score + pw.volatility * v_score,
                -100.0,
                100.0,
            )
        )

        pair_obj = next((p for p in (state.result.pairs if state.result else []) if p.spec.pair == pair_key), None)
        pair_tracked = [asdict(s) for s in state.tracked_signals if s.pair == pair_key]

        return {
            "pair": pair_key,
            "timeframe": tf_key,
            "decimals": dec,
            "pip_size": spec.pip_size,
            "tf_summary": {
                "total_score": round(tf_total, 1),
                "trend_score": round(t_score, 1),
                "strength_score": round(s_score, 1),
                "momentum_score": round(m_score, 1),
                "volatility_score": round(v_score, 1),
                "hma9": round(snap.hma9, dec),
                "hma21": round(snap.hma21, dec),
                "hma_status": snap.hma_status,
                "rsi14": round(snap.rsi14, 1),
                "dx14": round(snap.dx14, 1),
                "ravi": round(snap.ravi, 2),
                "vhf14": round(snap.vhf14, 2),
                "chop14": round(snap.chop14, 1),
                "ewo": round(snap.ewo, 2),
                "macd_accel_d2": round(snap.macd_accel_d2, 6),
                "tpo_poc": round(tpo_poc, dec),
                "tpo_vah": round(tpo_vah, dec),
                "tpo_val": round(tpo_val, dec),
                "adx14": round(snap.adx14, 1),
                "atr14": round(snap.atr14, dec),
                "atr_pips": round(snap.atr14 / spec.pip_size, 1),
                "vol_regime": snap.vol_regime,
            },
            "tpo_levels": {
                "poc": round(tpo_poc, dec),
                "vah": round(tpo_vah, dec),
                "val": round(tpo_val, dec),
            },
            "trade_plan": _serialize_pair(pair_obj)["trade_plan"] if pair_obj else None,
            "tracked_signals": pair_tracked,
            "candles": candles,
            "ema20": ema20_pts,
            "ema50": ema50_pts,
            "ema200": ema200_pts,
            "hma9": hma9_pts,
            "hma21": hma21_pts,
            "dc10_upper": dc10_upper_pts,
            "dc10_lower": dc10_lower_pts,
            "dc5_upper": dc5_upper_pts,
            "dc5_lower": dc5_lower_pts,
            "supertrend": supertrend_pts,
            "bb_upper": bb_upper_pts,
            "bb_lower": bb_lower_pts,
            "rsi14": rsi_pts,
            "macd_hist": macd_hist_pts,
            "macd_line": macd_line_pts,
            "macd_signal": macd_sig_pts,
            "ewo": ewo_pts,
        }

    @app.post("/api/refresh")
    def force_refresh(full: bool = Query(False, description="Full history reload vs fast 1m live tick")) -> Dict[str, Any]:
        if full:
            state.refresh(force=True)
        else:
            state.refresh_live_tick()
        return get_screener()

    @app.post("/api/risk")
    def update_risk(payload: RiskUpdatePayload) -> Dict[str, Any]:
        if payload.account_balance_usd <= 0 or not (0 < payload.risk_percent <= 50):
            raise HTTPException(status_code=400, detail="Invalid account balance or risk percent")
        state.config.risk.account_balance_usd = payload.account_balance_usd
        state.config.risk.risk_percent = payload.risk_percent
        state.config.risk.atr_timeframe = payload.atr_timeframe.upper()
        state.config.risk.sl_atr_multiplier = payload.sl_atr_multiplier
        state.config.save()
        state.recompute()
        return get_screener()

    @app.post("/api/watchlist")
    def toggle_watchlist(payload: WatchlistTogglePayload) -> Dict[str, Any]:
        pair = payload.pair.strip().upper()
        if pair not in UNIVERSE_MAP:
            raise HTTPException(status_code=404, detail=f"Unknown pair: {pair}")
        if pair in state.config.watchlist:
            state.config.watchlist.remove(pair)
        else:
            state.config.watchlist.append(pair)
        state.config.save()
        state.recompute()
        return get_screener()

    return app
