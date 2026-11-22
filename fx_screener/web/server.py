"""FastAPI Web Server and Interactive Charting API for the FX & Precious Metals Screener."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fx_screener.config import AppConfig
from fx_screener.data import load_or_fetch_universe
from fx_screener.engine.indicators import (
    bollinger_bands,
    compute_indicators,
    ema,
    macd,
    rsi,
)
from fx_screener.engine.scorer import (
    PairAnalysis,
    ScreenerResult,
    analyze_universe,
    score_momentum_pillar,
    score_trend_pillar,
    score_volatility_pillar,
)
from fx_screener.engine.strength import get_pair_strength_pillar_score
from fx_screener.universe import UNIVERSE_MAP


STATIC_DIR = Path(__file__).resolve().parent / "static"


class RiskUpdatePayload(BaseModel):
    account_balance_usd: float
    risk_percent: float
    atr_timeframe: str
    sl_atr_multiplier: float


class WatchlistTogglePayload(BaseModel):
    pair: str


class ScreenerWebState:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.lock = threading.Lock()
        self.config = config or AppConfig.load()
        self.universe_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.result: ScreenerResult | None = None
        self.last_updated: str = "-"
        self.refresh(force=False)

    def refresh(self, force: bool = False) -> None:
        with self.lock:
            self.universe_data, source = load_or_fetch_universe(
                self.config, force_refresh=force
            )
            self.result = analyze_universe(
                self.universe_data, self.config, source=source
            )
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_refresh_epoch = datetime.now().timestamp()

    def ensure_fresh(self) -> None:
        ttl = max(30, int(self.config.provider.auto_refresh_seconds or 300))
        now_ts = datetime.now().timestamp()
        if getattr(self, "last_refresh_epoch", 0.0) + ttl <= now_ts:
            self.refresh(force=True)

    def recompute(self) -> None:
        with self.lock:
            if not self.universe_data or not self.result:
                return
            self.result = analyze_universe(
                self.universe_data, self.config, source=self.result.source
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
            "adx14": round(ind.adx14, 1),
            "plus_di": round(ind.plus_di, 1),
            "minus_di": round(ind.minus_di, 1),
            "rsi14": round(ind.rsi14, 1),
            "macd_hist": ind.macd_hist,
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
        "h4": _serialize_timeframe_score(p.h4),
        "d1": _serialize_timeframe_score(p.d1),
        "w1": _serialize_timeframe_score(p.w1),
        "sparkline_d1": [round(v, dec) for v in p.d1_sparkline_closes],
        "sparkline_h4": [round(v, dec) for v in p.h4_sparkline_closes],
        "trade_plan": {
            "direction": tp.direction,
            "atr_timeframe": tp.atr_timeframe,
            "entry_price": round(tp.entry_price, dec),
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


def create_web_app(config: AppConfig | None = None) -> FastAPI:
    app = FastAPI(
        title="FX & Precious Metals Multi-Factor Screener UI",
        version="1.1.0",
    )
    state = ScreenerWebState(config=config)
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
        }

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
        close_s = work["Close"].astype(float)

        ema20_s = ema(close_s, 20)
        ema50_s = ema(close_s, 50)
        ema200_s = ema(close_s, min(200, max(20, len(close_s))))
        bb_u, bb_m, bb_l, _ = bollinger_bands(close_s, 20, 2.0)
        rsi_s = rsi(close_s, 14)
        macd_l, macd_sig, macd_h = macd(close_s, 12, 26, 9)

        candles: List[Dict[str, Any]] = []
        ema20_pts: List[Dict[str, Any]] = []
        ema50_pts: List[Dict[str, Any]] = []
        ema200_pts: List[Dict[str, Any]] = []
        bb_upper_pts: List[Dict[str, Any]] = []
        bb_lower_pts: List[Dict[str, Any]] = []
        rsi_pts: List[Dict[str, Any]] = []
        macd_hist_pts: List[Dict[str, Any]] = []
        macd_line_pts: List[Dict[str, Any]] = []
        macd_sig_pts: List[Dict[str, Any]] = []

        for idx, row in work.iterrows():
            ts = int(pd.Timestamp(idx).timestamp())
            o = round(float(row["Open"]), dec)
            h = round(float(row["High"]), dec)
            l = round(float(row["Low"]), dec)
            c = round(float(row["Close"]), dec)
            v = float(row.get("Volume", 0.0))

            candles.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
            ema20_pts.append({"time": ts, "value": round(float(ema20_s.loc[idx]), dec)})
            ema50_pts.append({"time": ts, "value": round(float(ema50_s.loc[idx]), dec)})
            ema200_pts.append({"time": ts, "value": round(float(ema200_s.loc[idx]), dec)})

            bu = float(bb_u.loc[idx])
            bl = float(bb_l.loc[idx])
            if not np.isnan(bu) and not np.isnan(bl):
                bb_upper_pts.append({"time": ts, "value": round(bu, dec)})
                bb_lower_pts.append({"time": ts, "value": round(bl, dec)})

            rv = float(rsi_s.loc[idx])
            if not np.isnan(rv):
                rsi_pts.append({"time": ts, "value": round(rv, 2)})

            mh = float(macd_h.loc[idx])
            ml = float(macd_l.loc[idx])
            ms = float(macd_sig.loc[idx])
            if not np.isnan(mh):
                macd_hist_pts.append(
                    {
                        "time": ts,
                        "value": round(mh, dec + 1),
                        "color": "rgba(16, 185, 129, 0.75)" if mh >= 0 else "rgba(239, 68, 68, 0.75)",
                    }
                )
                macd_line_pts.append({"time": ts, "value": round(ml, dec + 1)})
                macd_sig_pts.append({"time": ts, "value": round(ms, dec + 1)})

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
                "rsi14": round(snap.rsi14, 1),
                "adx14": round(snap.adx14, 1),
                "atr14": round(snap.atr14, dec),
                "atr_pips": round(snap.atr14 / spec.pip_size, 1),
                "vol_regime": snap.vol_regime,
            },
            "trade_plan": _serialize_pair(pair_obj)["trade_plan"] if pair_obj else None,
            "candles": candles,
            "ema20": ema20_pts,
            "ema50": ema50_pts,
            "ema200": ema200_pts,
            "bb_upper": bb_upper_pts,
            "bb_lower": bb_lower_pts,
            "rsi14": rsi_pts,
            "macd_hist": macd_hist_pts,
            "macd_line": macd_line_pts,
            "macd_signal": macd_sig_pts,
        }

    @app.post("/api/refresh")
    def force_refresh() -> Dict[str, Any]:
        state.refresh(force=True)
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
