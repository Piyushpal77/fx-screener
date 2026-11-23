"""4-Pillar Multi-Timeframe Composite Scoring & Confluence Engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from fx_screener.config import AppConfig
from fx_screener.engine.indicators import IndicatorSnapshot, compute_indicators
from fx_screener.engine.risk import TradePlan, build_trade_plan
from fx_screener.engine.strength import (
    CurrencyStrengthMatrix,
    compute_currency_strength_matrix,
    get_pair_strength_pillar_score,
)
from fx_screener.universe import InstrumentSpec, UNIVERSE, UNIVERSE_MAP


@dataclass
class TimeframeScore:
    timeframe: str
    trend_score: float       # -100..+100
    strength_score: float    # -100..+100
    momentum_score: float    # -100..+100
    volatility_score: float  # -100..+100
    total_score: float       # Weighted -100..+100
    bias: str                # "BULL" | "BEAR" | "NEUT"
    indicators: IndicatorSnapshot


@dataclass
class PairAnalysis:
    spec: InstrumentSpec
    price: float
    change_d1_pct: float
    h4: TimeframeScore
    d1: TimeframeScore
    w1: TimeframeScore
    pillar_trend: float
    pillar_strength: float
    pillar_momentum: float
    pillar_volatility: float
    raw_composite_score: float
    hma_h4_boost: float
    hma_trigger: str         # "★ H4 HMA TURN BULL" | "H4 HMA BULL" | "★ H4 HMA TURN BEAR" | "H4 HMA BEAR" | "PULLBACK (H4 BEAR)"
    is_fresh_hma_entry: bool # True ONLY when H4 pullback just finished and turned into a STRONG BUY / STRONG SELL
    composite_score: float
    signal: str              # "STRONG BUY" | "BUY" | "BUY (PULLBACK)" | "NEUTRAL" | "SELL (PULLBACK)" | "SELL" | "STRONG SELL"
    confluence: bool         # True when H4, D1, W1 all align beyond threshold
    confluence_label: str    # "★ BULL ALIGNED" | "★ BEAR ALIGNED" | "MIXED"
    vol_regime: str          # "SQUEEZE" | "BREAKOUT" | "EXPANDING" | "NORMAL"
    d1_sparkline_closes: List[float]
    h4_sparkline_closes: List[float]
    trade_plan: TradePlan
    is_watchlist: bool
    is_alert: bool


@dataclass
class ScreenerResult:
    pairs: List[PairAnalysis]
    strength_matrix: CurrencyStrengthMatrix
    source: str
    alert_pairs: List[str]


def score_trend_pillar(snap: IndicatorSnapshot) -> float:
    """Score Trend Pillar (-100 to +100) using Fast/Slow Hull Moving Averages (HMA 9/21), EMA 50/200 stack, DI+/-, and ADX."""
    points = 0.0
    # 1. Hull Moving Average (HMA 9 & HMA 21) low-lag directional structure (up to +/- 35)
    points += 12.0 if snap.close > snap.hma21 else -12.0
    points += 13.0 if snap.hma9 > snap.hma21 else -13.0
    # HMA(9) slope normalized by ATR% so low-volatility and high-volatility pairs scale evenly
    vol_norm = max(snap.atr_pct, 0.12)
    hma_slope_norm = (snap.hma9_slope_pct / vol_norm) * 28.0
    points += float(np.clip(hma_slope_norm, -10.0, 10.0))

    # 2. Macro EMA 20/50/200 stack structure (up to +/- 45)
    points += 10.0 if snap.close > snap.ema50 else -10.0
    points += 12.0 if snap.close > snap.ema200 else -12.0
    points += 11.0 if snap.ema20 > snap.ema50 else -11.0
    points += 12.0 if snap.ema50 > snap.ema200 else -12.0

    # 3. Directional movement DI+ vs DI- (up to +/- 20)
    di_spread = snap.plus_di - snap.minus_di
    points += float(np.clip(di_spread * 0.85, -20.0, 20.0))

    # ADX trend strength multiplier:
    # ADX >= 30 boosts trend conviction by up to 1.25x; ADX < 18 dampens range-bound noise to 0.65x
    if snap.adx14 >= 30.0:
        adx_mult = min(1.25, 1.0 + (snap.adx14 - 25.0) * 0.01)
    elif snap.adx14 < 18.0:
        adx_mult = max(0.60, 0.60 + (snap.adx14 / 18.0) * 0.35)
    else:
        adx_mult = 0.95 + ((snap.adx14 - 18.0) / 12.0) * 0.15

    return float(np.clip(points * adx_mult, -100.0, 100.0))


def score_momentum_pillar(snap: IndicatorSnapshot) -> float:
    """Score Momentum Pillar (-100 to +100) using RSI(14) zones and MACD histogram/crossover."""
    rsi_val = snap.rsi14
    # Healthy trend momentum peaks around RSI 62-72 (bullish) or 28-38 (bearish),
    # tapering slightly at extreme >80 / <20 exhaustion
    if rsi_val >= 50.0:
        if rsi_val <= 74.0:
            rsi_score = ((rsi_val - 50.0) / 24.0) * 55.0
        else:
            rsi_score = 55.0 - min(20.0, (rsi_val - 74.0) * 1.5)
    else:
        if rsi_val >= 26.0:
            rsi_score = -((50.0 - rsi_val) / 24.0) * 55.0
        else:
            rsi_score = -55.0 + min(20.0, (26.0 - rsi_val) * 1.5)

    # MACD component (up to +/- 45)
    atr_norm = max(snap.atr14, 1e-6)
    norm_hist = snap.macd_hist / atr_norm
    macd_score = float(np.clip(norm_hist * 85.0, -30.0, 30.0))

    # Histogram expansion / impulse bonus (+/- 15)
    hist_rising = snap.macd_hist > snap.macd_prev_hist
    if snap.macd_line > snap.macd_signal:
        macd_score += 15.0 if hist_rising else 7.0
    else:
        macd_score -= 15.0 if not hist_rising else 7.0

    return float(np.clip(rsi_score + macd_score, -100.0, 100.0))


def score_volatility_pillar(snap: IndicatorSnapshot, trend_direction: float) -> float:
    """Score Volatility & Structure Pillar (-100 to +100) using Bollinger %B, squeeze/breakout, and ATR ratio."""
    # Centered %B: 0.5 is midline, 1.0 is upper band, 0.0 is lower band
    b_centered = (snap.bb_percent_b - 0.5) * 2.0  # -1.0 at lower band, +1.0 at upper band
    base_dir_score = float(np.clip(b_centered * 55.0, -65.0, 65.0))

    # Volatility expansion multiplier
    vol_bonus = 0.0
    if snap.vol_regime == "BREAKOUT":
        vol_bonus = 35.0 if snap.bb_percent_b > 0.5 else -35.0
    elif snap.vol_regime == "EXPANDING":
        sign = 1.0 if (b_centered >= 0 or trend_direction >= 0) else -1.0
        vol_bonus = sign * min(25.0, (snap.atr_ratio - 1.0) * 40.0)
    elif snap.vol_regime == "SQUEEZE":
        # Coiled spring: modest directional bias aligned with trend
        sign = 1.0 if trend_direction >= 0 else -1.0
        vol_bonus = sign * 12.0

    return float(np.clip(base_dir_score + vol_bonus, -100.0, 100.0))


def compute_h4_hma_boost_and_trigger(
    h4_snap: IndicatorSnapshot,
    d1_score: float,
    w1_score: float,
) -> tuple[float, str, bool]:
    """Compute Smart Momentum Boost (+10..+15 pts) and Pullback Guard from 4-Hour HMA(9)/HMA(21).

    Evaluates 4H Hull Moving Average inflections against the macro D1/W1 trend so signals
    fire cleanly when a 4H pullback finishes and turns back into the daily/weekly trend.

    Returns:
        (boost_points, trigger_label, is_counter_trend_pullback)
    """
    macro_bias = 0.72 * d1_score + 0.28 * w1_score
    hma_bull_aligned = (h4_snap.hma9 > h4_snap.hma21) and (h4_snap.hma9 >= h4_snap.hma9_prev)
    hma_bear_aligned = (h4_snap.hma9 < h4_snap.hma21) and (h4_snap.hma9 <= h4_snap.hma9_prev)

    boost = 0.0
    is_pullback = False

    if macro_bias >= 12.0:
        # Macro Bullish trend on D1/W1
        if h4_snap.hma_bull_cross and h4_snap.close >= h4_snap.hma9:
            boost = 15.0
            trigger = "★ H4 HMA TURN BULL"
        elif hma_bull_aligned and h4_snap.close >= h4_snap.hma21:
            boost = 10.0
            trigger = "H4 HMA BULL"
        elif hma_bear_aligned or h4_snap.close < h4_snap.hma21:
            boost = -8.0
            is_pullback = True
            trigger = "PULLBACK (H4 BEAR)"
        else:
            trigger = f"H4 {h4_snap.hma_status}"
    elif macro_bias <= -12.0:
        # Macro Bearish trend on D1/W1
        if h4_snap.hma_bear_cross and h4_snap.close <= h4_snap.hma9:
            boost = -15.0
            trigger = "★ H4 HMA TURN BEAR"
        elif hma_bear_aligned and h4_snap.close <= h4_snap.hma21:
            boost = -10.0
            trigger = "H4 HMA BEAR"
        elif hma_bull_aligned or h4_snap.close > h4_snap.hma21:
            boost = 8.0
            is_pullback = True
            trigger = "PULLBACK (H4 BULL)"
        else:
            trigger = f"H4 {h4_snap.hma_status}"
    else:
        # Macro Neutral / Transitioning: allow early 4H HMA inflection to give a +6/-6 lead
        if h4_snap.hma_bull_cross:
            boost = 6.0
            trigger = "★ H4 HMA TURN BULL"
        elif h4_snap.hma_bear_cross:
            boost = -6.0
            trigger = "★ H4 HMA TURN BEAR"
        else:
            trigger = f"H4 {h4_snap.hma_status}"

    return boost, trigger, is_pullback


def classify_signal(score: float, config: AppConfig, is_pullback: bool = False) -> str:
    th = config.thresholds
    if score >= th.strong_buy:
        return "BUY (PULLBACK)" if is_pullback else "STRONG BUY"
    if score >= th.buy:
        return "BUY (PULLBACK)" if is_pullback else "BUY"
    if score <= th.strong_sell:
        return "SELL (PULLBACK)" if is_pullback else "STRONG SELL"
    if score <= th.sell:
        return "SELL (PULLBACK)" if is_pullback else "SELL"
    return "NEUTRAL"


def analyze_universe(
    universe_data: Dict[str, Dict[str, pd.DataFrame]],
    config: AppConfig,
    source: str = "yfinance",
) -> ScreenerResult:
    """Run full 4-pillar multi-timeframe analysis across all 30 instruments."""
    # Step 1: Compute indicator snapshots for all pairs and timeframes
    snapshots: Dict[str, Dict[str, IndicatorSnapshot]] = {}
    spot_prices: Dict[str, float] = {}

    for spec in UNIVERSE:
        pair = spec.pair
        tf_dfs = universe_data.get(pair)
        if not tf_dfs:
            continue
        snapshots[pair] = {}
        for tf in ("H4", "D1", "W1"):
            df = tf_dfs.get(tf)
            if df is not None and len(df) >= 5:
                snapshots[pair][tf] = compute_indicators(df)
        if "D1" in snapshots[pair]:
            spot_prices[pair] = snapshots[pair]["D1"].close

    # Step 2: Compute G8 Currency Strength Matrix
    strength_matrix = compute_currency_strength_matrix(snapshots, config.timeframe_weights)

    # Step 3: Score each pair across H4, D1, W1 and apply 4H HMA Pullback Trigger
    pw = config.pillar_weights
    tw = config.timeframe_weights
    conf_min = config.thresholds.confluence_min_tf_score
    watchlist_set = set(config.watchlist)

    analyses: List[PairAnalysis] = []
    alert_pairs: List[str] = []

    for spec in UNIVERSE:
        pair = spec.pair
        pair_snaps = snapshots.get(pair, {})
        if not all(tf in pair_snaps for tf in ("H4", "D1", "W1")):
            continue

        tf_scores: Dict[str, TimeframeScore] = {}
        for tf in ("H4", "D1", "W1"):
            snap = pair_snaps[tf]
            t_score = score_trend_pillar(snap)
            s_score = get_pair_strength_pillar_score(pair, tf, snap, strength_matrix)
            m_score = score_momentum_pillar(snap)
            v_score = score_volatility_pillar(snap, t_score)

            total = (
                pw.trend * t_score
                + pw.strength * s_score
                + pw.momentum * m_score
                + pw.volatility * v_score
            )
            total = float(np.clip(total, -100.0, 100.0))

            if total >= conf_min:
                bias = "BULL"
            elif total <= -conf_min:
                bias = "BEAR"
            else:
                bias = "NEUT"

            tf_scores[tf] = TimeframeScore(
                timeframe=tf,
                trend_score=t_score,
                strength_score=s_score,
                momentum_score=m_score,
                volatility_score=v_score,
                total_score=total,
                bias=bias,
                indicators=snap,
            )

        h4_s = tf_scores["H4"]
        d1_s = tf_scores["D1"]
        w1_s = tf_scores["W1"]

        pillar_trend = tw.D1 * d1_s.trend_score + tw.H4 * h4_s.trend_score + tw.W1 * w1_s.trend_score
        pillar_strength = tw.D1 * d1_s.strength_score + tw.H4 * h4_s.strength_score + tw.W1 * w1_s.strength_score
        pillar_momentum = tw.D1 * d1_s.momentum_score + tw.H4 * h4_s.momentum_score + tw.W1 * w1_s.momentum_score
        pillar_volatility = tw.D1 * d1_s.volatility_score + tw.H4 * h4_s.volatility_score + tw.W1 * w1_s.volatility_score

        raw_composite = (
            tw.D1 * d1_s.total_score
            + tw.H4 * h4_s.total_score
            + tw.W1 * w1_s.total_score
        )
        raw_composite = float(np.clip(raw_composite, -100.0, 100.0))

        hma_boost, hma_trigger, is_pullback = compute_h4_hma_boost_and_trigger(
            h4_s.indicators, d1_s.total_score, w1_s.total_score
        )
        composite = float(np.clip(raw_composite + hma_boost, -100.0, 100.0))

        all_bull = h4_s.total_score >= conf_min and d1_s.total_score >= conf_min and w1_s.total_score >= conf_min
        all_bear = h4_s.total_score <= -conf_min and d1_s.total_score <= -conf_min and w1_s.total_score <= -conf_min
        confluence = all_bull or all_bear
        if all_bull:
            confluence_label = "★ BULL ALIGNED"
        elif all_bear:
            confluence_label = "★ BEAR ALIGNED"
        else:
            confluence_label = "MIXED"

        signal = classify_signal(composite, config, is_pullback=is_pullback)
        is_fresh_hma_entry = bool(
            (signal == "STRONG BUY" and h4_s.indicators.hma_bull_cross)
            or (signal == "STRONG SELL" and h4_s.indicators.hma_bear_cross)
        )

        # Volatility regime (prioritize D1 or H4 squeeze/breakout)
        if d1_s.indicators.vol_regime in ("BREAKOUT", "SQUEEZE"):
            vol_regime = d1_s.indicators.vol_regime
        elif h4_s.indicators.vol_regime in ("BREAKOUT", "SQUEEZE"):
            vol_regime = f"H4 {h4_s.indicators.vol_regime}"
        else:
            vol_regime = d1_s.indicators.vol_regime

        # Trade Plan calculation (supports D1, H4, or H4_HMA dynamic trail stop)
        atr_tf = config.risk.atr_timeframe.upper()
        if atr_tf in ("H4", "H4_HMA"):
            active_atr = h4_s.indicators.atr14
        else:
            active_atr = d1_s.indicators.atr14

        trade_plan = build_trade_plan(
            pair=pair,
            composite_score=composite,
            entry_price=h4_s.indicators.close,
            atr_value=active_atr,
            risk_cfg=config.risk,
            spot_prices=spot_prices,
            hma21_h4=h4_s.indicators.hma21,
            atr_h4=h4_s.indicators.atr14,
        )

        d1_closes = [float(x) for x in universe_data[pair]["D1"]["Close"].tail(28).tolist()]
        h4_closes = [float(x) for x in universe_data[pair]["H4"]["Close"].tail(28).tolist()]

        is_alert = abs(composite) >= config.thresholds.alert_threshold
        if is_alert:
            alert_pairs.append(f"{pair} ({composite:+.1f})")

        analyses.append(
            PairAnalysis(
                spec=spec,
                price=h4_s.indicators.close,
                change_d1_pct=d1_s.indicators.change_pct,
                h4=h4_s,
                d1=d1_s,
                w1=w1_s,
                pillar_trend=pillar_trend,
                pillar_strength=pillar_strength,
                pillar_momentum=pillar_momentum,
                pillar_volatility=pillar_volatility,
                raw_composite_score=raw_composite,
                hma_h4_boost=hma_boost,
                hma_trigger=hma_trigger,
                is_fresh_hma_entry=is_fresh_hma_entry,
                composite_score=composite,
                signal=signal,
                confluence=confluence,
                confluence_label=confluence_label,
                vol_regime=vol_regime,
                d1_sparkline_closes=d1_closes,
                h4_sparkline_closes=h4_closes,
                trade_plan=trade_plan,
                is_watchlist=pair in watchlist_set,
                is_alert=is_alert,
            )
        )

    # Default sort: Watchlist pinned optional or by absolute conviction / score descending
    analyses.sort(key=lambda a: a.composite_score, reverse=True)

    return ScreenerResult(
        pairs=analyses,
        strength_matrix=strength_matrix,
        source=source,
        alert_pairs=alert_pairs,
    )
