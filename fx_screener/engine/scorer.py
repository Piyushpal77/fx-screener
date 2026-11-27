"""Fx_Screener2 4-Pillar Multi-Timeframe Composite Scoring, Early Entry & 4-Trigger ASAP Exit Engine.

- Pillar Weights:
  1. Early Trend & Fast Breakout (45%): HMA(9/21), Donchian(10), SuperTrend(10, 2.5), EMA 20/50/200,
     and Zero-Lag Efficiency Multiplier (35% DX(14) + 25% FX-RAVI(7,65) + 20% VHF(14) + 20% Inverted CHOP(14))
  2. Momentum Acceleration & Elliott Wave (30%): 2nd-Derivative MACD Δ²H, EWO(5,35), Causal ATR-ZigZag Wave, Constance Brown RSI(14)
  3. G8 Currency Strength Differential (15%): Base - Quote normalized basket score
  4. TPO Value Area & Volatility Squeeze (10%): TPO (VAH/POC/VAL) acceptance/breakout + TTM Squeeze
- Timeframe Weights: H4 (45%) + D1 (40%) + W1 (15%) + H1+H4+D1 Early Entry Alignment
- 4-Trigger ASAP Exit Engine: Flags ⚠ EXIT NOW immediately on H4 HMA reversal, Donchian(5) trail break, or Momentum Exhaustion climax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
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
    h1: Optional[TimeframeScore]
    h4: TimeframeScore
    d1: TimeframeScore
    w1: TimeframeScore
    pillar_trend: float
    pillar_strength: float
    pillar_momentum: float
    pillar_volatility: float
    raw_composite_score: float
    hma_h4_boost: float
    hma_trigger: str         # e.g. "★ H4 HMA TURN BULL" | "⚡ EARLY DONCHIAN(10) BRK"
    is_fresh_hma_entry: bool # True when early/pullback inflection fires into STRONG BUY / STRONG SELL or EARLY ENTRY
    is_early_entry: bool     # True when H1 + H4 + D1 align with fresh H4 momentum/trend inflection (before W1)
    top_pick_rank: int       # 1..5 for Top 5 Long, -1..-5 for Top 5 Short, 0 otherwise
    top_pick_label: str      # "★ TOP 1 LONG" | "★ TOP 1 SHORT" | ""
    asap_exit_flag: bool     # True if current market structure warns to exit an open position in the opposite/exhausted state
    asap_exit_reason: str    # e.g. "⚠ EXIT LONG: H4 HMA REVERSAL" | "⚠ EXIT: DONCHIAN(5) BREAK" | ""
    donchian5_trail_stop: float
    composite_score: float
    signal: str              # "STRONG BUY" | "★ EARLY BUY" | "BUY" | "BUY (PULLBACK)" | "NEUTRAL" | "SELL (PULLBACK)" | "SELL" | "★ EARLY SELL" | "STRONG SELL"
    confluence: bool         # True when H4, D1, W1 (or H1+H4+D1 early) align beyond threshold
    confluence_label: str    # "★ BULL ALIGNED" | "⚡ EARLY BULL (H1+H4+D1)" | "★ BEAR ALIGNED" | "⚡ EARLY BEAR (H1+H4+D1)" | "MIXED"
    vol_regime: str          # "SQUEEZE FIRE" | "TTM SQUEEZE" | "SQUEEZE" | "BREAKOUT" | "EXPANDING" | "NORMAL"
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
    """Score Pillar 1: Early Trend & Fast Breakout (-100 to +100).

    Blends Fast Donchian(10) breakout/midline structure, Hull Moving Averages HMA(9/21),
    SuperTrend(10, 2.5), EMA 20/50/200 stack, DI+/-, and the FX-calibrated Early Trend Efficiency Multiplier.
    """
    points = 0.0
    vol_norm = max(snap.atr_pct, 0.12)

    # 1. Fast Donchian(10) Breakout & Midline Position (up to +/- 22)
    if snap.donchian10_bull_brk:
        points += 22.0
    elif snap.donchian10_bear_brk:
        points -= 22.0
    else:
        points += 12.0 if snap.close >= snap.donchian10_mid else -12.0
        if snap.close > snap.donchian10_high:
            points += 8.0
        elif snap.close < snap.donchian10_low:
            points -= 8.0

    # 2. Hull Moving Average HMA(9) & HMA(21) Low-Lag Structure (up to +/- 28)
    points += 10.0 if snap.close > snap.hma21 else -10.0
    points += 10.0 if snap.hma9 > snap.hma21 else -10.0
    hma_slope_norm = (snap.hma9_slope_pct / vol_norm) * 24.0
    points += float(np.clip(hma_slope_norm, -8.0, 8.0))

    # 3. SuperTrend(10, 2.5) Structural Regime & Bar-1 Flip Bonus (up to +/- 18)
    points += 12.0 * snap.supertrend_dir
    if snap.supertrend_flipped_bull:
        points += 6.0
    elif snap.supertrend_flipped_bear:
        points -= 6.0

    # 4. Macro EMA 20/50/200 Alignment (up to +/- 20)
    points += 6.0 if snap.close > snap.ema50 else -6.0
    points += 7.0 if snap.ema20 > snap.ema50 else -7.0
    points += 7.0 if snap.ema50 > snap.ema200 else -7.0

    # 5. Wilder DI+ vs DI- Spread (up to +/- 12)
    di_spread = snap.plus_di - snap.minus_di
    points += float(np.clip(di_spread * 0.55, -12.0, 12.0))

    # 6. FX-Calibrated Zero-Lag Early Trend Efficiency Multiplier:
    #    - DX(14) (35%): unsmoothed Wilder directional movement
    #    - FX-RAVI(7, 65) (25%): calibrated for FX (0.15% floor threshold, 1.5% cap)
    #    - VHF(14) (20%): vertical vs horizontal efficiency
    #    - Inverted CHOP(14) (20%): <38.2 linear trend, + CHOP drop velocity bonus
    dx_eff = float(np.clip((snap.dx14 - 12.0) / 34.0, 0.0, 1.0))

    ravi_thresh = max(0.15, min(0.45, snap.atr_pct * 0.45))
    ravi_eff = float(np.clip(snap.ravi / min(1.50, ravi_thresh * 4.0), 0.0, 1.0))
    if snap.ravi_rising and snap.ravi >= 0.15:
        ravi_eff = min(1.0, ravi_eff + 0.18)

    vhf_eff = float(np.clip((snap.vhf14 - 0.20) / 0.32, 0.0, 1.0))
    chop_eff = float(np.clip((61.8 - snap.chop14) / (61.8 - 38.2), 0.0, 1.0))
    if snap.chop_falling:
        chop_eff = min(1.0, chop_eff + 0.15)

    trend_efficiency = (
        0.35 * dx_eff
        + 0.25 * ravi_eff
        + 0.20 * vhf_eff
        + 0.20 * chop_eff
    )
    conviction_mult = 0.68 + 0.62 * trend_efficiency

    return float(np.clip(points * conviction_mult, -100.0, 100.0))


def score_momentum_pillar(snap: IndicatorSnapshot) -> float:
    """Score Pillar 2: Momentum Acceleration & Elliott Wave (-100 to +100).

    Combines:
    - 2nd-Derivative MACD Histogram Acceleration (Δ²H) + Normalized MACD Histogram (up to +/- 38)
    - Elliott Wave Oscillator EWO(5, 35) & Causal ATR-ZigZag Wave 1-2-3 Bias (up to +/- 32)
    - Constance Brown RSI(14) Regimes (40-80 bull support, 20-60 bear resistance) (up to +/- 30)
    """
    atr_norm = max(snap.atr14, 1e-6)

    # 1. Constance Brown RSI(14) Regime Score (up to +/- 30)
    rsi_val = snap.rsi14
    if rsi_val >= 50.0:
        if rsi_val <= 76.0:
            rsi_score = ((rsi_val - 50.0) / 26.0) * 30.0
        else:
            rsi_score = 30.0 - min(14.0, (rsi_val - 76.0) * 1.4)
    else:
        if rsi_val >= 24.0:
            rsi_score = -((50.0 - rsi_val) / 26.0) * 30.0
        else:
            rsi_score = -30.0 + min(14.0, (24.0 - rsi_val) * 1.4)

    # 2. MACD Histogram + 2nd-Derivative Acceleration Δ²H (up to +/- 38)
    norm_hist = snap.macd_hist / atr_norm
    norm_d2 = snap.macd_accel_d2 / atr_norm
    macd_score = float(np.clip(norm_hist * 65.0, -20.0, 20.0))
    # Δ²H catches the exact bottom/top of pullbacks 2-4 bars before MACD crosses
    d2_score = float(np.clip(norm_d2 * 110.0, -12.0, 12.0))
    cross_bonus = 6.0 if snap.macd_line > snap.macd_signal else -6.0
    macd_total = macd_score + d2_score + cross_bonus

    # 3. Elliott Wave Oscillator EWO(5, 35) + Causal ATR-ZigZag Wave 1-2-3 (up to +/- 32)
    ewo_score = float(np.clip(snap.ewo * 11.0, -18.0, 18.0))
    if snap.ewo > 0 and snap.ewo_rising:
        ewo_score += 5.0
    elif snap.ewo < 0 and not snap.ewo_rising:
        ewo_score -= 5.0
    wave_score = 9.0 * float(snap.wave_bias)

    return float(np.clip(rsi_score + macd_total + ewo_score + wave_score, -100.0, 100.0))


def score_volatility_pillar(snap: IndicatorSnapshot, trend_direction: float) -> float:
    """Score Pillar 4: TPO Value Area (VAH/POC/VAL) & Volatility Squeeze (-100 to +100)."""
    # 1. Steidlmayer TPO Market Profile Value Area Acceptance / Initiative Breakout (up to +/- 55)
    tpo_score = 0.0
    if snap.tpo_vah > snap.tpo_val > 0:
        if snap.close > snap.tpo_vah:
            # Initiative breakout above 70% Value Area High
            tpo_score = 50.0
        elif snap.close < snap.tpo_val:
            # Initiative breakdown below 70% Value Area Low
            tpo_score = -50.0
        else:
            # Inside Value Area: score relative to Point of Control (POC)
            va_half = max(0.5 * (snap.tpo_vah - snap.tpo_val), 1e-8)
            tpo_score = float(np.clip(((snap.close - snap.tpo_poc) / va_half) * 28.0, -30.0, 30.0))

    # 2. TTM Squeeze (Bollinger inside Keltner) & Expansion Bonus (up to +/- 45)
    dir_sign = 1.0 if (trend_direction >= 0) else -1.0
    b_centered = (snap.bb_percent_b - 0.5) * 2.0
    vol_bonus = float(np.clip(b_centered * 20.0, -20.0, 20.0))

    if snap.ttm_squeeze_fired:
        vol_bonus += dir_sign * 30.0
    elif snap.vol_regime == "BREAKOUT":
        vol_bonus += 25.0 if snap.bb_percent_b > 0.5 else -25.0
    elif snap.ttm_squeeze_on or snap.vol_regime == "SQUEEZE":
        vol_bonus += dir_sign * 14.0
    elif snap.vol_regime == "EXPANDING":
        vol_bonus += dir_sign * min(18.0, (snap.atr_ratio - 1.0) * 35.0)

    return float(np.clip(tpo_score + vol_bonus, -100.0, 100.0))


def compute_h4_inflection_and_early_entry(
    h1_score: Optional[float],
    h4_s: TimeframeScore,
    d1_s: TimeframeScore,
    w1_s: TimeframeScore,
    conf_min: float,
) -> tuple[float, str, bool, bool]:
    """Evaluate 4H early inflection boost, pullback guard, and H1+H4+D1 Early Entry alignment.

    Returns:
        (boost_points, trigger_label, is_counter_trend_pullback, is_early_entry)
    """
    h4_snap = h4_s.indicators
    macro_bias = 0.72 * d1_s.total_score + 0.28 * w1_s.total_score
    hma_bull_aligned = (h4_snap.hma9 > h4_snap.hma21) and (h4_snap.hma9 >= h4_snap.hma9_prev)
    hma_bear_aligned = (h4_snap.hma9 < h4_snap.hma21) and (h4_snap.hma9 <= h4_snap.hma9_prev)

    bull_inflection = bool(
        h4_snap.hma_bull_cross
        or h4_snap.donchian10_bull_brk
        or h4_snap.supertrend_flipped_bull
        or (h4_snap.ttm_squeeze_fired and h4_s.total_score > 0)
    )
    bear_inflection = bool(
        h4_snap.hma_bear_cross
        or h4_snap.donchian10_bear_brk
        or h4_snap.supertrend_flipped_bear
        or (h4_snap.ttm_squeeze_fired and h4_s.total_score < 0)
    )

    # Check H1 + H4 + D1 early directional alignment (catches trend before W1 flips)
    h1_bull = (h1_score is not None and h1_score >= 15.0)
    h1_bear = (h1_score is not None and h1_score <= -15.0)
    early_bull = bool(h1_bull and h4_s.total_score >= conf_min and d1_s.total_score >= 12.0 and (bull_inflection or hma_bull_aligned))
    early_bear = bool(h1_bear and h4_s.total_score <= -conf_min and d1_s.total_score <= -12.0 and (bear_inflection or hma_bear_aligned))
    is_early_entry = early_bull or early_bear

    boost = 0.0
    is_pullback = False

    if macro_bias >= 10.0 or early_bull:
        if h4_snap.donchian10_bull_brk and h4_snap.close >= h4_snap.hma9:
            boost = 15.0
            trigger = "⚡ H4 DONCHIAN(10) BULL BRK"
        elif h4_snap.supertrend_flipped_bull:
            boost = 15.0
            trigger = "⚡ H4 SUPERTREND FLIP BULL"
        elif h4_snap.hma_bull_cross and h4_snap.close >= h4_snap.hma9:
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
    elif macro_bias <= -10.0 or early_bear:
        if h4_snap.donchian10_bear_brk and h4_snap.close <= h4_snap.hma9:
            boost = -15.0
            trigger = "⚡ H4 DONCHIAN(10) BEAR BRK"
        elif h4_snap.supertrend_flipped_bear:
            boost = -15.0
            trigger = "⚡ H4 SUPERTREND FLIP BEAR"
        elif h4_snap.hma_bear_cross and h4_snap.close <= h4_snap.hma9:
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
        if bull_inflection:
            boost = 8.0
            trigger = "★ H4 HMA TURN BULL"
        elif bear_inflection:
            boost = -8.0
            trigger = "★ H4 HMA TURN BEAR"
        else:
            trigger = f"H4 {h4_snap.hma_status}"

    return boost, trigger, is_pullback, is_early_entry


def evaluate_live_asap_exit_warning(
    composite: float,
    h4_snap: IndicatorSnapshot,
    d1_snap: IndicatorSnapshot,
) -> tuple[bool, str, float]:
    """Evaluate real-time ASAP Exit warning on the current H4 structure for open/existing trades.

    Returns (has_exit_warning, exit_reason, donchian5_trail_stop).
    """
    is_bull_bias = composite >= 0.0
    trail_stop = h4_snap.donchian5_low if is_bull_bias else h4_snap.donchian5_high

    if composite >= 20.0:
        # Pair is in Long bias — check if H4 structure just broke down or climaxed
        if h4_snap.close < h4_snap.donchian5_low:
            return True, "⚠ EXIT LONG: H4 DONCHIAN(5) TRAIL BREAK", trail_stop
        if h4_snap.close < h4_snap.hma21 and h4_snap.hma9 < h4_snap.hma9_prev:
            return True, "⚠ EXIT LONG: H4 HMA(9/21) REVERSAL", trail_stop
        if (
            h4_snap.rsi14 >= 72.0
            and h4_snap.macd_accel_d2 < 0
            and not h4_snap.ravi_rising
            and not h4_snap.chop_falling
        ):
            return True, "⚠ TAKE PROFIT: LONG MOMENTUM EXHAUSTION", trail_stop
    elif composite <= -20.0:
        # Pair is in Short bias — check if H4 structure just broke upward or climaxed
        if h4_snap.close > h4_snap.donchian5_high:
            return True, "⚠ EXIT SHORT: H4 DONCHIAN(5) TRAIL BREAK", trail_stop
        if h4_snap.close > h4_snap.hma21 and h4_snap.hma9 > h4_snap.hma9_prev:
            return True, "⚠ EXIT SHORT: H4 HMA(9/21) REVERSAL", trail_stop
        if (
            h4_snap.rsi14 <= 28.0
            and h4_snap.macd_accel_d2 > 0
            and not h4_snap.ravi_rising
            and not h4_snap.chop_falling
        ):
            return True, "⚠ TAKE PROFIT: SHORT MOMENTUM EXHAUSTION", trail_stop

    return False, "", trail_stop


def classify_signal(
    score: float,
    config: AppConfig,
    is_pullback: bool = False,
    is_early_entry: bool = False,
) -> str:
    th = config.thresholds
    if score >= th.strong_buy:
        return "BUY (PULLBACK)" if is_pullback else "STRONG BUY"
    if score >= th.buy:
        if is_pullback:
            return "BUY (PULLBACK)"
        return "★ EARLY BUY" if is_early_entry and score >= 40.0 else "BUY"
    if score <= th.strong_sell:
        return "SELL (PULLBACK)" if is_pullback else "STRONG SELL"
    if score <= th.sell:
        if is_pullback:
            return "SELL (PULLBACK)"
        return "★ EARLY SELL" if is_early_entry and score <= -40.0 else "SELL"
    return "NEUTRAL"


def analyze_universe(
    universe_data: Dict[str, Dict[str, pd.DataFrame]],
    config: AppConfig,
    source: str = "yfinance",
) -> ScreenerResult:
    """Run full Fx_Screener2 4-pillar multi-timeframe analysis across all 31 instruments."""
    snapshots: Dict[str, Dict[str, IndicatorSnapshot]] = {}
    spot_prices: Dict[str, float] = {}

    for spec in UNIVERSE:
        pair = spec.pair
        tf_dfs = universe_data.get(pair)
        if not tf_dfs:
            continue
        snapshots[pair] = {}
        for tf in ("H1", "H4", "D1", "W1"):
            df = tf_dfs.get(tf)
            if df is not None and len(df) >= 5:
                snapshots[pair][tf] = compute_indicators(df)
        if "H1" in snapshots[pair]:
            spot_prices[pair] = snapshots[pair]["H1"].close
        elif "D1" in snapshots[pair]:
            spot_prices[pair] = snapshots[pair]["D1"].close

    # Step 2: Compute G8 Currency Strength Matrix
    strength_matrix = compute_currency_strength_matrix(snapshots, config.timeframe_weights)

    # Step 3: Score each pair across H1, H4, D1, W1
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
        available_tfs = [tf for tf in ("H1", "H4", "D1", "W1") if tf in pair_snaps]
        for tf in available_tfs:
            snap = pair_snaps[tf]
            ref_strength_tf = "H4" if tf == "H1" else tf
            t_score = score_trend_pillar(snap)
            s_score = get_pair_strength_pillar_score(pair, ref_strength_tf, snap, strength_matrix)
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

        h1_s = tf_scores.get("H1")
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

        hma_boost, hma_trigger, is_pullback, is_early_entry = compute_h4_inflection_and_early_entry(
            h1_s.total_score if h1_s else None,
            h4_s,
            d1_s,
            w1_s,
            conf_min,
        )
        composite = float(np.clip(raw_composite + hma_boost, -100.0, 100.0))

        all_bull = h4_s.total_score >= conf_min and d1_s.total_score >= conf_min and w1_s.total_score >= conf_min
        all_bear = h4_s.total_score <= -conf_min and d1_s.total_score <= -conf_min and w1_s.total_score <= -conf_min
        confluence = all_bull or all_bear or is_early_entry
        if all_bull:
            confluence_label = "★ BULL ALIGNED"
        elif all_bear:
            confluence_label = "★ BEAR ALIGNED"
        elif is_early_entry and composite > 0:
            confluence_label = "⚡ EARLY BULL (H1+H4+D1)"
        elif is_early_entry and composite < 0:
            confluence_label = "⚡ EARLY BEAR (H1+H4+D1)"
        else:
            confluence_label = "MIXED"

        signal = classify_signal(
            composite, config, is_pullback=is_pullback, is_early_entry=is_early_entry
        )

        fresh_bull_trigger = bool(
            h4_s.indicators.hma_bull_cross
            or h4_s.indicators.donchian10_bull_brk
            or h4_s.indicators.supertrend_flipped_bull
        )
        fresh_bear_trigger = bool(
            h4_s.indicators.hma_bear_cross
            or h4_s.indicators.donchian10_bear_brk
            or h4_s.indicators.supertrend_flipped_bear
        )
        is_fresh_hma_entry = bool(
            (signal in ("STRONG BUY", "★ EARLY BUY") and fresh_bull_trigger)
            or (signal in ("STRONG SELL", "★ EARLY SELL") and fresh_bear_trigger)
        )

        asap_exit_flag, asap_exit_reason, dc5_trail = evaluate_live_asap_exit_warning(
            composite, h4_s.indicators, d1_s.indicators
        )

        # Volatility regime (prioritize TTM Squeeze Fire / Squeeze on H4 or D1)
        if h4_s.indicators.vol_regime in ("SQUEEZE FIRE", "TTM SQUEEZE"):
            vol_regime = f"H4 {h4_s.indicators.vol_regime}"
        elif d1_s.indicators.vol_regime in ("SQUEEZE FIRE", "TTM SQUEEZE", "BREAKOUT", "SQUEEZE"):
            vol_regime = d1_s.indicators.vol_regime
        elif h4_s.indicators.vol_regime in ("BREAKOUT", "SQUEEZE"):
            vol_regime = f"H4 {h4_s.indicators.vol_regime}"
        else:
            vol_regime = d1_s.indicators.vol_regime

        atr_tf = config.risk.atr_timeframe.upper()
        active_atr = h4_s.indicators.atr14 if atr_tf in ("H4", "H4_HMA") else d1_s.indicators.atr14

        h1_df = universe_data[pair].get("H1")
        if h1_df is not None and not h1_df.empty:
            live_price = float(h1_df["Close"].iloc[-1])
        else:
            live_price = h4_s.indicators.close

        trade_plan = build_trade_plan(
            pair=pair,
            composite_score=composite,
            entry_price=live_price,
            atr_value=active_atr,
            risk_cfg=config.risk,
            spot_prices=spot_prices,
            hma21_h4=h4_s.indicators.hma21,
            atr_h4=h4_s.indicators.atr14,
        )

        d1_closes = [float(x) for x in universe_data[pair]["D1"]["Close"].tail(28).tolist()]
        h4_closes = [float(x) for x in universe_data[pair]["H4"]["Close"].tail(28).tolist()]

        is_alert = abs(composite) >= config.thresholds.alert_threshold or asap_exit_flag
        if is_alert:
            alert_pairs.append(f"{pair} ({composite:+.1f})")

        analyses.append(
            PairAnalysis(
                spec=spec,
                price=live_price,
                change_d1_pct=d1_s.indicators.change_pct,
                h1=h1_s,
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
                is_early_entry=is_early_entry,
                top_pick_rank=0,
                top_pick_label="",
                asap_exit_flag=asap_exit_flag,
                asap_exit_reason=asap_exit_reason,
                donchian5_trail_stop=dc5_trail,
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

    # Sort descending by composite_score and assign Top 5 Long & Top 5 Short badges (no currency cap)
    analyses.sort(key=lambda a: a.composite_score, reverse=True)

    long_candidates = [a for a in analyses if a.composite_score >= config.thresholds.buy and "PULLBACK" not in a.signal]
    for idx, item in enumerate(long_candidates[:5], start=1):
        item.top_pick_rank = idx
        item.top_pick_label = f"★ TOP {idx} LONG"

    short_candidates = [
        a for a in reversed(analyses) if a.composite_score <= config.thresholds.sell and "PULLBACK" not in a.signal
    ]
    for idx, item in enumerate(short_candidates[:5], start=1):
        item.top_pick_rank = -idx
        item.top_pick_label = f"★ TOP {idx} SHORT"

    return ScreenerResult(
        pairs=analyses,
        strength_matrix=strength_matrix,
        source=source,
        alert_pairs=alert_pairs,
    )
