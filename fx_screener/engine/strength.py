"""G8 Currency Relative Strength Matrix Calculator (-100 to +100)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from fx_screener.config import TimeframeWeights
from fx_screener.engine.indicators import IndicatorSnapshot
from fx_screener.universe import FX_28_PAIRS, G8_CURRENCIES, UNIVERSE_MAP


@dataclass
class CurrencyStrengthEntry:
    currency: str
    h4_score: float
    d1_score: float
    w1_score: float
    composite_score: float
    rank: int


@dataclass
class CurrencyStrengthMatrix:
    by_timeframe: Dict[str, Dict[str, float]]  # {'H4': {'USD': 45.2, ...}, 'D1': ..., 'W1': ...}
    composite: Dict[str, float]                # {'USD': 52.1, ...}
    ranked: List[CurrencyStrengthEntry]        # Sorted strongest -> weakest
    top_pairings: List[Tuple[str, str, float]] # [(pair, direction, spread_score)]


def compute_currency_strength_matrix(
    snapshots: Dict[str, Dict[str, IndicatorSnapshot]],
    tf_weights: TimeframeWeights | None = None,
) -> CurrencyStrengthMatrix:
    """Compute relative strength (-100 to +100) for all 8 major currencies across H4, D1, W1."""
    weights = tf_weights or TimeframeWeights()
    by_tf: Dict[str, Dict[str, float]] = {}

    for tf in ("H4", "D1", "W1"):
        raw_scores: Dict[str, List[float]] = {c: [] for c in G8_CURRENCIES}

        for pair in FX_28_PAIRS:
            spec = UNIVERSE_MAP[pair]
            snap = snapshots.get(pair, {}).get(tf)
            if snap is None:
                continue

            # Blend normalized ROC(10), ROC(20), RSI deviation from 50, and HMA(9)/EMA(20) impulse
            ema20_dist_pct = ((snap.close - snap.ema20) / snap.ema20 * 100.0) if snap.ema20 else 0.0
            hma_dist_pct = ((snap.hma9 - snap.hma21) / snap.hma21 * 100.0) if snap.hma21 else 0.0
            rsi_centered = (snap.rsi14 - 50.0) / 25.0  # roughly -2..+2
            # Normalize ROC by ATR% so high-beta pairs don't skew the basket
            vol_norm = max(snap.atr_pct, 0.15)
            roc_signal = (0.6 * snap.roc10 + 0.4 * snap.roc20) / (vol_norm * 3.0)
            ema_signal = (0.5 * ema20_dist_pct + 0.5 * hma_dist_pct) / (vol_norm * 2.0)

            pair_impulse = float(np.clip(0.45 * roc_signal + 0.30 * rsi_centered + 0.25 * ema_signal, -2.5, 2.5))
            pair_score = (pair_impulse / 2.5) * 100.0

            raw_scores[spec.base].append(pair_score)
            raw_scores[spec.quote].append(-pair_score)

        tf_map: Dict[str, float] = {}
        for ccy in G8_CURRENCIES:
            vals = raw_scores[ccy]
            avg = float(np.mean(vals)) if vals else 0.0
            tf_map[ccy] = avg

        # Expand dynamic range across the G8 basket so strongest/weakest utilize -100..+100 cleanly
        max_abs = max((abs(v) for v in tf_map.values()), default=1.0)
        scale = min(1.85, 85.0 / max(max_abs, 1.0)) if max_abs > 0 else 1.0
        for ccy in G8_CURRENCIES:
            tf_map[ccy] = float(np.clip(tf_map[ccy] * scale, -100.0, 100.0))

        by_tf[tf] = tf_map

    composite: Dict[str, float] = {}
    for ccy in G8_CURRENCIES:
        comp = (
            weights.D1 * by_tf["D1"].get(ccy, 0.0)
            + weights.H4 * by_tf["H4"].get(ccy, 0.0)
            + weights.W1 * by_tf["W1"].get(ccy, 0.0)
        )
        composite[ccy] = float(np.clip(comp, -100.0, 100.0))

    sorted_ccys = sorted(G8_CURRENCIES, key=lambda c: composite[c], reverse=True)
    ranked: List[CurrencyStrengthEntry] = []
    for idx, ccy in enumerate(sorted_ccys, start=1):
        ranked.append(
            CurrencyStrengthEntry(
                currency=ccy,
                h4_score=by_tf["H4"][ccy],
                d1_score=by_tf["D1"][ccy],
                w1_score=by_tf["W1"][ccy],
                composite_score=composite[ccy],
                rank=idx,
            )
        )

    # Identify top strong-vs-weak currency pairings mapped to actual tradable G8 pairs
    pair_spreads: List[Tuple[str, str, float]] = []
    for pair in FX_28_PAIRS:
        spec = UNIVERSE_MAP[pair]
        diff = composite[spec.base] - composite[spec.quote]
        direction = "LONG" if diff >= 0 else "SHORT"
        pair_spreads.append((pair, direction, diff))

    pair_spreads.sort(key=lambda x: abs(x[2]), reverse=True)

    return CurrencyStrengthMatrix(
        by_timeframe=by_tf,
        composite=composite,
        ranked=ranked,
        top_pairings=pair_spreads[:5],
    )


def get_pair_strength_pillar_score(
    pair: str,
    timeframe: str,
    snap: IndicatorSnapshot,
    matrix: CurrencyStrengthMatrix,
) -> float:
    """Return the -100..+100 Currency Strength differential score for a given pair and timeframe."""
    spec = UNIVERSE_MAP[pair]
    tf_scores = matrix.by_timeframe.get(timeframe, {})

    if spec.base in tf_scores and spec.quote in tf_scores:
        diff = tf_scores[spec.base] - tf_scores[spec.quote]
        # Diff ranges roughly -170..+170; scale to -100..+100
        return float(np.clip(diff * 0.75, -100.0, 100.0))

    # For non-G8 pairs (e.g. XAU/USD, XAG/USD, USD/INR), compute intrinsic momentum vs USD basket score
    usd_score = tf_scores.get("USD", 0.0)
    vol_norm = max(snap.atr_pct, 0.25)
    pair_impulse = ((0.6 * snap.roc10 + 0.4 * snap.roc20) / (vol_norm * 2.5)) * 45.0 + (snap.rsi14 - 50.0) * 1.5
    pair_score = float(np.clip(pair_impulse, -100.0, 100.0))
    if spec.base == "USD":
        return float(np.clip(0.55 * pair_score + 0.45 * usd_score, -100.0, 100.0))
    return float(np.clip((pair_score - usd_score) * 0.75, -100.0, 100.0))
