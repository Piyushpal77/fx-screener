"""Vectorized technical indicators for the FX Screener (HMA, EMA, ADX, RSI, MACD, Bollinger, ATR)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass
class IndicatorSnapshot:
    close: float
    prev_close: float
    change_pct: float
    ema20: float
    ema50: float
    ema200: float
    ema20_slope_pct: float
    hma9: float
    hma21: float
    hma9_prev: float
    hma21_prev: float
    hma9_slope_pct: float
    hma21_slope_pct: float
    hma_bull_cross: bool      # True if HMA(9) crossed above HMA(21) or turned bullish within last 2 bars
    hma_bear_cross: bool      # True if HMA(9) crossed below HMA(21) or turned bearish within last 2 bars
    hma_status: str           # "HMA TURN BULL" | "HMA BULL" | "HMA TURN BEAR" | "HMA BEAR" | "HMA FLAT"
    adx14: float
    plus_di: float
    minus_di: float
    rsi14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    macd_prev_hist: float
    atr14: float
    atr_pct: float
    atr_ratio: float        # Current ATR(14) / 20-period SMA of ATR(14)
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_width: float
    bb_width_pct_rank: float  # 0..100 percentile of BB width over last 100 bars
    bb_percent_b: float       # (Close - Lower) / (Upper - Lower)
    roc10: float              # 10-bar rate of change (%)
    roc20: float              # 20-bar rate of change (%)
    vol_regime: str           # "SQUEEZE" | "BREAKOUT" | "EXPANDING" | "NORMAL"


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Vectorized Weighted Moving Average (WMA) with linear weights 1..period."""
    p = max(1, int(period))
    if p <= 1:
        return series.astype(float).copy()
    weights = np.arange(1, p + 1, dtype=float)
    w_sum = weights.sum()
    rolled = series.rolling(p, min_periods=1).apply(
        lambda x: float(np.dot(x, np.arange(1, len(x) + 1, dtype=float)) / (len(x) * (len(x) + 1) / 2.0)),
        raw=True,
    )
    return rolled


def hma(series: pd.Series, period: int = 9) -> pd.Series:
    """Hull Moving Average: HMA(n) = WMA(2 * WMA(n/2) - WMA(n), int(sqrt(n)))."""
    p = max(2, int(period))
    half_p = max(1, p // 2)
    sqrt_p = max(1, int(math.isqrt(p)))
    wma_half = wma(series, half_p)
    wma_full = wma(series, p)
    raw_hma = 2.0 * wma_half - wma_full
    return wma(raw_hma, sqrt_p)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().bfill()


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = df["High"]
    low = df["Low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr_s = atr(df, period).replace(0.0, np.nan)
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_s)
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_s)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * ((plus_di - minus_di).abs() / di_sum)
    adx_s = dx.ewm(alpha=1.0 / period, adjust=False).mean().fillna(20.0)
    return adx_s, plus_di.fillna(20.0), minus_di.fillna(20.0)


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period, min_periods=5).mean()
    std = series.rolling(period, min_periods=5).std().fillna(0.0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = ((upper - lower) / mid.replace(0.0, np.nan)).fillna(0.0)
    return upper, mid, lower, width


def compute_indicators(df: pd.DataFrame) -> IndicatorSnapshot:
    """Compute a complete IndicatorSnapshot for an OHLCV DataFrame."""
    if df is None or len(df) < 5:
        raise ValueError("Insufficient OHLCV bars to compute indicators.")

    close_s = df["Close"].astype(float)
    close = float(close_s.iloc[-1])
    prev_close = float(close_s.iloc[-2]) if len(close_s) >= 2 else close
    change_pct = ((close - prev_close) / prev_close * 100.0) if prev_close else 0.0

    ema20_s = ema(close_s, 20)
    ema50_s = ema(close_s, 50)
    ema200_s = ema(close_s, min(200, max(20, len(close_s))))

    ema20_now = float(ema20_s.iloc[-1])
    ema20_prev = float(ema20_s.iloc[-4]) if len(ema20_s) >= 4 else float(ema20_s.iloc[0])
    ema20_slope_pct = ((ema20_now - ema20_prev) / ema20_prev * 100.0) if ema20_prev else 0.0

    # Hull Moving Averages: Fast HMA(9) and Slow HMA(21)
    hma9_s = hma(close_s, 9)
    hma21_s = hma(close_s, 21)
    hma9_now = float(hma9_s.iloc[-1])
    hma9_prev = float(hma9_s.iloc[-2]) if len(hma9_s) >= 2 else hma9_now
    hma9_prev2 = float(hma9_s.iloc[-3]) if len(hma9_s) >= 3 else hma9_prev
    hma21_now = float(hma21_s.iloc[-1])
    hma21_prev = float(hma21_s.iloc[-2]) if len(hma21_s) >= 2 else hma21_now

    hma9_slope_pct = ((hma9_now - hma9_prev) / hma9_prev * 100.0) if hma9_prev else 0.0
    hma21_slope_pct = ((hma21_now - hma21_prev) / hma21_prev * 100.0) if hma21_prev else 0.0

    # Detect fresh HMA(9)/HMA(21) crossover or HMA(9) pullback resumption turn
    # A genuine bullish pullback turn requires either:
    # 1. HMA(9) crossing above HMA(21) on the current bar, OR
    # 2. Price / HMA(9) dipping into a pullback over the recent 3 bars (hma9_prev < hma9_prev2 or low dipped near/below HMA21)
    #    and HMA(9) now hooking upward with Close >= HMA(9)
    recent_lows = df["Low"].astype(float).iloc[-4:-1] if len(df) >= 4 else df["Low"].astype(float)
    recent_highs = df["High"].astype(float).iloc[-4:-1] if len(df) >= 4 else df["High"].astype(float)
    dipped_to_hma21 = bool((recent_lows <= hma21_now * 1.0015).any()) if len(recent_lows) > 0 else False
    rallied_to_hma21 = bool((recent_highs >= hma21_now * 0.9985).any()) if len(recent_highs) > 0 else False

    crossed_up = (hma9_now > hma21_now) and (hma9_prev <= hma21_prev)
    turned_up = (hma9_now > hma9_prev) and ((hma9_prev < hma9_prev2) or dipped_to_hma21) and (close >= hma9_now)
    hma_bull_cross = bool(crossed_up or turned_up)

    crossed_down = (hma9_now < hma21_now) and (hma9_prev >= hma21_prev)
    turned_down = (hma9_now < hma9_prev) and ((hma9_prev > hma9_prev2) or rallied_to_hma21) and (close <= hma9_now)
    hma_bear_cross = bool(crossed_down or turned_down)

    if hma_bull_cross and hma9_now >= hma21_now:
        hma_status = "HMA TURN BULL"
    elif hma_bear_cross and hma9_now <= hma21_now:
        hma_status = "HMA TURN BEAR"
    elif hma9_now > hma21_now and hma9_now >= hma9_prev:
        hma_status = "HMA BULL"
    elif hma9_now < hma21_now and hma9_now <= hma9_prev:
        hma_status = "HMA BEAR"
    elif hma9_now > hma9_prev:
        hma_status = "HMA RECOVER"
    elif hma9_now < hma9_prev:
        hma_status = "HMA PULLBACK"
    else:
        hma_status = "HMA FLAT"

    adx_s, plus_di_s, minus_di_s = adx(df, 14)
    rsi_s = rsi(close_s, 14)
    macd_l, macd_sig, macd_h = macd(close_s, 12, 26, 9)
    atr_s = atr(df, 14)
    atr_sma20 = atr_s.rolling(20, min_periods=5).mean().bfill()
    bb_u, bb_m, bb_l, bb_w = bollinger_bands(close_s, 20, 2.0)

    atr_val = float(atr_s.iloc[-1])
    atr_avg = float(atr_sma20.iloc[-1]) if float(atr_sma20.iloc[-1]) > 0 else atr_val
    atr_ratio = atr_val / atr_avg if atr_avg > 0 else 1.0

    recent_w = bb_w.tail(100).dropna()
    cur_w = float(bb_w.iloc[-1])
    if len(recent_w) > 5:
        bb_width_pct_rank = float((recent_w < cur_w).sum() / len(recent_w) * 100.0)
    else:
        bb_width_pct_rank = 50.0

    bb_upper_val = float(bb_u.iloc[-1])
    bb_lower_val = float(bb_l.iloc[-1])
    bb_span = bb_upper_val - bb_lower_val
    bb_percent_b = ((close - bb_lower_val) / bb_span) if bb_span > 0 else 0.5

    idx_10 = max(0, len(close_s) - 11)
    idx_20 = max(0, len(close_s) - 21)
    c10 = float(close_s.iloc[idx_10])
    c20 = float(close_s.iloc[idx_20])
    roc10 = ((close - c10) / c10 * 100.0) if c10 else 0.0
    roc20 = ((close - c20) / c20 * 100.0) if c20 else 0.0

    # Determine Volatility Regime
    if bb_percent_b > 1.0 or bb_percent_b < 0.0:
        vol_regime = "BREAKOUT"
    elif bb_width_pct_rank <= 22.0:
        vol_regime = "SQUEEZE"
    elif atr_ratio >= 1.25:
        vol_regime = "EXPANDING"
    else:
        vol_regime = "NORMAL"

    return IndicatorSnapshot(
        close=close,
        prev_close=prev_close,
        change_pct=change_pct,
        ema20=ema20_now,
        ema50=float(ema50_s.iloc[-1]),
        ema200=float(ema200_s.iloc[-1]),
        ema20_slope_pct=ema20_slope_pct,
        hma9=hma9_now,
        hma21=hma21_now,
        hma9_prev=hma9_prev,
        hma21_prev=hma21_prev,
        hma9_slope_pct=hma9_slope_pct,
        hma21_slope_pct=hma21_slope_pct,
        hma_bull_cross=hma_bull_cross,
        hma_bear_cross=hma_bear_cross,
        hma_status=hma_status,
        adx14=float(adx_s.iloc[-1]),
        plus_di=float(plus_di_s.iloc[-1]),
        minus_di=float(minus_di_s.iloc[-1]),
        rsi14=float(rsi_s.iloc[-1]),
        macd_line=float(macd_l.iloc[-1]),
        macd_signal=float(macd_sig.iloc[-1]),
        macd_hist=float(macd_h.iloc[-1]),
        macd_prev_hist=float(macd_h.iloc[-2]) if len(macd_h) >= 2 else float(macd_h.iloc[-1]),
        atr14=atr_val,
        atr_pct=(atr_val / close * 100.0) if close else 0.0,
        atr_ratio=atr_ratio,
        bb_upper=bb_upper_val,
        bb_mid=float(bb_m.iloc[-1]),
        bb_lower=bb_lower_val,
        bb_width=cur_w,
        bb_width_pct_rank=bb_width_pct_rank,
        bb_percent_b=bb_percent_b,
        roc10=roc10,
        roc20=roc20,
        vol_regime=vol_regime,
    )
