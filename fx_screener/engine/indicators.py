"""Vectorized technical indicators for Fx_Screener2:
- Fast Donchian(10) Entry & Donchian(5) Trailing Exit Channels
- Hull Moving Averages HMA(9) / HMA(21)
- SuperTrend(10, 2.5)
- Zero-Lag Early Trend Efficiency: DX(14), FX-RAVI(7, 65), VHF(14), CHOP(14)
- Momentum Acceleration & Elliott Wave: 2nd-Deriv MACD Δ²H, EWO(5, 35), Causal ATR-ZigZag Wave
- TPO Market Profile (VAH, POC, VAL) & TTM Bollinger/Keltner Squeeze
"""

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
    # Fast Donchian(10) Entry & Donchian(5) Exit Channels (prior-bar causal bounds)
    donchian10_high: float
    donchian10_low: float
    donchian10_mid: float
    donchian10_bull_brk: bool
    donchian10_bear_brk: bool
    donchian5_high: float
    donchian5_low: float
    # SuperTrend(10, 2.5)
    supertrend_line: float
    supertrend_dir: int       # +1 bullish, -1 bearish
    supertrend_flipped_bull: bool
    supertrend_flipped_bear: bool
    # Early Trend Efficiency Ensemble
    dx14: float               # Raw unsmoothed Wilder Directional Index DX(14) (0..100)
    ravi: float               # Chande Range Action Verification Index RAVI(7, 65) (%)
    ravi_rising: bool         # True if RAVI is expanding compared to 2 bars ago
    vhf14: float              # White Vertical Horizontal Filter VHF(14) (0..1)
    chop14: float             # Dreiss Choppiness Index CHOP(14) (0..100; <38.2 trend, >61.8 chop)
    chop_falling: bool        # True if CHOP(14) is dropping vs 2 bars ago (compression -> expansion)
    adx14: float              # Legacy smoothed ADX(14) retained for reference
    plus_di: float
    minus_di: float
    # Momentum Acceleration & Elliott Wave
    rsi14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    macd_prev_hist: float
    macd_accel_d2: float      # 2nd-derivative MACD histogram acceleration (Hist_t - 2*Hist_{t-1} + Hist_{t-2})
    ewo: float                # ATR-normalized Elliott Wave Oscillator: (SMA5 - SMA35) / ATR14
    ewo_rising: bool          # True if EWO is rising vs prior bar
    wave_bias: int            # Causal ATR-ZigZag impulse bias: +1 (bullish impulse/W3), -1 (bearish impulse/W3), 0
    # Volatility, TTM Squeeze & TPO Market Profile
    atr14: float
    atr_pct: float
    atr_ratio: float          # Current ATR(14) / 20-period SMA of ATR(14)
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_width: float
    bb_width_pct_rank: float  # 0..100 percentile of BB width over last 100 bars
    bb_percent_b: float       # (Close - Lower) / (Upper - Lower)
    ttm_squeeze_on: bool      # True if Bollinger(20, 2.0) is inside Keltner(20, 1.5x ATR)
    ttm_squeeze_fired: bool   # True if TTM squeeze just released on current/prior bar
    tpo_poc: float            # Time-Price Opportunity Point of Control
    tpo_vah: float            # TPO 70% Value Area High
    tpo_val: float            # TPO 70% Value Area Low
    roc10: float              # 10-bar rate of change (%)
    roc20: float              # 20-bar rate of change (%)
    vol_regime: str           # "TTM SQUEEZE" | "SQUEEZE FIRE" | "SQUEEZE" | "BREAKOUT" | "EXPANDING" | "NORMAL"


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Vectorized Weighted Moving Average (WMA) with linear weights 1..period."""
    p = max(1, int(period))
    if p <= 1:
        return series.astype(float).copy()
    return series.rolling(p, min_periods=1).apply(
        lambda x: float(np.dot(x, np.arange(1, len(x) + 1, dtype=float)) / (len(x) * (len(x) + 1) / 2.0)),
        raw=True,
    )


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


def donchian_channels(df: pd.DataFrame, period: int = 10) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Causal Donchian Channel using prior `period` bars (shifted by 1 so bar t can break out cleanly)."""
    p = max(2, int(period))
    high_s = df["High"].astype(float)
    low_s = df["Low"].astype(float)
    upper = high_s.shift(1).rolling(p, min_periods=1).max().bfill()
    lower = low_s.shift(1).rolling(p, min_periods=1).min().bfill()
    mid = 0.5 * (upper + lower)
    return upper, lower, mid


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 2.5
) -> tuple[pd.Series, pd.Series]:
    """Causal SuperTrend(period, multiplier) returning (supertrend_line, direction_series [+1/-1])."""
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    close = df["Close"].astype(float).to_numpy()
    atr_arr = atr(df, period).to_numpy()

    hl2 = 0.5 * (high + low)
    basic_ub = hl2 + multiplier * atr_arr
    basic_lb = hl2 - multiplier * atr_arr

    n = len(close)
    final_ub = np.copy(basic_ub)
    final_lb = np.copy(basic_lb)
    st_dir = np.ones(n, dtype=int)
    st_line = np.copy(final_lb)

    for i in range(1, n):
        if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        if st_dir[i - 1] == 1:
            if close[i] < final_lb[i]:
                st_dir[i] = -1
                st_line[i] = final_ub[i]
            else:
                st_dir[i] = 1
                st_line[i] = final_lb[i]
        else:
            if close[i] > final_ub[i]:
                st_dir[i] = 1
                st_line[i] = final_lb[i]
            else:
                st_dir[i] = -1
                st_line[i] = final_ub[i]

    return pd.Series(st_line, index=df.index), pd.Series(st_dir, index=df.index)


def directional_movement(
    df: pd.DataFrame, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute raw unsmoothed DX(14), smoothed ADX(14), +DI(14), and -DI(14)."""
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
    dx_s = (100.0 * ((plus_di - minus_di).abs() / di_sum)).fillna(20.0)
    adx_s = dx_s.ewm(alpha=1.0 / period, adjust=False).mean().fillna(20.0)
    return dx_s, adx_s, plus_di.fillna(20.0), minus_di.fillna(20.0)


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    _, adx_s, plus_di, minus_di = directional_movement(df, period)
    return adx_s, plus_di, minus_di


def ravi(series: pd.Series, short_period: int = 7, long_period: int = 65) -> pd.Series:
    """Tushar Chande's Range Action Verification Index: RAVI = 100 * |SMA(7) - SMA(65)| / SMA(65)."""
    eff_long = min(long_period, max(short_period + 2, len(series)))
    sma_short = series.rolling(short_period, min_periods=2).mean()
    sma_long = series.rolling(eff_long, min_periods=5).mean().replace(0.0, np.nan)
    out = 100.0 * ((sma_short - sma_long).abs() / sma_long)
    return out.fillna(0.0)


def vhf(series: pd.Series, period: int = 14) -> pd.Series:
    """Adam White's Vertical Horizontal Filter: VHF = (MaxClose_n - MinClose_n) / Sum(|Close_i - Close_{i-1}|)."""
    hcp = series.rolling(period, min_periods=3).max()
    lcp = series.rolling(period, min_periods=3).min()
    net_range = (hcp - lcp).abs()
    path_length = series.diff().abs().rolling(period, min_periods=3).sum().replace(0.0, np.nan)
    out = (net_range / path_length).clip(lower=0.0, upper=1.0)
    return out.fillna(0.35)


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """E.W. Dreiss Choppiness Index (0..100): < 38.2 linear trend, > 61.8 sideways consolidation."""
    tr = true_range(df)
    sum_tr = tr.rolling(period, min_periods=3).sum()
    max_high = df["High"].rolling(period, min_periods=3).max()
    min_low = df["Low"].rolling(period, min_periods=3).min()
    hl_range = (max_high - min_low).replace(0.0, np.nan)
    ratio = (sum_tr / hl_range).clip(lower=1.0)
    denom = math.log10(max(2, period))
    chop = 100.0 * (np.log10(ratio) / denom)
    return chop.clip(lower=0.0, upper=100.0).fillna(50.0)


def elliott_wave_oscillator(df: pd.DataFrame, fast: int = 5, slow: int = 35) -> pd.Series:
    """ATR-normalized Elliott Wave Oscillator: (SMA(5, HL2) - SMA(35, HL2)) / ATR(14)."""
    hl2 = 0.5 * (df["High"].astype(float) + df["Low"].astype(float))
    eff_slow = min(slow, max(fast + 2, len(hl2)))
    sma_f = hl2.rolling(fast, min_periods=2).mean()
    sma_s = hl2.rolling(eff_slow, min_periods=5).mean()
    atr_s = atr(df, 14).replace(0.0, np.nan)
    return ((sma_f - sma_s) / atr_s).fillna(0.0)


def causal_atr_zigzag_bias(df: pd.DataFrame, atr_mult: float = 2.0) -> int:
    """Causal ATR-ZigZag pivot sequence detector to identify Wave 1-2-3 bullish (+1) or bearish (-1) structure."""
    if len(df) < 15:
        return 0
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    close = df["Close"].astype(float).to_numpy()
    atr_arr = atr(df, 14).to_numpy()

    pivots: list[tuple[int, float, int]] = []  # (bar_idx, price, +1=High / -1=Low)
    direction = 1 if close[1] >= close[0] else -1
    ext_price = high[0] if direction == 1 else low[0]
    ext_idx = 0

    for i in range(1, len(close)):
        thresh = max(atr_arr[i] * atr_mult, 1e-8)
        if direction == 1:
            if high[i] >= ext_price:
                ext_price = high[i]
                ext_idx = i
            elif (ext_price - low[i]) >= thresh:
                pivots.append((ext_idx, ext_price, 1))
                direction = -1
                ext_price = low[i]
                ext_idx = i
        else:
            if low[i] <= ext_price:
                ext_price = low[i]
                ext_idx = i
            elif (high[i] - ext_price) >= thresh:
                pivots.append((ext_idx, ext_price, -1))
                direction = 1
                ext_price = high[i]
                ext_idx = i

    if len(pivots) < 3:
        return 1 if direction == 1 else -1

    # Inspect last 3 confirmed pivots for higher-low / lower-high Wave 1-2-3 progression
    p1, p2, p3 = pivots[-3], pivots[-2], pivots[-1]
    cur_c = close[-1]
    if p3[2] == -1 and p1[2] == -1:
        # Recent confirmed higher low (Wave 2 pullback holding above Wave 0 low) + price rising
        if p3[1] > p1[1] and cur_c > p3[1]:
            return 1
    if p3[2] == 1 and p1[2] == 1:
        # Recent confirmed lower high (Wave 2 bounce holding below Wave 0 high) + price falling
        if p3[1] < p1[1] and cur_c < p3[1]:
            return -1
    if p3[2] == 1 and p2[2] == -1 and cur_c > p3[1]:
        return 1
    if p3[2] == -1 and p2[2] == 1 and cur_c < p3[1]:
        return -1
    return 0


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period, min_periods=5).mean()
    std = series.rolling(period, min_periods=5).std().fillna(0.0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = ((upper - lower) / mid.replace(0.0, np.nan)).fillna(0.0)
    return upper, mid, lower, width


def keltner_squeeze(
    df: pd.DataFrame, period: int = 20, bb_std: float = 2.0, kc_mult: float = 1.5
) -> pd.Series:
    """TTM Squeeze state series: True when Bollinger(20, 2.0) is strictly inside Keltner(20, 1.5x ATR)."""
    close_s = df["Close"].astype(float)
    bb_u, _, bb_l, _ = bollinger_bands(close_s, period, bb_std)
    kc_mid = ema(close_s, period)
    kc_atr = atr(df, period)
    kc_u = kc_mid + kc_mult * kc_atr
    kc_l = kc_mid - kc_mult * kc_atr
    return (bb_u < kc_u) & (bb_l > kc_l)


def compute_tpo_profile(
    df: pd.DataFrame, lookback: int = 60, num_bins: int = 28, value_area_pct: float = 0.70
) -> tuple[float, float, float]:
    """Steidlmayer Time-Price Opportunity (TPO) Market Profile without requiring centralized volume.

    Counts how many bars traded through each price bin over the rolling `lookback` window,
    identifies the Point of Control (POC), and expands outward to capture 70% Value Area (VAH, VAL).
    """
    sub = df.tail(min(lookback, len(df)))
    if sub.empty:
        return 0.0, 0.0, 0.0

    low_min = float(sub["Low"].min())
    high_max = float(sub["High"].max())
    close_last = float(sub["Close"].iloc[-1])
    if not np.isfinite(low_min) or not np.isfinite(high_max) or high_max <= low_min:
        return close_last, close_last, close_last

    edges = np.linspace(low_min, high_max, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    lows = sub["Low"].astype(float).to_numpy()[:, None]
    highs = sub["High"].astype(float).to_numpy()[:, None]

    # A bar participates in bin j if [Low_i, High_i] overlaps [edge_j, edge_{j+1}]
    overlaps = (highs >= edges[:-1][None, :]) & (lows <= edges[1:][None, :])
    tpo_counts = overlaps.sum(axis=0).astype(float)
    total_tpo = float(tpo_counts.sum())
    if total_tpo <= 0:
        return close_last, close_last, close_last

    poc_idx = int(np.argmax(tpo_counts))
    poc = float(centers[poc_idx])

    target_tpo = total_tpo * value_area_pct
    cum_tpo = float(tpo_counts[poc_idx])
    lo_i = poc_idx
    hi_i = poc_idx

    while cum_tpo < target_tpo and (lo_i > 0 or hi_i < num_bins - 1):
        up_cnt = tpo_counts[hi_i + 1] if hi_i < num_bins - 1 else -1.0
        dn_cnt = tpo_counts[lo_i - 1] if lo_i > 0 else -1.0
        if up_cnt >= dn_cnt:
            hi_i += 1
            cum_tpo += float(tpo_counts[hi_i])
        else:
            lo_i -= 1
            cum_tpo += float(tpo_counts[lo_i])

    val = float(edges[lo_i])
    vah = float(edges[hi_i + 1])
    return poc, vah, val


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

    # Fast Donchian(10) Entry & Donchian(5) Exit Channels
    dc10_u, dc10_l, dc10_m = donchian_channels(df, 10)
    dc5_u, dc5_l, _ = donchian_channels(df, 5)
    donchian10_high = float(dc10_u.iloc[-1])
    donchian10_low = float(dc10_l.iloc[-1])
    donchian10_mid = float(dc10_m.iloc[-1])
    prev_dc10_u = float(dc10_u.iloc[-2]) if len(dc10_u) >= 2 else donchian10_high
    prev_dc10_l = float(dc10_l.iloc[-2]) if len(dc10_l) >= 2 else donchian10_low
    donchian10_bull_brk = bool((close > donchian10_high) and (prev_close <= prev_dc10_u))
    donchian10_bear_brk = bool((close < donchian10_low) and (prev_close >= prev_dc10_l))
    donchian5_high = float(dc5_u.iloc[-1])
    donchian5_low = float(dc5_l.iloc[-1])

    # SuperTrend(10, 2.5)
    st_line_s, st_dir_s = supertrend(df, 10, 2.5)
    supertrend_line = float(st_line_s.iloc[-1])
    supertrend_dir = int(st_dir_s.iloc[-1])
    prev_st_dir = int(st_dir_s.iloc[-2]) if len(st_dir_s) >= 2 else supertrend_dir
    supertrend_flipped_bull = bool(supertrend_dir == 1 and prev_st_dir == -1)
    supertrend_flipped_bear = bool(supertrend_dir == -1 and prev_st_dir == 1)

    # Early Trend Efficiency Ensemble
    dx_s, adx_s, plus_di_s, minus_di_s = directional_movement(df, 14)
    ravi_s = ravi(close_s, 7, 65)
    vhf_s = vhf(close_s, 14)
    chop_s = choppiness_index(df, 14)
    ravi_now = float(ravi_s.iloc[-1])
    ravi_prev2 = float(ravi_s.iloc[-3]) if len(ravi_s) >= 3 else float(ravi_s.iloc[0])
    ravi_rising = bool(ravi_now > ravi_prev2)
    chop_now = float(chop_s.iloc[-1])
    chop_prev2 = float(chop_s.iloc[-3]) if len(chop_s) >= 3 else float(chop_s.iloc[0])
    chop_falling = bool(chop_now < chop_prev2)

    # Momentum Acceleration & Elliott Wave
    rsi_s = rsi(close_s, 14)
    macd_l, macd_sig, macd_h = macd(close_s, 12, 26, 9)
    h_now = float(macd_h.iloc[-1])
    h_prev1 = float(macd_h.iloc[-2]) if len(macd_h) >= 2 else h_now
    h_prev2 = float(macd_h.iloc[-3]) if len(macd_h) >= 3 else h_prev1
    macd_accel_d2 = float(h_now - 2.0 * h_prev1 + h_prev2)

    ewo_s = elliott_wave_oscillator(df, 5, 35)
    ewo_now = float(ewo_s.iloc[-1])
    ewo_prev = float(ewo_s.iloc[-2]) if len(ewo_s) >= 2 else ewo_now
    ewo_rising = bool(ewo_now > ewo_prev)
    wave_bias = causal_atr_zigzag_bias(df, 2.0)

    # Volatility, TTM Squeeze & TPO Market Profile
    atr_s = atr(df, 14)
    atr_sma20 = atr_s.rolling(20, min_periods=5).mean().bfill()
    bb_u, bb_m, bb_l, bb_w = bollinger_bands(close_s, 20, 2.0)
    sqz_s = keltner_squeeze(df, 20, 2.0, 1.5)
    ttm_squeeze_on = bool(sqz_s.iloc[-1])
    prev_sqz = bool(sqz_s.iloc[-2]) if len(sqz_s) >= 2 else False
    prev2_sqz = bool(sqz_s.iloc[-3]) if len(sqz_s) >= 3 else False
    ttm_squeeze_fired = bool((not ttm_squeeze_on) and (prev_sqz or prev2_sqz))

    tpo_poc, tpo_vah, tpo_val = compute_tpo_profile(df, lookback=60, num_bins=28)

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
    if ttm_squeeze_fired:
        vol_regime = "SQUEEZE FIRE"
    elif ttm_squeeze_on:
        vol_regime = "TTM SQUEEZE"
    elif bb_percent_b > 1.0 or bb_percent_b < 0.0:
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
        donchian10_high=donchian10_high,
        donchian10_low=donchian10_low,
        donchian10_mid=donchian10_mid,
        donchian10_bull_brk=donchian10_bull_brk,
        donchian10_bear_brk=donchian10_bear_brk,
        donchian5_high=donchian5_high,
        donchian5_low=donchian5_low,
        supertrend_line=supertrend_line,
        supertrend_dir=supertrend_dir,
        supertrend_flipped_bull=supertrend_flipped_bull,
        supertrend_flipped_bear=supertrend_flipped_bear,
        dx14=float(dx_s.iloc[-1]),
        ravi=ravi_now,
        ravi_rising=ravi_rising,
        vhf14=float(vhf_s.iloc[-1]),
        chop14=chop_now,
        chop_falling=chop_falling,
        adx14=float(adx_s.iloc[-1]),
        plus_di=float(plus_di_s.iloc[-1]),
        minus_di=float(minus_di_s.iloc[-1]),
        rsi14=float(rsi_s.iloc[-1]),
        macd_line=float(macd_l.iloc[-1]),
        macd_signal=float(macd_sig.iloc[-1]),
        macd_hist=h_now,
        macd_prev_hist=h_prev1,
        macd_accel_d2=macd_accel_d2,
        ewo=ewo_now,
        ewo_rising=ewo_rising,
        wave_bias=wave_bias,
        atr14=atr_val,
        atr_pct=(atr_val / close * 100.0) if close else 0.0,
        atr_ratio=atr_ratio,
        bb_upper=bb_upper_val,
        bb_mid=float(bb_m.iloc[-1]),
        bb_lower=bb_lower_val,
        bb_width=cur_w,
        bb_width_pct_rank=bb_width_pct_rank,
        bb_percent_b=bb_percent_b,
        ttm_squeeze_on=ttm_squeeze_on,
        ttm_squeeze_fired=ttm_squeeze_fired,
        tpo_poc=tpo_poc,
        tpo_vah=tpo_vah,
        tpo_val=tpo_val,
        roc10=roc10,
        roc20=roc20,
        vol_regime=vol_regime,
    )
