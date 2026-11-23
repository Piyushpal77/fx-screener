"""ATR-based Trade Planner and Cross-Currency Position Sizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from fx_screener.config import RiskConfig
from fx_screener.universe import InstrumentSpec, UNIVERSE_MAP


@dataclass
class TradePlan:
    pair: str
    direction: str             # "LONG" | "SHORT"
    atr_timeframe: str         # "D1" | "H4" | "H4_HMA"
    stop_mode: str             # "ATR" | "H4_HMA21_TRAIL"
    entry_price: float
    hma21_h4_price: float
    atr_value: float
    atr_pips: float
    stop_loss_price: float
    stop_loss_pips: float
    tp1_price: float
    tp1_pips: float
    tp1_r: float
    tp2_price: float
    tp2_pips: float
    tp2_r: float
    tp3_price: float
    tp3_pips: float
    tp3_r: float
    account_balance_usd: float
    risk_percent: float
    risk_amount_usd: float
    pip_value_per_std_lot_usd: float
    standard_lots: float
    mini_lots: float
    micro_lots: float
    units: int


def compute_pip_value_per_standard_lot_usd(
    spec: InstrumentSpec,
    entry_price: float,
    spot_prices: Dict[str, float],
) -> float:
    """Compute the USD value of 1 pip for 1.00 standard lot across any G8 pair or XAU/XAG."""
    # In quote currency, 1 pip for 1 standard lot is always: contract_size * pip_size
    # Examples:
    # - EUR/USD: 100,000 * 0.0001 = 10.00 USD
    # - USD/JPY: 100,000 * 0.01 = 1,000 JPY
    # - XAU/USD: 100 oz * 0.10 = 10.00 USD
    # - XAG/USD: 5,000 oz * 0.01 = 50.00 USD
    pip_val_in_quote = spec.contract_size * spec.pip_size
    quote = spec.quote

    if quote == "USD":
        return pip_val_in_quote

    # Case 2: Base currency is USD (e.g. USD/JPY, USD/CAD, USD/CHF)
    if spec.base == "USD" and entry_price > 0:
        return pip_val_in_quote / entry_price

    # Case 3: Cross pair (e.g. EUR/GBP, GBP/JPY, AUD/CAD, EUR/AUD)
    # Look up QUOTE/USD or USD/QUOTE from spot_prices
    direct_pair = f"{quote}/USD"
    inverse_pair = f"USD/{quote}"

    if direct_pair in spot_prices and spot_prices[direct_pair] > 0:
        return pip_val_in_quote * spot_prices[direct_pair]
    if inverse_pair in spot_prices and spot_prices[inverse_pair] > 0:
        return pip_val_in_quote / spot_prices[inverse_pair]

    # Fallback approximation if neither quote/USD nor USD/quote is in map
    return 10.0


def build_trade_plan(
    pair: str,
    composite_score: float,
    entry_price: float,
    atr_value: float,
    risk_cfg: RiskConfig,
    spot_prices: Dict[str, float],
    hma21_h4: float | None = None,
    atr_h4: float | None = None,
) -> TradePlan:
    """Generate an actionable ATR or H4 HMA(21) Trail Stop-Loss, 1.5R/2R/3R Take-Profit, and USD Lot Size plan."""
    spec = UNIVERSE_MAP[pair]
    direction = "LONG" if composite_score >= 0 else "SHORT"
    sign = 1.0 if direction == "LONG" else -1.0

    atr_pips = (atr_value / spec.pip_size) if spec.pip_size > 0 else 0.0
    hma_ref = hma21_h4 if (hma21_h4 is not None and hma21_h4 > 0) else entry_price
    h4_atr_ref = atr_h4 if (atr_h4 is not None and atr_h4 > 0) else atr_value

    tf_mode = risk_cfg.atr_timeframe.upper()
    stop_mode = "ATR"

    if tf_mode == "H4_HMA" and hma21_h4 is not None and hma21_h4 > 0:
        # Dynamic H4 HMA(21) Trailing Stop with 0.5x H4 ATR breathing buffer,
        # bounded between [0.75x H4 ATR, sl_atr_multiplier * H4 ATR] to prevent ultra-tight whipsaws
        raw_hma_stop = hma21_h4 - sign * (0.5 * h4_atr_ref)
        raw_dist = sign * (entry_price - raw_hma_stop)
        min_dist = 0.75 * h4_atr_ref
        max_dist = max(min_dist, risk_cfg.sl_atr_multiplier * h4_atr_ref)
        if raw_dist >= min_dist:
            sl_distance = min(raw_dist, max_dist)
        else:
            sl_distance = min_dist
        stop_mode = "H4_HMA21_TRAIL"
    else:
        sl_distance = atr_value * risk_cfg.sl_atr_multiplier

    sl_pips = (sl_distance / spec.pip_size) if spec.pip_size > 0 else 0.0
    stop_loss_price = entry_price - sign * sl_distance

    r_mults: List[float] = list(risk_cfg.tp_r_multiples)
    while len(r_mults) < 3:
        r_mults.append(r_mults[-1] + 1.0 if r_mults else 1.5)

    tp1_dist = sl_distance * r_mults[0]
    tp2_dist = sl_distance * r_mults[1]
    tp3_dist = sl_distance * r_mults[2]

    tp1_price = entry_price + sign * tp1_dist
    tp2_price = entry_price + sign * tp2_dist
    tp3_price = entry_price + sign * tp3_dist

    tp1_pips = sl_pips * r_mults[0]
    tp2_pips = sl_pips * r_mults[1]
    tp3_pips = sl_pips * r_mults[2]

    risk_amount_usd = risk_cfg.account_balance_usd * (risk_cfg.risk_percent / 100.0)
    pip_val_std = compute_pip_value_per_standard_lot_usd(spec, entry_price, spot_prices)

    if sl_pips > 0 and pip_val_std > 0:
        standard_lots = risk_amount_usd / (sl_pips * pip_val_std)
    else:
        standard_lots = 0.0

    mini_lots = standard_lots * 10.0
    micro_lots = standard_lots * 100.0
    units = int(round(standard_lots * spec.contract_size))

    return TradePlan(
        pair=pair,
        direction=direction,
        atr_timeframe=risk_cfg.atr_timeframe,
        stop_mode=stop_mode,
        entry_price=entry_price,
        hma21_h4_price=hma_ref,
        atr_value=atr_value,
        atr_pips=atr_pips,
        stop_loss_price=stop_loss_price,
        stop_loss_pips=sl_pips,
        tp1_price=tp1_price,
        tp1_pips=tp1_pips,
        tp1_r=r_mults[0],
        tp2_price=tp2_price,
        tp2_pips=tp2_pips,
        tp2_r=r_mults[1],
        tp3_price=tp3_price,
        tp3_pips=tp3_pips,
        tp3_r=r_mults[2],
        account_balance_usd=risk_cfg.account_balance_usd,
        risk_percent=risk_cfg.risk_percent,
        risk_amount_usd=risk_amount_usd,
        pip_value_per_std_lot_usd=pip_val_std,
        standard_lots=standard_lots,
        mini_lots=mini_lots,
        micro_lots=micro_lots,
        units=units,
    )
