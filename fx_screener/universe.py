"""Universe of 28 G8 FX pairs and 2 Precious Metals (Gold & Silver)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


G8_CURRENCIES: List[str] = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]


@dataclass(frozen=True)
class InstrumentSpec:
    pair: str              # e.g. "EUR/USD"
    base: str              # e.g. "EUR"
    quote: str             # e.g. "USD"
    category: str          # "Major" | "Minor" | "Commodity"
    yf_ticker: str         # Primary Yahoo Finance ticker
    yf_fallback: str       # Fallback Yahoo Finance ticker
    oanda_instrument: str  # e.g. "EUR_USD"
    pip_size: float        # 0.0001 for standard, 0.01 for JPY/XAG, 0.10 for XAU
    contract_size: float   # 100,000 for FX, 100 oz for XAU, 5,000 oz for XAG
    decimals: int          # Display decimals


def _build_universe() -> List[InstrumentSpec]:
    majors = {"EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"}
    # Standard market convention order for the 28 G8 pairs
    ordered_pairs = [
        # 7 Majors
        ("EUR", "USD"),
        ("GBP", "USD"),
        ("USD", "JPY"),
        ("USD", "CHF"),
        ("AUD", "USD"),
        ("USD", "CAD"),
        ("NZD", "USD"),
        # 6 EUR crosses
        ("EUR", "GBP"),
        ("EUR", "JPY"),
        ("EUR", "CHF"),
        ("EUR", "AUD"),
        ("EUR", "CAD"),
        ("EUR", "NZD"),
        # 5 GBP crosses
        ("GBP", "JPY"),
        ("GBP", "CHF"),
        ("GBP", "AUD"),
        ("GBP", "CAD"),
        ("GBP", "NZD"),
        # 4 AUD crosses
        ("AUD", "JPY"),
        ("AUD", "NZD"),
        ("AUD", "CAD"),
        ("AUD", "CHF"),
        # 3 NZD crosses
        ("NZD", "JPY"),
        ("NZD", "CAD"),
        ("NZD", "CHF"),
        # 2 CAD crosses
        ("CAD", "JPY"),
        ("CAD", "CHF"),
        # 1 CHF cross
        ("CHF", "JPY"),
    ]

    specs: List[InstrumentSpec] = []
    for base, quote in ordered_pairs:
        pair = f"{base}/{quote}"
        is_jpy = quote == "JPY"
        specs.append(
            InstrumentSpec(
                pair=pair,
                base=base,
                quote=quote,
                category="Major" if pair in majors else "Minor",
                yf_ticker=f"{base}{quote}=X",
                yf_fallback=f"{base}{quote}=X",
                oanda_instrument=f"{base}_{quote}",
                pip_size=0.01 if is_jpy else 0.0001,
                contract_size=100_000.0,
                decimals=3 if is_jpy else 5,
            )
        )

    # Add Key Commodities: Gold (XAU/USD) and Silver (XAG/USD)
    specs.append(
        InstrumentSpec(
            pair="XAU/USD",
            base="XAU",
            quote="USD",
            category="Commodity",
            yf_ticker="GC=F",
            yf_fallback="XAUUSD=X",
            oanda_instrument="XAU_USD",
            pip_size=0.10,         # $0.10 per pip on Gold (10 pips = $1.00 move)
            contract_size=100.0,   # 1 standard lot = 100 troy oz
            decimals=2,
        )
    )
    specs.append(
        InstrumentSpec(
            pair="XAG/USD",
            base="XAG",
            quote="USD",
            category="Commodity",
            yf_ticker="SI=F",
            yf_fallback="XAGUSD=X",
            oanda_instrument="XAG_USD",
            pip_size=0.01,         # $0.01 per pip on Silver
            contract_size=5_000.0, # 1 standard lot = 5,000 troy oz
            decimals=3,
        )
    )
    return specs


UNIVERSE: List[InstrumentSpec] = _build_universe()
UNIVERSE_MAP: Dict[str, InstrumentSpec] = {spec.pair: spec for spec in UNIVERSE}
ALL_PAIRS: List[str] = [spec.pair for spec in UNIVERSE]
FX_28_PAIRS: List[str] = [spec.pair for spec in UNIVERSE if spec.category in ("Major", "Minor")]
