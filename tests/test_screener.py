"""Unit and integration tests for the Multi-Factor FX & Precious Metals Screener."""

from __future__ import annotations

from pathlib import Path

from fx_screener.config import AppConfig
from fx_screener.data.base import BASELINE_PRICES, SyntheticProvider
from fx_screener.data.cache import MarketDataCache
from fx_screener.engine.indicators import compute_indicators
from fx_screener.engine.risk import build_trade_plan, compute_pip_value_per_standard_lot_usd
from fx_screener.engine.scorer import analyze_universe
from fx_screener.ui.cli_renderer import export_screener_results, filter_pairs, make_sparkline
from fx_screener.universe import ALL_PAIRS, FX_28_PAIRS, G8_CURRENCIES, UNIVERSE, UNIVERSE_MAP


def test_universe_contains_28_fx_plus_gold_silver() -> None:
    assert len(UNIVERSE) == 30
    assert len(FX_28_PAIRS) == 28
    assert "XAU/USD" in ALL_PAIRS
    assert "XAG/USD" in ALL_PAIRS
    majors = [s for s in UNIVERSE if s.category == "Major"]
    minors = [s for s in UNIVERSE if s.category == "Minor"]
    commodities = [s for s in UNIVERSE if s.category == "Commodity"]
    assert len(majors) == 7
    assert len(minors) == 21
    assert len(commodities) == 2


def test_synthetic_provider_and_indicators() -> None:
    provider = SyntheticProvider(seed=101)
    data = provider.fetch_universe()
    assert len(data) == 30
    for pair in ("EUR/USD", "GBP/JPY", "XAU/USD", "XAG/USD"):
        assert set(data[pair].keys()) == {"H4", "D1", "W1"}
        snap_d1 = compute_indicators(data[pair]["D1"])
        assert snap_d1.close > 0
        assert 0.0 <= snap_d1.rsi14 <= 100.0
        assert snap_d1.atr14 > 0
        assert snap_d1.vol_regime in {"SQUEEZE", "BREAKOUT", "EXPANDING", "NORMAL"}


def test_analyze_universe_scoring_and_strength_matrix() -> None:
    cfg = AppConfig()
    data = SyntheticProvider(seed=42).fetch_universe()
    result = analyze_universe(data, cfg, source="synthetic")

    assert len(result.pairs) == 30
    assert len(result.strength_matrix.ranked) == len(G8_CURRENCIES)

    # Verify G8 currencies are sorted strongest to weakest
    scores = [e.composite_score for e in result.strength_matrix.ranked]
    assert scores == sorted(scores, reverse=True)
    for s in scores:
        assert -100.0 <= s <= 100.0

    # Verify pair composite scores and MTF weighting
    for p in result.pairs:
        assert -100.0 <= p.composite_score <= 100.0
        expected_comp = (
            cfg.timeframe_weights.D1 * p.d1.total_score
            + cfg.timeframe_weights.H4 * p.h4.total_score
            + cfg.timeframe_weights.W1 * p.w1.total_score
        )
        assert abs(p.composite_score - expected_comp) < 1e-4
        assert p.signal in {"STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"}


def test_cross_currency_pip_values_and_trade_planner() -> None:
    cfg = AppConfig()
    # 1. EUR/USD: 1 std lot = $10/pip
    eurusd_pip = compute_pip_value_per_standard_lot_usd(
        UNIVERSE_MAP["EUR/USD"], 1.0540, BASELINE_PRICES
    )
    assert abs(eurusd_pip - 10.0) < 1e-6

    # 2. USD/JPY at 150.00: 1 std lot = 1000 JPY / 150 = $6.6667/pip
    usdjpy_pip = compute_pip_value_per_standard_lot_usd(
        UNIVERSE_MAP["USD/JPY"], 150.00, BASELINE_PRICES
    )
    assert abs(usdjpy_pip - (1000.0 / 150.0)) < 1e-4

    # 3. GBP/JPY cross pair: uses USD/JPY from spot_prices
    gbpjpy_pip = compute_pip_value_per_standard_lot_usd(
        UNIVERSE_MAP["GBP/JPY"], 194.75, BASELINE_PRICES
    )
    assert abs(gbpjpy_pip - (1000.0 / BASELINE_PRICES["USD/JPY"])) < 1e-4

    # 4. Gold XAU/USD: 100 oz * $0.10 = $10/pip
    xau_pip = compute_pip_value_per_standard_lot_usd(
        UNIVERSE_MAP["XAU/USD"], 2650.0, BASELINE_PRICES
    )
    assert abs(xau_pip - 10.0) < 1e-6

    # Verify trade plan risk math ($10,000 @ 1% = $100 risk)
    plan = build_trade_plan(
        pair="EUR/USD",
        composite_score=68.0,
        entry_price=1.0500,
        atr_value=0.0080,  # 80 pips ATR -> 1.5x SL = 120 pips
        risk_cfg=cfg.risk,
        spot_prices=BASELINE_PRICES,
    )
    assert plan.direction == "LONG"
    assert abs(plan.stop_loss_pips - 120.0) < 1e-4
    assert abs(plan.stop_loss_price - (1.0500 - 0.0120)) < 1e-6
    assert abs(plan.tp2_pips - 240.0) < 1e-4
    # $100 risk / (120 pips * $10/pip) = 0.08333 standard lots
    assert abs(plan.standard_lots - (100.0 / 1200.0)) < 1e-4


def test_sqlite_cache_and_export(tmp_path: Path) -> None:
    db_file = tmp_path / "test_cache.sqlite3"
    cache = MarketDataCache(db_path=db_file, ttl_seconds=300)
    synth = SyntheticProvider(seed=7).fetch_universe(["EUR/USD", "XAU/USD"])

    assert not cache.is_fresh(["EUR/USD", "XAU/USD"])
    cache.save_universe(synth, provider_name="synthetic")
    assert cache.is_fresh(["EUR/USD", "XAU/USD"])

    loaded = cache.load_universe(["EUR/USD", "XAU/USD"])
    assert set(loaded.keys()) == {"EUR/USD", "XAU/USD"}
    assert len(loaded["EUR/USD"]["D1"]) == len(synth["EUR/USD"]["D1"])

    # Test export & filters
    cfg = AppConfig()
    full_data = SyntheticProvider(seed=42).fetch_universe()
    res = analyze_universe(full_data, cfg, source="synthetic")

    jpy_pairs = filter_pairs(res.pairs, filter_mode="all", currency_query="JPY")
    assert len(jpy_pairs) == 7

    csv_out = export_screener_results(res, jpy_pairs, tmp_path / "out.csv")
    json_out = export_screener_results(res, jpy_pairs, tmp_path / "out.json")
    assert csv_out.exists()
    assert json_out.exists()
    assert len(make_sparkline([1.0, 1.1, 1.05, 1.2, 1.15])) == 5
