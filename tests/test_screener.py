"""Unit and integration tests for BOTH fx_screener (v1) and fx_screener2 (v2)."""

from __future__ import annotations

from pathlib import Path

from fx_screener.config import AppConfig as AppConfigV1
from fx_screener.data.base import SyntheticProvider as SyntheticProviderV1
from fx_screener.engine.scorer import analyze_universe as analyze_universe_v1

from fx_screener2.config import AppConfig as AppConfigV2
from fx_screener2.data.base import BASELINE_PRICES, SyntheticProvider as SyntheticProviderV2
from fx_screener2.data.cache import MarketDataCache as MarketDataCacheV2
from fx_screener2.engine.indicators import compute_indicators as compute_indicators_v2
from fx_screener2.engine.risk import build_trade_plan, compute_pip_value_per_standard_lot_usd
from fx_screener2.engine.scorer import analyze_universe as analyze_universe_v2
from fx_screener2.engine.tracker import SignalTracker as SignalTrackerV2
from fx_screener2.ui.cli_renderer import export_screener_results, filter_pairs, make_sparkline
from fx_screener2.universe import ALL_PAIRS, FX_28_PAIRS, G8_CURRENCIES, UNIVERSE, UNIVERSE_MAP


def test_v1_screener_intact() -> None:
    cfg = AppConfigV1()
    assert abs(cfg.pillar_weights.trend - 0.35) < 1e-6
    assert abs(cfg.timeframe_weights.D1 - 0.50) < 1e-6
    data = SyntheticProviderV1(seed=42).fetch_universe(["EUR/USD", "GBP/JPY", "XAU/USD"])
    res = analyze_universe_v1(data, cfg, source="synthetic")
    assert len(res.pairs) == 3


def test_v2_universe_and_early_indicators() -> None:
    assert len(UNIVERSE) == 31
    assert len(FX_28_PAIRS) == 28
    assert "USD/INR" in ALL_PAIRS
    assert "XAU/USD" in ALL_PAIRS

    provider = SyntheticProviderV2(seed=101)
    data = provider.fetch_universe()
    assert len(data) == 31
    for pair in ("EUR/USD", "GBP/JPY", "USD/JPY", "USD/INR", "XAU/USD", "XAG/USD"):
        assert set(data[pair].keys()) == {"H1", "H4", "D1", "W1"}
        snap_d1 = compute_indicators_v2(data[pair]["D1"])
        assert snap_d1.close > 0
        assert 0.0 <= snap_d1.rsi14 <= 100.0
        assert 0.0 <= snap_d1.dx14 <= 100.0
        assert snap_d1.ravi >= 0.0
        assert 0.0 <= snap_d1.vhf14 <= 1.0
        assert 0.0 <= snap_d1.chop14 <= 100.0
        assert snap_d1.donchian10_high >= snap_d1.donchian10_low > 0
        assert snap_d1.donchian5_high >= snap_d1.donchian5_low > 0
        assert snap_d1.supertrend_dir in (-1, 1)
        assert snap_d1.tpo_vah >= snap_d1.tpo_poc >= snap_d1.tpo_val > 0
        assert snap_d1.atr14 > 0


def test_v2_scoring_weights_and_top_picks() -> None:
    cfg = AppConfigV2()
    assert abs(cfg.pillar_weights.trend - 0.45) < 1e-6
    assert abs(cfg.pillar_weights.momentum - 0.30) < 1e-6
    assert abs(cfg.pillar_weights.strength - 0.15) < 1e-6
    assert abs(cfg.pillar_weights.volatility - 0.10) < 1e-6
    assert abs(cfg.timeframe_weights.H4 - 0.45) < 1e-6
    assert abs(cfg.timeframe_weights.D1 - 0.40) < 1e-6
    assert abs(cfg.timeframe_weights.W1 - 0.15) < 1e-6

    data = SyntheticProviderV2(seed=42).fetch_universe()
    result = analyze_universe_v2(data, cfg, source="synthetic")
    assert len(result.pairs) == 31
    assert len(result.strength_matrix.ranked) == len(G8_CURRENCIES)

    for p in result.pairs:
        assert -100.0 <= p.composite_score <= 100.0
        assert p.donchian5_trail_stop > 0


def test_v2_web_app_and_asap_exit_engine(tmp_path: Path) -> None:
    from fx_screener2.web.server import create_web_app

    cfg = AppConfigV2()
    cfg.provider.active = "synthetic"
    tracker = SignalTrackerV2(
        db_path=tmp_path / "v2_test.sqlite3",
        csv_path=tmp_path / "v2_test.csv",
    )
    app = create_web_app(config=cfg, tracker=tracker)
    state = app.state.screener
    assert state.result is not None
    assert len(state.result.pairs) == 31

    iso_tracker = SignalTrackerV2(
        db_path=tmp_path / "v2_iso.sqlite3",
        csv_path=tmp_path / "v2_iso.csv",
    )
    long_pair = next(p for p in state.result.pairs if p.composite_score >= 25.0)
    long_pair.signal = "STRONG BUY"
    long_pair.is_fresh_hma_entry = True
    long_pair.hma_trigger = "⚡ H4 DONCHIAN(10) BULL BRK"

    signals = iso_tracker.sync_and_evaluate([long_pair], state.universe_data, source="synthetic", allow_synthetic=True)
    assert len(signals) == 1
    sig = signals[0]

    # Trigger ASAP Exit via H4 HMA(9/21) reversal before SL is hit
    mid_loss_price = sig.entry_price - 0.35 * (sig.entry_price - sig.stop_loss_price)
    long_pair.price = mid_loss_price
    long_pair.h4.indicators.hma21 = sig.entry_price + 0.0010
    long_pair.h4.indicators.hma9 = sig.entry_price - 0.0005
    long_pair.h4.indicators.hma9_prev = sig.entry_price + 0.0005

    updated = iso_tracker.sync_and_evaluate([long_pair], state.universe_data, source="synthetic", allow_synthetic=True)
    assert updated[0].asap_exit_alert is True
    assert "EXIT NOW" in updated[0].asap_exit_reason
