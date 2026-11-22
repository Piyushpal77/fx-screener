"""Configuration loader and persistence for the FX Screener."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class TimeframeWeights:
    D1: float = 0.50
    H4: float = 0.30
    W1: float = 0.20


@dataclass
class PillarWeights:
    trend: float = 0.35
    strength: float = 0.25
    momentum: float = 0.25
    volatility: float = 0.15


@dataclass
class ThresholdsConfig:
    strong_buy: float = 60.0
    buy: float = 25.0
    sell: float = -25.0
    strong_sell: float = -60.0
    confluence_min_tf_score: float = 20.0
    alert_threshold: float = 70.0


@dataclass
class RiskConfig:
    account_balance_usd: float = 10000.0
    risk_percent: float = 1.0
    atr_timeframe: str = "D1"  # "D1" or "H4"
    sl_atr_multiplier: float = 1.5
    tp_r_multiples: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0])


@dataclass
class ProviderConfig:
    active: str = "yfinance"
    cache_ttl_seconds: int = 300
    auto_refresh_seconds: int = 300
    oanda_api_key: str = ""
    oanda_account_id: str = ""
    oanda_environment: str = "practice"


@dataclass
class AppConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    timeframe_weights: TimeframeWeights = field(default_factory=TimeframeWeights)
    pillar_weights: PillarWeights = field(default_factory=PillarWeights)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    watchlist: List[str] = field(default_factory=lambda: ["EUR/USD", "GBP/JPY", "XAU/USD"])
    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "AppConfig":
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            return cls(config_path=cfg_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        prov_raw = raw.get("provider", {}) or {}
        oanda_raw = prov_raw.get("oanda", {}) or {}
        provider = ProviderConfig(
            active=prov_raw.get("active", "yfinance"),
            cache_ttl_seconds=int(prov_raw.get("cache_ttl_seconds", 300)),
            auto_refresh_seconds=int(prov_raw.get("auto_refresh_seconds", 300)),
            oanda_api_key=os.environ.get("OANDA_API_KEY", oanda_raw.get("api_key", "")),
            oanda_account_id=os.environ.get("OANDA_ACCOUNT_ID", oanda_raw.get("account_id", "")),
            oanda_environment=oanda_raw.get("environment", "practice"),
        )

        scoring_raw = raw.get("scoring", {}) or {}
        tf_raw = scoring_raw.get("timeframe_weights", {}) or {}
        tf_weights = TimeframeWeights(
            D1=float(tf_raw.get("D1", 0.50)),
            H4=float(tf_raw.get("H4", 0.30)),
            W1=float(tf_raw.get("W1", 0.20)),
        )

        pw_raw = scoring_raw.get("pillar_weights", {}) or {}
        pillar_weights = PillarWeights(
            trend=float(pw_raw.get("trend", 0.35)),
            strength=float(pw_raw.get("strength", 0.25)),
            momentum=float(pw_raw.get("momentum", 0.25)),
            volatility=float(pw_raw.get("volatility", 0.15)),
        )

        th_raw = scoring_raw.get("thresholds", {}) or {}
        thresholds = ThresholdsConfig(
            strong_buy=float(th_raw.get("strong_buy", 60.0)),
            buy=float(th_raw.get("buy", 25.0)),
            sell=float(th_raw.get("sell", -25.0)),
            strong_sell=float(th_raw.get("strong_sell", -60.0)),
            confluence_min_tf_score=float(th_raw.get("confluence_min_tf_score", 20.0)),
            alert_threshold=float(th_raw.get("alert_threshold", 70.0)),
        )

        risk_raw = raw.get("risk", {}) or {}
        risk = RiskConfig(
            account_balance_usd=float(risk_raw.get("account_balance_usd", 10000.0)),
            risk_percent=float(risk_raw.get("risk_percent", 1.0)),
            atr_timeframe=str(risk_raw.get("atr_timeframe", "D1")).upper(),
            sl_atr_multiplier=float(risk_raw.get("sl_atr_multiplier", 1.5)),
            tp_r_multiples=[float(x) for x in risk_raw.get("tp_r_multiples", [1.5, 2.0, 3.0])],
        )

        watchlist = list(raw.get("watchlist", ["EUR/USD", "GBP/JPY", "XAU/USD"]))

        return cls(
            provider=provider,
            timeframe_weights=tf_weights,
            pillar_weights=pillar_weights,
            thresholds=thresholds,
            risk=risk,
            watchlist=watchlist,
            config_path=cfg_path,
        )

    def save(self) -> None:
        data = {
            "provider": {
                "active": self.provider.active,
                "cache_ttl_seconds": self.provider.cache_ttl_seconds,
                "auto_refresh_seconds": self.provider.auto_refresh_seconds,
                "oanda": {
                    "api_key": self.provider.oanda_api_key,
                    "account_id": self.provider.oanda_account_id,
                    "environment": self.provider.oanda_environment,
                },
            },
            "scoring": {
                "timeframe_weights": {
                    "D1": self.timeframe_weights.D1,
                    "H4": self.timeframe_weights.H4,
                    "W1": self.timeframe_weights.W1,
                },
                "pillar_weights": {
                    "trend": self.pillar_weights.trend,
                    "strength": self.pillar_weights.strength,
                    "momentum": self.pillar_weights.momentum,
                    "volatility": self.pillar_weights.volatility,
                },
                "thresholds": {
                    "strong_buy": self.thresholds.strong_buy,
                    "buy": self.thresholds.buy,
                    "sell": self.thresholds.sell,
                    "strong_sell": self.thresholds.strong_sell,
                    "confluence_min_tf_score": self.thresholds.confluence_min_tf_score,
                    "alert_threshold": self.thresholds.alert_threshold,
                },
            },
            "risk": {
                "account_balance_usd": self.risk.account_balance_usd,
                "risk_percent": self.risk.risk_percent,
                "atr_timeframe": self.risk.atr_timeframe,
                "sl_atr_multiplier": self.risk.sl_atr_multiplier,
                "tp_r_multiples": self.risk.tp_r_multiples,
            },
            "watchlist": self.watchlist,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
