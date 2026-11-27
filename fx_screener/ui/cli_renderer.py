"""Rich CLI renderer, sparkline generator, and CSV/JSON exporter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fx_screener.engine.scorer import PairAnalysis, ScreenerResult
from fx_screener.engine.strength import CurrencyStrengthMatrix


SPARK_CHARS = " ▂▃▄▅▆▇█"


def make_sparkline(values: List[float], width: int = 24) -> str:
    if not values:
        return ""
    series = values[-width:]
    lo = min(series)
    hi = max(series)
    span = hi - lo
    if span <= 0:
        return "▄" * len(series)
    chars = []
    for v in series:
        idx = int(round(((v - lo) / span) * (len(SPARK_CHARS) - 2))) + 1
        idx = max(1, min(len(SPARK_CHARS) - 1, idx))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def make_score_bar(score: float, half_width: int = 8) -> Text:
    """Create a bidirectional Rich bar for scores from -100 to +100."""
    clamped = max(-100.0, min(100.0, score))
    filled = int(round((abs(clamped) / 100.0) * half_width))
    t = Text()
    if clamped >= 0:
        left = "·" * half_width
        right = "█" * filled + "·" * (half_width - filled)
        color = "bold bright_green" if clamped >= 60 else ("green" if clamped >= 20 else "dim white")
        t.append(left, style="dim")
        t.append("│", style="bright_black")
        t.append(right, style=color)
    else:
        left = "·" * (half_width - filled) + "█" * filled
        right = "·" * half_width
        color = "bold bright_red" if clamped <= -60 else ("red" if clamped <= -20 else "dim white")
        t.append(left, style=color)
        t.append("│", style="bright_black")
        t.append(right, style="dim")
    return t


def format_score_text(score: float) -> Text:
    if score >= 60:
        style = "bold bright_green"
    elif score >= 25:
        style = "green"
    elif score <= -60:
        style = "bold bright_red"
    elif score <= -25:
        style = "red"
    else:
        style = "yellow"
    return Text(f"{score:+6.1f}", style=style)


def format_signal_badge(signal: str) -> Text:
    styles = {
        "STRONG BUY": "bold black on bright_green",
        "BUY": "bold bright_green",
        "BUY (PULLBACK)": "bold yellow",
        "NEUTRAL": "dim white",
        "SELL (PULLBACK)": "bold yellow",
        "SELL": "bold bright_red",
        "STRONG SELL": "bold white on red",
    }
    return Text(f" {signal:^15} ", style=styles.get(signal, "white"))


def filter_pairs(
    pairs: Iterable[PairAnalysis],
    filter_mode: str = "all",
    currency_query: str = "",
) -> List[PairAnalysis]:
    out = list(pairs)
    mode = (filter_mode or "all").lower()
    if mode == "hma_turn":
        out = [p for p in out if "TURN" in p.hma_trigger]
    elif mode == "confluence":
        out = [p for p in out if p.confluence]
    if mode == "strong":
        out = [p for p in out if p.signal in ("STRONG BUY", "STRONG SELL")]
    elif mode == "squeeze":
        out = [p for p in out if "SQUEEZE" in p.vol_regime or "BREAKOUT" in p.vol_regime]
    elif mode == "watchlist":
        out = [p for p in out if p.is_watchlist]

    q = (currency_query or "").strip().upper()
    if q:
        out = [
            p
            for p in out
            if q in p.spec.pair.upper()
            or q == p.spec.base.upper()
            or q == p.spec.quote.upper()
            or q in p.signal.upper()
            or q in p.hma_trigger.upper()
        ]
    return out


def build_strength_rich_table(matrix: CurrencyStrengthMatrix) -> Table:
    tbl = Table(
        title="G8 Currency Relative Strength Matrix (-100 to +100)",
        expand=True,
        box=None,
        pad_edge=False,
    )
    tbl.add_column("#", justify="right", style="dim", width=2)
    tbl.add_column("CCY", style="bold cyan", width=4)
    tbl.add_column("Score", justify="right", width=7)
    tbl.add_column("Meter", width=17)
    tbl.add_column("H4", justify="right", width=6)
    tbl.add_column("D1", justify="right", width=6)
    tbl.add_column("W1", justify="right", width=6)

    for entry in matrix.ranked:
        tbl.add_row(
            str(entry.rank),
            entry.currency,
            format_score_text(entry.composite_score),
            make_score_bar(entry.composite_score, half_width=7),
            format_score_text(entry.h4_score),
            format_score_text(entry.d1_score),
            format_score_text(entry.w1_score),
        )
    return tbl


def render_cli_once(
    result: ScreenerResult,
    pairs: List[PairAnalysis],
    console: Console | None = None,
) -> None:
    """Print the G8 Currency Strength table and 30-Instrument Screener Table to stdout."""
    con = console or Console()

    # 1. G8 Currency Strength Panel
    strength_tbl = build_strength_rich_table(result.strength_matrix)
    top_ideas = "  |  ".join(
        f"[bold]{p}[/bold] ({d} Δ{s:+.0f})"
        for p, d, s in result.strength_matrix.top_pairings[:4]
    )
    con.print(
        Panel(
            strength_tbl,
            subtitle=f"[bold cyan]Top Divergence Pairings:[/bold cyan] {top_ideas}",
            border_style="cyan",
        )
    )

    # 2. Main Screener Table
    table = Table(
        title=f"FX & Precious Metals Multi-Factor Screener ({len(pairs)} Instruments | Source: {result.source})",
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("★", justify="center", no_wrap=True)
    table.add_column("Pair", style="bold white", no_wrap=True)
    table.add_column("Price", justify="right", no_wrap=True)
    table.add_column("D1%", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Signal", justify="center", no_wrap=True)
    table.add_column("H4", justify="right", no_wrap=True)
    table.add_column("D1", justify="right", no_wrap=True)
    table.add_column("W1", justify="right", no_wrap=True)
    table.add_column("Confluence", justify="center", no_wrap=True)
    table.add_column("DX/RAVI", justify="right", no_wrap=True)
    table.add_column("RSI", justify="right", no_wrap=True)
    table.add_column("Regime", justify="center", no_wrap=True)
    table.add_column("Trade Plan (1% Risk)", justify="left", no_wrap=True)

    for p in pairs:
        dec = p.spec.decimals
        chg_style = "green" if p.change_d1_pct >= 0 else "red"
        star = "[bold yellow]★[/bold yellow]" if p.is_watchlist else " "
        conf_style = (
            "bold bright_green"
            if "BULL" in p.confluence_label
            else ("bold bright_red" if "BEAR" in p.confluence_label else "dim")
        )
        vol_style = (
            "bold magenta"
            if "SQUEEZE" in p.vol_regime
            else ("bold bright_yellow" if "BREAKOUT" in p.vol_regime else "dim white")
        )
        tp = p.trade_plan
        plan_str = (
            f"{tp.direction:<5} SL:{tp.stop_loss_price:.{dec}f} "
            f"TP2:{tp.tp2_price:.{dec}f} ({tp.standard_lots:.2f}L)"
        )

        table.add_row(
            star,
            p.spec.pair,
            f"{p.price:.{dec}f}",
            Text(f"{p.change_d1_pct:+.2f}%", style=chg_style),
            format_score_text(p.composite_score),
            format_signal_badge(p.signal),
            format_score_text(p.h4.total_score),
            format_score_text(p.d1.total_score),
            format_score_text(p.w1.total_score),
            Text(p.confluence_label, style=conf_style),
            f"{p.d1.indicators.dx14:.0f}/{p.d1.indicators.ravi:.1f}%",
            f"{p.d1.indicators.rsi14:.1f}",
            Text(p.vol_regime, style=vol_style),
            plan_str,
        )

    con.print(table)


def export_screener_results(
    result: ScreenerResult,
    pairs: List[PairAnalysis],
    output_path: str | Path,
) -> Path:
    """Export screener rows to CSV or JSON based on file extension."""
    path = Path(output_path)
    rows = []
    for p in pairs:
        tp = p.trade_plan
        rows.append(
            {
                "pair": p.spec.pair,
                "category": p.spec.category,
                "watchlist": p.is_watchlist,
                "price": round(p.price, p.spec.decimals),
                "change_d1_pct": round(p.change_d1_pct, 3),
                "composite_score": round(p.composite_score, 2),
                "raw_composite_score": round(p.raw_composite_score, 2),
                "hma_h4_boost": round(p.hma_h4_boost, 2),
                "hma_trigger": p.hma_trigger,
                "signal": p.signal,
                "h4_score": round(p.h4.total_score, 2),
                "d1_score": round(p.d1.total_score, 2),
                "w1_score": round(p.w1.total_score, 2),
                "confluence": p.confluence,
                "confluence_label": p.confluence_label,
                "pillar_trend": round(p.pillar_trend, 2),
                "pillar_strength": round(p.pillar_strength, 2),
                "pillar_momentum": round(p.pillar_momentum, 2),
                "pillar_volatility": round(p.pillar_volatility, 2),
                "dx_d1": round(p.d1.indicators.dx14, 2),
                "ravi_d1": round(p.d1.indicators.ravi, 2),
                "vhf_d1": round(p.d1.indicators.vhf14, 2),
                "chop_d1": round(p.d1.indicators.chop14, 2),
                "adx_d1": round(p.d1.indicators.adx14, 2),
                "rsi_d1": round(p.d1.indicators.rsi14, 2),
                "vol_regime": p.vol_regime,
                "trade_direction": tp.direction,
                "atr_timeframe": tp.atr_timeframe,
                "atr_pips": round(tp.atr_pips, 1),
                "stop_loss_price": round(tp.stop_loss_price, p.spec.decimals),
                "stop_loss_pips": round(tp.stop_loss_pips, 1),
                "tp1_1_5r_price": round(tp.tp1_price, p.spec.decimals),
                "tp2_2_0r_price": round(tp.tp2_price, p.spec.decimals),
                "tp3_3_0r_price": round(tp.tp3_price, p.spec.decimals),
                "standard_lots": round(tp.standard_lots, 2),
                "units": tp.units,
            }
        )

    if path.suffix.lower() == ".json":
        payload = {
            "source": result.source,
            "currency_strength": [
                {
                    "rank": e.rank,
                    "currency": e.currency,
                    "composite_score": round(e.composite_score, 2),
                    "h4_score": round(e.h4_score, 2),
                    "d1_score": round(e.d1_score, 2),
                    "w1_score": round(e.w1_score, 2),
                }
                for e in result.strength_matrix.ranked
            ],
            "pairs": rows,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    return path
