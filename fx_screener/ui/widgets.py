"""Custom Textual Widgets for the FX Screener Dashboard."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from fx_screener.engine.scorer import PairAnalysis
from fx_screener.engine.strength import CurrencyStrengthMatrix
from fx_screener.ui.cli_renderer import (
    build_strength_rich_table,
    format_score_text,
    format_signal_badge,
    make_score_bar,
    make_sparkline,
)


class CurrencyStrengthWidget(Static):
    """Top-Right Panel displaying the G8 Currency Strength Matrix & Top Divergence Pairings."""

    def update_matrix(self, matrix: CurrencyStrengthMatrix) -> None:
        tbl = build_strength_rich_table(matrix)

        ideas_tbl = Table(box=None, expand=True, show_header=False, pad_edge=False)
        ideas_tbl.add_column("Idea", style="bold white")
        ideas_tbl.add_column("Dir", justify="center")
        ideas_tbl.add_column("Spread", justify="right")

        for pair, direction, spread in matrix.top_pairings[:4]:
            dir_style = "bold bright_green" if direction == "LONG" else "bold bright_red"
            ideas_tbl.add_row(
                pair,
                Text(direction, style=dir_style),
                Text(f"Δ {spread:+.1f}", style="cyan"),
            )

        group = Group(
            tbl,
            Text("\nTop Strong vs. Weak Pairings:", style="bold cyan"),
            ideas_tbl,
        )
        self.update(Panel(group, title="[bold cyan]G8 Currency Strength[/bold cyan]", border_style="cyan"))


class PairInspectorWidget(Static):
    """Bottom-Right Panel displaying 4-Pillar MTF Breakdown, Price Sparklines, and ATR Trade Ticket."""

    def update_pair(self, analysis: PairAnalysis) -> None:
        spec = analysis.spec
        dec = spec.decimals
        tp = analysis.trade_plan

        # Header line
        star = "★ " if analysis.is_watchlist else ""
        header = Text()
        header.append(f"{star}{spec.pair} ", style="bold bright_white")
        header.append(f"({spec.category})  ", style="dim")
        header.append(f"{analysis.price:.{dec}f} ", style="bold white")
        chg_color = "bright_green" if analysis.change_d1_pct >= 0 else "bright_red"
        header.append(f"({analysis.change_d1_pct:+.2f}%)  ", style=chg_color)
        header.append_text(format_signal_badge(analysis.signal))
        header.append(f"  Score: {analysis.composite_score:+.1f}", style="bold cyan")

        # Sparklines (D1 & H4)
        d1_spark = make_sparkline(analysis.d1_sparkline_closes, width=26)
        h4_spark = make_sparkline(analysis.h4_sparkline_closes, width=26)
        spark_tbl = Table(box=None, expand=True, show_header=False, pad_edge=False)
        spark_tbl.add_column("TF", width=8, style="bold cyan")
        spark_tbl.add_column("Spark", width=28)
        spark_tbl.add_column("EMA Stack", justify="right")

        d1_ind = analysis.d1.indicators
        h4_ind = analysis.h4.indicators
        d1_stack = "20>50>200 (BULL)" if (d1_ind.ema20 > d1_ind.ema50 > d1_ind.ema200) else (
            "20<50<200 (BEAR)" if (d1_ind.ema20 < d1_ind.ema50 < d1_ind.ema200) else "MIXED EMA"
        )
        h4_stack = "20>50>200 (BULL)" if (h4_ind.ema20 > h4_ind.ema50 > h4_ind.ema200) else (
            "20<50<200 (BEAR)" if (h4_ind.ema20 < h4_ind.ema50 < h4_ind.ema200) else "MIXED EMA"
        )
        spark_tbl.add_row("D1 (28d)", Text(d1_spark, style="bright_cyan"), Text(d1_stack, style="dim white"))
        spark_tbl.add_row("H4 (28b)", Text(h4_spark, style="cyan"), Text(h4_stack, style="dim white"))

        # 4-Pillar Multi-Timeframe Matrix
        pillar_tbl = Table(box=None, expand=True, pad_edge=False, header_style="bold cyan")
        pillar_tbl.add_column("Pillar (Wt)", width=15)
        pillar_tbl.add_column("H4 (45%)", justify="right", width=8)
        pillar_tbl.add_column("D1 (40%)", justify="right", width=8)
        pillar_tbl.add_column("W1 (15%)", justify="right", width=8)
        pillar_tbl.add_column("Comp", justify="right", width=8)
        pillar_tbl.add_column("Bar", width=13)

        rows = [
            ("Trend (45%)", analysis.h4.trend_score, analysis.d1.trend_score, analysis.w1.trend_score, analysis.pillar_trend),
            ("Momentum (30%)", analysis.h4.momentum_score, analysis.d1.momentum_score, analysis.w1.momentum_score, analysis.pillar_momentum),
            ("Strength (15%)", analysis.h4.strength_score, analysis.d1.strength_score, analysis.w1.strength_score, analysis.pillar_strength),
            ("TPO/Vol (10%)", analysis.h4.volatility_score, analysis.d1.volatility_score, analysis.w1.volatility_score, analysis.pillar_volatility),
            ("TOTAL SCORE", analysis.h4.total_score, analysis.d1.total_score, analysis.w1.total_score, analysis.composite_score),
        ]
        for label, h4_v, d1_v, w1_v, comp_v in rows:
            pillar_tbl.add_row(
                Text(label, style="bold white" if "TOTAL" in label else "white"),
                format_score_text(h4_v),
                format_score_text(d1_v),
                format_score_text(w1_v),
                format_score_text(comp_v),
                make_score_bar(comp_v, half_width=5),
            )

        # ATR Trade Plan & Position Sizer Box
        dir_color = "bold bright_green" if tp.direction == "LONG" else "bold bright_red"
        ticket_tbl = Table(box=None, expand=True, show_header=False, pad_edge=False)
        ticket_tbl.add_column("K1", style="dim", width=14)
        ticket_tbl.add_column("V1", width=18)
        ticket_tbl.add_column("K2", style="dim", width=14)
        ticket_tbl.add_column("V2", width=18)

        ticket_tbl.add_row(
            "Direction:",
            Text(f"{tp.direction} ({tp.atr_timeframe} ATR)", style=dir_color),
            "Account/Risk:",
            Text(f"${tp.account_balance_usd:,.0f} ({tp.risk_percent:.1f}% = ${tp.risk_amount_usd:,.0f})", style="bold white"),
        )
        ticket_tbl.add_row(
            "Entry Price:",
            Text(f"{tp.entry_price:.{dec}f}", style="bold white"),
            "Position Size:",
            Text(f"{tp.standard_lots:.2f} Lots ({tp.units:,} u)", style="bold bright_cyan"),
        )
        ticket_tbl.add_row(
            "Stop Loss:",
            Text(f"{tp.stop_loss_price:.{dec}f} (-{tp.stop_loss_pips:.1f}p)", style="bold bright_red"),
            f"TP1 ({tp.tp1_r:.1f}R):",
            Text(f"{tp.tp1_price:.{dec}f} (+{tp.tp1_pips:.1f}p)", style="green"),
        )
        ticket_tbl.add_row(
            f"TP2 ({tp.tp2_r:.1f}R):",
            Text(f"{tp.tp2_price:.{dec}f} (+{tp.tp2_pips:.1f}p)", style="bold bright_green"),
            f"TP3 ({tp.tp3_r:.1f}R):",
            Text(f"{tp.tp3_price:.{dec}f} (+{tp.tp3_pips:.1f}p)", style="bold bright_green"),
        )

        group = Group(
            header,
            Text(""),
            spark_tbl,
            Text(""),
            pillar_tbl,
            Text("\nATR Trade Plan & Position Sizer:", style="bold yellow"),
            ticket_tbl,
        )
        self.update(
            Panel(
                group,
                title=f"[bold yellow]Pair Inspector & Trade Ticket — {spec.pair}[/bold yellow]",
                border_style="yellow",
            )
        )
