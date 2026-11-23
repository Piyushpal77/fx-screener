"""Interactive Textual TUI Application for the Multi-Factor FX & Precious Metals Screener."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static

from fx_screener.config import AppConfig
from fx_screener.data import load_or_fetch_universe
from fx_screener.engine.scorer import PairAnalysis, ScreenerResult, analyze_universe
from fx_screener.engine.tracker import SignalTracker
from fx_screener.ui.cli_renderer import (
    export_screener_results,
    filter_pairs,
    format_score_text,
    format_signal_badge,
)
from fx_screener.ui.widgets import CurrencyStrengthWidget, PairInspectorWidget


class RiskConfigModal(ModalScreen[bool]):
    """Modal dialog to edit Account Balance (USD), Risk %, and ATR SL multiplier live."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        with Vertical(id="risk-modal"):
            yield Label("ATR Trade Planner & Risk Settings", id="modal-title")
            yield Label("Account Balance (USD):")
            yield Input(value=str(self.config.risk.account_balance_usd), id="inp-balance")
            yield Label("Risk Per Trade (%):")
            yield Input(value=str(self.config.risk.risk_percent), id="inp-risk")
            yield Label("Stop-Loss ATR Multiplier (e.g. 1.5):")
            yield Input(value=str(self.config.risk.sl_atr_multiplier), id="inp-mult")
            with Horizontal(id="modal-buttons"):
                yield Button("Save & Recalculate", variant="success", id="btn-save")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            try:
                bal = float(self.query_one("#inp-balance", Input).value)
                risk_pct = float(self.query_one("#inp-risk", Input).value)
                mult = float(self.query_one("#inp-mult", Input).value)
                if bal > 0 and 0 < risk_pct <= 25 and mult > 0:
                    self.config.risk.account_balance_usd = bal
                    self.config.risk.risk_percent = risk_pct
                    self.config.risk.sl_atr_multiplier = mult
                    self.config.save()
                    self.dismiss(True)
                    return
            except ValueError:
                pass
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class CurrencySearchModal(ModalScreen[Optional[str]]):
    """Modal dialog to filter pairs by currency or ticker substring (e.g., USD, JPY, XAU)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, current_query: str = "") -> None:
        super().__init__()
        self.current_query = current_query

    def compose(self) -> ComposeResult:
        with Vertical(id="search-modal"):
            yield Label("Filter by Currency / Pair / Signal (leave blank to clear):", id="modal-title")
            yield Input(
                value=self.current_query,
                placeholder="e.g. JPY, USD, XAU, BUY...",
                id="inp-query",
            )
            with Horizontal(id="modal-buttons"):
                yield Button("Apply", variant="primary", id="btn-apply")
                yield Button("Clear", variant="warning", id="btn-clear")

    def on_mount(self) -> None:
        self.query_one("#inp-query", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip().upper())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-apply":
            val = self.query_one("#inp-query", Input).value.strip().upper()
            self.dismiss(val)
        else:
            self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss(None)


class FxScreenerApp(App):
    """Textual Terminal Dashboard for the 30-Pair FX & Precious Metals Multi-Factor Screener."""

    TITLE = "FX & Precious Metals Multi-Factor Terminal Screener (H4 | D1 | W1)"

    CSS = """
    Screen {
        background: $surface;
    }
    #status-bar {
        dock: top;
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 1;
    }
    #main-split {
        height: 1fr;
    }
    #table-pane {
        width: 58%;
        height: 1fr;
        border: solid $primary;
    }
    #right-pane {
        width: 42%;
        height: 1fr;
    }
    #strength-widget {
        height: 16;
    }
    #inspector-widget {
        height: 1fr;
    }
    DataTable {
        height: 1fr;
    }
    #risk-modal, #search-modal {
        width: 56;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
        align: center middle;
    }
    #modal-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #modal-buttons {
        height: 3;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("1", "filter_mode('all')", "1:All(30)"),
        Binding("2", "filter_mode('confluence')", "2:Confluence★"),
        Binding("3", "filter_mode('strong')", "3:Strong(>=60)"),
        Binding("4", "filter_mode('squeeze')", "4:Squeeze/Brk"),
        Binding("5", "filter_mode('watchlist')", "5:Watchlist★"),
        Binding("slash", "search_currency", "/:Search CCY"),
        Binding("s", "cycle_sort", "S:Sort"),
        Binding("w", "toggle_watchlist", "W:Star★"),
        Binding("a", "toggle_atr_tf", "A:ATR(D1/H4)"),
        Binding("p", "open_risk_modal", "P:Risk/Lots"),
        Binding("r", "force_refresh", "R:Refresh"),
        Binding("e", "export_files", "E:Export CSV/JSON"),
        Binding("q", "quit", "Q:Quit"),
    ]

    SORT_MODES = [
        ("composite", "Composite Score"),
        ("abs_composite", "Conviction |Score|"),
        ("d1", "D1 Score"),
        ("h4", "H4 Score"),
        ("adx", "ADX(D1)"),
        ("change", "D1 Change%"),
    ]

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.universe_data = {}
        self.result: Optional[ScreenerResult] = None
        self.pair_map: Dict[str, PairAnalysis] = {}
        self.displayed_pairs: List[PairAnalysis] = []
        self.selected_pair: str = "EUR/USD"
        self.filter_mode_name: str = "all"
        self.currency_query: str = ""
        self.sort_idx: int = 0
        self.last_refresh_str: str = "-"
        self.seen_alerts: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Initializing FX Screener...", id="status-bar")
        with Horizontal(id="main-split"):
            with Container(id="table-pane"):
                yield DataTable(id="pairs-table", cursor_type="row", zebra_stripes=True)
            with Vertical(id="right-pane"):
                yield CurrencyStrengthWidget(id="strength-widget")
                yield PairInspectorWidget(id="inspector-widget")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#pairs-table", DataTable)
        table.add_columns(
            "★",
            "Pair",
            "Price",
            "D1%",
            "Score",
            "Signal",
            "H4",
            "D1",
            "W1",
            "Confluence",
            "ADX",
            "RSI",
            "Regime",
        )
        self.tracker = SignalTracker()
        self._refresh_data(force=False)
        if self.config.provider.auto_refresh_seconds > 0:
            self.set_interval(
                float(self.config.provider.auto_refresh_seconds),
                lambda: self._refresh_data(force=False),
            )

    def _refresh_data(self, force: bool = False) -> None:
        self.universe_data, source = load_or_fetch_universe(self.config, force_refresh=force)
        self.result = analyze_universe(self.universe_data, self.config, source=source)
        if hasattr(self, "tracker"):
            self.tracker.sync_and_evaluate(self.result.pairs, self.universe_data)
        self.last_refresh_str = datetime.now().strftime("%H:%M:%S")
        self._check_alerts()
        self._rebuild_view()

    def _check_alerts(self) -> None:
        if not self.result:
            return
        new_alerts = []
        for p in self.result.pairs:
            if p.is_alert and p.spec.pair not in self.seen_alerts:
                self.seen_alerts.add(p.spec.pair)
                new_alerts.append(f"{p.spec.pair} ({p.composite_score:+.1f} {p.signal})")
        if new_alerts:
            self.notify(
                "High-Conviction Signal: " + ", ".join(new_alerts[:4]),
                title="★ Score Threshold Alert",
                severity="information",
            )

    def _recompute_scores_only(self) -> None:
        if not self.universe_data or not self.result:
            return
        self.result = analyze_universe(
            self.universe_data, self.config, source=self.result.source
        )
        self._rebuild_view()

    def _sort_pairs(self, pairs: List[PairAnalysis]) -> List[PairAnalysis]:
        key_name, _ = self.SORT_MODES[self.sort_idx]
        if key_name == "composite":
            return sorted(pairs, key=lambda p: p.composite_score, reverse=True)
        if key_name == "abs_composite":
            return sorted(pairs, key=lambda p: abs(p.composite_score), reverse=True)
        if key_name == "d1":
            return sorted(pairs, key=lambda p: p.d1.total_score, reverse=True)
        if key_name == "h4":
            return sorted(pairs, key=lambda p: p.h4.total_score, reverse=True)
        if key_name == "adx":
            return sorted(pairs, key=lambda p: p.d1.indicators.adx14, reverse=True)
        if key_name == "change":
            return sorted(pairs, key=lambda p: p.change_d1_pct, reverse=True)
        return pairs

    def _rebuild_view(self) -> None:
        if not self.result:
            return

        self.pair_map = {p.spec.pair: p for p in self.result.pairs}
        filtered = filter_pairs(
            self.result.pairs,
            filter_mode=self.filter_mode_name,
            currency_query=self.currency_query,
        )
        self.displayed_pairs = self._sort_pairs(filtered)

        # Update status bar
        _, sort_label = self.SORT_MODES[self.sort_idx]
        q_label = f" | Search: {self.currency_query}" if self.currency_query else ""
        status_text = (
            f" Source: {self.result.source} | Updated: {self.last_refresh_str} | "
            f"Filter: {self.filter_mode_name.upper()} ({len(self.displayed_pairs)}/30){q_label} | "
            f"Sort: {sort_label} | ATR: {self.config.risk.atr_timeframe} ({self.config.risk.sl_atr_multiplier}x) | "
            f"Acct: ${self.config.risk.account_balance_usd:,.0f} @ {self.config.risk.risk_percent:.1f}%"
        )
        self.query_one("#status-bar", Static).update(status_text)

        # Update G8 Currency Strength widget
        self.query_one("#strength-widget", CurrencyStrengthWidget).update_matrix(
            self.result.strength_matrix
        )

        # Populate DataTable
        table = self.query_one("#pairs-table", DataTable)
        table.clear()

        target_row_idx = 0
        for idx, p in enumerate(self.displayed_pairs):
            if p.spec.pair == self.selected_pair:
                target_row_idx = idx
            dec = p.spec.decimals
            star = Text("★", style="bold yellow") if p.is_watchlist else Text(" ")
            chg_style = "green" if p.change_d1_pct >= 0 else "red"
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

            table.add_row(
                star,
                Text(p.spec.pair, style="bold white"),
                f"{p.price:.{dec}f}",
                Text(f"{p.change_d1_pct:+.2f}%", style=chg_style),
                format_score_text(p.composite_score),
                format_signal_badge(p.signal),
                format_score_text(p.h4.total_score),
                format_score_text(p.d1.total_score),
                format_score_text(p.w1.total_score),
                Text(p.confluence_label, style=conf_style),
                f"{p.d1.indicators.adx14:.1f}",
                f"{p.d1.indicators.rsi14:.1f}",
                Text(p.vol_regime, style=vol_style),
                key=p.spec.pair,
            )

        if self.displayed_pairs:
            self.selected_pair = self.displayed_pairs[target_row_idx].spec.pair
            table.move_cursor(row=target_row_idx)
            self.query_one("#inspector-widget", PairInspectorWidget).update_pair(
                self.displayed_pairs[target_row_idx]
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key and event.row_key.value in self.pair_map:
            self.selected_pair = str(event.row_key.value)
            self.query_one("#inspector-widget", PairInspectorWidget).update_pair(
                self.pair_map[self.selected_pair]
            )

    def action_filter_mode(self, mode: str) -> None:
        self.filter_mode_name = mode
        self._rebuild_view()

    def action_cycle_sort(self) -> None:
        self.sort_idx = (self.sort_idx + 1) % len(self.SORT_MODES)
        self._rebuild_view()

    def action_toggle_watchlist(self) -> None:
        if not self.selected_pair:
            return
        if self.selected_pair in self.config.watchlist:
            self.config.watchlist.remove(self.selected_pair)
            self.notify(f"Removed {self.selected_pair} from Watchlist ★")
        else:
            self.config.watchlist.append(self.selected_pair)
            self.notify(f"Added {self.selected_pair} to Watchlist ★")
        self.config.save()
        self._recompute_scores_only()

    def action_toggle_atr_tf(self) -> None:
        modes = ["D1", "H4", "H4_HMA"]
        cur = self.config.risk.atr_timeframe.upper()
        next_idx = (modes.index(cur) + 1) % len(modes) if cur in modes else 0
        self.config.risk.atr_timeframe = modes[next_idx]
        self.config.save()
        self.notify(f"Switched Trade Planner Stop Mode to {self.config.risk.atr_timeframe}")
        self._recompute_scores_only()

    def action_open_risk_modal(self) -> None:
        def _on_close(updated: bool | None) -> None:
            if updated:
                self.notify("Updated Risk & Position Sizing settings")
                self._recompute_scores_only()

        self.push_screen(RiskConfigModal(self.config), _on_close)

    def action_search_currency(self) -> None:
        def _on_search(query: Optional[str]) -> None:
            if query is not None:
                self.currency_query = query
                self._rebuild_view()

        self.push_screen(CurrencySearchModal(self.currency_query), _on_search)

    def action_force_refresh(self) -> None:
        self.notify("Refreshing market data across 30 instruments...")
        self._refresh_data(force=True)

    def action_export_files(self) -> None:
        if not self.result:
            return
        csv_path = export_screener_results(
            self.result, self.displayed_pairs, "screener_export.csv"
        )
        json_path = export_screener_results(
            self.result, self.displayed_pairs, "screener_export.json"
        )
        self.notify(f"Exported to {csv_path.name} and {json_path.name}")
