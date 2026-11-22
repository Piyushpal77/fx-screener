"""Main CLI and Textual TUI entry point for the FX & Precious Metals Screener."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from rich.console import Console

from fx_screener.config import AppConfig
from fx_screener.data import load_or_fetch_universe
from fx_screener.engine.scorer import analyze_universe
from fx_screener.ui.cli_renderer import (
    export_screener_results,
    filter_pairs,
    render_cli_once,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fx-screener",
        description="Multi-Factor FX & Precious Metals Terminal Screener (28 G8 Pairs + XAU/USD + XAG/USD)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to ./config.yaml)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run in headless one-shot CLI mode: print Rich table + G8 Strength Matrix and exit",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force fresh market data download, bypassing SQLite cache TTL",
    )
    parser.add_argument(
        "--provider",
        choices=["yfinance", "oanda", "synthetic"],
        default=None,
        help="Override market data provider (yfinance, oanda, synthetic)",
    )
    parser.add_argument(
        "--filter",
        choices=["all", "confluence", "strong", "squeeze", "watchlist"],
        default="all",
        help="Filter displayed pairs (all, confluence, strong, squeeze, watchlist)",
    )
    parser.add_argument(
        "--currency",
        type=str,
        default="",
        help="Filter by currency or pair substring (e.g. USD, JPY, XAU)",
    )
    parser.add_argument(
        "--atr-tf",
        choices=["D1", "H4", "d1", "h4"],
        default=None,
        help="Override ATR timeframe for Stop-Loss & Take-Profit plan (D1 or H4)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=None,
        help="Override account balance in USD for position sizing (default: 10000)",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=None,
        help="Override risk percent per trade (default: 1.0)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export filtered results to CSV or JSON file (e.g. screener.csv or screener.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AppConfig.load(args.config)

    if args.provider:
        config.provider.active = args.provider
    if args.atr_tf:
        config.risk.atr_timeframe = args.atr_tf.upper()
    if args.balance is not None and args.balance > 0:
        config.risk.account_balance_usd = args.balance
    if args.risk is not None and args.risk > 0:
        config.risk.risk_percent = args.risk

    if args.once or args.export:
        console = Console(width=165)
        universe_data, source = load_or_fetch_universe(config, force_refresh=args.refresh)
        result = analyze_universe(universe_data, config, source=source)
        filtered = filter_pairs(
            result.pairs,
            filter_mode=args.filter,
            currency_query=args.currency,
        )

        if args.once or not args.export:
            render_cli_once(result, filtered, console=console)

        if args.export:
            out_path = export_screener_results(result, filtered, Path(args.export))
            console.print(f"[bold green]✔ Exported {len(filtered)} instruments to:[/bold green] {out_path.resolve()}")
        return 0

    from fx_screener.ui.app import FxScreenerApp

    app = FxScreenerApp(config=config)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
