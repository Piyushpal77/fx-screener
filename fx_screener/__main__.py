"""Main Web UI, Textual TUI, and Rich CLI entry point for the FX & Precious Metals Screener."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import webbrowser
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
        description="Multi-Factor FX & Precious Metals Screener (Interactive Web UI, Textual TUI, & Rich CLI)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to ./config.yaml)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the interactive Web UI with candlestick charts (W1, D1, H4, H1) [Default mode]",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the interactive Textual Terminal UI inside PowerShell/terminal",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run in headless one-shot CLI mode: print Rich table + G8 Strength Matrix and exit",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the Web UI server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the Web UI server (default: 8050)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser when starting the Web UI server",
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

    # Mode 1: One-Shot CLI / Export
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

    # Mode 2: Terminal UI (Textual)
    if args.tui:
        from fx_screener.ui.app import FxScreenerApp

        app = FxScreenerApp(config=config)
        app.run()
        return 0

    # Mode 3 (Default): Professional Interactive Web UI with W1, D1, H4, H1 Charts
    import uvicorn
    from fx_screener.web.server import create_web_app

    web_app = create_web_app(config=config)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  ★ FX & Precious Metals Multi-Factor Terminal UI running at: {url}\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(web_app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
