"""4-Hour Strong Signal Tracker with SQLite persistence and automatic CSV Spreadsheet synchronization.

Automatically records STRONG BUY / STRONG SELL signals (and H4 HMA Pullback Triggers),
tracks their 4H price progression bar-by-bar (High/Low/Close), logs when TP1 (1.5R) is reached,
and closes the trade at TP2 (2.0R) as a WIN or at Stop-Loss as a LOSS.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd

from fx_screener.data.cache import DEFAULT_CACHE_DB
from fx_screener.engine.scorer import PairAnalysis
from fx_screener.universe import UNIVERSE_MAP


DEFAULT_SPREADSHEET_PATH = Path(__file__).resolve().parent.parent.parent / "strong_signals_tracker.csv"


@dataclass
class TrackedSignal:
    id: int
    pair: str
    direction: str             # "LONG" | "SHORT"
    signal_type: str           # "STRONG BUY" | "STRONG SELL"
    hma_trigger: str           # e.g. "★ H4 HMA TURN BULL"
    score_at_entry: float
    entry_time: str            # ISO UTC string of the 4H bar
    entry_ts: int              # Unix timestamp (seconds) of the 4H entry bar
    entry_price: float
    stop_loss_price: float
    stop_loss_pips: float
    tp1_price: float           # 1.5R
    tp2_price: float           # 2.0R (Target exit for full WIN)
    tp3_price: float           # 3.0R
    standard_lots: float
    risk_amount_usd: float
    status: str                # "ACTIVE" | "TP1_HIT" | "WIN_TP2" | "LOSS_SL"
    tp1_hit: bool
    tp1_hit_time: str
    current_h4_price: float
    last_h4_time: str
    max_favorable_pips: float
    max_adverse_pips: float
    bars_elapsed_h4: int
    exit_time: str
    exit_ts: int
    exit_price: float
    pnl_pips: float
    pnl_r: float
    pnl_usd: float


class SignalTracker:
    """Persists and updates Strong Signals on 4H candles and syncs to a CSV spreadsheet."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_CACHE_DB,
        csv_path: Path | str = DEFAULT_SPREADSHEET_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracked_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    hma_trigger TEXT NOT NULL,
                    score_at_entry REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_ts INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss_price REAL NOT NULL,
                    stop_loss_pips REAL NOT NULL,
                    tp1_price REAL NOT NULL,
                    tp2_price REAL NOT NULL,
                    tp3_price REAL NOT NULL,
                    standard_lots REAL NOT NULL,
                    risk_amount_usd REAL NOT NULL,
                    status TEXT NOT NULL,
                    tp1_hit INTEGER NOT NULL DEFAULT 0,
                    tp1_hit_time TEXT NOT NULL DEFAULT '',
                    current_h4_price REAL NOT NULL,
                    last_h4_time TEXT NOT NULL,
                    max_favorable_pips REAL NOT NULL DEFAULT 0.0,
                    max_adverse_pips REAL NOT NULL DEFAULT 0.0,
                    bars_elapsed_h4 INTEGER NOT NULL DEFAULT 0,
                    exit_time TEXT NOT NULL DEFAULT '',
                    exit_ts INTEGER NOT NULL DEFAULT 0,
                    exit_price REAL NOT NULL DEFAULT 0.0,
                    pnl_pips REAL NOT NULL DEFAULT 0.0,
                    pnl_r REAL NOT NULL DEFAULT 0.0,
                    pnl_usd REAL NOT NULL DEFAULT 0.0
                )
                """
            )

    def _row_to_signal(self, row: sqlite3.Row) -> TrackedSignal:
        return TrackedSignal(
            id=int(row["id"]),
            pair=str(row["pair"]),
            direction=str(row["direction"]),
            signal_type=str(row["signal_type"]),
            hma_trigger=str(row["hma_trigger"]),
            score_at_entry=float(row["score_at_entry"]),
            entry_time=str(row["entry_time"]),
            entry_ts=int(row["entry_ts"]),
            entry_price=float(row["entry_price"]),
            stop_loss_price=float(row["stop_loss_price"]),
            stop_loss_pips=float(row["stop_loss_pips"]),
            tp1_price=float(row["tp1_price"]),
            tp2_price=float(row["tp2_price"]),
            tp3_price=float(row["tp3_price"]),
            standard_lots=float(row["standard_lots"]),
            risk_amount_usd=float(row["risk_amount_usd"]),
            status=str(row["status"]),
            tp1_hit=bool(row["tp1_hit"]),
            tp1_hit_time=str(row["tp1_hit_time"]),
            current_h4_price=float(row["current_h4_price"]),
            last_h4_time=str(row["last_h4_time"]),
            max_favorable_pips=float(row["max_favorable_pips"]),
            max_adverse_pips=float(row["max_adverse_pips"]),
            bars_elapsed_h4=int(row["bars_elapsed_h4"]),
            exit_time=str(row["exit_time"]),
            exit_ts=int(row["exit_ts"]),
            exit_price=float(row["exit_price"]),
            pnl_pips=float(row["pnl_pips"]),
            pnl_r=float(row["pnl_r"]),
            pnl_usd=float(row["pnl_usd"]),
        )

    def sync_and_evaluate(
        self,
        pairs: List[PairAnalysis],
        universe_data: Dict[str, Dict[str, pd.DataFrame]],
        source: str = "yfinance",
        allow_synthetic: bool = False,
    ) -> List[TrackedSignal]:
        """1. Evaluate all open trades against 4H OHLC bars until SL or TP2 (2.0R) is hit.
        2. Register any new freshly timed H4 Hull Pullback STRONG BUY / STRONG SELL signals.
        3. Write the complete ledger to `strong_signals_tracker.csv`.
        """
        if not allow_synthetic and source.lower().startswith("synthetic"):
            return self.list_signals()

        with self._connect() as conn:
            # Clean up any legacy/stale open records that were not generated by an H4 Hull Pullback Turn
            conn.execute(
                "DELETE FROM tracked_signals WHERE status IN ('ACTIVE', 'TP1_HIT') AND hma_trigger NOT LIKE '%TURN%'"
            )

            pair_map = {p.spec.pair: p for p in pairs}

            # Purge any corrupted / synthetic records whose entry_price does not match the real 4H candle
            # at entry_ts (e.g. from an earlier synthetic fallback price like USD/CHF 0.88400)
            all_existing = conn.execute("SELECT id, pair, entry_ts, entry_price FROM tracked_signals").fetchall()
            for r in all_existing:
                sig_id = int(r["id"])
                sig_pair = str(r["pair"])
                sig_ts = int(r["entry_ts"])
                sig_entry = float(r["entry_price"])
                h4_df = universe_data.get(sig_pair, {}).get("H4")
                if h4_df is not None and not h4_df.empty and not h4_df.attrs.get("is_synthetic", False):
                    spec = UNIVERSE_MAP.get(sig_pair)
                    tol = (spec.pip_size * 5.0) if spec else 0.0005
                    ts_index = [int(pd.Timestamp(idx).timestamp()) for idx in h4_df.index]
                    if sig_ts in ts_index:
                        row_pos = ts_index.index(sig_ts)
                        bar_low = float(h4_df["Low"].iloc[row_pos])
                        bar_high = float(h4_df["High"].iloc[row_pos])
                        if sig_entry < (bar_low - tol) or sig_entry > (bar_high + tol):
                            conn.execute("DELETE FROM tracked_signals WHERE id = ?", (sig_id,))
                            continue
                pa = pair_map.get(sig_pair)
                if pa and pa.price > 0:
                    rel_diff = abs(sig_entry - pa.price) / pa.price
                    if rel_diff > 0.12:
                        conn.execute("DELETE FROM tracked_signals WHERE id = ?", (sig_id,))

            open_rows = conn.execute(
                "SELECT * FROM tracked_signals WHERE status IN ('ACTIVE', 'TP1_HIT') ORDER BY id ASC"
            ).fetchall()

            active_by_pair: Dict[str, TrackedSignal] = {}
            for row in open_rows:
                sig = self._row_to_signal(row)
                h4_df_sig = universe_data.get(sig.pair, {}).get("H4")
                if not allow_synthetic and h4_df_sig is not None and h4_df_sig.attrs.get("is_synthetic", False):
                    continue
                cur_analysis = pair_map.get(sig.pair)
                # If an active tracked trade has flipped direction in the unified Strong Signal engine,
                # close the old counter-signal so the tracker never contradicts the live Strong Signal
                if cur_analysis and (
                    (sig.direction == "LONG" and cur_analysis.composite_score <= -25.0)
                    or (sig.direction == "SHORT" and cur_analysis.composite_score >= 25.0)
                ):
                    conn.execute("DELETE FROM tracked_signals WHERE id = ?", (sig.id,))
                    continue

                updated = self._evaluate_signal_on_h4(sig, h4_df_sig)
                self._save_signal_update(conn, updated)
                if updated.status in ("ACTIVE", "TP1_HIT"):
                    active_by_pair[updated.pair] = updated

            # Register ONLY freshly timed H4 Hull Pullback Resumption STRONG BUY / STRONG SELL entries
            for p in pairs:
                if p.signal not in ("STRONG BUY", "STRONG SELL"):
                    continue
                if not p.is_fresh_hma_entry:
                    continue
                if p.spec.pair in active_by_pair:
                    continue

                h4_df = universe_data.get(p.spec.pair, {}).get("H4")
                if not allow_synthetic and h4_df is not None and h4_df.attrs.get("is_synthetic", False):
                    continue
                if h4_df is not None and not h4_df.empty:
                    last_idx = pd.Timestamp(h4_df.index[-1])
                    entry_ts = int(last_idx.timestamp())
                    entry_time = last_idx.strftime("%Y-%m-%d %H:%M UTC")
                    h4_close = float(h4_df["Close"].iloc[-1])
                else:
                    now_utc = datetime.now(timezone.utc)
                    entry_ts = int(now_utc.timestamp())
                    entry_time = now_utc.strftime("%Y-%m-%d %H:%M UTC")
                    h4_close = p.price

                tp = p.trade_plan
                conn.execute(
                    """
                    INSERT INTO tracked_signals (
                        pair, direction, signal_type, hma_trigger, score_at_entry,
                        entry_time, entry_ts, entry_price, stop_loss_price, stop_loss_pips,
                        tp1_price, tp2_price, tp3_price, standard_lots, risk_amount_usd,
                        status, tp1_hit, tp1_hit_time, current_h4_price, last_h4_time,
                        max_favorable_pips, max_adverse_pips, bars_elapsed_h4,
                        exit_time, exit_ts, exit_price, pnl_pips, pnl_r, pnl_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p.spec.pair,
                        tp.direction,
                        p.signal,
                        p.hma_trigger,
                        round(p.composite_score, 2),
                        entry_time,
                        entry_ts,
                        round(h4_close, p.spec.decimals),
                        round(tp.stop_loss_price, p.spec.decimals),
                        round(tp.stop_loss_pips, 1),
                        round(tp.tp1_price, p.spec.decimals),
                        round(tp.tp2_price, p.spec.decimals),
                        round(tp.tp3_price, p.spec.decimals),
                        round(tp.standard_lots, 2),
                        round(tp.risk_amount_usd, 2),
                        "ACTIVE",
                        0,
                        "",
                        round(h4_close, p.spec.decimals),
                        entry_time,
                        0.0,
                        0.0,
                        0,
                        "",
                        0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                )

            all_rows = conn.execute(
                "SELECT * FROM tracked_signals ORDER BY id DESC"
            ).fetchall()
            all_signals = [self._row_to_signal(r) for r in all_rows]

        self.export_csv(all_signals)
        return all_signals

    def _evaluate_signal_on_h4(
        self,
        sig: TrackedSignal,
        h4_df: Optional[pd.DataFrame],
    ) -> TrackedSignal:
        """Walk forward through all 4H bars after entry_ts to check SL, TP1 (1.5R), and TP2 (2.0R)."""
        if h4_df is None or h4_df.empty or sig.status in ("WIN_TP2", "LOSS_SL"):
            return sig

        spec = UNIVERSE_MAP.get(sig.pair)
        pip_size = spec.pip_size if spec else 0.0001
        dec = spec.decimals if spec else 5
        sl_pips = max(sig.stop_loss_pips, 0.1)

        work = h4_df.sort_index()
        bars_after = 0

        for idx, row in work.iterrows():
            ts = int(pd.Timestamp(idx).timestamp())
            time_str = pd.Timestamp(idx).strftime("%Y-%m-%d %H:%M UTC")
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            if ts < sig.entry_ts:
                continue

            if ts == sig.entry_ts:
                # Entry bar: update current price snapshot without triggering intrabar historical High/Low before entry
                sig.current_h4_price = round(close, dec)
                sig.last_h4_time = time_str
                continue

            bars_after += 1
            sig.bars_elapsed_h4 = bars_after
            sig.current_h4_price = round(close, dec)
            sig.last_h4_time = time_str

            if sig.direction == "LONG":
                fav_pips = (high - sig.entry_price) / pip_size
                adv_pips = max(0.0, (sig.entry_price - low) / pip_size)
                sig.max_favorable_pips = round(max(sig.max_favorable_pips, fav_pips), 1)
                sig.max_adverse_pips = round(max(sig.max_adverse_pips, adv_pips), 1)

                # Check Stop Loss hit first (conservative evaluation)
                if low <= sig.stop_loss_price:
                    sig.status = "LOSS_SL"
                    sig.exit_time = time_str
                    sig.exit_ts = ts
                    sig.exit_price = round(sig.stop_loss_price, dec)
                    sig.pnl_pips = round(-sig.stop_loss_pips, 1)
                    sig.pnl_r = -1.0
                    sig.pnl_usd = round(-sig.risk_amount_usd, 2)
                    return sig

                # Check TP1 (1.5R) milestone
                if not sig.tp1_hit and high >= sig.tp1_price:
                    sig.tp1_hit = True
                    sig.tp1_hit_time = time_str
                    sig.status = "TP1_HIT"

                # Check TP2 (2.0R) full target exit
                if high >= sig.tp2_price:
                    sig.status = "WIN_TP2"
                    sig.tp1_hit = True
                    if not sig.tp1_hit_time:
                        sig.tp1_hit_time = time_str
                    sig.exit_time = time_str
                    sig.exit_ts = ts
                    sig.exit_price = round(sig.tp2_price, dec)
                    win_pips = (sig.tp2_price - sig.entry_price) / pip_size
                    sig.pnl_pips = round(win_pips, 1)
                    sig.pnl_r = round(win_pips / sl_pips, 2)
                    sig.pnl_usd = round(sig.risk_amount_usd * sig.pnl_r, 2)
                    return sig

                # Floating PnL while open
                cur_pips = (close - sig.entry_price) / pip_size
                sig.pnl_pips = round(cur_pips, 1)
                sig.pnl_r = round(cur_pips / sl_pips, 2)
                sig.pnl_usd = round(sig.risk_amount_usd * sig.pnl_r, 2)

            else:
                # SHORT trade evaluation
                fav_pips = (sig.entry_price - low) / pip_size
                adv_pips = max(0.0, (high - sig.entry_price) / pip_size)
                sig.max_favorable_pips = round(max(sig.max_favorable_pips, fav_pips), 1)
                sig.max_adverse_pips = round(max(sig.max_adverse_pips, adv_pips), 1)

                # Check Stop Loss hit
                if high >= sig.stop_loss_price:
                    sig.status = "LOSS_SL"
                    sig.exit_time = time_str
                    sig.exit_ts = ts
                    sig.exit_price = round(sig.stop_loss_price, dec)
                    sig.pnl_pips = round(-sig.stop_loss_pips, 1)
                    sig.pnl_r = -1.0
                    sig.pnl_usd = round(-sig.risk_amount_usd, 2)
                    return sig

                # Check TP1 (1.5R) milestone
                if not sig.tp1_hit and low <= sig.tp1_price:
                    sig.tp1_hit = True
                    sig.tp1_hit_time = time_str
                    sig.status = "TP1_HIT"

                # Check TP2 (2.0R) full target exit
                if low <= sig.tp2_price:
                    sig.status = "WIN_TP2"
                    sig.tp1_hit = True
                    if not sig.tp1_hit_time:
                        sig.tp1_hit_time = time_str
                    sig.exit_time = time_str
                    sig.exit_ts = ts
                    sig.exit_price = round(sig.tp2_price, dec)
                    win_pips = (sig.entry_price - sig.tp2_price) / pip_size
                    sig.pnl_pips = round(win_pips, 1)
                    sig.pnl_r = round(win_pips / sl_pips, 2)
                    sig.pnl_usd = round(sig.risk_amount_usd * sig.pnl_r, 2)
                    return sig

                # Floating PnL while open
                cur_pips = (sig.entry_price - close) / pip_size
                sig.pnl_pips = round(cur_pips, 1)
                sig.pnl_r = round(cur_pips / sl_pips, 2)
                sig.pnl_usd = round(sig.risk_amount_usd * sig.pnl_r, 2)

        return sig

    def _save_signal_update(self, conn: sqlite3.Connection, sig: TrackedSignal) -> None:
        conn.execute(
            """
            UPDATE tracked_signals
            SET status = ?,
                tp1_hit = ?,
                tp1_hit_time = ?,
                current_h4_price = ?,
                last_h4_time = ?,
                max_favorable_pips = ?,
                max_adverse_pips = ?,
                bars_elapsed_h4 = ?,
                exit_time = ?,
                exit_ts = ?,
                exit_price = ?,
                pnl_pips = ?,
                pnl_r = ?,
                pnl_usd = ?
            WHERE id = ?
            """,
            (
                sig.status,
                1 if sig.tp1_hit else 0,
                sig.tp1_hit_time,
                sig.current_h4_price,
                sig.last_h4_time,
                sig.max_favorable_pips,
                sig.max_adverse_pips,
                sig.bars_elapsed_h4,
                sig.exit_time,
                sig.exit_ts,
                sig.exit_price,
                sig.pnl_pips,
                sig.pnl_r,
                sig.pnl_usd,
                sig.id,
            ),
        )

    def list_signals(self, pair: Optional[str] = None) -> List[TrackedSignal]:
        with self._connect() as conn:
            if pair:
                rows = conn.execute(
                    "SELECT * FROM tracked_signals WHERE pair = ? ORDER BY id DESC",
                    (pair.upper(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tracked_signals ORDER BY id DESC"
                ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def export_csv(self, signals: Optional[List[TrackedSignal]] = None) -> Path:
        """Write a clean, concise Excel-friendly CSV spreadsheet with the exact requested columns:
        Trade Suggested, Date of Recommendation, Entry Price, Target Price, Stop Loss Price,
        Live Price, Outcome / Status, and Date Reached.
        """
        rows = signals if signals is not None else self.list_signals()
        fieldnames = [
            "Trade Suggested",
            "Date of Recommendation",
            "Entry Price",
            "Target Price (TP2)",
            "Stop Loss Price",
            "Live Price",
            "Outcome / Status",
            "Date Reached",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in rows:
                spec = UNIVERSE_MAP.get(s.pair)
                dec = spec.decimals if spec else 5
                if s.status == "WIN_TP2":
                    outcome = "Take Profit Achieved (WIN)"
                elif s.status == "LOSS_SL":
                    outcome = "Stop Loss Reached (LOSS)"
                elif s.status == "TP1_HIT":
                    outcome = "In Progress (TP1 Reached, Tracking to Target)"
                else:
                    outcome = "In Progress (Live Tracking)"

                writer.writerow(
                    {
                        "Trade Suggested": f"{s.pair} — {s.signal_type} ({s.direction})",
                        "Date of Recommendation": s.entry_time,
                        "Entry Price": f"{s.entry_price:.{dec}f}",
                        "Target Price (TP2)": f"{s.tp2_price:.{dec}f}",
                        "Stop Loss Price": f"{s.stop_loss_price:.{dec}f}",
                        "Live Price": f"{s.current_h4_price:.{dec}f}",
                        "Outcome / Status": outcome,
                        "Date Reached": s.exit_time if s.exit_time else "-",
                    }
                )
        return self.csv_path

    def summary_stats(self, signals: Optional[List[TrackedSignal]] = None) -> Dict[str, Any]:
        rows = signals if signals is not None else self.list_signals()
        active = [s for s in rows if s.status in ("ACTIVE", "TP1_HIT")]
        wins = [s for s in rows if s.status == "WIN_TP2"]
        losses = [s for s in rows if s.status == "LOSS_SL"]
        closed_count = len(wins) + len(losses)
        win_rate = (len(wins) / closed_count * 100.0) if closed_count > 0 else 0.0
        realized_r = sum(s.pnl_r for s in wins) + sum(s.pnl_r for s in losses)
        floating_r = sum(s.pnl_r for s in active)
        realized_usd = sum(s.pnl_usd for s in wins) + sum(s.pnl_usd for s in losses)
        floating_usd = sum(s.pnl_usd for s in active)

        return {
            "total_tracked": len(rows),
            "active_count": len(active),
            "tp1_hit_open_count": sum(1 for s in active if s.tp1_hit),
            "win_count": len(wins),
            "loss_count": len(losses),
            "closed_count": closed_count,
            "win_rate_pct": round(win_rate, 1),
            "realized_r": round(realized_r, 2),
            "floating_r": round(floating_r, 2),
            "total_r": round(realized_r + floating_r, 2),
            "realized_usd": round(realized_usd, 2),
            "floating_usd": round(floating_usd, 2),
            "csv_path": str(self.csv_path),
        }
