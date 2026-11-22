"""Local SQLite cache for OHLCV bars, watchlist state, and alert tracking."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


DEFAULT_CACHE_DB = Path(__file__).resolve().parent.parent.parent / ".fx_screener_cache.sqlite3"


class MarketDataCache:
    """SQLite-backed OHLCV bar cache with configurable TTL."""

    def __init__(self, db_path: Path | str = DEFAULT_CACHE_DB, ttl_seconds: int = 300) -> None:
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv_bars (
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (pair, timeframe, ts)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_meta (
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    fetched_at REAL NOT NULL,
                    provider TEXT NOT NULL,
                    PRIMARY KEY (pair, timeframe)
                )
                """
            )

    def is_fresh(self, pairs: List[str], timeframes: tuple[str, ...] = ("H1", "H4", "D1", "W1")) -> bool:
        now = time.time()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pair, timeframe, fetched_at FROM cache_meta"
            ).fetchall()
        meta = {(r[0], r[1]): float(r[2]) for r in rows}
        for pair in pairs:
            for tf in timeframes:
                ts = meta.get((pair, tf))
                if ts is None or (now - ts) > self.ttl_seconds:
                    return False
        return True

    def last_updated_info(self) -> tuple[Optional[float], str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(fetched_at), provider FROM cache_meta"
            ).fetchone()
        if not row or row[0] is None:
            return None, "none"
        return float(row[0]), str(row[1] or "cache")

    def save_universe(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        provider_name: str = "yfinance",
    ) -> None:
        now = time.time()
        bar_rows = []
        meta_rows = []

        for pair, tf_dict in data.items():
            for tf, df in tf_dict.items():
                if df is None or df.empty:
                    continue
                meta_rows.append((pair, tf, now, provider_name))
                for idx, row in df.iterrows():
                    ts_str = pd.Timestamp(idx).isoformat()
                    bar_rows.append(
                        (
                            pair,
                            tf,
                            ts_str,
                            float(row["Open"]),
                            float(row["High"]),
                            float(row["Low"]),
                            float(row["Close"]),
                            float(row.get("Volume", 0.0)),
                        )
                    )

        if not bar_rows:
            return

        with self._connect() as conn:
            for pair, tf, _, _ in meta_rows:
                conn.execute(
                    "DELETE FROM ohlcv_bars WHERE pair = ? AND timeframe = ?",
                    (pair, tf),
                )
            conn.executemany(
                """
                INSERT OR REPLACE INTO ohlcv_bars
                (pair, timeframe, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                bar_rows,
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO cache_meta
                (pair, timeframe, fetched_at, provider)
                VALUES (?, ?, ?, ?)
                """,
                meta_rows,
            )

    def load_universe(
        self, pairs: List[str], timeframes: tuple[str, ...] = ("H1", "H4", "D1", "W1")
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        result: Dict[str, Dict[str, pd.DataFrame]] = {p: {} for p in pairs}
        with self._connect() as conn:
            for pair in pairs:
                for tf in timeframes:
                    rows = conn.execute(
                        """
                        SELECT ts, open, high, low, close, volume
                        FROM ohlcv_bars
                        WHERE pair = ? AND timeframe = ?
                        ORDER BY ts ASC
                        """,
                        (pair, tf),
                    ).fetchall()
                    if not rows:
                        continue
                    df = pd.DataFrame(
                        rows,
                        columns=["ts", "Open", "High", "Low", "Close", "Volume"],
                    )
                    df["ts"] = pd.to_datetime(df["ts"], utc=True)
                    df = df.set_index("ts")
                    result[pair][tf] = df
        # Keep pairs that have at least H4, D1, W1
        return {
            p: tfs
            for p, tfs in result.items()
            if all(req in tfs for req in ("H4", "D1", "W1"))
        }
