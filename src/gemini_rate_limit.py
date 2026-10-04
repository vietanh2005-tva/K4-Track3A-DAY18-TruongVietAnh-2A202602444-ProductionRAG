"""Coordinate the lab's Gemini request budget across Python processes."""
from pathlib import Path
import os
import sqlite3
import time


def wait_for_gemini(model):
    gap = float(os.getenv("GEMINI_REQUEST_INTERVAL", "4.5"))
    if gap <= 0:
        raise ValueError("GEMINI_REQUEST_INTERVAL must be positive")
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_dir / "gemini_requests.sqlite", timeout=30) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS reservations (model TEXT PRIMARY KEY, next_time REAL NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT next_time FROM reservations WHERE model=?", (model,)).fetchone()
        now = time.time()
        reserved = max(now, row[0] if row else now)
        conn.execute("INSERT OR REPLACE INTO reservations VALUES (?, ?)", (model, reserved + gap))
        conn.commit()
    delay = reserved - now
    if delay > 0:
        time.sleep(delay)
