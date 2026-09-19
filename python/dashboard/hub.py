"""Thread-safe latest training tick for the dashboard."""

from __future__ import annotations

import threading
from collections import deque

_lock = threading.Lock()
_latest: dict | None = None
_history: deque[dict] = deque(maxlen=240)


def reset() -> None:
    global _latest
    with _lock:
        _latest = None
        _history.clear()


def publish(tick: dict) -> dict:
    global _latest
    with _lock:
        _latest = tick
        _history.append(tick)
    return tick


def latest() -> dict | None:
    with _lock:
        return _latest


def history() -> list[dict]:
    with _lock:
        return list(_history)
