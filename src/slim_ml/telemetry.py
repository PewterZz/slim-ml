from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class TokenEvent:
    t: float
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class TokRateMeter:
    def __init__(self, window: int = 64):
        self._timestamps: deque[float] = deque(maxlen=window)
        self._first: Optional[float] = None
        self._count = 0

    def tick(self) -> None:
        now = time.monotonic()
        if self._first is None:
            self._first = now
        self._timestamps.append(now)
        self._count += 1

    def rolling_tps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span

    def mean_tps(self) -> float:
        if self._first is None or self._count < 2:
            return 0.0
        elapsed = time.monotonic() - self._first
        return (self._count - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def count(self) -> int:
        return self._count


class JsonlRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)

    def record(self, kind: str, **payload: Any) -> None:
        self._fh.write(json.dumps(asdict(TokenEvent(t=time.time(), kind=kind, payload=payload))) + "\n")

    def close(self) -> None:
        self._fh.close()


class NullRecorder:
    def record(self, kind: str, **payload: Any) -> None: ...
    def close(self) -> None: ...
