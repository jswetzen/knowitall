"""Env-gated, near-zero-cost tool profiling.

When `KNOWITALL_PROFILE=1`, an instrumented tool emits exactly one structured
line to stderr per call, breaking total latency into named stages. The point is
to answer one question without guessing: for a slow `query_memory`, is the time
in the Ollama embed round-trip, the LanceDB vector scan, or Kùzu? Those have
completely different fixes (network vs. ANN index vs. graph query), so we
measure before we cut.

When profiling is off, `profile()` returns a no-op recorder: no clock reads, no
allocations beyond the recorder itself, no output. Instrumentation can stay in
the hot path permanently.

Usage:

    with profile("query_memory") as p:
        with p.stage("embed"):
            vec = await embed(...)
        with p.stage("lance"):
            table = search.to_arrow()
        p.count(rows=len(table))

Emitted line (stderr):

    knowitall.profile tool=query_memory total_ms=812.4 embed_ms=780.1 \
        lance_ms=30.2 kuzu_ms=1.8 rows=7
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator

from server import config


class _NullRecorder:
    """No-op recorder used when profiling is disabled. Cheap and inert."""

    @contextmanager
    def stage(self, _name: str) -> Iterator[None]:
        yield

    def count(self, **_counts: int) -> None:
        pass


class _Recorder:
    def __init__(self, tool: str) -> None:
        self._tool = tool
        self._stages: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            # Accumulate so a stage entered in a loop sums rather than clobbers.
            self._stages[name] = (
                self._stages.get(name, 0.0)
                + (time.perf_counter() - start) * 1000.0
            )

    def count(self, **counts: int) -> None:
        self._counts.update(counts)

    def emit(self) -> None:
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        parts = [f"knowitall.profile tool={self._tool}", f"total_ms={total_ms:.1f}"]
        parts += [f"{name}_ms={ms:.1f}" for name, ms in self._stages.items()]
        parts += [f"{name}={n}" for name, n in self._counts.items()]
        print(" ".join(parts), file=sys.stderr, flush=True)


@contextmanager
def profile(tool: str) -> Iterator[_Recorder | _NullRecorder]:
    """Time a tool call by stage. No-op unless KNOWITALL_PROFILE is set."""
    if not config.settings.profile:
        yield _NullRecorder()
        return
    rec = _Recorder(tool)
    try:
        yield rec
    finally:
        rec.emit()
