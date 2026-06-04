from __future__ import annotations

import asyncio
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import kuzu
import lancedb
import pyarrow as pa

from schema.migrate import apply_migrations
from server import config

# An IVF index needs enough rows to train meaningful partitions; below this we
# leave the table index-free (a flat scan over a few hundred vectors is fast).
_MIN_ROWS_FOR_INDEX = 256


def _embeddings_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("node_type", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
            pa.field("project_id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
            pa.field("model_version", pa.string()),
            pa.field("retracted_at", pa.timestamp("us", tz="UTC")),
        ]
    )


@dataclass
class AppState:
    kuzu_db: kuzu.Database
    lance_db: "lancedb.DBConnection"
    embeddings: "lancedb.table.Table"
    http: httpx.AsyncClient
    # Inserts since the last optimize pass. record() bumps this and triggers
    # maintenance every config.maint_interval rows so the index/fragments
    # don't drift between restarts.
    inserts_since_optimize: int = 0
    # Held reference to the in-flight background optimize task. Both prevents
    # GC of a bare create_task and acts as the "already running" guard so we
    # never stack concurrent optimize() passes on the same table.
    _optimize_task: "asyncio.Task[None] | None" = field(
        default=None, repr=False
    )

    def kuzu_conn(self) -> kuzu.Connection:
        return kuzu.Connection(self.kuzu_db)

    def note_insert_and_maybe_optimize(self) -> None:
        """Call after each embedding insert. Every Nth row, kick off a
        background optimize so record() can return without waiting on it.

        optimize() is blocking LanceDB I/O, so it runs via asyncio.to_thread
        on a worker thread rather than on the event loop. If a previous pass
        is still running we skip (and don't reset the counter), so the work
        coalesces instead of piling up under a burst of writes.
        """
        interval = config.settings.maint_interval
        if interval <= 0:
            return
        self.inserts_since_optimize += 1
        if self.inserts_since_optimize < interval:
            return
        if self._optimize_task is not None and not self._optimize_task.done():
            # Previous optimize still running; let it finish, try again next
            # insert. Counter stays at/above interval so we retry promptly.
            return
        self.inserts_since_optimize = 0
        self._optimize_task = asyncio.create_task(self._run_optimize())

    async def _run_optimize(self) -> None:
        try:
            await asyncio.to_thread(optimize_embeddings, self.embeddings)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"knowitall.maint background optimize failed: {exc}",
                  file=sys.stderr, flush=True)

    async def aclose(self) -> None:
        # Let an in-flight optimize finish so we don't tear down its thread
        # mid-write; it's bounded and idempotent so the wait is short.
        if self._optimize_task is not None and not self._optimize_task.done():
            try:
                await self._optimize_task
            except Exception:  # pragma: no cover - already logged in _run_optimize
                pass
        await self.http.aclose()


def build_state() -> AppState:
    s = config.settings
    data_dir = Path(s.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    kuzu_db = kuzu.Database(str(data_dir / "kuzu"))
    apply_migrations(kuzu_db)

    lance_db = lancedb.connect(str(data_dir / "lance"))
    tables = lance_db.list_tables()
    # v2 cuts over: drop the v1 `episodes` table if it lingers. User has no
    # data to migrate (per PLAN_V2.md), so this is safe.
    if "episodes" in tables:
        lance_db.drop_table("episodes")
    if "embeddings" in tables:
        embeddings = lance_db.open_table("embeddings")
    else:
        embeddings = lance_db.create_table(
            "embeddings",
            schema=_embeddings_schema(s.embedding_dim),
            exist_ok=True,
        )

    # Startup maintenance: compact fragments + (re)index. Once per deploy;
    # self-heals a table that fragmented or grew since the last boot.
    optimize_embeddings(embeddings)

    http = httpx.AsyncClient(base_url=s.ollama_url, timeout=5.0)
    return AppState(
        kuzu_db=kuzu_db, lance_db=lance_db, embeddings=embeddings, http=http
    )


def _has_vector_index(table: "lancedb.table.Table") -> bool:
    try:
        return any("vector" in idx.columns for idx in table.list_indices())
    except Exception:
        return False


def optimize_embeddings(table: "lancedb.table.Table") -> None:
    """Compact fragments and (re)build the IVF_PQ vector index.

    Two problems this fixes, both observed via KNOWITALL_PROFILE:
      - record() appends one row per call, so the table accrues many tiny
        fragments; `optimize()` compacts them, which is what keeps both
        writes and scans fast.
      - query_memory() with no ANN index does a brute-force flat scan
        (~300ms for 8 hits). An IVF_PQ index turns that into a partition
        probe.

    Safe to call repeatedly. No-op below _MIN_ROWS_FOR_INDEX rows, where a
    flat scan is already fast and IVF training would be ill-conditioned.
    """
    try:
        rows = table.count_rows()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"knowitall.maint optimize skipped: count failed: {exc}",
              file=sys.stderr, flush=True)
        return
    if rows < _MIN_ROWS_FOR_INDEX:
        return

    # Compact fragments + merge any pending index deltas. Cheap relative to a
    # full reindex and the main lever for write/scan latency.
    try:
        table.optimize()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"knowitall.maint optimize() failed: {exc}",
              file=sys.stderr, flush=True)

    # Size partitions to ~sqrt(rows): the IVF rule of thumb. num_sub_vectors
    # must divide the embedding dim (768 / 96 = 8). replace=True rebuilds in
    # place so growth is reflected on each maintenance pass.
    num_partitions = max(1, min(256, int(math.sqrt(rows))))
    try:
        table.create_index(
            metric="l2",
            num_partitions=num_partitions,
            num_sub_vectors=96,
            index_type="IVF_PQ",
            replace=True,
        )
        print(
            f"knowitall.maint indexed rows={rows} "
            f"num_partitions={num_partitions}",
            file=sys.stderr, flush=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"knowitall.maint create_index failed: {exc}",
              file=sys.stderr, flush=True)


async def embed(http: httpx.AsyncClient, text: str) -> list[float]:
    """One embedding call, with one retry on transport error."""
    payload = {"model": config.settings.ollama_model, "input": text}
    for attempt in range(2):
        try:
            r = await http.post("/api/embed", json=payload)
            r.raise_for_status()
            data = r.json()
            embeddings = data.get("embeddings") or [data.get("embedding")]
            vec = embeddings[0]
            if vec is None:
                raise RuntimeError(f"Ollama returned no embedding: {data}")
            return vec
        except (httpx.TransportError, httpx.ReadTimeout):
            if attempt == 1:
                raise
            await asyncio.sleep(0.2)
    raise RuntimeError("unreachable")
