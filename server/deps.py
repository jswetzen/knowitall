from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx
import kuzu
import lancedb
import pyarrow as pa

from schema.migrate import apply_migrations
from server import config


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

    def kuzu_conn(self) -> kuzu.Connection:
        return kuzu.Connection(self.kuzu_db)

    async def aclose(self) -> None:
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
        )

    http = httpx.AsyncClient(base_url=s.ollama_url, timeout=5.0)
    return AppState(
        kuzu_db=kuzu_db, lance_db=lance_db, embeddings=embeddings, http=http
    )


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
