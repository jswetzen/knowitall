from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
from mcp.server.fastmcp import FastMCP

from server import config
from server.deps import AppState, embed

MUTATING_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|DROP|REMOVE|DETACH|COPY|ALTER|INSTALL|LOAD|ATTACH|CALL)\b",
    re.IGNORECASE,
)


def _resolve_project(state: AppState, hint: str | None) -> str | None:
    if not hint:
        return None
    conn = state.kuzu_conn()
    result = conn.execute(
        "MATCH (p:Project {name: $name}) RETURN p.id LIMIT 1",
        {"name": hint},
    )
    if result.has_next():
        return str(result.get_next()[0])
    project_id = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Project {id: $id, name: $name, status: 'active', created_at: $ts})",
        {"id": project_id, "name": hint, "ts": datetime.now(timezone.utc)},
    )
    return project_id


def register_tools(mcp: FastMCP, state: AppState) -> None:
    @mcp.tool()
    async def store_episode(
        text: str, kind: str, project_hint: str | None = None
    ) -> dict[str, Any]:
        """Embed text and store as an episode in LanceDB.

        Returns the episode id and the resolved project id (created if hint was novel).
        """
        vec = await embed(state.http, text)
        if len(vec) != config.settings.embedding_dim:
            raise RuntimeError(
                f"embedding dim mismatch: got {len(vec)}, configured {config.settings.embedding_dim}"
            )
        project_id = _resolve_project(state, project_hint)
        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        batch = pa.table(
            {
                "id": [episode_id],
                "text": [text],
                "vector": pa.array(
                    [vec],
                    type=pa.list_(pa.float32(), list_size=config.settings.embedding_dim),
                ),
                "kind": [kind],
                "project_id": [project_id],
                "conversation_id": [None],
                "created_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
                "model_version": [config.settings.ollama_model],
            },
            schema=state.episodes.schema,
        )
        state.episodes.add(batch)
        return {"id": episode_id, "project_id": project_id}

    @mcp.tool()
    async def query_memory(
        query: str, project_hint: str | None = None, k: int = 10
    ) -> list[dict[str, Any]]:
        """Vector search over stored episodes. Optional project filter."""
        vec = await embed(state.http, query)
        search = state.episodes.search(vec).limit(k)
        if project_hint:
            project_id = _resolve_project(state, project_hint)
            if project_id is None:
                return []
            search = search.where(f"project_id = '{project_id}'", prefilter=True)
        table = search.to_arrow()
        rows: list[dict[str, Any]] = []
        for record in table.to_pylist():
            created = record.get("created_at")
            rows.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "kind": record["kind"],
                    "project_id": record["project_id"],
                    "created_at": created.isoformat() if created is not None else None,
                    "score": float(record["_distance"]),
                }
            )
        return rows

    @mcp.tool()
    async def cypher(query: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        """Read-only Kùzu Cypher passthrough. Mutating keywords are rejected."""
        if MUTATING_PATTERN.search(query):
            raise ValueError(
                "Mutating Cypher rejected. Allowed: read-only MATCH/RETURN/WHERE/WITH/ORDER/LIMIT."
            )
        conn = state.kuzu_conn()
        result = conn.execute(query, params or {})
        rows: list[list[Any]] = []
        while result.has_next():
            rows.append([_coerce(v) for v in result.get_next()])
        return rows


def _coerce(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v
