"""Anchor resolution + lazy stub creation + ANCHORED_TO edge writes.

The `record` MCP tool accepts a list of typed anchor JSON objects (Commit /
File / Symbol / Project / Concept / Person). For each one we either look up
the existing graph node by natural key (sha / email / (repo,path) / name) or
create a sparse stub from whatever fields the client supplied. Then we write
one bi-temporal ANCHORED_TO edge from the source memory node to the anchor.

This module is the only place where memory nodes acquire graph context. Bulk
git ingestion is no longer an MCP tool; helpers from `ingest/git_extractor`
are imported here so commit/file/person stubs share a single code path.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import kuzu

from ingest.git_extractor import (
    _upsert_file as _upsert_file_node,
    _upsert_person as _upsert_person_node,
)

EXTRACTOR_NAME = "mcp"
EXTRACTOR_VERSION = "v2"

# Memory-node kinds that can author ANCHORED_TO edges.
ANCHOR_SOURCE_LABELS = {"Episode", "Decision", "Task", "Idea", "Note"}


def resolve_project(conn: kuzu.Connection, hint: str | None) -> str | None:
    """Look up a Project by name or create one. Returns its id, or None."""
    if not hint:
        return None
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


def _resolve_repo_by_name(conn: kuzu.Connection, name: str | None) -> str | None:
    """Best-effort Repo lookup by remote_url or path tail. None if not found.

    Anchors arrive with a free-form `repo` field (usually a slug). We try a
    suffix match against remote_url and against the resolved local path. If
    that misses we create a sparse Repo stub keyed on the slug so the File
    anchor still has somewhere to live.
    """
    if not name:
        return None
    # Try remote_url contains slug, then path contains slug.
    result = conn.execute(
        "MATCH (r:Repo) WHERE r.remote_url CONTAINS $n OR r.path CONTAINS $n "
        "RETURN r.id LIMIT 1",
        {"n": name},
    )
    if result.has_next():
        return str(result.get_next()[0])
    return None


def _create_repo_stub(conn: kuzu.Connection, name: str) -> str:
    repo_id = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Repo {id: $id, path: $p, remote_url: NULL, default_branch: NULL})",
        {"id": repo_id, "p": name},
    )
    return repo_id


def _resolve_or_create_repo(conn: kuzu.Connection, name: str | None) -> str | None:
    if not name:
        return None
    existing = _resolve_repo_by_name(conn, name)
    if existing:
        return existing
    return _create_repo_stub(conn, name)


def _resolve_commit(
    conn: kuzu.Connection, sha: str, anchor: dict[str, Any], now: datetime
) -> tuple[str, str]:
    """Return (node_label, primary_key_value). MERGE the Commit by sha."""
    result = conn.execute(
        "MATCH (c:Commit {sha: $sha}) RETURN c.sha LIMIT 1", {"sha": sha}
    )
    if result.has_next():
        return ("Commit", str(result.get_next()[0]))
    # Sparse stub: clients may omit message/authored_at; idempotent backfill happens
    # if the same sha re-appears with richer data (handled lazily; not required).
    repo_id = _resolve_or_create_repo(conn, anchor.get("repo"))
    authored_at = anchor.get("authored_at")
    if isinstance(authored_at, str):
        try:
            authored_at = datetime.fromisoformat(authored_at.replace("Z", "+00:00"))
        except ValueError:
            authored_at = None
    if authored_at is None:
        authored_at = now
    conn.execute(
        "CREATE (:Commit {sha: $sha, repo_id: $rid, message: $m, authored_at: $ts})",
        {
            "sha": sha,
            "rid": repo_id,
            "m": anchor.get("message") or "",
            "ts": authored_at,
        },
    )
    return ("Commit", sha)


def _resolve_file(
    conn: kuzu.Connection, anchor: dict[str, Any]
) -> tuple[str, str]:
    path = anchor.get("path")
    if not path:
        raise ValueError("file anchor requires 'path'")
    repo_id = _resolve_or_create_repo(conn, anchor.get("repo")) or ""
    fid, _ = _upsert_file_node(conn, {}, repo_id, path)
    return ("File", fid)


def _resolve_symbol(
    conn: kuzu.Connection, anchor: dict[str, Any]
) -> tuple[str, str]:
    name = anchor.get("name")
    file_path = anchor.get("file")
    if not name or not file_path:
        raise ValueError("symbol anchor requires 'name' and 'file'")
    # Ensure underlying File exists.
    file_label, file_id = _resolve_file(
        conn, {"repo": anchor.get("repo"), "path": file_path}
    )
    line = anchor.get("line")
    result = conn.execute(
        "MATCH (s:Symbol {file_id: $fid, name: $n}) RETURN s.id LIMIT 1",
        {"fid": file_id, "n": name},
    )
    if result.has_next():
        return ("Symbol", str(result.get_next()[0]))
    sid = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Symbol {id: $id, file_id: $fid, name: $n, kind: $k, line: $l})",
        {
            "id": sid,
            "fid": file_id,
            "n": name,
            "k": anchor.get("symbol_kind") or "",
            "l": line if isinstance(line, int) else None,
        },
    )
    return ("Symbol", sid)


def _resolve_project_anchor(
    conn: kuzu.Connection, anchor: dict[str, Any]
) -> tuple[str, str]:
    name = anchor.get("name")
    if not name:
        raise ValueError("project anchor requires 'name'")
    pid = resolve_project(conn, name)
    assert pid is not None
    return ("Project", pid)


def _resolve_concept(
    conn: kuzu.Connection, anchor: dict[str, Any]
) -> tuple[str, str]:
    name = anchor.get("name")
    if not name:
        raise ValueError("concept anchor requires 'name'")
    result = conn.execute(
        "MATCH (c:Concept {name: $n}) RETURN c.id LIMIT 1", {"n": name}
    )
    if result.has_next():
        return ("Concept", str(result.get_next()[0]))
    cid = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Concept {id: $id, name: $n})", {"id": cid, "n": name}
    )
    return ("Concept", cid)


def _resolve_person(
    conn: kuzu.Connection, anchor: dict[str, Any]
) -> tuple[str, str]:
    email = anchor.get("email")
    if not email:
        raise ValueError("person anchor requires 'email'")
    name = anchor.get("name") or email.split("@")[0]
    pid, _ = _upsert_person_node(conn, {}, name, email)
    return ("Person", pid)


_RESOLVERS = {
    "file": _resolve_file,
    "symbol": _resolve_symbol,
    "project": _resolve_project_anchor,
    "concept": _resolve_concept,
    "person": _resolve_person,
}


def resolve_anchor(
    conn: kuzu.Connection, anchor: dict[str, Any], now: datetime
) -> tuple[str, str]:
    """Resolve a single anchor JSON to (target_label, target_pk_value).

    Creates a stub if no matching node exists. Idempotent: re-resolving the
    same anchor reuses the existing node.
    """
    kind = anchor.get("kind")
    if not kind:
        raise ValueError(f"anchor missing 'kind': {anchor}")
    if kind == "commit":
        sha = anchor.get("sha")
        if not sha:
            raise ValueError("commit anchor requires 'sha'")
        return _resolve_commit(conn, sha, anchor, now)
    resolver = _RESOLVERS.get(kind)
    if resolver is None:
        raise ValueError(f"unknown anchor kind: {kind}")
    return resolver(conn, anchor)


def _pk_field(label: str) -> str:
    return "sha" if label == "Commit" else "id"


def write_anchor_edge(
    conn: kuzu.Connection,
    source_label: str,
    source_id: str,
    target_label: str,
    target_pk: str,
    now: datetime,
) -> None:
    """Create one ANCHORED_TO edge if not already present."""
    if source_label not in ANCHOR_SOURCE_LABELS:
        raise ValueError(f"invalid anchor source: {source_label}")
    target_field = _pk_field(target_label)
    existing = conn.execute(
        f"MATCH (s:{source_label} {{id: $sid}})-[e:ANCHORED_TO]->"
        f"(t:{target_label} {{{target_field}: $tid}}) RETURN e LIMIT 1",
        {"sid": source_id, "tid": target_pk},
    )
    if existing.has_next():
        return
    conn.execute(
        f"MATCH (s:{source_label} {{id: $sid}}), "
        f"(t:{target_label} {{{target_field}: $tid}}) "
        "CREATE (s)-[:ANCHORED_TO {valid_from: $now, valid_to: NULL, "
        "recorded_at: $now, source_extractor: $ex, extractor_version: $exv}]->(t)",
        {
            "sid": source_id,
            "tid": target_pk,
            "now": now,
            "ex": EXTRACTOR_NAME,
            "exv": EXTRACTOR_VERSION,
        },
    )


def apply_anchors(
    conn: kuzu.Connection,
    source_label: str,
    source_id: str,
    anchors: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, str]]:
    """Resolve each anchor and link it. Returns a summary list."""
    out: list[dict[str, str]] = []
    for a in anchors or []:
        target_label, target_pk = resolve_anchor(conn, a, now)
        write_anchor_edge(
            conn, source_label, source_id, target_label, target_pk, now
        )
        out.append({"target_label": target_label, "target_id": target_pk})
    return out
