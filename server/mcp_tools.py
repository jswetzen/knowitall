from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
from mcp.server.fastmcp import FastMCP

from server import config
from server.anchors import (
    ANCHOR_SOURCE_LABELS,
    apply_anchors,
    resolve_project,
)
from server.deps import AppState, embed

MUTATING_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|DROP|REMOVE|DETACH|COPY|ALTER|INSTALL|LOAD|ATTACH|CALL)\b",
    re.IGNORECASE,
)

# Summary length used by list_memories and Note.title clipping. One constant,
# one rule: summaries are first-N-of-body when not explicitly stored.
SUMMARY_MAX_LEN = 200

# kind -> (graph_label_or_None, node_type_for_embeddings, embeds?)
# Episode-flavored kinds become Episode nodes carrying `kind=`.
KIND_TO_LABEL: dict[str, str] = {
    "decision": "Decision",
    "task": "Task",
    "idea": "Idea",
    "note": "Note",
    "summary": "Episode",
    "blocker": "Episode",
    "fact": "Episode",
    "episode": "Episode",
}

# What we record as `node_type` on the embeddings row. Differs from graph label
# only for Episode kinds, which preserve their sub-kind through `kind` column.
NODE_TYPE_BY_KIND: dict[str, str] = {
    "decision": "decision",
    "task": "task",
    "idea": "idea",
    "note": "note",
    "summary": "episode",
    "blocker": "episode",
    "fact": "episode",
    "episode": "episode",
}

# Memory-bearing graph labels and the kind string `list_memories`/`query_memory`
# expose for filtering. Order matches a stable enumeration over the five tables.
MEMORY_KINDS: tuple[str, ...] = ("decision", "task", "idea", "note", "episode")
MEMORY_KIND_TO_LABEL: dict[str, str] = {
    "decision": "Decision",
    "task": "Task",
    "idea": "Idea",
    "note": "Note",
    "episode": "Episode",
}

# Per-label MATCH config for `list_memories` / `get_memory`. Each label exposes
# a `created_at`-shaped timestamp under a different column name, so we project
# it under a uniform alias. `body_field` is the column whose first
# SUMMARY_MAX_LEN chars become the fallback summary. `summary_field` is the
# stored summary column when present; Note has no parallel `summary` column
# because `title` already serves that purpose, so `summary_field == "title"`
# and `_summarize_with_fallback` resolves it without a body fallback.
_LIST_FIELDS: dict[str, dict[str, str | None]] = {
    "Decision": {"created_at_field": "decided_at", "body_field": "body",  "summary_field": "summary"},
    "Task":     {"created_at_field": "created_at", "body_field": "body",  "summary_field": "summary"},
    "Idea":     {"created_at_field": "created_at", "body_field": "body",  "summary_field": "summary"},
    "Note":     {"created_at_field": "created_at", "body_field": "title", "summary_field": None},
    "Episode":  {"created_at_field": "created_at", "body_field": "body",  "summary_field": "summary"},
}

# Allowed `kind` strings in `record(..., relates_to=[{"kind": ..., "id": ...}])`
# → the graph edge label they translate to. SUPERSEDES_MEMORY is named that
# way so it doesn't collide with v1's Decision→Decision SUPERSEDES (left
# intact per the v3 schema decision).
MEMORY_EDGE_KIND_TO_LABEL: dict[str, str] = {
    "supersedes":  "SUPERSEDES_MEMORY",
    "refines":     "REFINES",
    "contradicts": "CONTRADICTS",
    "relates_to":  "RELATES_TO_MEMORY",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _insert_embedding_row(
    state: AppState,
    *,
    row_id: str,
    node_type: str,
    text: str,
    vec: list[float],
    project_id: str | None,
    kind: str | None,
    created_at: datetime,
) -> None:
    batch = pa.table(
        {
            "id": [row_id],
            "node_type": [node_type],
            "text": [text],
            "vector": pa.array(
                [vec],
                type=pa.list_(pa.float32(), list_size=config.settings.embedding_dim),
            ),
            "project_id": [project_id],
            "kind": [kind],
            "created_at": pa.array([created_at], type=pa.timestamp("us", tz="UTC")),
            "model_version": [config.settings.ollama_model],
            "retracted_at": pa.array([None], type=pa.timestamp("us", tz="UTC")),
        },
        schema=state.embeddings.schema,
    )
    state.embeddings.add(batch)


def _create_graph_node(
    conn,
    *,
    label: str,
    node_id: str,
    body: str,
    kind: str,
    created_at: datetime,
    summary: str | None = None,
) -> None:
    """Insert a memory-bearing node. `summary` is stored only on labels with
    a real `summary` column (Decision/Task/Idea/Episode). Note ignores the
    arg because its `title` column already serves the title-shaped role —
    Note's title is computed as the first SUMMARY_MAX_LEN chars of body if
    no explicit summary was passed, else the explicit summary (also clipped).
    """
    if label == "Episode":
        conn.execute(
            "CREATE (:Episode {id: $id, body: $b, kind: $k, created_at: $ts, "
            "model_version: $mv, retracted_at: NULL, amended_at: NULL, "
            "summary: $s})",
            {
                "id": node_id,
                "b": body,
                "k": kind,
                "ts": created_at,
                "mv": config.settings.ollama_model,
                "s": summary,
            },
        )
    elif label == "Decision":
        conn.execute(
            "CREATE (:Decision {id: $id, body: $b, decided_at: $ts, "
            "retracted_at: NULL, amended_at: NULL, summary: $s})",
            {"id": node_id, "b": body, "ts": created_at, "s": summary},
        )
    elif label == "Task":
        conn.execute(
            "CREATE (:Task {id: $id, body: $b, status: 'open', created_at: $ts, "
            "closed_at: NULL, retracted_at: NULL, amended_at: NULL, summary: $s})",
            {"id": node_id, "b": body, "ts": created_at, "s": summary},
        )
    elif label == "Idea":
        conn.execute(
            "CREATE (:Idea {id: $id, body: $b, status: 'open', created_at: $ts, "
            "died_at: NULL, retracted_at: NULL, amended_at: NULL, summary: $s})",
            {"id": node_id, "b": body, "ts": created_at, "s": summary},
        )
    elif label == "Note":
        # Note has no separate summary column — title is the summary. Prefer
        # the explicit summary if given, else clip the body.
        title = (summary if summary is not None else body)[:SUMMARY_MAX_LEN]
        conn.execute(
            "CREATE (:Note {id: $id, path: NULL, title: $b, created_at: $ts, "
            "retracted_at: NULL, amended_at: NULL})",
            {"id": node_id, "b": title, "ts": created_at},
        )
    else:
        raise ValueError(f"unknown graph label: {label}")


def _lookup_node_label(conn, node_id: str) -> str | None:
    """Find which memory-bearing label owns this id. None if unknown/retracted."""
    for label in ANCHOR_SOURCE_LABELS:
        result = conn.execute(
            f"MATCH (n:{label} {{id: $id}}) RETURN n.id LIMIT 1", {"id": node_id}
        )
        if result.has_next():
            return label
    return None


def _summarize(value: str | None) -> str | None:
    """Title-shape: first SUMMARY_MAX_LEN chars of value (or None)."""
    if value is None:
        return None
    if len(value) <= SUMMARY_MAX_LEN:
        return value
    return value[:SUMMARY_MAX_LEN]


def _effective_summary(
    label: str, stored_summary: str | None, body_value: str | None
) -> str | None:
    """Resolve the summary surfaced to MCP callers.

    Note has no separate column — `body_value` is already the title and is
    the canonical summary.

    For other labels: prefer the explicit stored `summary` if present, else
    fall back to the first SUMMARY_MAX_LEN chars of body. This keeps the
    contract stable across legacy rows (no summary column data) and new
    rows recorded with an explicit summary.
    """
    if label == "Note":
        return body_value  # already clipped at insert
    if stored_summary is not None:
        return stored_summary
    return _summarize(body_value)


def _list_one_label(
    conn,
    label: str,
    *,
    project_id: str | None,
    include_retracted: bool,
) -> list[dict[str, Any]]:
    """Return raw rows for one memory label. Sort + slice happen in Python."""
    fields = _LIST_FIELDS[label]
    ts_field = fields["created_at_field"]
    body_field = fields["body_field"]
    summary_field = fields["summary_field"]
    node_type = "episode" if label == "Episode" else label.lower()

    clauses: list[str] = []
    if not include_retracted:
        clauses.append("n.retracted_at IS NULL")

    # Note has no separate summary column — read body_field twice as a
    # uniform stand-in (the second slot becomes `stored_summary`, which we
    # then ignore via `_effective_summary("Note", ...)`).
    summary_select = f"n.{summary_field}" if summary_field else f"n.{body_field}"

    params: dict[str, Any] = {}
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    if project_id is not None:
        # Filter to a known project; we already know its id, no need to re-project.
        match_clause = (
            f"MATCH (n:{label})-[:ANCHORED_TO]->(p:Project {{id: $pid}}){where}"
        )
        params["pid"] = project_id
        return_project_clause = "$pid AS pid"
    else:
        # No filter — OPTIONAL MATCH so nodes without a Project anchor still appear.
        # WHERE must precede OPTIONAL MATCH in Kuzu; placing it after binds to
        # the optional pattern and silently nulls out p.id when the WHERE clause
        # is supposed to constrain n only.
        match_clause = (
            f"MATCH (n:{label}){where} "
            f"OPTIONAL MATCH (n)-[:ANCHORED_TO]->(p:Project)"
        )
        return_project_clause = "p.id AS pid"

    # Alias both columns: Note has body_field == summary_field == "title",
    # and Kuzu rejects duplicate result column names.
    query = (
        f"{match_clause} "
        f"RETURN n.id, n.{body_field} AS body_value, "
        f"{summary_select} AS stored_summary, n.{ts_field}, "
        f"n.retracted_at, {return_project_clause}"
    )
    result = conn.execute(query, params)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    while result.has_next():
        node_id, body_value, stored_summary, created_at, retracted_at, pid = (
            result.get_next()
        )
        # OPTIONAL MATCH can yield duplicates if a node has multiple Project
        # anchors. Dedup keeping first hit — project_id is informational, not
        # canonical (the graph holds the full set).
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        rows.append(
            {
                "id": node_id,
                "node_type": node_type,
                "summary": _effective_summary(label, stored_summary, body_value),
                "project_id": pid,
                "created_at": created_at,
                "retracted_at": retracted_at,
            }
        )
    return rows


def _write_memory_edge(
    conn,
    source_label: str,
    source_id: str,
    target_label: str,
    target_id: str,
    edge_label: str,
    now: datetime,
) -> None:
    """Create one memory→memory edge if not already present.

    Idempotent: re-issuing the same (source, target, edge) is a no-op. Uses
    the same bi-temporal columns every other edge in the schema uses.
    """
    existing = conn.execute(
        f"MATCH (s:{source_label} {{id: $sid}})-[e:{edge_label}]->"
        f"(t:{target_label} {{id: $tid}}) RETURN e LIMIT 1",
        {"sid": source_id, "tid": target_id},
    )
    if existing.has_next():
        return
    conn.execute(
        f"MATCH (s:{source_label} {{id: $sid}}), "
        f"(t:{target_label} {{id: $tid}}) "
        f"CREATE (s)-[:{edge_label} {{valid_from: $now, valid_to: NULL, "
        "recorded_at: $now, source_extractor: 'mcp', extractor_version: 'v3'}]->(t)",
        {"sid": source_id, "tid": target_id, "now": now},
    )


def _apply_relates_to(
    conn,
    source_label: str,
    source_id: str,
    relates_to: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, str]]:
    """Validate + write each memory→memory edge. Returns a summary list.

    Each entry must be {"kind": <one of MEMORY_EDGE_KIND_TO_LABEL>, "id":
    <target memory node id>}. Validation raises ValueError on unknown kind
    or unknown / non-memory target id — fail at write-time, not silently.
    """
    out: list[dict[str, str]] = []
    for spec in relates_to:
        kind = spec.get("kind")
        target_id = spec.get("id")
        if kind not in MEMORY_EDGE_KIND_TO_LABEL:
            raise ValueError(
                f"unknown relates_to kind {kind!r}. "
                f"Allowed: {sorted(MEMORY_EDGE_KIND_TO_LABEL)}"
            )
        if not target_id:
            raise ValueError(f"relates_to entry missing 'id': {spec}")
        target_label = _lookup_node_label(conn, target_id)
        if target_label is None:
            raise ValueError(
                f"relates_to target {target_id!r} is not a memory node"
            )
        edge_label = MEMORY_EDGE_KIND_TO_LABEL[kind]
        _write_memory_edge(
            conn, source_label, source_id, target_label, target_id,
            edge_label, now,
        )
        out.append({"kind": kind, "target_id": target_id, "target_label": target_label})
    return out


def _expand_neighbors(
    conn, source_label: str, source_id: str, hops: int
) -> list[dict[str, Any]]:
    """Return outbound ANCHORED_TO neighbors (1 hop). Hops>1 currently no-op extra."""
    if hops <= 0:
        return []
    result = conn.execute(
        f"MATCH (s:{source_label} {{id: $id}})-[:ANCHORED_TO]->(n) "
        "RETURN label(n), n LIMIT 25",
        {"id": source_id},
    )
    neighbors: list[dict[str, Any]] = []
    while result.has_next():
        row = result.get_next()
        label = row[0]
        node = row[1] if isinstance(row[1], dict) else {}
        summary: dict[str, Any] = {"label": label}
        # Common identifying fields per label.
        for key in ("id", "sha", "name", "path", "email"):
            v = node.get(key)
            if v is not None:
                summary[key] = v
        neighbors.append(summary)
    return neighbors


def register_tools(mcp: FastMCP, state: AppState) -> None:
    @mcp.tool()
    async def record(
        kind: str,
        body: str,
        project_hint: str | None = None,
        anchors: list[dict[str, Any]] | None = None,
        summary: str | None = None,
        relates_to: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Record a durable memory: decision / task / idea / note / summary /
        blocker / fact / episode.

        Write tool. One polymorphic entry point — pick `kind` per the taxonomy
        below; the server creates the right graph node and a matching embedding
        row so future `query_memory` calls can find it semantically.

        Use this when something would be useful in a FUTURE Claude Code session,
        after the current context window is gone. Good triggers:
          - The user states something durable about themselves or the project
            ("we always do X", "next we want Y", "Z is the blocker").
          - A design decision is made — capture the choice AND the rationale.
          - A bug is tracked down — capture the root cause and the fix.
          - A feature is finished — capture what was built.
          - Session is wrapping up: proactively ASK before storing a `summary`.

        Do NOT use for: transient debug output, things already captured in git
        or code, scratchpad thinking, or speculative ideas the user hasn't
        endorsed.

        kind taxonomy:
          - "decision" | "task" | "idea" | "note": become first-class graph
            nodes (citable, expandable, can be the target of edges).
          - "summary" | "blocker" | "fact" | "episode": become Episode nodes
            carrying the kind, for less structural / more narrative content.

        body: self-contained prose. Must be readable without surrounding chat.

        project_hint: project NAME (not id). If novel, a Project node is
        created. Omit to leave unattached. Stored on the embedding row for
        filter pushdown AND linked as an ANCHORED_TO edge in the graph.

        anchors: list of typed JSON objects citing where this memory came
        from. Shapes:
          {"kind": "commit", "sha": "abc123", "repo": "myrepo",
           "message": "...", "authored_at": "2026-05-15T...",
           "author_email": "..."}
          {"kind": "file",   "repo": "myrepo", "path": "server/app.py"}
          {"kind": "symbol", "repo": "myrepo", "file": "server/app.py",
           "name": "create_app", "line": 14}
          {"kind": "project", "name": "myrepo"}
          {"kind": "concept", "name": "rate limiting"}
          {"kind": "person",  "email": "claude@swetzen.com"}
        Existing nodes are reused by natural key (sha / (repo,path) / email /
        name). Sparse anchors are accepted — backfill is a client concern.

        summary: optional ≤200-char title-shaped string surfaced by
        `list_memories` and `get_memory`. If omitted, those tools fall
        back to the first 200 chars of body. Validation rejects strings
        longer than SUMMARY_MAX_LEN — fail at write-time rather than
        silently truncate. For `kind="note"` the summary writes to the
        existing `title` column (Note has no parallel summary field).

        relates_to: optional list of memory→memory edges to write. Each
        entry: {"kind": "supersedes"|"refines"|"contradicts"|"relates_to",
        "id": "<existing memory node id>"}. Target ids are validated
        (must resolve to a memory-bearing node) — bad ids raise
        ValueError before any state changes.

        Returns: {"id", "node_type", "project_id", "anchored": [...],
        "related": [...]}.
        """
        if kind not in KIND_TO_LABEL:
            raise ValueError(
                f"unknown kind '{kind}'. Allowed: {sorted(KIND_TO_LABEL)}"
            )
        if summary is not None and len(summary) > SUMMARY_MAX_LEN:
            raise ValueError(
                f"summary too long ({len(summary)} > {SUMMARY_MAX_LEN}). "
                "Trim at the call site rather than silently truncating."
            )
        label = KIND_TO_LABEL[kind]
        node_type = NODE_TYPE_BY_KIND[kind]

        vec = await embed(state.http, body)
        if len(vec) != config.settings.embedding_dim:
            raise RuntimeError(
                f"embedding dim mismatch: got {len(vec)}, "
                f"configured {config.settings.embedding_dim}"
            )

        conn = state.kuzu_conn()
        now = _now()
        project_id = resolve_project(conn, project_hint)
        node_id = str(uuid.uuid4())

        _create_graph_node(
            conn,
            label=label,
            node_id=node_id,
            body=body,
            kind=kind,
            created_at=now,
            summary=summary,
        )

        # Project anchor: prepend so project shows up in ANCHORED_TO graph too.
        anchor_list: list[dict[str, Any]] = []
        if project_hint:
            anchor_list.append({"kind": "project", "name": project_hint})
        if anchors:
            anchor_list.extend(anchors)
        anchored = apply_anchors(conn, label, node_id, anchor_list, now)

        related: list[dict[str, str]] = []
        if relates_to:
            related = _apply_relates_to(conn, label, node_id, relates_to, now)

        _insert_embedding_row(
            state,
            row_id=node_id,
            node_type=node_type,
            text=body,
            vec=vec,
            project_id=project_id,
            kind=kind,
            created_at=now,
        )

        return {
            "id": node_id,
            "node_type": node_type,
            "project_id": project_id,
            "anchored": anchored,
            "related": related,
        }

    @mcp.tool()
    async def update_todo(
        id: str,
        status: str,
        anchors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Transition a Task's status (and optionally cite anchors).

        Write tool. For `kind=task` nodes only. Use when a task is started,
        blocked, completed, or abandoned. Optional anchors record what closed
        or blocked the task — typically a commit anchor for "done".

        status: free-form, but conventional values are
          "open" | "in_progress" | "blocked" | "done" | "abandoned".
        When status=="done" and a commit anchor is provided, a CLOSED_BY edge
        is also written from Task to Commit.

        Returns: {"id", "status", "anchored": [...]}.
        """
        conn = state.kuzu_conn()
        now = _now()
        # Verify Task exists and isn't retracted.
        result = conn.execute(
            "MATCH (t:Task {id: $id}) RETURN t.retracted_at LIMIT 1", {"id": id}
        )
        if not result.has_next():
            raise ValueError(f"no Task with id={id}")
        row = result.get_next()
        if row[0] is not None:
            raise ValueError(f"Task {id} is retracted")

        closed_at_clause = ", t.closed_at = $now" if status == "done" else ""
        conn.execute(
            f"MATCH (t:Task {{id: $id}}) SET t.status = $s{closed_at_clause}",
            {"id": id, "s": status, "now": now},
        )

        anchored = apply_anchors(conn, "Task", id, anchors or [], now)

        # If marking done and a commit anchor was provided, also write CLOSED_BY.
        if status == "done" and anchors:
            for a in anchors:
                if a.get("kind") == "commit" and a.get("sha"):
                    sha = a["sha"]
                    existing = conn.execute(
                        "MATCH (t:Task {id: $id})-[e:CLOSED_BY]->(c:Commit {sha: $sha}) "
                        "RETURN e LIMIT 1",
                        {"id": id, "sha": sha},
                    )
                    if existing.has_next():
                        continue
                    conn.execute(
                        "MATCH (t:Task {id: $id}), (c:Commit {sha: $sha}) "
                        "CREATE (t)-[:CLOSED_BY {valid_from: $now, valid_to: NULL, "
                        "recorded_at: $now, source_extractor: 'mcp', "
                        "extractor_version: 'v2'}]->(c)",
                        {"id": id, "sha": sha, "now": now},
                    )

        return {"id": id, "status": status, "anchored": anchored}

    @mcp.tool()
    async def forget(id: str, reason: str) -> dict[str, Any]:
        """Soft-delete a memory node: sets retracted_at; default queries hide it.

        Write tool. Use when the user says a stored memory is wrong, obsolete,
        or duplicate. The node and its edges remain in the graph; only the
        `retracted_at` timestamp is set. `query_memory(..., include_retracted=
        True)` will surface it again. There is no hard delete.

        Args:
          id: the node id returned by `record`.
          reason: short string captured on the audit trail (currently
                  embedded in the retraction; future versions may write a
                  separate Retraction node).

        Returns: {"id", "retracted_at": ISO timestamp, "label": <node label>}.
        """
        conn = state.kuzu_conn()
        label = _lookup_node_label(conn, id)
        if label is None:
            raise ValueError(f"no retractable node with id={id}")
        now = _now()
        conn.execute(
            f"MATCH (n:{label} {{id: $id}}) SET n.retracted_at = $now",
            {"id": id, "now": now},
        )
        # Mirror onto the LanceDB row. LanceDB update API is `update(where=..., values=...)`.
        try:
            state.embeddings.update(
                where=f"id = '{id}'",
                values={"retracted_at": now},
            )
        except Exception:
            # If the row isn't in embeddings (e.g. a kind that doesn't embed),
            # the graph-side retraction is still authoritative.
            pass
        return {
            "id": id,
            "label": label,
            "retracted_at": now.isoformat(),
            "reason": reason,
        }

    @mcp.tool()
    async def amend(
        id: str,
        body: str | None = None,
        summary: str | None = None,
        add_anchors: list[dict[str, Any]] | None = None,
        remove_anchors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """In-place edit of a memory node. Preserves id; re-embeds when body changes.

        Write tool. Use this when an existing memory needs to be revised —
        a renamed concept, a sharpened decision, an updated task body.
        Unlike `record`+`forget`, this keeps the id stable so anchors and
        any inbound memory→memory edges (RELATES_TO_MEMORY, etc.) keep
        pointing at the right thing.

        Args:
          id: the node id returned by `record` / `list_memories`.
          body: if provided, replaces the stored body AND re-embeds. The
            old embedding row is deleted and a fresh one inserted, so
            subsequent `query_memory` calls reflect the new wording.
          summary: if provided, replaces the stored summary. Pure graph
            SET — no re-embed (summary is not part of the embedded text).
            For `kind=note` the value writes to `title`. Validated against
            SUMMARY_MAX_LEN.
          add_anchors: list of typed anchor JSON (same shapes as
            `record`'s `anchors` arg) to add. Idempotent — existing
            ANCHORED_TO edges are not duplicated.
          remove_anchors: list of {"target_label", "target_id"} pairs
            (the shape returned in `record(...)["anchored"]`) to detach.
            Only the ANCHORED_TO edge is removed; the target node stays.

        Retracted nodes are rejected — to revise a retracted memory,
        either record a new one or un-retract via direct cypher (not yet
        a typed mutation).

        Returns: {"id", "node_type", "amended_at", "re_embedded": bool,
        "added": [...], "removed": [...]}.
        """
        if summary is not None and len(summary) > SUMMARY_MAX_LEN:
            raise ValueError(
                f"summary too long ({len(summary)} > {SUMMARY_MAX_LEN})."
            )
        if body is None and summary is None and not add_anchors and not remove_anchors:
            raise ValueError(
                "amend requires at least one of body / summary / add_anchors / remove_anchors"
            )

        conn = state.kuzu_conn()
        label = _lookup_node_label(conn, id)
        if label is None:
            raise ValueError(f"no memory node with id={id}")

        # Reject retracted: addressable (get_memory still works) but not editable.
        retracted_check = conn.execute(
            f"MATCH (n:{label} {{id: $id}}) RETURN n.retracted_at LIMIT 1",
            {"id": id},
        )
        if retracted_check.has_next() and retracted_check.get_next()[0] is not None:
            raise ValueError(f"cannot amend retracted node {id}")

        now = _now()
        fields = _LIST_FIELDS[label]
        body_field = fields["body_field"]
        summary_field = fields["summary_field"]
        node_type = "episode" if label == "Episode" else label.lower()

        re_embedded = False
        if body is not None:
            # Note's body lives in `title` and is clipped at SUMMARY_MAX_LEN.
            stored_body = body[:SUMMARY_MAX_LEN] if label == "Note" else body
            conn.execute(
                f"MATCH (n:{label} {{id: $id}}) SET n.{body_field} = $b",
                {"id": id, "b": stored_body},
            )
            # Delete-then-insert in LanceDB. Updating the fixed-size vector
            # column via `update(values={vector: ...})` is not battle-tested;
            # the safe path is to drop the old row entirely and re-embed.
            # Only memory kinds that embed land in LanceDB — Note doesn't
            # appear in the embeddings table by way of `record`, so a delete
            # over a missing id is a harmless no-op.
            try:
                state.embeddings.delete(f"id = '{id}'")
            except Exception:
                pass
            if label != "Note":
                vec = await embed(state.http, body)
                if len(vec) != config.settings.embedding_dim:
                    raise RuntimeError(
                        f"embedding dim mismatch: got {len(vec)}, "
                        f"configured {config.settings.embedding_dim}"
                    )
                # project_id is the optional ANCHORED_TO Project — re-derive.
                pid_res = conn.execute(
                    f"MATCH (n:{label} {{id: $id}})-[:ANCHORED_TO]->(p:Project) "
                    f"RETURN p.id LIMIT 1",
                    {"id": id},
                )
                project_id = (
                    str(pid_res.get_next()[0]) if pid_res.has_next() else None
                )
                kind_res = conn.execute(
                    f"MATCH (n:{label} {{id: $id}}) "
                    f"RETURN {'n.kind' if label == 'Episode' else 'NULL'} LIMIT 1",
                    {"id": id},
                )
                row = kind_res.get_next() if kind_res.has_next() else [None]
                stored_kind = row[0] if label == "Episode" else None
                _insert_embedding_row(
                    state,
                    row_id=id,
                    node_type=node_type,
                    text=body,
                    vec=vec,
                    project_id=project_id,
                    kind=stored_kind,
                    created_at=now,
                )
                re_embedded = True

        if summary is not None:
            if summary_field is not None:
                conn.execute(
                    f"MATCH (n:{label} {{id: $id}}) SET n.{summary_field} = $s",
                    {"id": id, "s": summary},
                )
            else:
                # Note: summary writes route to title (clipped).
                conn.execute(
                    "MATCH (n:Note {id: $id}) SET n.title = $s",
                    {"id": id, "s": summary[:SUMMARY_MAX_LEN]},
                )

        added: list[dict[str, str]] = []
        if add_anchors:
            added = apply_anchors(conn, label, id, add_anchors, now)

        removed: list[dict[str, str]] = []
        if remove_anchors:
            for spec in remove_anchors:
                target_label = spec.get("target_label")
                target_id = spec.get("target_id")
                if not target_label or not target_id:
                    raise ValueError(
                        f"remove_anchors entry needs target_label + target_id: {spec}"
                    )
                # Commit uses sha as PK; everything else uses id.
                pk_field = "sha" if target_label == "Commit" else "id"
                conn.execute(
                    f"MATCH (s:{label} {{id: $sid}})-[e:ANCHORED_TO]->"
                    f"(t:{target_label} {{{pk_field}: $tid}}) DELETE e",
                    {"sid": id, "tid": target_id},
                )
                removed.append({"target_label": target_label, "target_id": target_id})

        conn.execute(
            f"MATCH (n:{label} {{id: $id}}) SET n.amended_at = $now",
            {"id": id, "now": now},
        )

        return {
            "id": id,
            "node_type": node_type,
            "amended_at": now.isoformat(),
            "re_embedded": re_embedded,
            "added": added,
            "removed": removed,
        }

    @mcp.tool()
    async def query_memory(
        query: str,
        project_hint: str | None = None,
        k: int = 10,
        expand_hops: int = 0,
        include_retracted: bool = False,
        node_types: list[str] | None = None,
        snippet_chars: int = 240,
    ) -> list[dict[str, Any]]:
        """Semantic search over recorded memory; optional 1-hop graph expansion.

        Read tool. Retrieves passages recorded via `record`, ranks by
        embedding distance. By default returns truncated text and no
        neighbors — both are opt-in to keep the MCP output budget small.

        When to call:
          - The user asks "where is X at?" / "what did we decide about Y?"
          - Session start when the user references prior work.
          - Before saying you don't know something project-specific.

        Phrasing tip: noun phrases beat questions. "auth service location"
        beats "where is the auth service".

        Args:
          query: free-text. Embedded; ranked by vector distance.
          project_hint: project NAME (not id). Restricts to that project's
            embedding rows. If the project name doesn't exist, returns [].
          k: top-k hits to return (default 10).
          expand_hops: 0 (default) skips neighbor expansion entirely; 1
            includes outbound ANCHORED_TO targets. Values >1 are rejected
            (the multi-hop walk was never implemented — set to 1 if you
            want neighbors, or use `cypher` for deeper traversal).
          include_retracted: default False; pass True to include soft-deleted
            entries.
          node_types: filter by node_type list, e.g. ["decision","task"].
            Allowed values: "decision","task","idea","note","episode".
          snippet_chars: 240 (default) truncates each hit's text to that
            many chars with a trailing ellipsis. Pass 0 for full bodies
            (use `get_memory(id)` for a single full body without a query).

        Returns: [{"hit": {id,text,kind,node_type,project_id,score,created_at,
        retracted_at}, "neighbors": [{label, ...identifying fields}]}].
        Lower score == closer match. When expand_hops=0, "neighbors" is [].
        """
        if expand_hops not in (0, 1):
            raise ValueError(
                f"expand_hops must be 0 or 1, got {expand_hops}. "
                "Multi-hop traversal is not implemented; use cypher for it."
            )
        vec = await embed(state.http, query)
        conn = state.kuzu_conn()

        clauses: list[str] = []
        if project_hint:
            project_id = resolve_project(conn, project_hint)
            if project_id is None:
                return []
            clauses.append(f"project_id = '{project_id}'")
        if not include_retracted:
            clauses.append("retracted_at IS NULL")
        if node_types:
            quoted = ", ".join(f"'{nt}'" for nt in node_types)
            clauses.append(f"node_type IN ({quoted})")

        search = state.embeddings.search(vec).limit(k)
        if clauses:
            search = search.where(" AND ".join(clauses), prefilter=True)
        table = search.to_arrow()

        rows: list[dict[str, Any]] = []
        for record_row in table.to_pylist():
            node_id = record_row["id"]
            node_type = record_row["node_type"]
            label = (
                "Episode"
                if node_type == "episode"
                else node_type.capitalize()
            )
            text_value = record_row["text"]
            if snippet_chars > 0 and text_value is not None and len(text_value) > snippet_chars:
                text_value = text_value[:snippet_chars] + "…"
            hit = {
                "id": node_id,
                "text": text_value,
                "kind": record_row.get("kind"),
                "node_type": node_type,
                "project_id": record_row.get("project_id"),
                "created_at": (
                    record_row["created_at"].isoformat()
                    if record_row.get("created_at") is not None
                    else None
                ),
                "retracted_at": (
                    record_row["retracted_at"].isoformat()
                    if record_row.get("retracted_at") is not None
                    else None
                ),
                "score": float(record_row["_distance"]),
            }
            neighbors = (
                _expand_neighbors(conn, label, node_id, expand_hops)
                if expand_hops > 0
                else []
            )
            rows.append({"hit": hit, "neighbors": neighbors})
        return rows

    @mcp.tool()
    async def cypher(
        query: str, params: dict[str, Any] | None = None
    ) -> list[list[Any]]:
        """Read-only Kùzu Cypher passthrough — graph queries against the v2 schema.

        Read tool (structural). Use for who/what/when over the graph (authors,
        commits, files, projects, anchor links). Use `query_memory` for free-
        text semantic recall.

        v2 schema additions on top of v0/v1:
          Nodes: Episode(id, body, kind, created_at, model_version, retracted_at).
          Retracted_at columns on Decision, Task, Idea, Note.
          Edges: ANCHORED_TO (bi-temporal generic citation) from Episode |
            Decision | Task | Idea | Note → Commit | File | Symbol | Project |
            Concept | Person.

        Carried over from v0/v1: Project, Repo, Commit, File, Symbol, Note,
        Conversation, Decision, Concept, Person, Task, Idea. Edges PART_OF,
        IN_REPO, AUTHORED, MODIFIED, DEFINED_IN, CALLS, IMPORTS, DEPENDS_ON,
        BLOCKS, SUPERSEDES, GRADUATED_TO, ALIAS_OF, RELATES_TO, DECIDED_IN,
        DROPPED, CLOSED_BY, TOUCHED_BY, BELONGS_TO, MENTIONED_IN.

        Mutating keywords (CREATE/MERGE/DELETE/SET/DROP/REMOVE/DETACH/COPY/
        ALTER/INSTALL/LOAD/ATTACH/CALL) are rejected. Writes go through
        `record`, `update_todo`, or `forget`.

        Example: `MATCH (d:Decision)-[:ANCHORED_TO]->(f:File)
                  RETURN d.body, f.path`.
        """
        if MUTATING_PATTERN.search(query):
            raise ValueError(
                "Mutating Cypher rejected. Allowed: read-only "
                "MATCH/RETURN/WHERE/WITH/ORDER/LIMIT."
            )
        conn = state.kuzu_conn()
        result = conn.execute(query, params or {})
        rows: list[list[Any]] = []
        while result.has_next():
            rows.append([_coerce(v) for v in result.get_next()])
        return rows

    @mcp.tool()
    async def list_memories(
        kind: str | None = None,
        project_hint: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at_desc",
        include_retracted: bool = False,
    ) -> list[dict[str, Any]]:
        """Enumerate memory nodes without semantic ranking. Returns summaries.

        Read tool (enumeration). The third navigation primitive alongside
        `query_memory` (semantic) and `cypher` (structural). Use when you
        want to see *what's in here* without a query in mind — e.g. "what
        ideas do I have on project X?", "what tasks are open?".

        Bodies are NOT returned. Each row carries a `summary` (first ~200
        chars of body, or for Note the existing title). Use `get_memory`
        with a returned id to pull a full body when needed.

        Args:
          kind: one of "decision","task","idea","note","episode" (or None
            for all). Episode-flavored kinds (summary/blocker/fact) all
            live under "episode" here — use `query_memory` with node_types
            for sub-kind discrimination.
          project_hint: project NAME (not id). Filters via ANCHORED_TO
            Project edge. If the project name doesn't exist, returns [].
          limit: max rows returned (default 50). Applied after sort.
          offset: rows to skip (default 0). For paging.
          order_by: "created_at_desc" (default) | "created_at_asc".
          include_retracted: default False; pass True to include
            soft-deleted entries.

        Returns: [{id, node_type, summary, project_id, created_at,
        retracted_at}]. created_at/retracted_at are ISO strings or None.
        """
        if kind is not None and kind not in MEMORY_KIND_TO_LABEL:
            raise ValueError(
                f"unknown kind '{kind}'. Allowed: {sorted(MEMORY_KIND_TO_LABEL)} or None"
            )
        if order_by not in ("created_at_desc", "created_at_asc"):
            raise ValueError(
                f"unknown order_by '{order_by}'. Allowed: created_at_desc | created_at_asc"
            )

        conn = state.kuzu_conn()

        if project_hint is not None:
            result = conn.execute(
                "MATCH (p:Project {name: $n}) RETURN p.id LIMIT 1",
                {"n": project_hint},
            )
            if not result.has_next():
                return []
            project_id: str | None = str(result.get_next()[0])
        else:
            project_id = None

        labels = (
            [MEMORY_KIND_TO_LABEL[kind]] if kind is not None else
            [MEMORY_KIND_TO_LABEL[k] for k in MEMORY_KINDS]
        )

        all_rows: list[dict[str, Any]] = []
        for label in labels:
            all_rows.extend(
                _list_one_label(
                    conn,
                    label,
                    project_id=project_id,
                    include_retracted=include_retracted,
                )
            )

        # Sort, then page. Rows with NULL created_at sort to the end either way.
        reverse = order_by == "created_at_desc"
        all_rows.sort(
            key=lambda r: (r["created_at"] is None, r["created_at"] or datetime.min),
            reverse=reverse,
        )
        sliced = all_rows[offset : offset + limit]

        for row in sliced:
            row["created_at"] = (
                row["created_at"].isoformat() if row["created_at"] is not None else None
            )
            row["retracted_at"] = (
                row["retracted_at"].isoformat()
                if row["retracted_at"] is not None
                else None
            )
        return sliced

    @mcp.tool()
    async def get_memory(
        id: str,
        include_neighbors: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a memory node by id. Returns full body + metadata, or None.

        Read tool. Use when you have an id (e.g. from `list_memories`,
        `record`, or a prior `query_memory` hit) and want the full body
        without re-running a semantic query. Retracted nodes ARE returned
        with `retracted_at` populated — addressable even when not editable.

        Args:
          id: the node id returned by `record` / `list_memories` / etc.
          include_neighbors: if True, also returns 1-hop ANCHORED_TO
            neighbors (same shape as `query_memory` neighbors).

        Returns: {id, node_type, body, summary, project_id, created_at,
        retracted_at, neighbors?} or None if no memory has that id.
        """
        conn = state.kuzu_conn()
        label = _lookup_node_label(conn, id)
        if label is None:
            return None

        fields = _LIST_FIELDS[label]
        ts_field = fields["created_at_field"]
        body_field = fields["body_field"]
        summary_field = fields["summary_field"]
        node_type = "episode" if label == "Episode" else label.lower()

        summary_select = (
            f"n.{summary_field}" if summary_field else f"n.{body_field}"
        )
        result = conn.execute(
            f"MATCH (n:{label} {{id: $id}}) "
            f"OPTIONAL MATCH (n)-[:ANCHORED_TO]->(p:Project) "
            f"RETURN n.{body_field} AS body_value, "
            f"{summary_select} AS stored_summary, n.{ts_field}, "
            f"n.retracted_at, p.id LIMIT 1",
            {"id": id},
        )
        if not result.has_next():
            return None
        body_value, stored_summary, created_at, retracted_at, project_id = (
            result.get_next()
        )

        out: dict[str, Any] = {
            "id": id,
            "node_type": node_type,
            "body": body_value,
            "summary": _effective_summary(label, stored_summary, body_value),
            "project_id": project_id,
            "created_at": created_at.isoformat() if created_at is not None else None,
            "retracted_at": (
                retracted_at.isoformat() if retracted_at is not None else None
            ),
        }
        if include_neighbors:
            out["neighbors"] = _expand_neighbors(conn, label, id, 1)
        return out
