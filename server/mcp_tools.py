from __future__ import annotations

import re
import uuid
import warnings
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
from server.profiling import profile

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
    "solution": "Episode",
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
    "solution": "episode",
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
# because `title` already serves that purpose, so `summary_field` is None and
# `_effective_summary` resolves it from `body_field` (== "title") instead.
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
    "blocks":      "BLOCKS",
}

# `relates_to` kinds with endpoint constraints. Absence from this dict means
# any memory-bearing label is allowed on both sides. BLOCKS is Task→Task only
# because that's how the v1 edge is declared in the schema; widening it would
# need a destructive DROP+CREATE.
MEMORY_EDGE_ENDPOINTS: dict[str, tuple[set[str], set[str]]] = {
    "blocks": ({"Task"}, {"Task"}),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _parse_iso_datetime(param_name: str, value: str | None) -> datetime | None:
    """Parse an optional ISO-8601 string, raising a clear ValueError on
    malformed input rather than letting `fromisoformat` raise inline."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(
            f"{param_name} is not a valid ISO-8601 datetime: {value!r} ({e})"
        )


def _date_window_clause(
    columns: list[str],
    since: datetime | None,
    until: datetime | None,
    params: dict[str, Any],
    prefix: str,
) -> str:
    """Kùzu Cypher fragment matching if ANY of `columns` falls in [since,
    until] (either bound optional) — i.e. "touched" by any of several
    timestamp columns (e.g. created_at OR amended_at), not just one.
    Bound values are added to `params` under `{prefix}_since`/
    `{prefix}_until`. Returns "" (no-op) if both bounds are None.
    """
    if since is None and until is None:
        return ""
    if since is not None:
        params[f"{prefix}_since"] = since
    if until is not None:
        params[f"{prefix}_until"] = until
    per_column = []
    for col in columns:
        bounds = []
        if since is not None:
            bounds.append(f"{col} >= ${prefix}_since")
        if until is not None:
            bounds.append(f"{col} <= ${prefix}_until")
        per_column.append("(" + " AND ".join(bounds) + ")")
    return "(" + " OR ".join(per_column) + ")"


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

    History: Note was originally designed (PLAN.md v1) as a pointer to an
    external markdown file via a `path` field, with `title` just a label.
    That importer was never built — `path` is always NULL — so in practice
    every Note is ad-hoc prose via `record(kind="note")`, capped at a
    title's length. PLAN.md's Note row is annotated SUPERSEDED accordingly.
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
            "CREATE (:Idea {id: $id, body: $b, created_at: $ts, "
            "retracted_at: NULL, amended_at: NULL, summary: $s})",
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


# Anchor-hint kinds that query_memory/list_memories accept as a scope. Each
# value is the graph node label the name resolves against. We keep this small
# on purpose: Project and Concept are the two anchor types keyed purely by
# `name`, which is the only stable handle a caller can type. Repo/Symbol/
# Person have composite keys; if they become useful as scopes, add a separate
# parameter shape rather than overloading `name`.
ANCHOR_HINT_KINDS: dict[str, str] = {
    "project": "Project",
    "concept": "Concept",
}


def _resolve_anchor_hint_ids(
    conn, anchor_hint: dict[str, Any] | None
) -> set[str] | None:
    """Resolve an anchor_hint to the set of memory ids ANCHORED_TO that node.

    Returns:
      None — no hint provided, caller should not filter.
      set() (empty) — hint provided but the anchor node doesn't exist, OR
        the node exists but has no memories anchored. Caller should treat
        as "no matches" and short-circuit.
      {id1, id2, ...} — memory ids to filter by.

    Read-only: does NOT create stub anchor nodes (that's record-time only).
    """
    if anchor_hint is None:
        return None
    kind = anchor_hint.get("kind")
    name = anchor_hint.get("name")
    if not kind or not name:
        raise ValueError(
            "anchor_hint requires 'kind' and 'name', got: "
            f"{anchor_hint!r}"
        )
    target_label = ANCHOR_HINT_KINDS.get(kind)
    if target_label is None:
        raise ValueError(
            f"anchor_hint kind '{kind}' not supported. "
            f"Allowed: {sorted(ANCHOR_HINT_KINDS)}"
        )
    # One pass per memory label: collect ids of memory nodes anchored to the
    # named anchor target. Kuzu rel-table traversal is the source of truth;
    # the embedding row's denormalized project_id is incidental.
    ids: set[str] = set()
    for label in MEMORY_KIND_TO_LABEL.values():
        result = conn.execute(
            f"MATCH (n:{label})-[:ANCHORED_TO]->(t:{target_label} "
            f"{{name: $name}}) RETURN n.id",
            {"name": name},
        )
        while result.has_next():
            ids.add(str(result.get_next()[0]))
    return ids


def _coerce_legacy_project_hint(
    project_hint: str | None, anchor_hint: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Resolve the (project_hint, anchor_hint) pair to a single anchor_hint.

    project_hint is the v3 spelling; anchor_hint is the v4 generalization.
    Passing both is an error — callers must pick one. Passing project_hint
    alone is auto-rewritten to {"kind":"project","name":project_hint} and
    emits a DeprecationWarning so callers migrate before the alias is
    removed.
    """
    if project_hint is not None and anchor_hint is not None:
        raise ValueError(
            "pass either project_hint or anchor_hint, not both. "
            "project_hint is the deprecated alias."
        )
    if project_hint is not None:
        warnings.warn(
            "project_hint is deprecated; pass "
            'anchor_hint={"kind":"project","name":<hint>} instead. '
            "project_hint will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return {"kind": "project", "name": project_hint}
    return anchor_hint


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
    scope_ids: set[str] | None,
    include_retracted: bool,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return raw rows for one memory label. Sort + slice happen in Python.

    scope_ids: if not None, restrict to nodes whose id is in this set. The
    caller (list_memories) resolves anchor_hint once into a single id set
    spanning every memory-bearing label. If None, list everything.

    since/until: if given, keep only rows whose creation timestamp OR
    amended_at falls in [since, until] — "touched" during the window, not
    just created in it. Either bound may be omitted.

    project_id surfaced in the row is informational — it reads any Project
    anchor on the node via OPTIONAL MATCH. Memories anchored to multiple
    Projects still appear once (deduped); the graph is the source of truth
    for the full anchor set.
    """
    fields = _LIST_FIELDS[label]
    ts_field = fields["created_at_field"]
    body_field = fields["body_field"]
    summary_field = fields["summary_field"]
    node_type = "episode" if label == "Episode" else label.lower()

    clauses: list[str] = []
    params: dict[str, Any] = {}
    if not include_retracted:
        clauses.append("n.retracted_at IS NULL")
    if scope_ids is not None:
        if not scope_ids:
            return []
        quoted = ", ".join(f"'{i}'" for i in scope_ids)
        clauses.append(f"n.id IN [{quoted}]")
    window_clause = _date_window_clause(
        [f"n.{ts_field}", "n.amended_at"], since, until, params, "w"
    )
    if window_clause:
        clauses.append(window_clause)

    summary_select = f"n.{summary_field}" if summary_field else f"n.{body_field}"
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    # WHERE must precede OPTIONAL MATCH in Kuzu; placing it after binds to
    # the optional pattern and silently nulls out p.id when WHERE is meant
    # to constrain n only.
    match_clause = (
        f"MATCH (n:{label}){where} "
        f"OPTIONAL MATCH (n)-[:ANCHORED_TO]->(p:Project)"
    )

    query = (
        f"{match_clause} "
        f"RETURN n.id, n.{body_field} AS body_value, "
        f"{summary_select} AS stored_summary, n.{ts_field}, "
        f"n.retracted_at, n.amended_at, p.id AS pid"
    )
    result = conn.execute(query, params)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    while result.has_next():
        node_id, body_value, stored_summary, created_at, retracted_at, amended_at, pid = (
            result.get_next()
        )
        # OPTIONAL MATCH yields duplicates when a node has multiple Project
        # anchors. Dedup keeping first hit.
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
                "amended_at": amended_at,
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
        endpoints = MEMORY_EDGE_ENDPOINTS.get(kind)
        if endpoints is not None:
            allowed_from, allowed_to = endpoints
            if source_label not in allowed_from or target_label not in allowed_to:
                raise ValueError(
                    f"relates_to kind {kind!r} requires "
                    f"{sorted(allowed_from)}→{sorted(allowed_to)}, got "
                    f"{source_label}→{target_label}"
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
        blocker / fact / solution / episode.

        Write tool. One polymorphic entry point — pick `kind` per the taxonomy
        below; the server creates the right graph node and a matching embedding
        row so future `query_memory` calls can find it semantically.

        Memories are SHARED across every AI tool connected to this knowitall
        server (Claude Code, Codex, Cursor, etc.). Write for the next agent
        — possibly running in a different tool — not just for future-you in
        this session.

        Use this when something would be useful in a FUTURE session,
        after the current context window is gone. Good triggers:
          - The user states something durable about themselves or the project
            ("we always do X", "next we want Y", "Z is the blocker").
          - A design decision is made — capture the choice AND the rationale.
          - A bug is tracked down — capture the root cause and the fix.
          - A tricky env/setup/import/config issue was resolved — capture it
            as `kind="solution"`. Future-you (or another AI tool) WILL hit
            this again; the lookup is only useful if the entry is findable.
          - A feature is finished — capture what was built.
          - Session is wrapping up: proactively ASK before storing a `summary`.

        Do NOT use for: transient debug output, things already captured in git
        or code, scratchpad thinking, or speculative ideas the user hasn't
        endorsed.

        kind taxonomy:
          - "decision" | "task" | "idea" | "note": become first-class graph
            nodes (citable, expandable, can be the target of edges).
          - "summary" | "blocker" | "fact" | "solution" | "episode": become
            Episode nodes carrying the kind, for less structural / more
            narrative content.
          - "note" is title-only: <=200 chars, no long body column. A body
            over 200 chars is REJECTED — use "fact"/"idea"/"decision"/"task"
            for anything longer than a title (see PLAN.md for why `note`
            is shaped this way).

        kind="solution" body shape (FOLLOW THIS — semantic retrieval depends
        on it):
          Line 1: the verbatim error string or exact symptom (paste it,
                  don't paraphrase). This is what future-you will search.
          Line 2: the command / context that surfaces it.
          Body:   the fix, and how you verified it worked.
          Last line: `Discoverable keywords: <3-5 paraphrases the future
                  searcher might type>` — embedding models cluster on
                  lexical neighbors, so paraphrases on the page widen
                  the recall surface.
        After recording, run `query_memory` with the symptom phrasing you
        expect future-you to use. If your entry isn't top-1, amend the body
        until it is. If a similar solution already exists, `amend` it
        instead of recording a duplicate — duplicates split the embedding
        signal and bury the right answer.

        body: self-contained prose. Must be readable without surrounding chat.

        project_hint: project NAME (not id). If novel, a Project node is
        created. Omit to leave unattached.

        anchors: typed JSON citations, e.g. {"kind":"commit","sha":...,
        "repo":...}, {"kind":"file","repo":...,"path":...},
        {"kind":"project","name":...}, {"kind":"concept","name":...}
        (less common: "symbol", "person" — shapes in README). Existing
        nodes are reused by natural key. Pass MULTIPLE project/concept
        anchors to scope shared/cross-cutting knowledge (an internal
        library, a framework gotcha) so it surfaces under every anchor's
        `query_memory`/`list_memories` hint — worked examples in README's
        "Scoping memories across projects".

        summary: optional ≤200-char title-shaped string surfaced by
        `list_memories` and `get_memory`. If omitted, those tools fall
        back to the first 200 chars of body. Validation rejects strings
        longer than 200 — fail at write-time rather than silently
        truncate. For `kind="note"` the summary writes to the
        existing `title` column (Note has no parallel summary field) —
        since a Note IS its title, `summary` and `body` are two names for
        the same ≤200-char string on a Note.

        relates_to: optional list of memory→memory edges to write. Each
        entry: {"kind": "supersedes"|"refines"|"contradicts"|"relates_to"|
        "blocks", "id": "<existing memory node id>"}. Target ids are
        validated (must resolve to a memory-bearing node) — bad ids
        raise ValueError before any state changes. "blocks" additionally
        requires both endpoints be Task nodes (the underlying BLOCKS
        edge is Task→Task only).

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
        if label == "Note" and len(body) > SUMMARY_MAX_LEN:
            raise ValueError(
                f"note body is {len(body)} chars but a Note only stores a "
                f"{SUMMARY_MAX_LEN}-char title; the remainder would be "
                "dropped. Use kind=\"fact\" (or \"idea\"/\"decision\"/"
                "\"task\") for content longer than a title, and reserve "
                "kind=\"note\" for short title-shaped labels."
            )

        with profile("record") as p:
            with p.stage("embed"):
                vec = await embed(state.http, body)
            if len(vec) != config.settings.embedding_dim:
                raise RuntimeError(
                    f"embedding dim mismatch: got {len(vec)}, "
                    f"configured {config.settings.embedding_dim}"
                )

            conn = state.kuzu_conn()
            now = _now()
            with p.stage("kuzu"):
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

                # Project anchor: prepend so project shows up in ANCHORED_TO graph.
                anchor_list: list[dict[str, Any]] = []
                if project_hint:
                    anchor_list.append({"kind": "project", "name": project_hint})
                if anchors:
                    anchor_list.extend(anchors)
                anchored = apply_anchors(conn, label, node_id, anchor_list, now)

                related: list[dict[str, str]] = []
                if relates_to:
                    related = _apply_relates_to(
                        conn, label, node_id, relates_to, now
                    )

            with p.stage("lance"):
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

            # Every maint_interval inserts this kicks off compaction/reindex on
            # a background thread and returns immediately, so the optimize cost
            # is NOT in this call's profile line.
            state.note_insert_and_maybe_optimize()

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
        anchors: same shape as `record`'s `anchors` — a list of typed JSON
          citations keyed on "kind" (NOT "type"), e.g.
          [{"kind":"commit","sha":"abc123","repo":"knowitall"}]. See
          `record`'s docstring for the per-kind required fields.
        When status=="done" and a commit anchor is provided, a CLOSED_BY edge
        is also written from Task to Commit. Every call bumps `amended_at`,
        so a date-window query (`list_memories`'s since/until) catches
        status transitions, not just body/summary edits.

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
            f"MATCH (t:Task {{id: $id}}) SET t.status = $s, "
            f"t.amended_at = $now{closed_at_clause}",
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
    async def amend(
        id: str,
        body: str | None = None,
        summary: str | None = None,
        add_anchors: list[dict[str, Any]] | None = None,
        remove_anchors: list[dict[str, Any]] | None = None,
        retract: bool | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """In-place edit of a memory node. Preserves id; re-embeds when body changes.

        Write tool. Use this when an existing memory needs to be revised —
        a renamed concept, a sharpened decision, an updated task body — or
        retracted. Unlike recording a replacement, this keeps the id stable
        so anchors and any inbound memory→memory edges (RELATES_TO_MEMORY,
        etc.) keep pointing at the right thing.

        Args:
          id: the node id returned by `record` / `list_memories`.
          body: if provided, replaces the stored body AND re-embeds. The
            old embedding row is deleted and a fresh one inserted, so
            subsequent `query_memory` calls reflect the new wording. For
            `kind="note"`, body must stay <=200 chars (a Note is
            title-only) — a longer amended body is REJECTED; record a new
            "fact"/"idea"/"decision"/"task" instead of growing a note
            past a title.
          summary: if provided, replaces the stored summary. Pure graph
            SET — no re-embed (summary is not part of the embedded text).
            For `kind=note` the value writes to `title`. Must be <=200
            chars, same limit `record`'s `summary` enforces.
          add_anchors: list of typed anchor JSON (same shapes as
            `record`'s `anchors` arg) to add. Idempotent — existing
            ANCHORED_TO edges are not duplicated.
          remove_anchors: list of {"target_label", "target_id"} pairs
            (the shape returned in `record(...)["anchored"]`) to detach.
            Only the ANCHORED_TO edge is removed; the target node stays.
          retract: True soft-deletes (sets retracted_at; default queries
            hide it; `query_memory(..., include_retracted=True)` surfaces
            it again). False un-retracts. Composable with body/summary/
            anchors in the SAME call — e.g. un-retract and correct at
            once: `amend(id, retract=False, body="corrected text")`. A
            retracted node rejects every OTHER kind of edit (body/summary/
            anchors) unless this same call also passes retract=False.
          reason: short string persisted alongside a retraction (only
            meaningful together with retract=True — passing it otherwise
            raises ValueError rather than silently dropping it).

        Returns: {"id", "node_type", "amended_at", "re_embedded": bool,
        "added": [...], "removed": [...], "retracted_at", "retract_reason"}.
        """
        if summary is not None and len(summary) > SUMMARY_MAX_LEN:
            raise ValueError(
                f"summary too long ({len(summary)} > {SUMMARY_MAX_LEN})."
            )
        if reason is not None and not retract:
            raise ValueError(
                "reason is only meaningful together with retract=True "
                f"(got retract={retract!r})"
            )
        if (
            body is None and summary is None and not add_anchors
            and not remove_anchors and retract is None
        ):
            raise ValueError(
                "amend requires at least one of body / summary / add_anchors "
                "/ remove_anchors / retract"
            )

        conn = state.kuzu_conn()
        label = _lookup_node_label(conn, id)
        if label is None:
            raise ValueError(f"no memory node with id={id}")

        # Retracted nodes reject every edit except un-retracting (retract=
        # False) — addressable via get_memory either way, but not silently
        # editable while retracted.
        retracted_check = conn.execute(
            f"MATCH (n:{label} {{id: $id}}) RETURN n.retracted_at LIMIT 1",
            {"id": id},
        )
        is_retracted = (
            retracted_check.has_next() and retracted_check.get_next()[0] is not None
        )
        if is_retracted and retract is not False:
            raise ValueError(
                f"cannot amend retracted node {id} — pass retract=False in "
                "this same call to un-retract it first, or a separate call."
            )

        now = _now()
        fields = _LIST_FIELDS[label]
        body_field = fields["body_field"]
        summary_field = fields["summary_field"]
        node_type = "episode" if label == "Episode" else label.lower()

        re_embedded = False
        if body is not None:
            if label == "Note" and len(body) > SUMMARY_MAX_LEN:
                raise ValueError(
                    f"amended note body is {len(body)} chars but a Note "
                    f"only stores a {SUMMARY_MAX_LEN}-char title. To keep "
                    "longer content, record a new kind=\"fact\"/\"idea\"/"
                    "\"decision\"/\"task\" instead; don't grow a note past "
                    "a title."
                )
            # Note's body lives in `title`, clipped at SUMMARY_MAX_LEN — a
            # no-op now that the length is validated above.
            stored_body = body[:SUMMARY_MAX_LEN] if label == "Note" else body
            conn.execute(
                f"MATCH (n:{label} {{id: $id}}) SET n.{body_field} = $b",
                {"id": id, "b": stored_body},
            )
            # Delete-then-insert in LanceDB. Updating the fixed-size vector
            # column via `update(values={vector: ...})` is not battle-tested;
            # the safe path is to drop the old row entirely and re-embed.
            # `record()` embeds every kind unconditionally, Note included
            # (see `_insert_embedding_row` call in `record()`), so there is
            # always a row here to replace — re-embed below runs for every
            # label, or the delete above would silently destroy a note's
            # only full-text copy (this used to be skipped for Note, which
            # was the actual bug: it deleted the row and never put one back).
            try:
                state.embeddings.delete(f"id = '{id}'")
            except Exception:
                pass
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
            # Episode preserves its sub-kind (fact/solution/...); every
            # other label's Lance `kind` column mirrors node_type, same as
            # record() does (kind=kind there, which equals node_type for
            # graph-node kinds).
            stored_kind = row[0] if label == "Episode" else node_type
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

        if retract is not None:
            if retract:
                conn.execute(
                    f"MATCH (n:{label} {{id: $id}}) SET n.retracted_at = $now, "
                    f"n.retract_reason = $reason",
                    {"id": id, "now": now, "reason": reason},
                )
                lance_retracted_at: datetime | None = now
            else:
                conn.execute(
                    f"MATCH (n:{label} {{id: $id}}) SET n.retracted_at = NULL, "
                    f"n.retract_reason = NULL",
                    {"id": id},
                )
                lance_retracted_at = None
            # Mirror onto the LanceDB row. Harmless no-op if no row exists
            # (e.g. a kind that doesn't embed, or a prior amend deleted it).
            try:
                state.embeddings.update(
                    where=f"id = '{id}'",
                    values={"retracted_at": lance_retracted_at},
                )
            except Exception:
                pass

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
            "retracted_at": now.isoformat() if retract else None,
            "retract_reason": reason if retract else None,
        }

    @mcp.tool()
    async def query_memory(
        query: str,
        project_hint: str | None = None,
        anchor_hint: dict[str, Any] | None = None,
        k: int = 10,
        expand_hops: int = 0,
        include_retracted: bool = False,
        node_types: list[str] | None = None,
        snippet_chars: int = 240,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over recorded memory; optional 1-hop graph expansion.

        Read tool. Retrieves passages recorded via `record`, ranks by
        embedding distance. By default returns truncated text and no
        neighbors — both are opt-in to keep the MCP output budget small.

        Memory is shared across every AI tool connected to this knowitall
        server — call PROACTIVELY, not just when asked: session start when
        the user references prior work; before a second attempt at a
        failing env/setup/config fix (paste the LITERAL error string —
        exact tokens beat paraphrases); before recommending a non-obvious
        workaround; for a library/framework quirk, try
        anchor_hint={"kind":"concept","name":<library>} first.

        Hygiene: if a hit names a specific file, function, or flag, VERIFY
        it still exists before recommending — memory can outlive the code
        it references.

        Args:
          query: free-text. Embedded; ranked by vector distance. Noun
            phrases beat questions ("auth service location" > "where is
            the auth service").
          anchor_hint: scope to memory anchored to a graph node, e.g.
            {"kind":"project","name":"knowitall"} (internal projects) or
            {"kind":"concept","name":"kuzu"} (libraries/frameworks/cross-
            cutting topics). A memory anchored to several projects/
            concepts surfaces under each (see `record`'s SCOPING doc). []
            if the named node doesn't exist.
          project_hint: DEPRECATED alias for
            anchor_hint={"kind":"project","name":<hint>}. Passing both
            raises ValueError.
          k: top-k hits to return (default 10).
          expand_hops: 0 (default) skips neighbor expansion; 1 includes
            outbound ANCHORED_TO targets. >1 rejected — use `cypher` for
            deeper traversal.
          include_retracted: default False; pass True to include soft-
            deleted entries.
          node_types: filter by node_type, e.g. ["decision","task"].
            Allowed: "decision","task","idea","note","episode". Does NOT
            reach the Episode sub-kind (summary/blocker/fact/solution —
            all share node_type="episode"); for "just solutions" drop to
            `cypher`: `MATCH (e:Episode {kind:'solution'}) RETURN e.id,
            e.body` (structural, no ranking/score).
          snippet_chars: 240 (default) truncates text with a trailing
            ellipsis. 0 for full bodies (or use `get_memory(id)` directly).
          since / until: optional ISO-8601 bounds (either may be omitted)
            on the embedding row's `created_at` — which is the LAST
            (re-)embed time, not necessarily the original record time,
            since `amend(body=...)` re-embeds in place. For a pure
            enumeration window ("what did I work on this week") prefer
            `list_memories(since=..., until=...)`, which checks the
            graph's real created_at OR amended_at; this is the semantic-
            search equivalent when you also need a query.

        Returns: [{"hit": {id,text,kind,node_type,project_id,score,created_at,
        retracted_at}, "neighbors": [{label, ...identifying fields}]}].
        Lower score == closer match. When expand_hops=0, "neighbors" is [].
        """
        if expand_hops not in (0, 1):
            raise ValueError(
                f"expand_hops must be 0 or 1, got {expand_hops}. "
                "Multi-hop traversal is not implemented; use cypher for it."
            )
        since_dt = _parse_iso_datetime("since", since)
        until_dt = _parse_iso_datetime("until", until)
        if since_dt is not None and until_dt is not None and since_dt > until_dt:
            raise ValueError(f"since ({since}) is after until ({until})")
        with profile("query_memory") as p:
            with p.stage("embed"):
                vec = await embed(state.http, query)
            return _query_memory_search(
                state, p, vec, k, expand_hops, snippet_chars,
                project_hint, anchor_hint, node_types, include_retracted,
                since_dt, until_dt,
            )

    def _query_memory_search(
        state: AppState,
        p: Any,
        vec: list[float],
        k: int,
        expand_hops: int,
        snippet_chars: int,
        project_hint: str | None,
        anchor_hint: dict[str, Any] | None,
        node_types: list[str] | None,
        include_retracted: bool,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        conn = state.kuzu_conn()

        effective_hint = _coerce_legacy_project_hint(project_hint, anchor_hint)

        clauses: list[str] = []
        if effective_hint is not None:
            scoped_ids = _resolve_anchor_hint_ids(conn, effective_hint)
            if not scoped_ids:
                # Unknown anchor target, or anchor exists with zero memories.
                return []
            quoted = ", ".join(f"'{i}'" for i in scoped_ids)
            clauses.append(f"id IN ({quoted})")
        if not include_retracted:
            clauses.append("retracted_at IS NULL")
        if node_types:
            quoted = ", ".join(f"'{nt}'" for nt in node_types)
            clauses.append(f"node_type IN ({quoted})")
        if since is not None:
            clauses.append(f"created_at >= timestamp '{since.isoformat()}'")
        if until is not None:
            clauses.append(f"created_at <= timestamp '{until.isoformat()}'")

        with p.stage("lance"):
            search = state.embeddings.search(vec).limit(k)
            if clauses:
                search = search.where(" AND ".join(clauses), prefilter=True)
            table = search.to_arrow()
        p.count(rows=table.num_rows)

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
            if expand_hops > 0:
                with p.stage("kuzu"):
                    neighbors = _expand_neighbors(
                        conn, label, node_id, expand_hops
                    )
            else:
                neighbors = []
            rows.append({"hit": hit, "neighbors": neighbors})
        return rows

    @mcp.tool()
    async def cypher(
        query: str, params: dict[str, Any] | None = None
    ) -> list[list[Any]]:
        """Read-only Kùzu Cypher passthrough — graph queries against the v3 schema.

        Read tool (structural). Use for who/what/when over the graph (anchor
        links, supersedes/refines/blocks chains, closed-by-commit). Use
        `query_memory` for free-text semantic recall.

        Nodes (memory-bearing): Episode, Decision, Task, Idea, Note.
          All carry: id, retracted_at, retract_reason, amended_at.
          Decision/Task/Idea/Episode also have a `summary` column; Note
          uses `title`.
        Nodes (anchor targets): Project, Commit, File, Symbol, Concept,
          Person, Repo.

        Edges currently WRITTEN by the MCP tools (these have data):
          - ANCHORED_TO: memory → {Project|Commit|File|Symbol|Concept|Person}.
            Generic citation. Written by `record(anchors=)`,
            `amend(add_anchors=)`, `update_todo(anchors=)`.
          - SUPERSEDES_MEMORY, REFINES, CONTRADICTS, RELATES_TO_MEMORY:
            memory → memory. Written by `record(relates_to=...)`.
          - BLOCKS: Task → Task. Written by `record(relates_to=[{
            "kind":"blocks", "id":...}])` when both endpoints are Tasks.
          - CLOSED_BY: Task → Commit. Written by `update_todo(status=
            "done", anchors=[{"kind":"commit", "sha":...}])`.

        Edges declared in the schema but NOT YET WRITTEN by any tool
        (querying them returns empty until ingestion lands): PART_OF,
        IN_REPO, AUTHORED, MODIFIED, DEFINED_IN, CALLS, IMPORTS,
        DEPENDS_ON, GRADUATED_TO, ALIAS_OF, RELATES_TO (Concept→Concept),
        DECIDED_IN, DROPPED, TOUCHED_BY, BELONGS_TO, MENTIONED_IN, and
        the v1 SUPERSEDES (Decision→Decision; superseded by
        SUPERSEDES_MEMORY).

        Mutating keywords (CREATE/MERGE/DELETE/SET/DROP/REMOVE/DETACH/COPY/
        ALTER/INSTALL/LOAD/ATTACH/CALL) are rejected. Writes go through
        `record`, `amend` (also handles retract/un-retract), or
        `update_todo`.

        Example: `MATCH (d:Decision)-[:ANCHORED_TO]->(f:File)
                  RETURN d.body, f.path`.
        """
        if MUTATING_PATTERN.search(query):
            raise ValueError(
                "Mutating Cypher rejected. Allowed: read-only "
                "MATCH/RETURN/WHERE/WITH/ORDER/LIMIT."
            )
        with profile("cypher") as p:
            conn = state.kuzu_conn()
            with p.stage("kuzu"):
                result = conn.execute(query, params or {})
                rows: list[list[Any]] = []
                while result.has_next():
                    rows.append([_coerce(v) for v in result.get_next()])
            p.count(rows=len(rows))
            return rows

    @mcp.tool()
    async def list_memories(
        kind: str | None = None,
        project_hint: str | None = None,
        anchor_hint: dict[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at_desc",
        include_retracted: bool = False,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate memory nodes without semantic ranking. Returns summaries.

        Read tool (enumeration). The third navigation primitive alongside
        `query_memory` (semantic) and `cypher` (structural). Use when you
        want to see *what's in here* without a query in mind — e.g. "what
        ideas do I have on project X?", "what tasks are open?", "what did
        I work on this week?" (since="2026-07-29").

        Bodies are NOT returned. Each row carries a `summary` (first ~200
        chars of body, or for Note the existing title). Use `get_memory`
        with a returned id to pull a full body when needed.

        Args:
          kind: one of "decision","task","idea","note","episode" (or None
            for all). Episode-flavored kinds (summary/blocker/fact) all
            live under "episode" here — use `query_memory` with node_types
            for sub-kind discrimination.
          anchor_hint: scope to memory anchored to a specific graph node.
            Shapes:
              {"kind":"project", "name":"knowitall"}  — internal projects
              {"kind":"concept", "name":"kuzu"}       — public libraries,
                frameworks, or cross-cutting topics
            Resolves via the ANCHORED_TO graph, so a memory anchored to
            multiple projects/concepts surfaces under each. If the named
            node doesn't exist, returns [].
          project_hint: DEPRECATED alias for
            anchor_hint={"kind":"project","name":<hint>}. Passing both
            raises ValueError.
          limit: max rows returned (default 50). Applied after sort.
          offset: rows to skip (default 0). For paging.
          order_by: "created_at_desc" (default) | "created_at_asc".
          include_retracted: default False; pass True to include
            soft-deleted entries.
          since / until: optional ISO-8601 bounds (either may be omitted).
            Keeps a memory if its creation time OR `amended_at` falls in
            [since, until] — i.e. "touched" during the window, not just
            created in it, so an old Decision you revised this week still
            shows up. `update_todo` also bumps `amended_at`, so a status
            transition counts as "touched" too.

        Returns: [{id, node_type, summary, project_id, created_at,
        amended_at, retracted_at}]. Timestamps are ISO strings or None.
        """
        if kind is not None and kind not in MEMORY_KIND_TO_LABEL:
            raise ValueError(
                f"unknown kind '{kind}'. Allowed: {sorted(MEMORY_KIND_TO_LABEL)} or None"
            )
        if order_by not in ("created_at_desc", "created_at_asc"):
            raise ValueError(
                f"unknown order_by '{order_by}'. Allowed: created_at_desc | created_at_asc"
            )
        since_dt = _parse_iso_datetime("since", since)
        until_dt = _parse_iso_datetime("until", until)
        if since_dt is not None and until_dt is not None and since_dt > until_dt:
            raise ValueError(f"since ({since}) is after until ({until})")

        conn = state.kuzu_conn()

        effective_hint = _coerce_legacy_project_hint(project_hint, anchor_hint)
        if effective_hint is not None:
            scope_ids: set[str] | None = _resolve_anchor_hint_ids(
                conn, effective_hint
            )
            # Empty set = anchor node missing OR has zero memories. Both → [].
            if not scope_ids:
                return []
        else:
            scope_ids = None

        labels = (
            [MEMORY_KIND_TO_LABEL[kind]] if kind is not None else
            [MEMORY_KIND_TO_LABEL[k] for k in MEMORY_KINDS]
        )

        with profile("list_memories") as p:
            all_rows: list[dict[str, Any]] = []
            with p.stage("kuzu"):
                for label in labels:
                    all_rows.extend(
                        _list_one_label(
                            conn,
                            label,
                            scope_ids=scope_ids,
                            include_retracted=include_retracted,
                            since=since_dt,
                            until=until_dt,
                        )
                    )

            # Sort, then page. NULL created_at sorts to the end either way.
            with p.stage("sort"):
                reverse = order_by == "created_at_desc"
                all_rows.sort(
                    key=lambda r: (
                        r["created_at"] is None,
                        r["created_at"] or datetime.min,
                    ),
                    reverse=reverse,
                )
            # scanned = total rows pulled before paging; the gap between this
            # and `limit` is the wasted work (no LIMIT pushed into Cypher).
            p.count(scanned=len(all_rows), labels=len(labels))
        sliced = all_rows[offset : offset + limit]

        for row in sliced:
            row["created_at"] = (
                row["created_at"].isoformat() if row["created_at"] is not None else None
            )
            row["amended_at"] = (
                row["amended_at"].isoformat() if row["amended_at"] is not None else None
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
        if label == "Note" and body_value is not None and len(body_value) >= SUMMARY_MAX_LEN:
            # Legacy over-length note (recorded/amended before the
            # reject-loud validation existed): the graph title is clipped
            # at SUMMARY_MAX_LEN, but `record()` always wrote the full body
            # into the Lance embeddings row's `text` column too. Recover it
            # if that row is still there — best-effort, degrades to the
            # clipped title if not. Filters on `text` (model-independent),
            # so a stale `model_version` or a retracted row doesn't matter.
            try:
                lance_rows = (
                    state.embeddings.search()
                    .where(f"id = '{id}'")
                    .select(["text"])
                    .limit(1)
                    .to_arrow()
                )
            except Exception:
                lance_rows = None
            if lance_rows is not None and lance_rows.num_rows:
                full_text = lance_rows.column("text")[0].as_py()
                if full_text and len(full_text) > len(body_value):
                    out["body"] = full_text
        if include_neighbors:
            out["neighbors"] = _expand_neighbors(conn, label, id, 1)
        return out
