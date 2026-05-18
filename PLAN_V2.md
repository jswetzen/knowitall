# knowitall v2 — MCP surface refactor

## Context

Through ideation on 2026-05-15 → 2026-05-16 we landed on a redesign:

- **Drop bulk git ingestion.** The current `ingest_repo` MCP tool passes a path that must live inside the server container, breaking the cross-machine deployment that's the actual target. Replace with lazy-creation: graph nodes for commits/files/people are created on demand, by reference, when other memory objects anchor to them.
- **One polymorphic write tool.** `record(kind, body, project_hint, anchors)` covers decision/task/idea/note + episode-flavored kinds. Smaller surface for smaller models; taxonomy grows without API churn.
- **Embed everything.** A single LanceDB `embeddings` table with `node_type`, so one similarity search ranks across Episodes, Decisions, Tasks, Ideas, Notes, Concepts.
- **Graph-Enhanced Vector as default retrieval.** `query_memory` returns hits + 1-hop neighborhood, not flat texts.
- **Soft undo.** `forget(id, reason)` sets `retracted_at`; default queries exclude.
- **Anchors as a typed JSON argument**, not inline markup — Claude writes anchors structurally. Natural-prose extraction is a deferred enhancement.
- **MCP prompts** ship alongside tools: `status`, `capture`, `provenance`, `reflect`.

User has no stored data, so v2 is a hard cut — no data migration.

## In scope

Schema v2 DDL · LanceDB union `embeddings` table · `record` / `update_todo` / `forget` / extended `query_memory` · removal of `ingest_repo` from MCP surface · 4 MCP prompts · test rewrite for the new surface · README + STATUS refresh.

## Out of scope (deferred)

SessionStart hook · curator skill (separate artifact) · BM25+RRF · tree-sitter · natural-prose anchor extraction · re-embed-on-model-swap tooling.

---

## Schema changes

### Kuzu — new `schema/v2.cypher`

Additive on top of v0/v1 (kept intact, applied incrementally by `schema/migrate.py:apply_migrations`).

New node:
- `Episode(id STRING PK, body STRING, kind STRING, project_id STRING, created_at TIMESTAMP, model_version STRING, retracted_at TIMESTAMP)`

New rel: `ANCHORED_TO` — bi-temporal generic citation edge from Episode/Decision/Task/Idea/Note → Commit/File/Symbol/Project/Concept/Person. Confirm Kuzu's multi-pair `FROM x|y|z TO a|b|c` syntax against 0.9+ docs at impl time; if unsupported, expand to per-target rel tables (`ANCHORED_TO_FILE`, `ANCHORED_TO_COMMIT`, …).

Add `retracted_at TIMESTAMP` to existing `Decision`, `Task`, `Idea`, `Note` via `ALTER NODE TABLE ... ADD`. Confirm Kuzu supports this; if not, fall back to a separate `Retraction` node + `RETRACTS` edge.

Project attachment uses an **edge**, not a denormalized `project_id` column on the graph nodes. The graph is the point — bi-temporal `BELONGS_TO` lets us handle project renames / supersession / merges cleanly. `BELONGS_TO` is already defined in `v1.cypher` as `FROM File TO Project`; generalize to multi-pair `FROM Episode|Decision|Task|Idea|Note|File TO Project` if Kuzu 0.9+ supports multi-pair FROM. Otherwise add per-type rels (`EPISODE_OF`, `DECISION_OF`, …) or one rel per source kind. The LanceDB row separately keeps `project_id` because LanceDB is a columnar store and that's where filter pushdown happens — that's not denormalization in the graph, it's metadata on the embedding row.

### LanceDB — replace `episodes` with `embeddings`

In `server/deps.py:build_state` (currently lines ~16–28 define the `episodes` schema), create one union table:

```
embeddings:
  id STRING (PK, == Kuzu node id)
  node_type STRING   -- episode | decision | task | idea | note | concept
  text STRING        -- canonical embedded text
  vector fixed_size_list<float32, embedding_dim>
  project_id STRING (nullable)
  kind STRING (nullable)   -- preserves soft sub-enum for episode-flavored rows
  created_at TIMESTAMP (UTC)
  model_version STRING
  retracted_at TIMESTAMP (nullable)
```

If a legacy `episodes` table exists in the data dir, drop it on startup. (User has no data; safe.)

What gets embedded:
- `episode|summary|blocker|fact|note|decision|task|idea`: the `body`.
- `concept`: the `name`.
- `commit|file|symbol|person`: **not embedded** (low signal, churn).

---

## Tool surface (final)

| Tool | Status | Shape |
|---|---|---|
| `record(kind, body, project_hint=None, anchors=[])` | replaces `store_episode` | returns `{id, node_type, project_id}` |
| `update_todo(id, status, anchors=[])` | new | for `kind=task` only; transitions status, optionally adds CLOSED_BY/BLOCKS edges via anchors |
| `forget(id, reason)` | new | soft undo; sets `retracted_at` on Kuzu node + mirror row |
| `query_memory(query, project_hint=None, k=10, expand_hops=1, include_retracted=False, node_types=None)` | extended | returns `[{hit, neighbors:[…]}]` |
| `cypher(query, params=None)` | unchanged | read-only |
| ~~`ingest_repo`~~ | **removed** | — |

### `kind` enum (hard, validated server-side)

- Graph-node kinds — get stable IDs, citable, expandable: `decision`, `task`, `idea`, `note`
- Episode-flavored — become `Episode` nodes carrying `kind=`: `summary`, `blocker`, `fact`, `episode`

### Anchor protocol

`anchors` is a list of typed JSON objects. Examples:

```python
{"kind": "commit", "sha": "abc1234", "repo": "knowitall",
 "message": "...", "authored_at": "2026-05-15T...", "author_email": "..."}
{"kind": "file",    "repo": "knowitall", "path": "server/app.py"}
{"kind": "symbol",  "repo": "knowitall", "file": "server/app.py", "name": "create_app", "line": 14}
{"kind": "project", "name": "knowitall"}
{"kind": "concept", "name": "rate limiting"}
{"kind": "person",  "email": "claude@swetzen.com"}
```

Server resolves each anchor: existing node by natural key (sha / email / (repo,path) / name) → reuse; else create stub from whatever fields the client provided. Creates one `ANCHORED_TO` edge per anchor. Enrichment (e.g. `git show` to flesh out a commit sha) is the client's job via Bash.

---

## MCP prompts (`server/mcp_prompts.py`, new)

Registered via `mcp.prompt()`. Surfaced to Claude Code as `/knowitall:*`.

- `status(project)` — Claude runs cypher + query_memory templates, returns a markdown digest (recent decisions, open tasks, recent commits via local git, recent episodes).
- `capture(scope="session")` — Claude proposes a batch of records based on the conversation so far; user confirms in bulk. Embodies the "batched approval, not constant asking" stance.
- `provenance(anchor)` — finds all records anchored to a given file/commit/concept; expands 2-hop.
- `reflect(project, last_n=20)` — pulls last N episodes; Claude proposes a `summary` record for user endorsement.

---

## Critical files

### Modified
- `server/mcp_tools.py` — rewrite registrations: `record`, `update_todo`, `forget`, extended `query_memory`, `cypher`. Remove `ingest_repo`.
- `server/deps.py` — replace `episodes` table init with `embeddings` union table; preserve `embed()` and `AppState` shape (additive only).
- `server/app.py` — call `register_prompts(mcp, state)` after `register_tools`. Reconsider `stateless_http=True` (line 16) per STATUS.md — likely flip to `False` and re-test Claude Code handshake. Also fix the trailing-slash quirk: current `app.mount("/mcp", mcp.streamable_http_app())` + `streamable_http_path="/"` forces `/mcp/` as the canonical endpoint; rewire so `/mcp` (no slash) works directly — try `streamable_http_path=""` first; if FastMCP rejects empty path, replace the `.mount` with an ASGI forwarder or explicit FastAPI route. Once fixed, README and STATUS update to drop the trailing slash from examples.
- `client/cli.py` — replace `store` subcommand with `record` (+ anchor JSON); add `forget`; update `query` output to surface neighbors. `health` and `cypher` unchanged.
- `tests/test_mcp_tools.py` — rewrite for the new surface. Reuse the `_Recorder` + `tools` fixture pattern (line 24).
- `README.md` — update tool table; remove `ingest_repo`; note `/knowitall:*` prompts.
- `STATUS.md` — refresh with v2 completion notes after merge.

### New
- `schema/v2.cypher` — additive DDL (Episode node, retracted_at columns, ANCHORED_TO edge).
- `server/anchors.py` — anchor resolution + lazy stub creation + ANCHORED_TO edge writes.
- `server/mcp_prompts.py` — prompt registrations.

### Reused (no edits)
- `ingest/git_extractor.py:_resolve_repo` (line 55), `_upsert_person` (line 135), `_upsert_file` (line 158), `_create_commit_if_missing` (line 185) — import from `server/anchors.py` for commit/file/person stub creation.
- `server/mcp_tools.py:_resolve_project` (line 21) — already exists, lift into `server/anchors.py` or leave and import.
- `schema/migrate.py:apply_migrations` (line 33) — picks up `v2.cypher` automatically.
- `tests/conftest.py` fixtures (`isolated_data_dir`, `requires_ollama`) — unchanged.

### Kept but no longer MCP-surfaced
- `ingest/git_extractor.py:ingest_repo` (line 265) — still callable internally; helpers are reused for anchor stubs. `tests/test_git_extractor.py` stays green.

---

## Verification

### End-to-end happy path (fresh data dir)
1. `uv run uvicorn --factory server.app:create_app --host 127.0.0.1 --port 8765`.
2. `curl /healthz` returns `model_version`, embeddings row count 0.
3. `python -m client.cli record --kind decision --body "Kuzu over Neo4j: embedded, no JVM" --project knowitall --anchor '{"kind":"file","repo":"knowitall","path":"PLAN.md"}'` → returns id + node_type=decision.
4. `python -m client.cli query "graph database choice" --project knowitall` → returns the decision **plus** the anchored File node as a neighbor.
5. `python -m client.cli forget <id> --reason "duplicate"` → 200 OK.
6. Re-query → result excludes retracted; `--include-retracted` flag includes it.
7. `python -m client.cli cypher 'MATCH (d:Decision)-[:ANCHORED_TO]->(f:File) RETURN d.body, f.path'` → returns the link.

### Test suite
- `uv run pytest tests/test_mcp_tools.py` — covers `record` (each kind), `update_todo`, `forget` (retracted-at filtering), `query_memory` (with/without retracted, with neighbors, node-type filter), `cypher` mutation-block enforcement, anchor stub creation idempotency.
- `uv run pytest tests/test_git_extractor.py` — still green (extractor unchanged).

### Claude Code smoke
- Register `http://<server>:8765/mcp/` per README. Try `/knowitall:status knowitall` — expect markdown back. Try `record` from a session with anchors round-tripping.
- If Claude Code still fails to connect: flip `stateless_http=False` per STATUS.md hypothesis, rebuild, retry.

---

## Risks / decisions to make at impl time

1. **Kuzu multi-pair rel table** (`FROM x|y|z TO …`) — confirm in 0.9+ docs. Fall back to per-target rels if needed.
2. **Kuzu `ALTER NODE TABLE ADD COLUMN`** — confirm support for adding `retracted_at` to existing node tables. Fall back to a `Retraction` node + `RETRACTS` edge.
3. **`stateless_http` + endpoint URL** — flip `stateless_http=False`, fix the wiring so `/mcp` (no trailing slash) is the canonical endpoint, test against Claude Code, document the working setting in README. Both are likely contributors to STATUS.md's connection failure.
4. **Anchor stub completeness** — commit anchors without enrichment fields are accepted (sparse stub). Decide: should the server reject if message/authored_at missing, or accept and let the client backfill later? Default: accept sparse; idempotent updates fill in.
