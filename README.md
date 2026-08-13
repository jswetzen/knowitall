# knowitall

Persistent memory for Claude Code. Self-hosted, single-user, hobby-grade.

## Why

Claude Code forgets everything between sessions. Re-explaining architecture,
past decisions, blockers, and what touches what is the bottleneck for any
long-running project. knowitall is the layer that remembers, so you can ask
"where is project X at?" or "I just realized Y would need this" and have
Claude already know what X is, which repo it lives in, what was last
decided, and what else it touches.

Multiple AI tools (Claude Code, Codex, Cursor, …) connected to the same
knowitall instance share decisions, tasks, and solutions — so a fix
discovered in one tool isn't lost when you switch to another. Per-tool
auto-memory (e.g. Claude Code's `MEMORY.md`) still handles tool-specific
working preferences; knowitall is the cross-tool engineering layer.

Not a chatbot personalization layer (mem0). Not an agent framework
(Letta/MemGPT). Not pure vector RAG. Engineering memory for IDE
assistants — graph + vector, with code, decisions, and solutions as
first-class nodes.

## What

A small HTTP MCP server you run on your own box. Claude Code connects to it
over your LAN/VPN with a bearer token.

- **Graph:** Kùzu (embedded, file-based, Cypher). Closed schema spanning
  ~13 node types and ~24 bi-temporal edges, including a generic
  `ANCHORED_TO` citation edge. See `PLAN.md` §3 and `PLAN_V2.md` for
  design history — where they disagree with `schema/*.cypher` or the
  `cypher` tool's docstring, the schema files and docstring win.
- **Vectors:** LanceDB (embedded, Arrow-native). One union `embeddings`
  table — one similarity search ranks across Episodes, Decisions, Tasks,
  Ideas, Notes, Concepts.
- **Embeddings:** your Ollama instance (`nomic-embed-text-v2-moe`, 768-d).
  No LLM inference on the server — Claude itself reasons over retrieved
  passages.
- **Auth:** static bearer token over HTTP.

### MCP tools

| Tool | Purpose |
|---|---|
| `record(kind, body, project_hint, anchors, summary, relates_to)` | Save a durable memory. `kind` ∈ {decision, task, idea, note, summary, blocker, fact, solution, episode}. **`note` is a short title-only label (≤200 chars); a longer body is rejected — use `fact`/`idea`/`decision`/`task` instead.** `solution` is for env/setup/config gotchas — lead the body with the verbatim error string. `anchors` are typed JSON citations (commit/file/symbol/project/concept/person) that create graph edges. Optional `summary` is a ≤200-char title-shaped string (falls back to first 200 of body). Optional `relates_to` writes memory→memory edges (kinds: supersedes/refines/contradicts/relates_to/blocks). |
| `query_memory(query, project_hint, k, expand_hops=0, snippet_chars=240, include_retracted, node_types, since, until)` | Semantic search. Defaults: bodies clipped to 240 chars + ellipsis, no neighbor expansion. Pass `expand_hops=1` for ANCHORED_TO neighbors; pass `snippet_chars=0` for full bodies. `node_types` doesn't reach Episode sub-kinds (summary/blocker/fact/solution all share `node_type="episode"`) — use `cypher` to filter by `kind` if you need "just solutions". Optional `since`/`until` (ISO-8601) bound the embedding row's last-(re-)embed time. |
| `list_memories(kind, project_hint, limit, offset, order_by, include_retracted, since, until)` | Enumerate memories without semantic ranking. Returns summaries only; use `get_memory` to fetch full bodies. Optional `since`/`until` (ISO-8601, either omittable) keep a memory if its created-at OR `amended_at` falls in the window — "what did I work on this week?" is `list_memories(since="2026-07-29")`. |
| `get_memory(id, include_neighbors)` | Fetch a memory by id. Returns full body + summary + metadata, including `amended_at`/`amend_reason` and `retracted_at`/`retract_reason`. Surfaces retracted nodes rather than hiding them. `id` accepts an unambiguous prefix of ≥8 chars — the short form that appears in chat — and always returns the full id. |
| `amend(id, body, summary, add_anchors, remove_anchors, retract, reason)` | In-place edit preserving id. Body changes trigger re-embed; summary updates skip it. `retract=True` soft-deletes (sets `retracted_at` + persists `reason` as `retract_reason`); `retract=False` un-retracts — composable with body/summary/anchors in the same call. A retracted node rejects every other edit unless the same call also un-retracts it. On any non-retracting amend, `reason` persists as `amend_reason` next to `amended_at` — worth setting on a correcting amend, since the body diff can't say why. Rejections are batched: one error lists every problem in the payload. |
| `update_todo(id, status, anchors)` | Transition a Task's status (bumps `amended_at` too). Done + commit anchor also writes `CLOSED_BY`. `anchors` uses the same `{"kind": ...}` shape as `record` — note `kind`, not `type`. Accepts an id prefix like `get_memory`. |
| `cypher(query, params)` | Read-only Cypher passthrough over the graph. |

#### Scoping memories across projects

`record`'s `anchors` accepts multiple `{"kind":"project",...}`/`{"kind":"concept",...}` entries — there's no "primary" project; the graph is the source of truth and the memory surfaces under every anchor's `query_memory`/`list_memories` hint. Two idioms:

- **Internal library used across repos** (e.g. an in-house `mycelium` package consumed by `aa-SDK` and `powerfactors-api`):
  `anchors=[{"kind":"project","name":"mycelium"}, {"kind":"project","name":"aa-SDK"}, {"kind":"project","name":"powerfactors-api"}]`.
  Future `query_memory(anchor_hint={"kind":"project","name":X})` finds it under any of those names.
- **Public library / framework knowledge** (a Kùzu pitfall, a Pydantic recipe, a fix you don't want to re-derive): tag with a concept anchor named after the library/topic, plus optionally the consumer project(s):
  `anchors=[{"kind":"concept","name":"kuzu"}, {"kind":"project","name":"knowitall"}]`.
  Future `query_memory(anchor_hint={"kind":"concept","name":"kuzu"})` finds it regardless of which repo you're in next.

#### Calendar (Cango) shims

Thin shims over the sibling `cango-daemon` (Unix socket at `KNOWITALL_CANGO_SOCKET`, default `/run/cango/cango.sock`). No calendar logic lives in knowitall; these marshal JSON-RPC and pass the daemon's answer through. If the daemon is down or the socket is missing, each returns `{"error": "cango_unavailable", "reason": ...}` — memory tools are unaffected.

| Tool | Purpose |
|---|---|
| `check_availability(start, end, people)` | "Can we go?" → `free` / `soft_conflict` / `hard_conflict` verdict + conflicts for a window across family calendars. |
| `find_free_slot(duration_minutes, between_start, between_end, people, working_hours)` | Candidate free windows of a given length within a range. |
| `list_events(start, end, people, extended, exclude_roles, limit, offset)` | Events in a window with their resolved roles; compact by default, `extended` adds Exchange ids + the full `resolved_by` trace. |
| `explain_event(event_id)` | Layer-by-layer trace of how one event's role was resolved. |
| `list_series(source_id)` | Recent recurring series on a source — input for adding attendance/fan-out rules. |
| `create_event(source_id, title, start, end, all_day, occupants)` | Write an event to a `writable` CalDAV source. `occupants` adds per-event attendees (ATTENDEE for people with an email; others returned as `unwritten_occupants`). |
| `list_rules(include_retracted)` | List the agent-managed tiebreaker rules — the mutable `state.db` layer that replaced static `rules.yaml`. |
| `record_rule(match, role, reason, effect, occupants)` | Add a tiebreaker rule. `effect`: `self` (default) / `mask` (out-of-office) / `fanout` (add `occupants` → household fan-out). |
| `amend_rule(id, match, role, reason, effect, occupants)` | Edit a rule in place, keeping its id stable. |
| `forget_rule(id, reason)` | Retract a rule (soft delete; kept as a tombstone for audit/undo). |

**Fan-out worked example** — "the whole family may attend Saras läger; flag it, don't hard-block the kids":
`record_rule(match={"series_id":"lager-2026"}, role="soft", effect="fanout", occupants=["family"])`.

### MCP prompts

Surfaced to Claude Code as `/knowitall:*`:

| Prompt | Purpose |
|---|---|
| `/knowitall:status <project>` | Markdown digest: recent decisions, open tasks, recent episodes, recent commits. |
| `/knowitall:capture` | Propose a batch of `record` calls for end-of-session approval. |
| `/knowitall:provenance <anchor>` | Find everything anchored to a file / commit / concept; expand 2 hops. |
| `/knowitall:reflect <project> [last_n]` | Draft a session-summary record from recent episodes. |

### Status

v2 MCP surface refactor landed. See `STATUS.md` for current open items.

## How — run it

### Locally (no container)

```bash
cp .env.example .env                 # set KNOWITALL_TOKEN
uv sync
uv run uvicorn --factory server.app:create_app --host 127.0.0.1 --port 8765
```

### Via podman compose

```bash
cp .env.example deploy/.env          # set KNOWITALL_TOKEN
cd deploy
podman compose --env-file .env up -d
curl http://127.0.0.1:8765/healthz
```

Data persists to `./data/` (bind-mounted into the container at `/data`).

### Register with Claude Code

```bash
claude mcp add --transport http knowitall \
    http://<server-ip>:8765/mcp \
    --header "Authorization: Bearer <KNOWITALL_TOKEN>"
```

`/mcp` is canonical — no trailing slash, no 307 redirect. Equivalent JSON
if you prefer editing `~/.claude/settings.json` directly:

```json
{
  "mcpServers": {
    "knowitall": {
      "type": "http",
      "url": "http://<server-ip>:8765/mcp",
      "headers": { "Authorization": "Bearer <KNOWITALL_TOKEN>" }
    }
  }
}
```

### Smoke test from the command line

```bash
KNOWITALL_TOKEN=<token> uv run python -m client.cli record \
    --kind note --body "the new auth lives in cmd/auth-svc" \
    --project knowitall \
    --anchor '{"kind":"file","repo":"knowitall","path":"cmd/auth-svc/main.go"}'

KNOWITALL_TOKEN=<token> uv run python -m client.cli query \
    "auth service location" --project knowitall
```

### Tests

```bash
uv run pytest        # unit (mocked Ollama) + e2e (skipped if Ollama unreachable)
```

## Layout

```
server/         FastAPI + FastMCP app, MCP tools + prompts, bearer middleware
schema/         Kùzu DDL (v0, v1, v2) + idempotent migration runner
ingest/         Structural extractors — git today, tree-sitter later.
                git_extractor is now an internal helper (not an MCP tool);
                its commit/file/person upserts back the lazy anchor stubs.
client/cli.py   Tiny CLI MCP client for smoke testing
server/anchors.py     Anchor resolution + lazy stub creation + ANCHORED_TO writes
server/cango.py       Calendar (Cango) shims — JSON-RPC over the daemon's Unix socket
server/mcp_prompts.py /knowitall:* prompts (status, capture, provenance, reflect)
tests/          Unit + e2e
deploy/         Dockerfile + docker-compose.yml
```

## Roadmap

Near-term (next few slices, à la carte):

- **`SessionStart` recall endpoint** — async hook that returns markdown
  (open todos, last decisions, recent commits) for `additionalContext`.
- **BM25 + vector RRF retrieval** — currently vector-only.
- **Tree-sitter ingestion** — one language to start; symbol-level graph.
- **Natural-prose anchor extraction** — today anchors are structural JSON;
  parse mentions like `(see auth.py:42)` into `{kind:"symbol",...}`.
- **Re-embed-on-model-swap tool** — `model_version` is recorded; no migration
  helper yet.

Forced, not optional — **replace or fork Kùzu**:

Kùzu was archived 2025-10-10 (Apple acquired Kùzu Inc.), with 0.11.3 the final
release. That alone would only be a slow-burn concern, but our store also
carries real damage: `ALTER TABLE ... DROP` leaves chunks that segfault on a
full checkpoint, so `Idea` cannot be rewritten by the engine that damaged it.
Normal operation is unaffected — details and measurements in
`docs/kuzu-drop-column-hazard.md`.

Two shapes of answer, both open:

- **A maintained fork.** The code is MIT, and several exist —
  [LadybugDB](https://gdotv.com/blog/kuzu-legacy-embedded-graph-database-landscape/),
  [Vela-Engineering/kuzu](https://vela.partners/blog/kuzudb-ai-agent-memory-graph-database)
  (multi-writer, aimed at agent memory), and Kineviz's *bighorn*. Cheapest path
  by far: same Cypher, same embedded model, likely a drop-in swap. But a fork
  off 0.11.3 inherits this bug unless it has actually been fixed there — our
  repro was never filed upstream before the archive, so **check the fork
  against `tests/test_schema_migrate.py`'s xfail before committing to it.**
- **A different engine.** Migration guides exist for
  [ArcadeDB](https://arcadedb.com/blog/from-kuzudb-to-arcadedb-migration-guide/)
  and [FalkorDB](https://www.falkordb.com/blog/kuzudb-to-falkordb-migration/).
  More work, and both are servers rather than embedded, which changes the
  deployment story.

Either way the migration is the moment to rebuild `Idea` cleanly, since every
row gets read out and rewritten anyway — that repairs the damage as a side
effect. The evaluation criterion that actually matters here is therefore: *can
the candidate rewrite a table that Kùzu 0.11.3 cannot?*

Longer-term (only if it earns its keep): PPR retrieval seeded from
query-extracted entities, Joplin importer, idea-graveyard query,
`graduate_idea` to promote ideas to projects, `/consolidate` for summary
nodes.

Explicitly **deferred or dropped**: LLM extraction at write time,
dedup/consolidate passes, conversation-turn auto-firehose, multi-user
auth, web UI. Full rationale in `PLAN.md` §2 and §8.

## Design rules (worth knowing before contributing)

- Closed schema. New node/edge types require a conscious migration, not
  drift from extractors.
- Bi-temporal edges from day one (`valid_from`, `valid_to`, `recorded_at`,
  `source_extractor`, `extractor_version`).
- No LLM extraction at write time. Structural extractors only.
- Embeddings always tagged with `model_version`; re-embed is a planned
  ops task.
- Hooks degrade gracefully — memory failures never block a Claude Code
  session.

See `PLAN.md` for the full design history and decisions.
