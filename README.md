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
  ~14 node types and ~20 bi-temporal edges, including a generic
  `ANCHORED_TO` citation edge. See `PLAN.md` §3 and `PLAN_V2.md`.
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
| `record(kind, body, project_hint, anchors, summary, relates_to)` | Save a durable memory. `kind` ∈ {decision, task, idea, note, summary, blocker, fact, solution, episode}. `solution` is for env/setup/config gotchas — lead the body with the verbatim error string. `anchors` are typed JSON citations (commit/file/symbol/project/concept/person) that create graph edges. Optional `summary` is a ≤200-char title-shaped string (falls back to first 200 of body). Optional `relates_to` writes memory→memory edges (kinds: supersedes/refines/contradicts/relates_to). |
| `query_memory(query, project_hint, k, expand_hops=0, snippet_chars=240, include_retracted, node_types)` | Semantic search. Defaults: bodies clipped to 240 chars + ellipsis, no neighbor expansion. Pass `expand_hops=1` for ANCHORED_TO neighbors; pass `snippet_chars=0` for full bodies. |
| `list_memories(kind, project_hint, limit, offset, order_by, include_retracted)` | Enumerate memories without semantic ranking. Returns summaries only; use `get_memory` to fetch full bodies. |
| `get_memory(id, include_neighbors)` | Fetch a memory by id. Returns full body + summary + metadata. Surfaces retracted nodes with `retracted_at` populated. |
| `amend(id, body, summary, add_anchors, remove_anchors)` | In-place edit preserving id. Body changes trigger re-embed; summary updates skip it. Rejects retracted nodes. |
| `update_todo(id, status, anchors)` | Transition a Task's status. Done + commit anchor also writes `CLOSED_BY`. |
| `forget(id, reason)` | Soft undo — sets `retracted_at`; default queries hide it. |
| `cypher(query, params)` | Read-only Cypher passthrough over the graph. |

#### Calendar (Cango) shims

Thin shims over the sibling `cango-daemon` (Unix socket at `KNOWITALL_CANGO_SOCKET`, default `/run/cango/cango.sock`). No calendar logic lives in knowitall; these marshal JSON-RPC and pass the daemon's answer through. If the daemon is down or the socket is missing, each returns `{"error": "cango_unavailable", "reason": ...}` — memory tools are unaffected.

| Tool | Purpose |
|---|---|
| `check_availability(start, end, people)` | "Can we go?" → `free` / `soft_conflict` / `hard_conflict` verdict + conflicts for a window across family calendars. |
| `find_free_slot(duration_minutes, between_start, between_end, people, working_hours)` | Candidate free windows of a given length within a range. |
| `list_events(start, end, people)` | Events in a window with their resolved roles and `resolved_by` trace reason. |
| `explain_event(event_id)` | Layer-by-layer trace of how one event's role was resolved. |
| `list_series(source_id)` | Recent recurring series on a source — input for adding attendance edges. |

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
