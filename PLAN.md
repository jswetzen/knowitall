# knowitall — Handover Plan

Personal RAG / persistent memory layer for Claude Code. Self-hosted, graph + vector,
single-user, hobby-grade.

This document is the source of truth for an agent picking up implementation in a fresh
session. Read top to bottom once; it's deliberately self-contained.

---

## 1. What this is and why

**Problem.** Claude Code forgets everything between sessions. Re-explaining architecture,
quirks, decisions, current status of long-running projects every time is the bottleneck.
The user wants to say *"where is project X at?"* or *"just realized, Y would need this"*
and have Claude know what X is, which repo it lives in, what was last decided, and what
else touches it.

**Solution.** A self-hosted memory service that holds:

- Long-running project state (status, decisions, blockers)
- Cross-project links (which projects/files/concepts touch each other)
- Conversation history (Claude Code turns, attributed to projects)
- Todos and embryonic project ideas (with a graveyard for ideas that die)
- Code structure (tree-sitter symbols, git history) — deferred past week 1

Exposed to Claude Code via an HTTP MCP server on the home network. Embeddings run
locally on the user's Ollama. No LLM inference on the server side — Claude itself does
all reasoning over retrieved content.

**Name.** `knowitall`. MCP server name `mcp-knowitall`, slash command `/knowitall`.

---

## 2. Decisions already locked in

These came out of a long Q&A with the user. Do not relitigate without reason.

### Stack

| Layer | Choice | Why |
|---|---|---|
| Graph | **Kùzu** (embedded, Cypher) | Single-user, no JVM, file-based, syncs/backs up as a directory |
| Vectors | **LanceDB** (embedded) | Arrow-native, file-based, pairs philosophically with Kùzu |
| Embeddings | **Ollama** → `nomic-embed-text-v2-moe` | Already installed on user's server; small MoE model (475M, F16), good MTEB |
| Server | **Python + FastAPI** | Best Kùzu + LanceDB + MCP SDK support |
| MCP transport | **Streamable HTTP** (fallback SSE if needed) | LAN/VPN access from multiple machines; stdio won't work cross-machine |
| Auth | Static bearer token in `Authorization` header | "A few of my own machines"; OAuth is overkill |
| Container | **docker-compose / podman-compose**, bind-mounted data dir | User will deploy on dev box first, then Proxmox LXC |
| Backup | External concern (Proxmox host-layer / LXC backups) | No restic in-container |

### Hosting reality

- One live host (home server) at static internal IP. Second server is DR only.
- Ollama already running at `http://192.168.1.33:11434`, no auth, has `nomic-embed-text-v2-moe:latest` available.
- User reaches the server from workstation on LAN, laptop over VPN.
- Static internal IP for MCP endpoint. Bearer token in user-level `~/.claude/settings.json`.

### Design rules

- **Closed schema discipline.** ~13 node types, ~18 edge types. New types require a
  conscious migration, not extraction-driven sprawl.
- **Bi-temporal edges from day one.** `valid_from`, `valid_to`, `recorded_at` on every
  edge. Retrofitting is painful.
- **No LLM extraction at write time.** Structural extraction only (git, tree-sitter,
  frontmatter, raw episodes + embeddings). Sonnet is good enough to reason over raw
  passages at read time. Revisit only if retrieval clearly suffers.
- **No dedup / no consolidation passes** in v1. The user explicitly dropped the
  "fan-out / dedup / consolidate" pattern.
- **Embeddings always tagged with `model_version`.** Re-embed on model swap is a planned
  ops task, not an afterthought.
- **Soft aliases, never hard merges.** `ALIAS_OF` edges; retrieval merges at query time.
  (Not v1, but reserved in the schema vocabulary.)
- **Hooks are async and non-blocking.** Memory failures degrade gracefully — they never
  break a Claude Code session.

### Explicitly deferred

- LLM entity extraction (`Decision`, `Concept` from prose)
- Tree-sitter ingestion (month 1, not week 1)
- Dedup / soft aliasing
- GraphRAG-style community summaries
- Time-bound reminders, schedulers, push notifications
- External capture endpoint (mobile)
- Joplin / GitHub / Linear / calendar / email sync
- Multi-machine read replicas
- Web UI
- Restic / in-container backups
- Multi-user auth

### Explicitly dropped (do not propose)

- Neo4j (overkill for single user, JVM, ceremony)
- mem0 (wrong shape — chatbot personalization, not engineering memory)
- Letta / MemGPT (agent framework, not a memory MCP)
- Pure-vector-only (the graph earns its keep for multi-hop + temporal queries that are
  exactly the user's primary use case)
- Fan-out / dedup / consolidate pattern (dropped by user mid-design)

---

## 3. Schema v1 (the closed schema)

DDL lives in `schema/v1.cypher`. Migrations are forward-only files applied in order.

### Node types (~13)

| Node | Purpose | Key fields |
|---|---|---|
| `Project` | A long-running effort. May span 0–N repos | `id, name, status, created_at` |
| `Repo` | A git repository on disk | `id, path, remote_url, default_branch` |
| `File` | A file in a repo | `id, repo_id, path` |
| `Symbol` | Function/class/etc. from tree-sitter | `id, file_id, name, kind, line` |
| `Commit` | A git commit | `sha, repo_id, message, authored_at` |
| `Note` | A markdown note (Joplin/`~/notes`) | `id, path, title, created_at` |
| `Conversation` | One Claude Code session | `session_uuid, project_id, started_at` |
| `ConversationTurn` | One user↔assistant exchange | `id, conversation_id, role, text, ts` |
| `Decision` | A recorded decision | `id, body, decided_at` (manual or future-extracted) |
| `Concept` | A free-form topic | `id, name` |
| `Person` | Git author / mentioned human | `id, name, email` |
| `Task` | A todo | `id, body, status (open/done/dropped), created_at, closed_at` |
| `Idea` | Embryonic project idea | `id, body, status (incubating/graduated/dropped), created_at, died_at` |

### Edge types (~18), all bi-temporal

`PART_OF`, `BELONGS_TO`, `MENTIONED_IN`, `AUTHORED`, `MODIFIED`, `DEFINED_IN`,
`CALLS`, `IMPORTS`, `DEPENDS_ON`, `BLOCKS`, `SUPERSEDES`, `GRADUATED_TO`,
`ALIAS_OF` (reserved, not used in v1), `RELATES_TO`, `DECIDED_IN`, `DROPPED`,
`CLOSED_BY`, `TOUCHED_BY`.

Every edge carries `valid_from TIMESTAMP`, `valid_to TIMESTAMP NULL`,
`recorded_at TIMESTAMP`, plus optional `source_extractor` and `extractor_version`
for structural extractor provenance.

### Week-1 subset of the schema

Only ship these three node types and two edges in v0; v1 lands during month 1.

- Nodes: `Project`, `Repo`, `Commit`
- Edges: `Repo -PART_OF-> Project`, `Commit -IN-> Repo`

Plus a single LanceDB table `episodes` with columns:
`id, text, vector, kind, project_id, conversation_id, created_at, model_version`.

---

## 4. MCP tool surface (week 1)

All tools live behind a Streamable-HTTP MCP server, bearer-token auth.

### `store_episode(text: str, kind: str, project_hint: str | None = None) -> { id, project_id }`

- Embeds `text` via Ollama `nomic-embed-text-v2-moe`.
- Writes to LanceDB `episodes`.
- If `project_hint` resolves to an existing `Project`, links it. If not and `project_hint`
  is provided, creates a new `Project`. If absent, leaves `project_id` null.
- Returns episode id and resolved project id.

### `query_memory(query: str, project_hint: str | None = None, k: int = 10) -> [{ id, text, kind, score, project_id, created_at }]`

- Embeds query, ANN search over LanceDB.
- If `project_hint` provided, filter to that project (post-filter is fine at this scale).
- Returns top-k passages with metadata.

### `cypher(query: str, params: dict = {}) -> [rows]`

- Raw Kùzu passthrough. Read-only enforced by string check (block `CREATE`, `MERGE`,
  `DELETE`, `SET`, `DROP` for now — week 1 only). Mutation will get its own tool later.
- Useful for Claude to drill into structural questions.

### Health endpoint (not MCP, plain HTTP)

`GET /healthz` → `{ status: "ok", kuzu_rows: N, lance_rows: N, model_version: "..." }`.

### Auth

Every request must carry `Authorization: Bearer <token>`. Token loaded from env
`KNOWITALL_TOKEN`. Generate once, write to `.env` (gitignored), copy into
`~/.claude/settings.json` on each client machine.

---

## 5. Claude Code integration

### MCP registration (user-level `~/.claude/settings.json`)

```json
{
  "mcpServers": {
    "knowitall": {
      "type": "http",
      "url": "http://192.168.1.33:8765/mcp",
      "headers": { "Authorization": "Bearer <KNOWITALL_TOKEN>" }
    }
  }
}
```

Verify the user's Claude Code build supports `type: "http"` (Streamable HTTP). If only
SSE is available, the same URL with `type: "sse"` works against an SSE-mode server —
the FastMCP SDK supports both, expose both transports if cheap.

### Hooks (added month 1, not week 1)

**`SessionStart`** — async HTTP POST to `/recall/session_start` with `CLAUDE_PROJECT_DIR`
in payload. Server returns a short markdown block (open todos, last 3 decisions, last
5 commits) for `additionalContext`. Hard budget: <200ms, <50 lines.

**`PostToolUse`** — async fire-and-forget POST to `/ingest/turn` with the turn payload.
Server creates `ConversationTurn`, embeds, links to project. Hook MUST be `async: true`
to avoid blocking the UI.

Both hook endpoints are HTTP, same bearer token. Defined later — week 1 only needs the
MCP tools.

### Slash command (later)

`.claude/commands/knowitall.md` — wraps `query_memory` with `CLAUDE_PROJECT_DIR` as the
implicit `project_hint`.

---

## 6. Week-1 walking skeleton — checklist

The goal is: from a Claude Code session, Claude can `store_episode("...")`, and a later
session can `query_memory("...")` and get it back. Nothing more. Nothing about hooks,
tree-sitter, or todos in week 1 — those land in month 1.

### Prereqs to verify in the new session

```bash
# Container tooling
podman --version
podman-compose --version  # or: docker compose version

# Python (uv is already available; python3 was missing in last session)
uv --version

# Ollama reachable and has the model
curl -sS http://192.168.1.33:11434/api/tags | grep nomic-embed-text-v2-moe
# Expected: model is present (confirmed in previous session)
```

### Repo state at handover

- Path: `/workspace/knowitall`
- Git: initialized, on `main`, **no commits yet**, working tree clean except this
  `PLAN.md` we're about to add.
- The repo will be renamed/re-purposed as `knowitall`. Keep the path
  `/workspace/knowitall` for now; rename later if desired.

### Step-by-step

1. **Scaffold the project**
   - `pyproject.toml` with `uv`-managed deps: `fastapi`, `uvicorn`, `mcp` (official SDK),
     `kuzu`, `lancedb`, `pyarrow`, `httpx`, `pydantic-settings`, `pytest`, `pytest-asyncio`,
     `ruff`.
   - Directory layout:
     ```
     server/        FastAPI + MCP app
       app.py         Entrypoint, mounts MCP + /healthz
       mcp_tools.py   The three MCP tools
       deps.py        Kùzu + LanceDB + Ollama clients
       auth.py        Bearer-token middleware
       config.py      Settings (env-driven)
     schema/
       v0.cypher      Week-1 DDL (Project, Repo, Commit, 2 edges)
       migrate.py     Apply DDL idempotently
     ingest/        (empty in week 1)
     client/
       cli.py         `knowitall query/store` for local testing
     tests/
       test_mcp_tools.py
       test_e2e.py
     deploy/
       Dockerfile
       docker-compose.yml
     .env.example
     .gitignore
     README.md
     ```

2. **Pull / verify the embedding model**
   - Confirmed available: `nomic-embed-text-v2-moe:latest` on `192.168.1.33:11434`.
   - Smoke test:
     ```bash
     curl -sS http://192.168.1.33:11434/api/embed \
       -d '{"model":"nomic-embed-text-v2-moe","input":"hello"}' | head -c 500
     ```
   - Note the vector dimension from the response and hardcode it in the LanceDB schema.
     (Expected 768 for nomic-embed v2, **verify, don't assume**.)

3. **Kùzu schema v0**
   - Write `schema/v0.cypher` with three node tables and two rel tables. Include the
     bi-temporal edge fields even though week-1 doesn't query them.
   - `schema/migrate.py` opens the DB at `KNOWITALL_DATA_DIR/kuzu`, applies each
     `vN.cypher` in order, records applied versions in a `_migrations` node table.

4. **LanceDB `episodes` table**
   - Create on first startup if missing. Columns:
     `id: str, text: str, vector: fixed_size_list<float32, DIM>, kind: str,
      project_id: str | null, conversation_id: str | null, created_at: timestamp,
      model_version: str`.
   - `model_version` value: `"nomic-embed-text-v2-moe:latest"` (or the digest if you
     want to be strict — digest is in `/api/tags` output).

5. **Embedding client**
   - `server/deps.py` has an `async embed(text: str) -> list[float]` calling
     `POST /api/embed` on Ollama. Single-call only in week 1; batching later.
   - 5s timeout, one retry on connection error, surface errors clearly.

6. **MCP server (Streamable HTTP)**
   - Use the official `mcp` Python SDK with `FastMCP`. Mount under FastAPI at `/mcp`.
   - Implement the three tools from §4.
   - Bearer-token middleware applied to both `/mcp/*` and `/healthz` (well, healthz can
     be open — easier for compose health checks).

7. **Cypher tool safety**
   - Reject queries containing (case-insensitive, word-boundary)
     `CREATE|MERGE|DELETE|SET|DROP|REMOVE|DETACH|COPY`. Crude but week-1 sufficient.
     Better grammar-aware checking is a month-2 task if `cypher` proves useful.

8. **Tests**
   - `test_mcp_tools.py`: spin up Kùzu + LanceDB in `tmp_path`, mock Ollama with a
     deterministic 768-d vector, exercise each tool function directly.
   - `test_e2e.py`: start the FastAPI app in-process, hit `/mcp` over HTTP with the MCP
     client SDK, do store→query round-trip. Real Kùzu + LanceDB, real Ollama (skip if
     `KNOWITALL_OLLAMA_URL` not reachable).

9. **Containerization**
   - `Dockerfile`: python:3.12-slim base, `uv sync --frozen`, copy app, run uvicorn.
   - `docker-compose.yml`: one service `knowitall`, bind mount `./data:/data`, env
     `KNOWITALL_DATA_DIR=/data`, `KNOWITALL_OLLAMA_URL=http://192.168.1.33:11434`,
     `KNOWITALL_TOKEN` from `.env`, port `8765:8765`. Health check hits `/healthz`.
   - Verify with `podman compose up`, curl `/healthz`, curl the MCP endpoint, see store
     and query work end-to-end.

10. **README**
    - How to run (`cp .env.example .env`, set `KNOWITALL_TOKEN`, `podman compose up`).
    - The Claude Code `~/.claude/settings.json` snippet.
    - One paragraph explaining what this is (link to this PLAN.md for depth).

11. **Commits**
    - Commit at clean checkpoints, not at the end. Suggested boundaries:
      1. scaffold + pyproject + empty schema
      2. schema v0 + migrate + Kùzu/LanceDB plumbing
      3. embedding client + Ollama integration
      4. MCP tools + auth
      5. tests passing
      6. docker-compose green

### Exit criterion

`podman compose up`, then from any HTTP client (curl or a Claude Code session):

```
POST /mcp  (store_episode with text="the new auth lives in cmd/auth-svc", kind="note", project_hint="knowitall")
POST /mcp  (query_memory with query="auth service location", project_hint="knowitall")
```

returns the stored text in the top results. That's week-1 done.

---

## 7. Risk and known unknowns

- **MCP transport compatibility.** Streamable HTTP is the 2025 standard but some
  Claude Code versions still expect SSE. Expose both via FastMCP if it's a one-line
  cost; otherwise pick Streamable HTTP and adjust if the user reports a handshake
  failure.
- **Kùzu API churn.** Kùzu is pre-1.0; pin the version in `pyproject.toml`. If a Python
  API call doesn't match docs, check the installed version's docs specifically.
- **LanceDB filter pushdown.** Project-hint filtering should use SQL-style `where`
  clauses on LanceDB; verify the column type matches (string vs. categorical).
- **Embedding dimension.** Confirmed model is MoE, but verify dim from the live API
  response, do not hardcode 768 without checking.
- **Cypher tool is a footgun.** The string-blocklist is crude. If the user goes wild
  with `cypher` calls, harden this before exposing to a less-trusted context.
- **No tests for the MCP protocol layer itself in week 1** — we rely on the SDK being
  correct. End-to-end test via the SDK's client is the proxy for protocol correctness.

---

## 8. Month-1 additions (do not start in week 1)

For continuity once week 1 lands. Detailed design in the conversation, summary here:

- Full schema v1 (all 13 node types, all 18 edges, bi-temporal everywhere)
- Tree-sitter ingestion (one language to start — pick whichever language dominates the
  user's repos)
- Git extractor (`Commit`, `Person`, `MODIFIED` edges)
- Frontmatter / markdown extractor for notes
- `ConversationTurn` ingestion via `PostToolUse` async hook
- `remember_todo` / `update_todo` MCP tools, `Task` and `Idea` nodes
- `SessionStart` hook that injects open todos + recent decisions + recent commits
- `/knowitall` slash command
- BM25 + vector RRF retrieval (currently just vector)
- 1-hop graph expansion on retrieved nodes

## 9. Later (months 2–3, only if needed)

- PPR retrieval seeded from query-extracted entities (HippoRAG-style)
- Code-specific embedder split (if `nomic-v2-moe` proves weak on code)
- Joplin importer
- `/consolidate` slash command for `Summary` nodes
- `graduate_idea(id, project)` to promote ideas to projects
- Idea graveyard query

---

## 10. Conversation context the next agent should know

- The user is a power user, comfortable with self-hosting, runs Proxmox and has two
  servers (one live, one DR-only). They're aware of restic now but it's external to
  the project — Proxmox/LXC backups handle persistence at the host layer.
- Maintenance tolerance is **high** — this is the user's hobby. Complexity is fine if
  it earns its keep. Boring tech still preferred where it makes no difference.
- The user spent the design phase asking pointed questions and dropped one mid-design
  pattern (fan-out/dedup/consolidate) once they reconsidered. Be willing to do the same.
- The user is on Claude Code, primarily Opus, with Sonnet considered "really good" —
  do not hesitate to defer reasoning to the LLM rather than building extraction logic
  on the server side.
- `MEMORY.md` auto-memory is **not** what the user wants to extend; knowitall is a
  parallel system, not a replacement for built-in auto-memory.
- The user reads code well. Verbose comments are unwelcome. Closed schema discipline
  is a stated value — resist accreting node/edge types without a real need.

---

## 11. First commands to run in the new session

```bash
cd /workspace/knowitall
ls -la
git status
git log --oneline | head -5     # should be empty / "no commits yet"
podman --version
curl -sS http://192.168.1.33:11434/api/tags | grep -o '"name":"[^"]*"' | head -20
uv --version
```

If all green, proceed to step 1 of §6.
