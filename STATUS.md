# knowitall — Current state (2026-05-17)

Companion to `PLAN.md` and `PLAN_V2.md`. Snapshot of where we are; refresh
when picking up.

## What's done

**v2 MCP surface refactor — uncommitted** (working tree on `main`, post `a151b0c`).

Implements every "In scope" item in `PLAN_V2.md`:

- `schema/v2.cypher` lands the additive DDL: `Episode` node, `retracted_at`
  columns on `Decision`/`Task`/`Idea`/`Note`, and the multi-pair
  `ANCHORED_TO` rel (memory-bearing → graph-anchor). Confirmed against
  Kuzu 0.11.3: `CREATE REL TABLE` with multiple `FROM x TO y` pairs and
  `ALTER TABLE ... ADD IF NOT EXISTS ...` both supported as-is.
- `server/deps.py` replaces the v1 `episodes` table with the union
  `embeddings` table (`id`, `node_type`, `text`, `vector`, `project_id`,
  `kind`, `created_at`, `model_version`, `retracted_at`). Legacy
  `episodes` table is dropped on startup (user has no data; safe).
- `server/anchors.py` is the single resolution path for anchors. Reuses
  `_upsert_file` / `_upsert_person` from `ingest/git_extractor.py`; adds
  resolvers for commit / symbol / project / concept stubs.
- `server/mcp_tools.py` rewrites the tool surface: `record`,
  `update_todo`, `forget`, extended `query_memory` (1-hop neighbors,
  retracted filter, node_types filter), `cypher` (unchanged semantics,
  refreshed docstring). `ingest_repo` is no longer registered. `kind`
  enum validated server-side.
- `server/mcp_prompts.py` registers `/knowitall:status`,
  `/knowitall:capture`, `/knowitall:provenance`, `/knowitall:reflect`.
- `server/app.py` flips `stateless_http=False` and mounts the FastMCP
  ASGI app at `/` (with `streamable_http_path="/mcp"`) so `/mcp` is the
  canonical endpoint — no trailing-slash redirect. `/healthz` is
  registered first so route precedence keeps it reachable.
- `client/cli.py` replaces `store` with `record` (`--anchor` JSON,
  repeatable), adds `forget` and `update-todo`, updates `query` to
  surface neighbors and accept `--hops`, `--include-retracted`, and
  `--node-type`.
- `tests/test_mcp_tools.py` rewritten for the new surface; 22 cases
  green. `tests/test_e2e.py` updated to call `record` and assert the new
  tool set. `tests/test_git_extractor.py` left untouched and still green.
- README updated: tool table, prompt table, no-trailing-slash registration
  line, new CLI examples.

## Verification

- `uv run pytest -q` → **27 passed** (unit + e2e via MCP client over real
  uvicorn + real Ollama).
- Manual smoke against `TestClient`: `/healthz` returns 200; POST to
  `/mcp` (no trailing slash) reaches the FastMCP transport layer.

## What's open

1. **Confirm Claude Code can connect.** The two STATUS hypotheses
   (`stateless_http=True`, trailing-slash quirk) are both addressed in
   code, but the actual Claude Code handshake hasn't been retried since
   the rewrite. Next session: `claude --debug` against the rebuilt
   container.
2. **Container rebuild.** `podman compose --env-file deploy/.env up -d
   --build` from `deploy/`. The Dockerfile is unchanged; rebuild picks
   up the new server code.
3. **Deferred per PLAN_V2.md (out of scope here):** SessionStart hook,
   curator skill, BM25+RRF, tree-sitter, natural-prose anchor
   extraction, re-embed-on-model-swap.

## Risks parked at impl time

- **Sparse commit anchors.** A `commit` anchor without `message` or
  `authored_at` is accepted and the stub stores `""` / `now()`. Backfill
  is the client's job. If we ever see real divergence between stub and
  real-ingest data, add an idempotent enrichment pass.
- **`update_todo` + non-`done` statuses.** Free-form string today;
  conventional set is documented in the docstring but not enforced.

## Useful local context (unchanged from prior STATUS)

- Token lives in `deploy/.env` as `KNOWITALL_TOKEN=...` (gitignored at
  root and in `deploy/`).
- Ollama: `http://192.168.1.33:11434`, model `nomic-embed-text-v2-moe`,
  dim 768.
- Container: `podman ps` should show `knowitall` on `:8765`.
- `uv.lock` is in `.gitignore` (deliberate).
- Local venv is Python 3.14; container is Python 3.12. Tests should be
  run via the venv (`uv run pytest`); container is for deployment only.
