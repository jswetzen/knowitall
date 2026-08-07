# Plan: migrate to `mcp` 2.0

Not started. Parking the investigation so a future session can pick this up without
re-deriving it. Context: 2026-08-07, after fixing `deploy/Dockerfile` to build from a
frozen `mcp>=1.2,<2.0` lock (see git log around that date) — this repo runs 1.27.1 in
production on mikro. This doc is about the *next* step, upgrading past that pin
on purpose, not a bug.

## Why bother

`mcp<2.0` is a ceiling, not a resting place. Upstream 2.0 is where new features land;
staying pinned indefinitely means drifting further from what `mcp` docs/examples assume.

## Scope, already traced

Confirmed by installing 2.0.0 in a scratch container and introspecting every `mcp.*`
symbol this repo imports:

| Old (1.x) | New (2.0) | Where |
|---|---|---|
| `mcp.server.fastmcp.FastMCP` | `mcp.server.mcpserver.MCPServer` | `server/app.py`, `server/mcp_tools.py`, `server/mcp_prompts.py`, `server/cango.py` — import + type hints only |
| `mcp.client.streamable_http.streamablehttp_client` | `mcp.client.streamable_http.streamable_http_client` | `client/cli.py` |

`@mcp.tool()` / `@mcp.prompt()` decorator signatures are unchanged for how this codebase
calls them (no args, or `name=`/`title=`/`description=`) — the ~15 decorated functions in
`mcp_tools.py`/`mcp_prompts.py`/`cango.py` need **only the import line changed**.

The one structural change is `server/app.py`. Today:

```python
mcp = FastMCP("knowitall", stateless_http=False, streamable_http_path="/mcp",
              transport_security=TransportSecuritySettings(...))
```

`MCPServer.__init__` in 2.0 no longer takes `stateless_http` / `streamable_http_path` /
`transport_security` — they moved to `streamable_http_app()` / `run_streamable_http_async()`:

```python
mcp = MCPServer("knowitall")
...
app.mount("/", mcp.streamable_http_app(
    stateless_http=False, streamable_http_path="/mcp",
    transport_security=TransportSecuritySettings(...),
))
```

`mcp.session_manager` and the lifespan wiring (`async with mcp.session_manager.run(): ...`)
are unchanged.

## Do NOT flip `stateless_http` while doing this

`stateless_http=False` is a deliberate, already-commented decision in `server/app.py`:
*"Claude Code expects the Mcp-Session-Id header flow."* This migration is an import-path
bump, not a chance to also go stateless — keep it `False` unless a separate investigation
confirms the Claude Code MCP client no longer needs session continuity.

2.0 does add a real stateless-*ish* primitive worth knowing about but not using here:
`RequestStateSecurity` / `RequestStateBoundary` — an opaque AES-256-GCM-sealed
`requestState` token the client round-trips per request instead of the server holding
session state in memory, aimed at horizontally-scaled multi-worker deployments.
`MCPServer` installs `RequestStateSecurity.ephemeral()` by default, which is explicitly
single-process only ("multi-instance deployments must share a key via `keys=[...]`").
Irrelevant to knowitall: it's one podman container on one CT. Don't reach for it.

## What's NOT yet audited

2.0's `MCPServer.__init__` gained several new concepts this investigation didn't chase
down: `Extension`, `ResourceSecurity`, `SubscriptionBus`, `middleware=`, `cache_hints=`.
None of them are used today, so the migration doesn't *need* them, but skim the 2.0
changelog for anything that silently changes default behavior (e.g. `ResourceSecurity`'s
defaults — `reject_path_traversal=True` etc. — look like they were opt-out in 1.x and are
now always-on; confirm that's not a breaking behavior change for any resource read path).

## Steps, when picked up

1. Bump `pyproject.toml` to `mcp>=2.0,<3.0`, `uv lock`.
2. Fix the import/type-hint lines listed above (mechanical, ~5 lines) and the
   `server/app.py` constructor restructure.
3. Fix `client/cli.py`'s `streamablehttp_client` → `streamable_http_client` rename.
4. `pytest tests/test_cango.py -q` plus the full suite.
5. Local smoke: `uv run uvicorn --factory server.app:create_app ...`, hit `/healthz`,
   drive one tool call end-to-end over real streamable HTTP (not just import-level).
6. Ship through the documented path: push to `main`, then mikro-iac's
   `./deploy.sh redeploy knowitall` (see `.claude/skills/ship-calendar-stack` there).
7. Verify Claude Code's actual MCP client reconnects and session flow still works —
   this is the one behavior the import-rename can't prove by itself.

## Also noticed in passing, unrelated to this plan

`STATUS.md`'s "Useful local context" section still says *"`uv.lock` is in `.gitignore`
(deliberate)"* — that's now stale; `uv.lock` was un-gitignored and committed as part of
the Dockerfile reproducibility fix mentioned above. Worth a line-fix next time STATUS.md
is touched, not urgent enough to justify a solo commit.
