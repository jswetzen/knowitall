# knowitall

Personal RAG / persistent memory MCP server for Claude Code. Self-hosted, single-user,
graph (Kùzu) + vector (LanceDB) + embeddings (Ollama). See [PLAN.md](PLAN.md) for the
full design.

This is the week-1 walking skeleton: three MCP tools (`store_episode`, `query_memory`,
`cypher`) exposed over Streamable HTTP, bearer-token auth.

## Run locally (no container)

```bash
cp .env.example .env                 # then edit KNOWITALL_TOKEN
uv sync
uv run uvicorn --factory server.app:create_app --host 127.0.0.1 --port 8765
```

## Run via podman compose

```bash
cp .env.example deploy/.env          # edit KNOWITALL_TOKEN
cd deploy
podman compose --env-file .env up -d
curl http://127.0.0.1:8765/healthz
```

Data persists to `./data/` (bind-mounted into the container at `/data`).

## Try it

```bash
KNOWITALL_TOKEN=<token> uv run python -m client.cli store \
    "the new auth lives in cmd/auth-svc" --kind note --project knowitall
KNOWITALL_TOKEN=<token> uv run python -m client.cli query \
    "auth service location" --project knowitall
```

## Register with Claude Code

Add to `~/.claude/settings.json`:

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

## Tests

```bash
uv run pytest        # unit + e2e (e2e requires Ollama reachable)
```

## What's in here

```
server/         FastAPI + FastMCP app, three tools, bearer-token middleware
schema/         Kùzu DDL (v0.cypher) + idempotent migration runner
client/cli.py   Tiny CLI MCP client for smoke testing
tests/          Unit tests (mocked Ollama) + e2e test (real Ollama)
deploy/         Dockerfile + docker-compose.yml
```
