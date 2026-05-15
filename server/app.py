from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from server.auth import BearerTokenMiddleware
from server import config
from server.deps import build_state
from server.mcp_tools import register_tools


def create_app() -> FastAPI:
    state = build_state()
    mcp = FastMCP("knowitall", stateless_http=True, streamable_http_path="/")
    register_tools(mcp, state)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                await state.aclose()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(BearerTokenMiddleware)
    app.mount("/mcp", mcp.streamable_http_app())

    @app.get("/healthz")
    async def healthz():
        kuzu_rows = 0
        result = state.kuzu_conn().execute("MATCH (p:Project) RETURN count(p)")
        if result.has_next():
            kuzu_rows = int(result.get_next()[0])
        lance_rows = state.episodes.count_rows()
        return {
            "status": "ok",
            "kuzu_projects": kuzu_rows,
            "lance_episodes": lance_rows,
            "model_version": config.settings.ollama_model,
            "embedding_dim": config.settings.embedding_dim,
        }

    return app
