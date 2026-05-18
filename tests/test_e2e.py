"""End-to-end test via the MCP client SDK over real HTTP.

Requires Ollama reachable at KNOWITALL_OLLAMA_URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tests.conftest import requires_ollama


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def live_server(isolated_data_dir):
    from server.app import create_app

    port = _free_port()
    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    deadline = time.time() + 10
    async with httpx.AsyncClient() as probe:
        while time.time() < deadline:
            try:
                r = await probe.get(
                    f"http://127.0.0.1:{port}/healthz", timeout=0.5
                )
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("server failed to come up")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)


@requires_ollama
async def test_store_query_via_mcp_client(live_server):
    headers = {"Authorization": "Bearer test-token"}
    async with streamablehttp_client(f"{live_server}/mcp", headers=headers) as (
        read,
        write,
        _,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert {
                "record",
                "query_memory",
                "cypher",
                "forget",
                "update_todo",
            } <= tool_names

            await session.call_tool(
                "record",
                {
                    "kind": "note",
                    "body": "fixture: ferrets favor floral fixtures",
                    "project_hint": "fixturetest",
                },
            )

            result = await session.call_tool(
                "query_memory",
                {"query": "ferrets floral", "project_hint": "fixturetest"},
            )
            text_blocks = [
                b.text for b in result.content if getattr(b, "type", None) == "text"
            ]
            assert any("ferrets favor floral" in t for t in text_blocks)
