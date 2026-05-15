from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from server.deps import build_state
from server.mcp_tools import register_tools


class _Recorder:
    def __init__(self):
        self.tools: dict = {}

    def tool(self, *args, **kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


@pytest.fixture
def tools(isolated_data_dir):
    state = build_state()
    recorder = _Recorder()
    register_tools(recorder, state)
    yield recorder.tools, state


@pytest.fixture
def fake_embed():
    vec = [0.01 * (i % 100) for i in range(768)]
    with patch("server.mcp_tools.embed", new=AsyncMock(return_value=vec)) as m:
        yield m


async def test_store_then_query_roundtrip(tools, fake_embed):
    fns, _ = tools
    stored = await fns["store_episode"](
        text="alpha bravo charlie", kind="note", project_hint="proj1"
    )
    assert stored["id"]
    assert stored["project_id"]

    rows = await fns["query_memory"](query="anything", project_hint="proj1")
    assert len(rows) == 1
    assert rows[0]["text"] == "alpha bravo charlie"
    assert rows[0]["project_id"] == stored["project_id"]


async def test_project_filter_excludes_others(tools, fake_embed):
    fns, _ = tools
    await fns["store_episode"](text="in proj1", kind="note", project_hint="proj1")
    await fns["store_episode"](text="in proj2", kind="note", project_hint="proj2")

    rows = await fns["query_memory"](query="x", project_hint="proj1")
    assert [r["text"] for r in rows] == ["in proj1"]


async def test_cypher_rejects_mutations(tools):
    fns, _ = tools
    with pytest.raises(ValueError):
        await fns["cypher"]("CREATE (:Project {id: 'x', name: 'evil'})")


async def test_cypher_read_works(tools, fake_embed):
    fns, _ = tools
    await fns["store_episode"](text="hi", kind="note", project_hint="hello-proj")
    rows = await fns["cypher"]("MATCH (p:Project) RETURN p.name")
    names = [row[0] for row in rows]
    assert "hello-proj" in names


async def test_embedding_dim_mismatch_raises(tools):
    fns, _ = tools
    with patch("server.mcp_tools.embed", new=AsyncMock(return_value=[0.0] * 10)):
        with pytest.raises(RuntimeError, match="embedding dim mismatch"):
            await fns["store_episode"](text="x", kind="note")
