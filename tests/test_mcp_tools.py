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


# ----------------- record + query roundtrip -----------------


async def test_record_decision_roundtrip(tools, fake_embed):
    fns, _ = tools
    stored = await fns["record"](
        kind="decision",
        body="Pick Kuzu over Neo4j: embedded, no JVM",
        project_hint="proj1",
    )
    assert stored["id"]
    assert stored["node_type"] == "decision"
    assert stored["project_id"]
    # Project auto-anchored.
    labels = [a["target_label"] for a in stored["anchored"]]
    assert "Project" in labels

    rows = await fns["query_memory"](query="anything", project_hint="proj1")
    assert len(rows) == 1
    assert rows[0]["hit"]["text"].startswith("Pick Kuzu")
    assert rows[0]["hit"]["node_type"] == "decision"
    assert any(n["label"] == "Project" for n in rows[0]["neighbors"])


@pytest.mark.parametrize(
    "kind,expected_node_type",
    [
        ("decision", "decision"),
        ("task", "task"),
        ("idea", "idea"),
        ("note", "note"),
        ("summary", "episode"),
        ("blocker", "episode"),
        ("fact", "episode"),
        ("episode", "episode"),
    ],
)
async def test_record_each_kind(tools, fake_embed, kind, expected_node_type):
    fns, _ = tools
    out = await fns["record"](kind=kind, body=f"body for {kind}")
    assert out["node_type"] == expected_node_type


async def test_record_rejects_unknown_kind(tools, fake_embed):
    fns, _ = tools
    with pytest.raises(ValueError, match="unknown kind"):
        await fns["record"](kind="garbage", body="x")


async def test_record_with_file_anchor_creates_link(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](
        kind="decision",
        body="Move auth out of monolith",
        project_hint="alpha",
        anchors=[{"kind": "file", "repo": "alpha", "path": "server/auth.py"}],
    )
    # Project + File anchors.
    labels = sorted(a["target_label"] for a in out["anchored"])
    assert labels == ["File", "Project"]

    rows = await fns["query_memory"](query="x", project_hint="alpha")
    assert len(rows) == 1
    neighbors = {n["label"] for n in rows[0]["neighbors"]}
    assert "File" in neighbors


# ----------------- project filter -----------------


async def test_project_filter_excludes_others(tools, fake_embed):
    fns, _ = tools
    await fns["record"](kind="note", body="in proj1", project_hint="proj1")
    await fns["record"](kind="note", body="in proj2", project_hint="proj2")

    rows = await fns["query_memory"](query="x", project_hint="proj1")
    texts = [r["hit"]["text"] for r in rows]
    assert texts == ["in proj1"]


# ----------------- node_types filter -----------------


async def test_node_types_filter(tools, fake_embed):
    fns, _ = tools
    await fns["record"](kind="decision", body="d1", project_hint="p")
    await fns["record"](kind="task", body="t1", project_hint="p")
    await fns["record"](kind="fact", body="f1", project_hint="p")

    decisions = await fns["query_memory"](
        query="x", project_hint="p", node_types=["decision"]
    )
    assert [r["hit"]["node_type"] for r in decisions] == ["decision"]

    eps = await fns["query_memory"](
        query="x", project_hint="p", node_types=["episode"]
    )
    assert [r["hit"]["node_type"] for r in eps] == ["episode"]


# ----------------- forget / retracted_at -----------------


async def test_forget_hides_from_default_query(tools, fake_embed):
    fns, _ = tools
    stored = await fns["record"](
        kind="note", body="to forget", project_hint="px"
    )
    out = await fns["forget"](id=stored["id"], reason="dup")
    assert out["label"] == "Note"
    assert out["retracted_at"]

    rows = await fns["query_memory"](query="x", project_hint="px")
    assert rows == []

    rows = await fns["query_memory"](
        query="x", project_hint="px", include_retracted=True
    )
    assert len(rows) == 1
    assert rows[0]["hit"]["retracted_at"] is not None


async def test_forget_unknown_id_raises(tools):
    fns, _ = tools
    with pytest.raises(ValueError, match="no retractable node"):
        await fns["forget"](id="nope", reason="r")


# ----------------- update_todo -----------------


async def test_update_todo_transitions_status(tools, fake_embed):
    fns, _ = tools
    stored = await fns["record"](kind="task", body="ship thing")
    out = await fns["update_todo"](id=stored["id"], status="done")
    assert out["status"] == "done"

    rows = await fns["cypher"](
        "MATCH (t:Task {id: $id}) RETURN t.status", {"id": stored["id"]}
    )
    assert rows[0][0] == "done"


async def test_update_todo_with_commit_creates_closed_by(tools, fake_embed):
    fns, _ = tools
    stored = await fns["record"](kind="task", body="fix bug")
    await fns["update_todo"](
        id=stored["id"],
        status="done",
        anchors=[{"kind": "commit", "sha": "deadbeef", "repo": "x"}],
    )
    rows = await fns["cypher"](
        "MATCH (t:Task {id: $id})-[:CLOSED_BY]->(c:Commit) RETURN c.sha",
        {"id": stored["id"]},
    )
    assert rows == [["deadbeef"]]


async def test_update_todo_unknown_id(tools):
    fns, _ = tools
    with pytest.raises(ValueError, match="no Task"):
        await fns["update_todo"](id="nope", status="done")


# ----------------- cypher gating -----------------


async def test_cypher_rejects_mutations(tools):
    fns, _ = tools
    with pytest.raises(ValueError):
        await fns["cypher"]("CREATE (:Project {id: 'x', name: 'evil'})")


async def test_cypher_read_works(tools, fake_embed):
    fns, _ = tools
    await fns["record"](kind="note", body="hi", project_hint="hello-proj")
    rows = await fns["cypher"]("MATCH (p:Project) RETURN p.name")
    names = [row[0] for row in rows]
    assert "hello-proj" in names


# ----------------- anchor idempotency -----------------


async def test_anchor_stub_idempotent_across_records(tools, fake_embed):
    fns, state = tools
    await fns["record"](
        kind="decision",
        body="d1",
        anchors=[{"kind": "file", "repo": "r", "path": "a.py"}],
    )
    await fns["record"](
        kind="note",
        body="n1",
        anchors=[{"kind": "file", "repo": "r", "path": "a.py"}],
    )
    rows = await fns["cypher"](
        "MATCH (f:File {path: 'a.py'}) RETURN count(f)"
    )
    assert rows[0][0] == 1


# ----------------- embedding dim guard -----------------


async def test_embedding_dim_mismatch_raises(tools):
    fns, _ = tools
    with patch("server.mcp_tools.embed", new=AsyncMock(return_value=[0.0] * 10)):
        with pytest.raises(RuntimeError, match="embedding dim mismatch"):
            await fns["record"](kind="note", body="x")


# ----------------- query_memory snippet_chars -----------------


async def test_query_memory_default_returns_full_body(tools, fake_embed):
    fns, _ = tools
    long_body = "x" * 1000
    await fns["record"](kind="idea", body=long_body, project_hint="sp")
    rows = await fns["query_memory"](query="x", project_hint="sp")
    assert rows[0]["hit"]["text"] == long_body


async def test_query_memory_snippet_chars_truncates(tools, fake_embed):
    fns, _ = tools
    long_body = "x" * 1000
    await fns["record"](kind="idea", body=long_body, project_hint="sp")
    rows = await fns["query_memory"](
        query="x", project_hint="sp", snippet_chars=240
    )
    text = rows[0]["hit"]["text"]
    assert text.endswith("…")
    assert len(text) == 241  # 240 chars + ellipsis


async def test_query_memory_snippet_chars_no_op_when_shorter(tools, fake_embed):
    fns, _ = tools
    await fns["record"](kind="idea", body="short", project_hint="sp")
    rows = await fns["query_memory"](
        query="x", project_hint="sp", snippet_chars=240
    )
    assert rows[0]["hit"]["text"] == "short"  # no ellipsis added


# ----------------- list_memories -----------------


async def test_list_memories_returns_summaries(tools, fake_embed):
    fns, _ = tools
    await fns["record"](kind="idea", body="first idea body", project_hint="lp")
    await fns["record"](kind="idea", body="second idea body", project_hint="lp")
    rows = await fns["list_memories"](kind="idea", project_hint="lp")
    assert len(rows) == 2
    summaries = sorted(r["summary"] for r in rows)
    assert summaries == ["first idea body", "second idea body"]
    # No "body" field — list returns summaries only.
    assert "body" not in rows[0]
    assert rows[0]["node_type"] == "idea"


async def test_list_memories_summary_clipped_to_200(tools, fake_embed):
    fns, _ = tools
    long_body = "a" * 500
    await fns["record"](kind="idea", body=long_body, project_hint="lp")
    rows = await fns["list_memories"](kind="idea", project_hint="lp")
    assert len(rows[0]["summary"]) == 200


async def test_list_memories_note_uses_title(tools, fake_embed):
    fns, _ = tools
    # Note's body is clipped to 200 chars and stored in `title` — list_memories
    # must read it back without double-clipping or mismatched fields.
    await fns["record"](kind="note", body="note about something", project_hint="lp")
    rows = await fns["list_memories"](kind="note", project_hint="lp")
    assert rows[0]["summary"] == "note about something"


async def test_list_memories_all_kinds_when_no_filter(tools, fake_embed):
    fns, _ = tools
    await fns["record"](kind="decision", body="d", project_hint="lp")
    await fns["record"](kind="task", body="t", project_hint="lp")
    await fns["record"](kind="idea", body="i", project_hint="lp")
    await fns["record"](kind="note", body="n", project_hint="lp")
    await fns["record"](kind="fact", body="f", project_hint="lp")  # → episode
    rows = await fns["list_memories"](project_hint="lp")
    node_types = sorted(r["node_type"] for r in rows)
    assert node_types == ["decision", "episode", "idea", "note", "task"]


async def test_list_memories_unknown_project_returns_empty(tools, fake_embed):
    fns, _ = tools
    await fns["record"](kind="idea", body="i", project_hint="real")
    rows = await fns["list_memories"](project_hint="nonexistent")
    assert rows == []


async def test_list_memories_unknown_kind_raises(tools):
    fns, _ = tools
    with pytest.raises(ValueError, match="unknown kind"):
        await fns["list_memories"](kind="bogus")


async def test_list_memories_hides_retracted_by_default(tools, fake_embed):
    fns, _ = tools
    stored = await fns["record"](kind="idea", body="goner", project_hint="lp")
    await fns["forget"](id=stored["id"], reason="x")
    rows = await fns["list_memories"](kind="idea", project_hint="lp")
    assert rows == []
    rows = await fns["list_memories"](
        kind="idea", project_hint="lp", include_retracted=True
    )
    assert len(rows) == 1
    assert rows[0]["retracted_at"] is not None


async def test_list_memories_order_and_paging(tools, fake_embed):
    fns, _ = tools
    ids = []
    for i in range(5):
        out = await fns["record"](kind="idea", body=f"idea {i}", project_hint="lp")
        ids.append(out["id"])

    desc = await fns["list_memories"](
        kind="idea", project_hint="lp", order_by="created_at_desc"
    )
    asc = await fns["list_memories"](
        kind="idea", project_hint="lp", order_by="created_at_asc"
    )
    assert [r["id"] for r in desc] == list(reversed([r["id"] for r in asc]))

    paged = await fns["list_memories"](
        kind="idea", project_hint="lp", limit=2, offset=1, order_by="created_at_asc"
    )
    assert [r["id"] for r in paged] == [ids[1], ids[2]]


async def test_list_memories_includes_project_id_without_filter(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="idea", body="i", project_hint="lp")
    rows = await fns["list_memories"](kind="idea")
    assert rows[0]["project_id"] == out["project_id"]


# ----------------- get_memory -----------------


async def test_get_memory_returns_full_body(tools, fake_embed):
    fns, _ = tools
    long_body = "x" * 500
    out = await fns["record"](kind="idea", body=long_body, project_hint="gp")
    got = await fns["get_memory"](id=out["id"])
    assert got is not None
    assert got["body"] == long_body
    assert got["summary"] == "x" * 200
    assert got["node_type"] == "idea"
    assert got["project_id"] == out["project_id"]
    assert "neighbors" not in got


async def test_get_memory_unknown_id_returns_none(tools):
    fns, _ = tools
    assert await fns["get_memory"](id="does-not-exist") is None


async def test_get_memory_returns_retracted(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="idea", body="i", project_hint="gp")
    await fns["forget"](id=out["id"], reason="r")
    got = await fns["get_memory"](id=out["id"])
    assert got is not None
    assert got["retracted_at"] is not None


async def test_get_memory_include_neighbors(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](
        kind="decision",
        body="d",
        project_hint="gp",
        anchors=[{"kind": "file", "repo": "r", "path": "f.py"}],
    )
    got = await fns["get_memory"](id=out["id"], include_neighbors=True)
    assert got is not None
    labels = {n["label"] for n in got["neighbors"]}
    assert "Project" in labels
    assert "File" in labels


async def test_get_memory_works_for_each_kind(tools, fake_embed):
    fns, _ = tools
    for kind in ("decision", "task", "idea", "note", "fact"):
        out = await fns["record"](kind=kind, body=f"body {kind}", project_hint="ap")
        got = await fns["get_memory"](id=out["id"])
        assert got is not None
        if kind == "note":
            # Note stores body as title (clipped); get_memory reads title back.
            assert got["body"] == f"body {kind}"
        else:
            assert got["body"] == f"body {kind}"


# ----------------- record summary -----------------


async def test_record_with_explicit_summary(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](
        kind="idea",
        body="a much longer body that says many things and rambles",
        summary="short hand",
        project_hint="sp",
    )
    got = await fns["get_memory"](id=out["id"])
    assert got["summary"] == "short hand"
    assert got["body"].startswith("a much longer body")


async def test_record_summary_falls_back_to_body(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="idea", body="just body", project_hint="sp")
    got = await fns["get_memory"](id=out["id"])
    assert got["summary"] == "just body"


async def test_record_rejects_oversize_summary(tools, fake_embed):
    fns, _ = tools
    with pytest.raises(ValueError, match="summary too long"):
        await fns["record"](kind="idea", body="b", summary="x" * 201)


async def test_record_summary_routes_to_note_title(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](
        kind="note", body="longer body content", summary="short title"
    )
    got = await fns["get_memory"](id=out["id"])
    # Note has no separate summary column — title IS the summary, and the
    # explicit summary should win over body-fallback.
    assert got["body"] == "short title"
    assert got["summary"] == "short title"


# ----------------- relates_to edges on record -----------------


async def test_record_relates_to_supersedes_writes_edge(tools, fake_embed):
    fns, _ = tools
    a = await fns["record"](kind="idea", body="EFD is good", project_hint="rp")
    b = await fns["record"](
        kind="idea",
        body="PFD is the new name",
        project_hint="rp",
        relates_to=[{"kind": "supersedes", "id": a["id"]}],
    )
    assert b["related"] == [
        {"kind": "supersedes", "target_id": a["id"], "target_label": "Idea"}
    ]
    rows = await fns["cypher"](
        "MATCH (newer:Idea)-[:SUPERSEDES_MEMORY]->(older:Idea) "
        "WHERE newer.id = $nid RETURN older.id",
        {"nid": b["id"]},
    )
    assert rows == [[a["id"]]]


async def test_record_relates_to_each_kind(tools, fake_embed):
    fns, _ = tools
    a = await fns["record"](kind="idea", body="base", project_hint="rp")
    for spec_kind, edge_label in [
        ("refines", "REFINES"),
        ("contradicts", "CONTRADICTS"),
        ("relates_to", "RELATES_TO_MEMORY"),
    ]:
        b = await fns["record"](
            kind="idea",
            body=f"linked via {spec_kind}",
            project_hint="rp",
            relates_to=[{"kind": spec_kind, "id": a["id"]}],
        )
        rows = await fns["cypher"](
            f"MATCH (n:Idea)-[:{edge_label}]->(t:Idea) WHERE n.id = $nid RETURN t.id",
            {"nid": b["id"]},
        )
        assert rows == [[a["id"]]]


async def test_record_relates_to_unknown_kind_raises(tools, fake_embed):
    fns, _ = tools
    a = await fns["record"](kind="idea", body="x")
    with pytest.raises(ValueError, match="unknown relates_to kind"):
        await fns["record"](
            kind="idea",
            body="y",
            relates_to=[{"kind": "bogus", "id": a["id"]}],
        )


async def test_record_relates_to_unknown_target_raises(tools, fake_embed):
    fns, _ = tools
    with pytest.raises(ValueError, match="not a memory node"):
        await fns["record"](
            kind="idea",
            body="y",
            relates_to=[{"kind": "supersedes", "id": "no-such-id"}],
        )


# ----------------- amend -----------------


async def test_amend_body_re_embeds(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="idea", body="EFD is good", project_hint="ap")
    res = await fns["amend"](id=out["id"], body="PFD is the renamed concept")
    assert res["re_embedded"] is True
    assert res["amended_at"]

    got = await fns["get_memory"](id=out["id"])
    assert got["body"] == "PFD is the renamed concept"
    # id preserved.
    assert got["id"] == out["id"]

    # Embedding row exists exactly once for this id with the new text.
    rows = await fns["query_memory"](query="x", project_hint="ap")
    matching = [r for r in rows if r["hit"]["id"] == out["id"]]
    assert len(matching) == 1
    assert matching[0]["hit"]["text"] == "PFD is the renamed concept"


async def test_amend_summary_only_no_re_embed(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="idea", body="body unchanged", project_hint="ap")
    call_count_before = fake_embed.call_count
    res = await fns["amend"](id=out["id"], summary="tight title")
    assert res["re_embedded"] is False
    # No new embed call — summary updates don't touch the vector.
    assert fake_embed.call_count == call_count_before

    got = await fns["get_memory"](id=out["id"])
    assert got["summary"] == "tight title"
    assert got["body"] == "body unchanged"
    assert got["retracted_at"] is None


async def test_amend_rejects_retracted(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="idea", body="x")
    await fns["forget"](id=out["id"], reason="r")
    with pytest.raises(ValueError, match="cannot amend retracted"):
        await fns["amend"](id=out["id"], body="new")


async def test_amend_unknown_id_raises(tools):
    fns, _ = tools
    with pytest.raises(ValueError, match="no memory node"):
        await fns["amend"](id="nope", body="x")


async def test_amend_requires_at_least_one_change(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="idea", body="x")
    with pytest.raises(ValueError, match="at least one"):
        await fns["amend"](id=out["id"])


async def test_amend_rejects_oversize_summary(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="idea", body="x")
    with pytest.raises(ValueError, match="summary too long"):
        await fns["amend"](id=out["id"], summary="y" * 201)


async def test_amend_add_anchors(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="decision", body="d", project_hint="ap")
    res = await fns["amend"](
        id=out["id"],
        add_anchors=[{"kind": "file", "repo": "r", "path": "x.py"}],
    )
    labels = [a["target_label"] for a in res["added"]]
    assert "File" in labels

    got = await fns["get_memory"](id=out["id"], include_neighbors=True)
    neighbor_labels = {n["label"] for n in got["neighbors"]}
    assert "File" in neighbor_labels


async def test_amend_remove_anchors(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](
        kind="decision",
        body="d",
        project_hint="ap",
        anchors=[{"kind": "file", "repo": "r", "path": "x.py"}],
    )
    file_anchor = next(a for a in out["anchored"] if a["target_label"] == "File")
    res = await fns["amend"](
        id=out["id"],
        remove_anchors=[
            {"target_label": "File", "target_id": file_anchor["target_id"]}
        ],
    )
    assert res["removed"] == [
        {"target_label": "File", "target_id": file_anchor["target_id"]}
    ]

    got = await fns["get_memory"](id=out["id"], include_neighbors=True)
    neighbor_labels = {n["label"] for n in got["neighbors"]}
    assert "File" not in neighbor_labels


async def test_amend_preserves_project_anchor_through_re_embed(tools, fake_embed):
    fns, _ = tools
    out = await fns["record"](kind="idea", body="orig", project_hint="ap")
    await fns["amend"](id=out["id"], body="revised")
    # project_id must survive the delete-then-insert into LanceDB.
    rows = await fns["query_memory"](query="x", project_hint="ap")
    matching = [r for r in rows if r["hit"]["id"] == out["id"]]
    assert len(matching) == 1
    assert matching[0]["hit"]["project_id"] == out["project_id"]


async def test_amend_episode_preserves_kind_through_re_embed(tools, fake_embed):
    fns, _ = tools
    # `kind="fact"` is an Episode flavor; amend must preserve that sub-kind
    # on the re-inserted embeddings row.
    out = await fns["record"](kind="fact", body="f", project_hint="ap")
    await fns["amend"](id=out["id"], body="f2")
    rows = await fns["query_memory"](query="x", project_hint="ap")
    matching = [r for r in rows if r["hit"]["id"] == out["id"]]
    assert len(matching) == 1
    assert matching[0]["hit"]["kind"] == "fact"


# ----------------- EFD→PFD rename, end-to-end (the original incident) -----------------


async def test_efd_to_pfd_rename_via_amend(tools, fake_embed):
    """The triggering pain point: rename an idea while keeping queries finding it."""
    fns, _ = tools
    out = await fns["record"](kind="idea", body="EFD is good", project_hint="rp")
    await fns["amend"](
        id=out["id"], body="PFD is good", summary="PFD (renamed from EFD)"
    )
    # id preserved.
    got = await fns["get_memory"](id=out["id"])
    assert got["id"] == out["id"]
    assert got["body"] == "PFD is good"
    assert got["summary"] == "PFD (renamed from EFD)"
    # list_memories shows the new summary, not first-of-body.
    listed = await fns["list_memories"](kind="idea", project_hint="rp")
    assert listed[0]["summary"] == "PFD (renamed from EFD)"


async def test_efd_to_pfd_rename_via_supersedes(tools, fake_embed):
    """Alternative path when you want both versions discoverable."""
    fns, _ = tools
    efd = await fns["record"](kind="idea", body="EFD is good", project_hint="rp")
    pfd = await fns["record"](
        kind="idea",
        body="PFD is good",
        project_hint="rp",
        relates_to=[{"kind": "supersedes", "id": efd["id"]}],
    )
    await fns["forget"](id=efd["id"], reason="renamed to PFD")
    # Edge persists across retraction; the graph still shows the lineage.
    rows = await fns["cypher"](
        "MATCH (newer:Idea {id: $nid})-[:SUPERSEDES_MEMORY]->(older:Idea) "
        "RETURN older.id",
        {"nid": pfd["id"]},
    )
    assert rows == [[efd["id"]]]
