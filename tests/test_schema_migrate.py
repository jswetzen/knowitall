"""Upgrade-path tests for schema migrations.

The regular test suite always starts from a fresh DB, so `apply_migrations`
runs all of v1+v2+v3 in one shot — that exercises "fresh install" but not
"upgrade an existing DB." This file does the harder thing: build a DB with
only v1+v2 applied, populate it with rows in every memory label, then
re-open with v3 in place and confirm:

  - the v3 migration applies without errors,
  - pre-existing rows survive,
  - the new columns default cleanly (NULL) on legacy rows,
  - new rows recorded after the upgrade can use the new columns.

This is the safety net for the real-data upgrade we don't simulate in CI.
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import kuzu
import pytest

from schema import migrate as migrate_module


def _stage_schemas(staging_dir: Path, versions: list[int]) -> None:
    """Copy only the requested vN.cypher files into staging_dir."""
    real_schema_dir = Path(migrate_module.__file__).parent
    for v in versions:
        src = real_schema_dir / f"v{v}.cypher"
        shutil.copy(src, staging_dir / src.name)


def _all_schema_versions() -> list[int]:
    """Every vN.cypher that exists, so upgrade tests don't pin to a version.

    A real deploy runs `apply_migrations`, which applies everything pending —
    not just the version that happened to be newest when a test was written.
    Staging a fixed list made "upgrade to current" silently mean "upgrade to
    v3" as the schema moved on.
    """
    real_schema_dir = Path(migrate_module.__file__).parent
    return sorted(
        int(m.group(1))
        for p in real_schema_dir.glob("v*.cypher")
        if (m := migrate_module.MIGRATION_RE.match(p.name))
    )


@pytest.fixture
def staged_schema_dir(tmp_path, monkeypatch):
    """Returns a Path that callers populate with selected vN.cypher files.

    Monkeypatches `schema.migrate.SCHEMA_DIR` to point at it so
    `apply_migrations` only sees what's staged.
    """
    staging = tmp_path / "schemas"
    staging.mkdir()
    monkeypatch.setattr(migrate_module, "SCHEMA_DIR", staging)
    return staging


def _seed_v1_v2_rows(conn: kuzu.Connection) -> dict[str, str]:
    """Populate one row per memory label using only v1+v2 columns. Returns
    a {label: id} map so the upgrade test can assert these survive."""
    now = datetime.now(timezone.utc)
    ids: dict[str, str] = {}

    ids["Decision"] = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Decision {id: $id, body: 'pre-v3 decision', decided_at: $t, "
        "retracted_at: NULL})",
        {"id": ids["Decision"], "t": now},
    )

    ids["Task"] = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Task {id: $id, body: 'pre-v3 task', status: 'open', "
        "created_at: $t, closed_at: NULL, retracted_at: NULL})",
        {"id": ids["Task"], "t": now},
    )

    ids["Idea"] = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Idea {id: $id, body: 'pre-v3 idea', status: 'open', "
        "created_at: $t, died_at: NULL, retracted_at: NULL})",
        {"id": ids["Idea"], "t": now},
    )

    ids["Note"] = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Note {id: $id, path: NULL, title: 'pre-v3 note', "
        "created_at: $t, retracted_at: NULL})",
        {"id": ids["Note"], "t": now},
    )

    ids["Episode"] = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Episode {id: $id, body: 'pre-v3 episode', kind: 'fact', "
        "created_at: $t, model_version: 'test', retracted_at: NULL})",
        {"id": ids["Episode"], "t": now},
    )

    # An ANCHORED_TO edge too — exercises that v3's new edges don't disturb
    # the existing edge graph.
    pid = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Project {id: $id, name: 'pre-v3 proj', status: 'active', created_at: $t})",
        {"id": pid, "t": now},
    )
    conn.execute(
        "MATCH (i:Idea {id: $iid}), (p:Project {id: $pid}) "
        "CREATE (i)-[:ANCHORED_TO {valid_from: $t, valid_to: NULL, "
        "recorded_at: $t, source_extractor: 'test', extractor_version: 'v2'}]->(p)",
        {"iid": ids["Idea"], "pid": pid, "t": now},
    )
    ids["_project"] = pid
    return ids


def test_v3_upgrade_preserves_v1_v2_rows(staged_schema_dir, tmp_path):
    """Build a DB with only v1+v2, populate it, then apply v3 and verify."""
    # --- Phase 1: v1 + v2 only ---
    _stage_schemas(staged_schema_dir, [0, 1, 2])
    db_path = tmp_path / "kuzu"
    db = kuzu.Database(str(db_path))
    applied = migrate_module.apply_migrations(db)
    assert applied == [0, 1, 2]

    conn = kuzu.Connection(db)
    ids = _seed_v1_v2_rows(conn)
    # Drop the handle so the next phase can reopen cleanly. (Kuzu is fine
    # with a single Database per process; we re-use it below.)
    del conn

    # --- Phase 2: stage v3 alongside, re-run migrations ---
    _stage_schemas(staged_schema_dir, [3])
    applied = migrate_module.apply_migrations(db)
    assert applied == [3]

    conn = kuzu.Connection(db)

    # Pre-existing rows survive, with new columns defaulting to NULL.
    for label, expected_body_or_title in [
        ("Decision", "pre-v3 decision"),
        ("Task", "pre-v3 task"),
        ("Idea", "pre-v3 idea"),
        ("Episode", "pre-v3 episode"),
    ]:
        result = conn.execute(
            f"MATCH (n:{label} {{id: $id}}) "
            f"RETURN n.body, n.summary, n.amended_at",
            {"id": ids[label]},
        )
        body, summary, amended_at = result.get_next()
        assert body == expected_body_or_title, f"{label} body lost during upgrade"
        assert summary is None, f"{label}.summary should default to NULL on legacy rows"
        assert amended_at is None, f"{label}.amended_at should default to NULL"

    # Note has no summary column — only amended_at is new.
    result = conn.execute(
        "MATCH (n:Note {id: $id}) RETURN n.title, n.amended_at",
        {"id": ids["Note"]},
    )
    title, amended_at = result.get_next()
    assert title == "pre-v3 note"
    assert amended_at is None

    # The pre-existing ANCHORED_TO edge survives.
    result = conn.execute(
        "MATCH (i:Idea {id: $iid})-[:ANCHORED_TO]->(p:Project) RETURN p.id",
        {"iid": ids["Idea"]},
    )
    assert result.has_next()
    assert str(result.get_next()[0]) == ids["_project"]


def test_v3_upgrade_then_new_columns_usable(staged_schema_dir, tmp_path):
    """After upgrade, new rows can populate the new columns and new edges."""
    _stage_schemas(staged_schema_dir, [0, 1, 2])
    db_path = tmp_path / "kuzu"
    db = kuzu.Database(str(db_path))
    migrate_module.apply_migrations(db)
    conn = kuzu.Connection(db)
    legacy_idea = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    conn.execute(
        "CREATE (:Idea {id: $id, body: 'legacy', status: 'open', "
        "created_at: $t, died_at: NULL, retracted_at: NULL})",
        {"id": legacy_idea, "t": now},
    )

    _stage_schemas(staged_schema_dir, [3])
    migrate_module.apply_migrations(db)
    conn = kuzu.Connection(db)

    # New row with summary + amended_at populated.
    new_idea = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Idea {id: $id, body: 'post-v3 body', summary: 'post-v3 summary', "
        "status: 'open', created_at: $t, died_at: NULL, retracted_at: NULL, "
        "amended_at: $t})",
        {"id": new_idea, "t": now},
    )

    # Write a SUPERSEDES_MEMORY edge from the new idea to the legacy one.
    conn.execute(
        "MATCH (newer:Idea {id: $nid}), (older:Idea {id: $oid}) "
        "CREATE (newer)-[:SUPERSEDES_MEMORY {valid_from: $t, valid_to: NULL, "
        "recorded_at: $t, source_extractor: 'test', extractor_version: 'v3'}]->(older)",
        {"nid": new_idea, "oid": legacy_idea, "t": now},
    )

    # Verify the legacy idea — which has NULL in the new columns — still
    # participates in the new edge type without complaint.
    result = conn.execute(
        "MATCH (newer:Idea)-[:SUPERSEDES_MEMORY]->(older:Idea) "
        "WHERE older.id = $oid RETURN newer.id, newer.summary, older.summary",
        {"oid": legacy_idea},
    )
    nid, new_summary, old_summary = result.get_next()
    assert str(nid) == new_idea
    assert new_summary == "post-v3 summary"
    assert old_summary is None  # legacy row, NULL in the new column


def test_v4_drops_idea_status_and_died_at(staged_schema_dir, tmp_path):
    """v4 retires Idea.status/died_at — written once as 'open' at creation
    and never transitioned by any tool. Confirm the columns are gone and
    the rest of a legacy Idea row (created under v0-v3) survives."""
    _stage_schemas(staged_schema_dir, [0, 1, 2, 3])
    db_path = tmp_path / "kuzu"
    db = kuzu.Database(str(db_path))
    applied = migrate_module.apply_migrations(db)
    assert applied == [0, 1, 2, 3]

    conn = kuzu.Connection(db)
    legacy_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    conn.execute(
        "CREATE (:Idea {id: $id, body: 'legacy idea', status: 'open', "
        "created_at: $t, died_at: NULL, retracted_at: NULL, summary: NULL, "
        "amended_at: NULL})",
        {"id": legacy_id, "t": now},
    )
    del conn

    _stage_schemas(staged_schema_dir, [4])
    applied = migrate_module.apply_migrations(db)
    assert applied == [4]

    conn = kuzu.Connection(db)
    result = conn.execute(
        "MATCH (i:Idea {id: $id}) RETURN i.body, i.created_at, i.retracted_at",
        {"id": legacy_id},
    )
    assert result.has_next()
    body, created_at, retracted_at = result.get_next()
    assert body == "legacy idea"
    assert created_at is not None
    assert retracted_at is None

    # status/died_at no longer exist as columns.
    with pytest.raises(RuntimeError, match="status"):
        conn.execute("MATCH (i:Idea {id: $id}) RETURN i.status", {"id": legacy_id})
    with pytest.raises(RuntimeError, match="died_at"):
        conn.execute("MATCH (i:Idea {id: $id}) RETURN i.died_at", {"id": legacy_id})


def test_v3_upgrade_then_record_amend_via_mcp(
    staged_schema_dir, tmp_path, monkeypatch
):
    """End-to-end: stage v1+v2 DB, seed a legacy row, upgrade to v3, then use
    the MCP `amend` tool to revise the legacy row. This is the closest CI
    proxy for the real upgrade scenario — a user's existing memories getting
    revised after a deploy."""
    # Phase 1: v1+v2 only, seed a legacy idea.
    _stage_schemas(staged_schema_dir, [0, 1, 2])
    monkeypatch.setenv("KNOWITALL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("KNOWITALL_TOKEN", "test-token")
    from server import config as cfg
    cfg.settings = cfg.Settings()
    from server.deps import build_state
    from server.mcp_tools import register_tools

    state = build_state()
    # A Decision, not an Idea: v4 DROPs Idea.status/died_at, and amending a
    # row that predates a dropped column segfaults Kuzu 0.11.3 on checkpoint
    # (see test_amending_a_row_that_predates_v4s_drop_survives_checkpoint).
    # Decision is untouched by v4, so this test exercises what it's actually
    # for — a legacy row revised after a full upgrade — without riding on
    # that bug. Legacy-Idea survival through v4 is covered by
    # test_v4_drops_idea_status_and_died_at.
    legacy_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    conn = state.kuzu_conn()
    conn.execute(
        "CREATE (:Decision {id: $id, body: 'legacy body', decided_at: $t, "
        "retracted_at: NULL})",
        {"id": legacy_id, "t": now},
    )
    # Close before reopening the same path in phase 2 rather than leaning on
    # refcounting to do it at the right moment.
    conn.close()
    state.kuzu_db.close()
    del state, conn

    # Phase 2: stage every remaining migration (what a deploy actually
    # applies), rebuild state (which triggers migration).
    _stage_schemas(staged_schema_dir, _all_schema_versions())
    state = build_state()

    class _Recorder:
        def __init__(self):
            self.tools = {}
        def tool(self, *a, **kw):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    rec = _Recorder()
    register_tools(rec, state)

    # The legacy row is addressable via get_memory. Drive the async tool
    # explicitly with asyncio.run — this test isn't async itself because the
    # migration phases need to happen synchronously between fixtures.
    import asyncio
    got = asyncio.run(rec.tools["get_memory"](id=legacy_id))
    assert got is not None
    assert got["body"] == "legacy body"
    assert got["summary"] == "legacy body"  # falls back to body since no stored summary

    # Amend the legacy row — must work even though its summary/amended_at started NULL.
    vec = [0.01 * (i % 100) for i in range(768)]
    with patch("server.mcp_tools.embed", new=AsyncMock(return_value=vec)):
        res = asyncio.run(
            rec.tools["amend"](
                id=legacy_id,
                body="revised body",
                summary="revised summary",
            )
        )
    assert res["re_embedded"] is True

    got = asyncio.run(rec.tools["get_memory"](id=legacy_id))
    assert got["body"] == "revised body"
    assert got["summary"] == "revised summary"


# --- known Kuzu bug: updating a row that predates a dropped column ---

# Runs in a subprocess: the failure mode is SIGSEGV inside Kuzu's checkpoint,
# which pytest cannot trap in-process — it would take the whole session down.
_DROPPED_COLUMN_REPRO = """
import shutil, sys, uuid
from datetime import datetime, timezone
from pathlib import Path
import kuzu
from schema import migrate as migrate_module

mode, tmp = sys.argv[1], Path(sys.argv[2])
real = Path(migrate_module.__file__).parent
staging = tmp / "schemas"
staging.mkdir()
migrate_module.SCHEMA_DIR = staging


def stage(versions):
    for v in versions:
        shutil.copy(real / f"v{v}.cypher", staging / f"v{v}.cypher")


now = datetime.now(timezone.utc)
stage([0, 1, 2, 3])
db = kuzu.Database(str(tmp / "kuzu"))
conn = kuzu.Connection(db)
migrate_module.apply_migrations(db)
row_id = str(uuid.uuid4())
LEGACY = (
    "CREATE (:Idea {id: $id, body: 'legacy', status: 'open', created_at: $t, "
    "died_at: NULL, retracted_at: NULL, summary: NULL, amended_at: NULL})"
)
POST_DROP = (
    "CREATE (:Idea {id: $id, body: 'fresh', created_at: $t, "
    "retracted_at: NULL, summary: NULL, amended_at: NULL})"
)
if mode == "legacy":
    conn.execute(LEGACY, {"id": row_id, "t": now})
conn.close()
db.close()
del conn, db

stage([4] if mode != "no_drop" else [])
db = kuzu.Database(str(tmp / "kuzu"))
conn = kuzu.Connection(db)
migrate_module.apply_migrations(db)
if mode == "post_drop":
    conn.execute(POST_DROP, {"id": row_id, "t": now})
elif mode == "no_drop":
    conn.execute(LEGACY, {"id": row_id, "t": now})
conn.execute(
    "MATCH (n:Idea {id: $id}) SET n.body = 'revised', n.amended_at = $t",
    {"id": row_id, "t": now},
)
# Exiting here destroys the Database, which checkpoints. That is where the
# segfault lands for mode="legacy".
"""


def _run_dropped_column_repro(mode: str, tmp_path: Path) -> int:
    """Run the repro in a clean interpreter. Returns its exit code."""
    import subprocess
    import sys

    workdir = tmp_path / mode
    workdir.mkdir()
    repo_root = Path(migrate_module.__file__).parent.parent
    return subprocess.run(
        [sys.executable, "-c", _DROPPED_COLUMN_REPRO, mode, str(workdir)],
        cwd=repo_root,
        capture_output=True,
    ).returncode


@pytest.mark.parametrize("mode", ["post_drop", "no_drop"])
def test_updating_a_row_not_predating_v4s_drop_checkpoints_cleanly(mode, tmp_path):
    """Control cases for the xfail below — these must stay clean."""
    assert _run_dropped_column_repro(mode, tmp_path) == 0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Kuzu 0.11.3 segfaults in ChunkedNodeGroup::scanCommitted during "
        "checkpoint when a row created BEFORE an ALTER TABLE ... DROP is "
        "updated afterwards. v4 drops Idea.status/died_at, so every Idea "
        "recorded before v4 shipped is affected: amending one crashes the "
        "process on close. Only Idea is affected (v4 is the only DROP). "
        "No upstream fix exists or is coming — kuzudb/kuzu was archived on "
        "2025-10-10 and 0.11.3 (2025-10-10) is the final release, with no "
        "matching issue filed before the archive. A CHECKPOINT after the "
        "DROP fixes this on toy databases but hangs indefinitely on a "
        "real one, so it is not a usable remedy — see docs/kuzu-drop-"
        "column-hazard.md. Mitigation for now is operational: don't "
        "amend a pre-v4 Idea in the same process that applied v4."
    ),
)
def test_updating_a_row_predating_v4s_drop_checkpoints_cleanly(tmp_path):
    assert _run_dropped_column_repro("legacy", tmp_path) == 0
