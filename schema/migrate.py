from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import kuzu

SCHEMA_DIR = Path(__file__).parent
MIGRATION_RE = re.compile(r"^v(\d+)\.cypher$")


def _ensure_migrations_table(conn: kuzu.Connection) -> None:
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS _migrations("
        "version INT64 PRIMARY KEY, applied_at TIMESTAMP)"
    )


def _applied_versions(conn: kuzu.Connection) -> set[int]:
    result = conn.execute("MATCH (m:_migrations) RETURN m.version")
    return {int(row[0]) for row in result}


def _split_statements(text: str) -> list[str]:
    # Strip // line comments, split on ';'. Naive but week-1 sufficient.
    cleaned = "\n".join(
        line.split("//", 1)[0] for line in text.splitlines()
    )
    return [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]


def apply_migrations(db: kuzu.Database) -> list[int]:
    conn = kuzu.Connection(db)
    _ensure_migrations_table(conn)
    applied = _applied_versions(conn)
    newly_applied: list[int] = []
    migrations = sorted(
        (int(m.group(1)), p)
        for p in SCHEMA_DIR.glob("v*.cypher")
        if (m := MIGRATION_RE.match(p.name))
    )
    for version, path in migrations:
        if version in applied:
            continue
        for stmt in _split_statements(path.read_text()):
            conn.execute(stmt)
        conn.execute(
            "CREATE (:_migrations {version: $v, applied_at: $ts})",
            {"v": version, "ts": datetime.now(timezone.utc)},
        )
        newly_applied.append(version)
    return newly_applied
