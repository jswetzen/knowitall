"""Git history → knowitall graph.

Reads `git log` metadata via subprocess and writes Commit/Person/File nodes
plus AUTHORED/IN_REPO/MODIFIED edges into Kùzu. No source code or diffs are
stored — only commit metadata and file paths.

Idempotent: nodes are deduped by stable keys (sha, email, repo+path); edges
are deduped by their endpoint pair. Re-running on the same repo is a no-op
for already-ingested commits, and incrementally picks up new ones.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kuzu

from server.deps import AppState

# Field separator (\x1f, ASCII Unit Separator) inside a commit header line;
# record separator (\x1e, ASCII Record Separator) at the start of each commit.
# Neither character appears in author names, emails, dates, subjects, or
# filenames in any sane git repo.
_REC = "\x1e"
_FLD = "\x1f"
_GIT_FORMAT = f"{_REC}COMMIT{_FLD}%H{_FLD}%an{_FLD}%ae{_FLD}%aI{_FLD}%s"

_ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})([+-]\d{2}):?(\d{2})$")


def _run_git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _parse_iso(ts: str) -> datetime:
    # git %aI emits e.g. 2026-05-15T13:07:00+02:00 ; datetime.fromisoformat
    # handles this on 3.11+, but normalize defensively.
    m = _ISO_RE.match(ts)
    if m:
        ts = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _resolve_repo(conn: kuzu.Connection, repo_path: Path) -> tuple[str, str | None, str | None]:
    """Return (repo_id, remote_url, default_branch). MERGE the Repo node."""
    try:
        remote_url = _run_git(repo_path, "remote", "get-url", "origin").strip() or None
    except subprocess.CalledProcessError:
        remote_url = None
    try:
        default_branch = _run_git(
            repo_path, "symbolic-ref", "--short", "HEAD"
        ).strip() or None
    except subprocess.CalledProcessError:
        default_branch = None
    abs_path = str(repo_path.resolve())

    # Dedupe by remote_url if available, else by absolute path.
    if remote_url:
        result = conn.execute(
            "MATCH (r:Repo {remote_url: $url}) RETURN r.id LIMIT 1",
            {"url": remote_url},
        )
    else:
        result = conn.execute(
            "MATCH (r:Repo {path: $p}) RETURN r.id LIMIT 1",
            {"p": abs_path},
        )
    if result.has_next():
        return str(result.get_next()[0]), remote_url, default_branch

    repo_id = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Repo {id: $id, path: $path, remote_url: $url, default_branch: $br})",
        {"id": repo_id, "path": abs_path, "url": remote_url, "br": default_branch},
    )
    return repo_id, remote_url, default_branch


def _resolve_project(conn: kuzu.Connection, hint: str | None) -> str | None:
    if not hint:
        return None
    result = conn.execute(
        "MATCH (p:Project {name: $name}) RETURN p.id LIMIT 1",
        {"name": hint},
    )
    if result.has_next():
        return str(result.get_next()[0])
    project_id = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Project {id: $id, name: $name, status: 'active', created_at: $ts})",
        {"id": project_id, "name": hint, "ts": datetime.now(timezone.utc)},
    )
    return project_id


def _link_part_of(conn: kuzu.Connection, repo_id: str, project_id: str, now: datetime) -> None:
    existing = conn.execute(
        "MATCH (r:Repo {id: $rid})-[e:PART_OF]->(p:Project {id: $pid}) RETURN e LIMIT 1",
        {"rid": repo_id, "pid": project_id},
    )
    if existing.has_next():
        return
    conn.execute(
        "MATCH (r:Repo {id: $rid}), (p:Project {id: $pid}) "
        "CREATE (r)-[:PART_OF {valid_from: $now, valid_to: NULL, recorded_at: $now, "
        "source_extractor: 'git', extractor_version: 'v1'}]->(p)",
        {"rid": repo_id, "pid": project_id, "now": now},
    )


def _watermark(conn: kuzu.Connection, repo_id: str) -> datetime | None:
    result = conn.execute(
        "MATCH (c:Commit {repo_id: $rid}) RETURN max(c.authored_at)",
        {"rid": repo_id},
    )
    if result.has_next():
        v = result.get_next()[0]
        if isinstance(v, datetime):
            return v
    return None


def _upsert_person(
    conn: kuzu.Connection, cache: dict[str, tuple[str, bool]], name: str, email: str
) -> tuple[str, bool]:
    """Return (person_id, created)."""
    if email in cache:
        return cache[email][0], False
    result = conn.execute(
        "MATCH (p:Person {email: $e}) RETURN p.id LIMIT 1",
        {"e": email},
    )
    if result.has_next():
        pid = str(result.get_next()[0])
        cache[email] = (pid, False)
        return pid, False
    pid = str(uuid.uuid4())
    conn.execute(
        "CREATE (:Person {id: $id, name: $name, email: $email})",
        {"id": pid, "name": name, "email": email},
    )
    cache[email] = (pid, True)
    return pid, True


def _upsert_file(
    conn: kuzu.Connection,
    cache: dict[tuple[str, str], tuple[str, bool]],
    repo_id: str,
    path: str,
) -> tuple[str, bool]:
    """Return (file_id, created)."""
    key = (repo_id, path)
    if key in cache:
        return cache[key][0], False
    result = conn.execute(
        "MATCH (f:File {repo_id: $rid, path: $p}) RETURN f.id LIMIT 1",
        {"rid": repo_id, "p": path},
    )
    if result.has_next():
        fid = str(result.get_next()[0])
        cache[key] = (fid, False)
        return fid, False
    fid = str(uuid.uuid4())
    conn.execute(
        "CREATE (:File {id: $id, repo_id: $rid, path: $p})",
        {"id": fid, "rid": repo_id, "p": path},
    )
    cache[key] = (fid, True)
    return fid, True


def _create_commit_if_missing(
    conn: kuzu.Connection, sha: str, repo_id: str, message: str, authored_at: datetime
) -> bool:
    result = conn.execute(
        "MATCH (c:Commit {sha: $sha}) RETURN c.sha LIMIT 1",
        {"sha": sha},
    )
    if result.has_next():
        return False
    conn.execute(
        "CREATE (:Commit {sha: $sha, repo_id: $rid, message: $m, authored_at: $ts})",
        {"sha": sha, "rid": repo_id, "m": message, "ts": authored_at},
    )
    return True


def _link_authored(
    conn: kuzu.Connection, person_id: str, sha: str, authored_at: datetime, now: datetime
) -> None:
    existing = conn.execute(
        "MATCH (p:Person {id: $pid})-[e:AUTHORED]->(c:Commit {sha: $sha}) RETURN e LIMIT 1",
        {"pid": person_id, "sha": sha},
    )
    if existing.has_next():
        return
    conn.execute(
        "MATCH (p:Person {id: $pid}), (c:Commit {sha: $sha}) "
        "CREATE (p)-[:AUTHORED {valid_from: $vf, valid_to: NULL, recorded_at: $now, "
        "source_extractor: 'git', extractor_version: 'v1'}]->(c)",
        {"pid": person_id, "sha": sha, "vf": authored_at, "now": now},
    )


def _link_in_repo(
    conn: kuzu.Connection, sha: str, repo_id: str, authored_at: datetime, now: datetime
) -> None:
    existing = conn.execute(
        "MATCH (c:Commit {sha: $sha})-[e:IN_REPO]->(r:Repo {id: $rid}) RETURN e LIMIT 1",
        {"sha": sha, "rid": repo_id},
    )
    if existing.has_next():
        return
    conn.execute(
        "MATCH (c:Commit {sha: $sha}), (r:Repo {id: $rid}) "
        "CREATE (c)-[:IN_REPO {valid_from: $vf, valid_to: NULL, recorded_at: $now, "
        "source_extractor: 'git', extractor_version: 'v1'}]->(r)",
        {"sha": sha, "rid": repo_id, "vf": authored_at, "now": now},
    )


def _link_modified(
    conn: kuzu.Connection, sha: str, file_id: str, authored_at: datetime, now: datetime
) -> None:
    existing = conn.execute(
        "MATCH (c:Commit {sha: $sha})-[e:MODIFIED]->(f:File {id: $fid}) RETURN e LIMIT 1",
        {"sha": sha, "fid": file_id},
    )
    if existing.has_next():
        return
    conn.execute(
        "MATCH (c:Commit {sha: $sha}), (f:File {id: $fid}) "
        "CREATE (c)-[:MODIFIED {valid_from: $vf, valid_to: NULL, recorded_at: $now, "
        "source_extractor: 'git', extractor_version: 'v1'}]->(f)",
        {"sha": sha, "fid": file_id, "vf": authored_at, "now": now},
    )


def _git_log(repo_path: Path, since: datetime | None) -> str:
    args = [
        "log",
        "--no-merges",
        f"--pretty=format:{_GIT_FORMAT}",
        "--name-only",
    ]
    if since is not None:
        # +1 second to exclude the watermark commit itself.
        args.append(f"--since={since.isoformat()}")
    return _run_git(repo_path, *args)


def ingest_repo(
    state: AppState, path: str, project_hint: str | None = None
) -> dict[str, Any]:
    repo_path = Path(path)
    if not (repo_path / ".git").exists():
        # Allow bare repos and worktrees; fall back to `git rev-parse`.
        try:
            _run_git(repo_path, "rev-parse", "--git-dir")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"not a git repository: {path}") from e

    conn = state.kuzu_conn()
    now = datetime.now(timezone.utc)

    repo_id, _remote, _branch = _resolve_repo(conn, repo_path)
    project_id = _resolve_project(conn, project_hint)
    if project_id:
        _link_part_of(conn, repo_id, project_id, now)

    since = _watermark(conn, repo_id)
    raw = _git_log(repo_path, since)
    chunks = [c for c in raw.split(_REC) if c.strip()]

    person_cache: dict[str, tuple[str, bool]] = {}
    file_cache: dict[tuple[str, str], tuple[str, bool]] = {}
    commits_added = 0
    persons_added = 0
    files_added = 0

    for chunk in chunks:
        lines = chunk.split("\n")
        header = lines[0]
        if not header.startswith("COMMIT" + _FLD):
            continue
        _, sha, author_name, author_email, authored_iso, subject = header.split(
            _FLD, 5
        )
        authored_at = _parse_iso(authored_iso)
        file_paths = [ln for ln in lines[1:] if ln.strip()]

        person_id, p_created = _upsert_person(conn, person_cache, author_name, author_email)
        if p_created:
            persons_added += 1

        if _create_commit_if_missing(conn, sha, repo_id, subject, authored_at):
            commits_added += 1

        _link_authored(conn, person_id, sha, authored_at, now)
        _link_in_repo(conn, sha, repo_id, authored_at, now)

        for p in file_paths:
            file_id, f_created = _upsert_file(conn, file_cache, repo_id, p)
            if f_created:
                files_added += 1
            _link_modified(conn, sha, file_id, authored_at, now)

    return {
        "repo_id": repo_id,
        "project_id": project_id,
        "commits_added": commits_added,
        "files_added": files_added,
        "persons_added": persons_added,
    }
