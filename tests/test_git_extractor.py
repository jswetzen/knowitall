from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "alice@example.com")
    _git(repo, "config", "user.name", "Alice")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "a.txt").write_text("alpha\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "add a")

    (repo / "b.txt").write_text("bravo\n")
    (repo / "a.txt").write_text("alpha v2\n")
    _git(repo, "add", "a.txt", "b.txt")
    _git(repo, "commit", "-q", "-m", "add b, bump a")

    _git(repo, "config", "user.email", "bob@example.com")
    _git(repo, "config", "user.name", "Bob")
    (repo / "c.txt").write_text("charlie\n")
    _git(repo, "add", "c.txt")
    _git(repo, "commit", "-q", "-m", "add c")

    return repo


def test_extractor_basic(synthetic_repo, isolated_data_dir):
    from ingest.git_extractor import ingest_repo
    from server.deps import build_state

    state = build_state()
    result = ingest_repo(state, str(synthetic_repo), project_hint="syntest")

    assert result["commits_added"] == 3
    assert result["persons_added"] == 2  # alice + bob
    assert result["files_added"] == 3  # a, b, c
    assert result["project_id"] is not None

    conn = state.kuzu_conn()
    assert conn.execute("MATCH (c:Commit) RETURN count(c)").get_next()[0] == 3
    assert conn.execute("MATCH (p:Person) RETURN count(p)").get_next()[0] == 2
    assert conn.execute("MATCH (f:File) RETURN count(f)").get_next()[0] == 3
    assert conn.execute("MATCH ()-[e:AUTHORED]->() RETURN count(e)").get_next()[0] == 3
    assert conn.execute("MATCH ()-[e:IN_REPO]->() RETURN count(e)").get_next()[0] == 3
    # 1 file in c1, 2 in c2, 1 in c3 = 4 MODIFIED edges
    assert conn.execute("MATCH ()-[e:MODIFIED]->() RETURN count(e)").get_next()[0] == 4
    assert conn.execute("MATCH ()-[e:PART_OF]->() RETURN count(e)").get_next()[0] == 1


def test_extractor_idempotent(synthetic_repo, isolated_data_dir):
    from ingest.git_extractor import ingest_repo
    from server.deps import build_state

    state = build_state()
    r1 = ingest_repo(state, str(synthetic_repo), project_hint="syntest")
    r2 = ingest_repo(state, str(synthetic_repo), project_hint="syntest")

    assert r1["repo_id"] == r2["repo_id"]
    assert r1["project_id"] == r2["project_id"]
    assert r2["commits_added"] == 0
    assert r2["files_added"] == 0
    assert r2["persons_added"] == 0


def test_extractor_picks_up_new_commits(synthetic_repo, isolated_data_dir):
    from ingest.git_extractor import ingest_repo
    from server.deps import build_state

    state = build_state()
    ingest_repo(state, str(synthetic_repo), project_hint="syntest")

    (synthetic_repo / "d.txt").write_text("delta\n")
    _git(synthetic_repo, "add", "d.txt")
    _git(synthetic_repo, "commit", "-q", "-m", "add d")

    r = ingest_repo(state, str(synthetic_repo), project_hint="syntest")
    assert r["commits_added"] == 1
    assert r["files_added"] == 1
    assert r["persons_added"] == 0  # bob already exists


def test_extractor_rejects_non_repo(tmp_path, isolated_data_dir):
    from ingest.git_extractor import ingest_repo
    from server.deps import build_state

    state = build_state()
    with pytest.raises(ValueError, match="not a git repository"):
        ingest_repo(state, str(tmp_path), project_hint="nope")
