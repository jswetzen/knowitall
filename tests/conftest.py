from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KNOWITALL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("KNOWITALL_TOKEN", "test-token")
    # Reload settings so the env vars take effect for any newly-built state.
    from server import config as cfg

    cfg.settings = cfg.Settings()
    return tmp_path


def _ollama_reachable() -> bool:
    import httpx

    url = os.environ.get("KNOWITALL_OLLAMA_URL", "http://192.168.1.33:11434")
    try:
        r = httpx.get(f"{url}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(), reason="Ollama not reachable"
)
