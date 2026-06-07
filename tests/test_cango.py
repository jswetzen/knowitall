"""Tests for the cango calendar shims.

These never touch a real cango-daemon. A tiny in-process asyncio Unix-socket
server speaks the same length-prefixed JSON-RPC framing the daemon uses
(packages/daemon/src/framing.ts), so we exercise the real wire path: connect,
encode a frame, decode the reply, map errors.
"""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import pytest

from server import config as cfg
from server.cango import register_cango_tools


class _Recorder:
    """Same shape as the recorder in test_mcp_tools: capture @mcp.tool() fns."""

    def __init__(self):
        self.tools: dict = {}

    def tool(self, *args, **kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


class FakeDaemon:
    """A length-prefixed JSON-RPC server over a Unix socket.

    `responder(method, params) -> dict` produces the JSON-RPC `result` (or, if
    it returns something under an `__error__` key, an RPC error). It also
    records every request it received for assertions.
    """

    def __init__(self, socket_path: str, responder):
        self.socket_path = socket_path
        self.responder = responder
        self.requests: list[dict] = []
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(
            self._handle, path=self.socket_path
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        header = await reader.readexactly(4)
        (length,) = struct.unpack(">I", header)
        body = await reader.readexactly(length)
        request = json.loads(body.decode("utf-8"))
        self.requests.append(request)

        outcome = self.responder(request["method"], request.get("params", {}))
        if isinstance(outcome, dict) and "__error__" in outcome:
            payload = {"jsonrpc": "2.0", "id": request["id"], "error": outcome["__error__"]}
        else:
            payload = {"jsonrpc": "2.0", "id": request["id"], "result": outcome}

        out = json.dumps(payload).encode("utf-8")
        writer.write(struct.pack(">I", len(out)) + out)
        await writer.drain()
        writer.close()


@pytest.fixture
def tools():
    recorder = _Recorder()
    register_cango_tools(recorder)
    return recorder.tools


@pytest.fixture
def socket_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    path = str(tmp_path / "cango.sock")
    monkeypatch.setattr(cfg.settings, "cango_socket", path)
    return path


async def _with_daemon(socket_path: str, responder) -> FakeDaemon:
    daemon = FakeDaemon(socket_path, responder)
    await daemon.start()
    return daemon


# ----------------- happy path: verdict round-trip -----------------


async def test_check_availability_roundtrip(tools, socket_path):
    verdict = {
        "verdict": "hard_conflict",
        "conflicts": [
            {
                "person": {"id": "wife", "name": "Wife"},
                "event": {"id": "e1", "title": "Shift", "resolved_role": "hard"},
                "overlap_minutes": 60,
            }
        ],
        "degraded": False,
        "stale_sources": [],
    }
    daemon = await _with_daemon(socket_path, lambda m, p: verdict)
    try:
        result = await tools["check_availability"](
            start="2026-06-14T09:00:00Z",
            end="2026-06-14T11:00:00Z",
            people=["wife"],
        )
    finally:
        await daemon.stop()

    assert result == verdict
    # Method name and params reached the daemon as the daemon's schema expects.
    assert len(daemon.requests) == 1
    req = daemon.requests[0]
    assert req["method"] == "checkAvailability"
    assert req["jsonrpc"] == "2.0"
    assert req["params"] == {
        "start": "2026-06-14T09:00:00Z",
        "end": "2026-06-14T11:00:00Z",
        "people": ["wife"],
    }


async def test_none_params_are_filtered(tools, socket_path):
    """Optional args left at None must not reach the daemon as explicit nulls."""
    daemon = await _with_daemon(socket_path, lambda m, p: {"verdict": "free"})
    try:
        await tools["check_availability"](
            start="2026-06-14T09:00:00Z", end="2026-06-14T11:00:00Z"
        )
    finally:
        await daemon.stop()

    assert daemon.requests[0]["params"] == {
        "start": "2026-06-14T09:00:00Z",
        "end": "2026-06-14T11:00:00Z",
    }
    assert "people" not in daemon.requests[0]["params"]


async def test_find_free_slot_nests_between(tools, socket_path):
    daemon = await _with_daemon(socket_path, lambda m, p: {"slots": []})
    try:
        await tools["find_free_slot"](
            duration_minutes=90,
            between_start="2026-06-20T00:00:00Z",
            between_end="2026-06-21T00:00:00Z",
            working_hours={"start": "09:00", "end": "17:00"},
        )
    finally:
        await daemon.stop()

    params = daemon.requests[0]["params"]
    assert daemon.requests[0]["method"] == "findFreeSlot"
    assert params["duration_minutes"] == 90
    assert params["between"] == {
        "start": "2026-06-20T00:00:00Z",
        "end": "2026-06-21T00:00:00Z",
    }
    assert params["working_hours"] == {"start": "09:00", "end": "17:00"}
    assert "people" not in params  # None filtered


async def test_create_event_roundtrip(tools, socket_path):
    created = {
        "event": {
            "id": "evt-1",
            "source_id": "src-cal",
            "title": "Torpkonferensen",
            "start": "2026-06-16T00:00:00.000Z",
            "end": "2026-06-21T00:00:00.000Z",
            "all_day": True,
            "resolved_role": "hard",
        },
        "degraded": False,
        "stale_sources": [],
    }
    daemon = await _with_daemon(socket_path, lambda m, p: created)
    try:
        result = await tools["create_event"](
            source_id="src-cal",
            title="Torpkonferensen",
            start="2026-06-16T00:00:00Z",
            end="2026-06-21T00:00:00Z",
            all_day=True,
        )
    finally:
        await daemon.stop()

    assert result == created
    req = daemon.requests[0]
    assert req["method"] == "createEvent"
    assert req["params"] == {
        "source_id": "src-cal",
        "title": "Torpkonferensen",
        "start": "2026-06-16T00:00:00Z",
        "end": "2026-06-21T00:00:00Z",
        "all_day": True,
    }


async def test_create_event_not_writable_maps_to_unavailable(tools, socket_path):
    """The daemon rejects writes to a non-writable source; surface it structured."""

    def responder(method, params):
        return {"__error__": {"code": -32005, "message": "source not writable: src-ro"}}

    daemon = await _with_daemon(socket_path, responder)
    try:
        result = await tools["create_event"](
            source_id="src-ro",
            title="Nope",
            start="2026-06-16T09:00:00Z",
            end="2026-06-16T10:00:00Z",
        )
    finally:
        await daemon.stop()

    assert result["error"] == "cango_unavailable"
    assert "not writable" in result["reason"]


async def test_explain_event_and_list_series_method_names(tools, socket_path):
    seen: list[str] = []

    def responder(method, params):
        seen.append(method)
        return {"ok": True}

    daemon = await _with_daemon(socket_path, responder)
    try:
        await tools["explain_event"](event_id="e1")
        await tools["list_series"](source_id="src1")
        await tools["list_events"](start="2026-06-01T00:00:00Z", end="2026-06-02T00:00:00Z")
    finally:
        await daemon.stop()

    assert seen == ["explainEvent", "listSeries", "listEvents"]


async def test_list_events_forwards_compact_params(tools, socket_path):
    """Default call is compact (extended=False); unset optionals are filtered."""
    daemon = await _with_daemon(socket_path, lambda m, p: {"events": []})
    try:
        await tools["list_events"](
            start="2026-06-01T00:00:00Z", end="2026-06-02T00:00:00Z"
        )
    finally:
        await daemon.stop()

    params = daemon.requests[0]["params"]
    assert params == {
        "start": "2026-06-01T00:00:00Z",
        "end": "2026-06-02T00:00:00Z",
        "extended": False,
    }
    # None-valued optionals never reach the daemon's schema.
    for k in ("people", "exclude_roles", "limit", "offset"):
        assert k not in params


async def test_list_events_forwards_all_params(tools, socket_path):
    daemon = await _with_daemon(socket_path, lambda m, p: {"events": []})
    try:
        await tools["list_events"](
            start="2026-06-01T00:00:00Z",
            end="2026-07-01T00:00:00Z",
            people=["johan"],
            extended=True,
            exclude_roles=["soft"],
            limit=50,
            offset=10,
        )
    finally:
        await daemon.stop()

    assert daemon.requests[0]["params"] == {
        "start": "2026-06-01T00:00:00Z",
        "end": "2026-07-01T00:00:00Z",
        "people": ["johan"],
        "extended": True,
        "exclude_roles": ["soft"],
        "limit": 50,
        "offset": 10,
    }


# ----------------- failure modes -----------------


async def test_daemon_down_returns_structured_error(tools, socket_path):
    """No daemon listening: every shim returns cango_unavailable, never raises."""
    result = await tools["check_availability"](
        start="2026-06-14T09:00:00Z", end="2026-06-14T11:00:00Z"
    )
    assert result["error"] == "cango_unavailable"
    assert socket_path in result["reason"]


async def test_rpc_error_maps_to_unavailable(tools, socket_path):
    """A daemon-level RPC error (e.g. unknown event id) becomes cango_unavailable."""

    def responder(method, params):
        return {"__error__": {"code": -32004, "message": "event not found: e9"}}

    daemon = await _with_daemon(socket_path, responder)
    try:
        result = await tools["explain_event"](event_id="e9")
    finally:
        await daemon.stop()

    assert result["error"] == "cango_unavailable"
    assert "event not found: e9" in result["reason"]


async def test_only_expected_tools_registered(tools):
    """Read shims + the create_event write shim. reloadConfig/health excluded."""
    assert set(tools) == {
        "check_availability",
        "find_free_slot",
        "list_events",
        "explain_event",
        "list_series",
        "create_event",
    }
