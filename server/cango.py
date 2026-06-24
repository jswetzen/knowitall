"""Cango calendar shims.

knowitall is the only MCP surface family agents discover. The calendar logic
lives in a sibling Bun process, `cango-daemon`, reachable over a Unix socket.
These tools are thin shims: they marshal args into a JSON-RPC request, send it
over the socket, and return the daemon's response verbatim.

No calendar logic lives here. No `family.yaml` is read here. The boundary is
the socket: if the daemon is down or the socket is missing, every shim returns
a structured ``{"error": "cango_unavailable", "reason": ...}`` and knowitall's
memory tools keep working regardless.

Wire protocol (matches packages/daemon/src/framing.ts in the calendar-assistant
repo): length-prefixed JSON-RPC 2.0. Each frame is a 4-byte big-endian uint32
byte length followed by that many bytes of UTF-8 JSON.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

from mcp.server.fastmcp import FastMCP

from server import config

# Mirror the daemon's FrameDecoder cap (16 MiB) so a corrupt/hostile length
# prefix can't make us allocate unboundedly.
_MAX_FRAME_BYTES = 16 * 1024 * 1024

# Connecting and the full request/response round-trip each get a budget. The
# daemon serves from a warm SQLite cache, so these are generous.
_CONNECT_TIMEOUT_S = 2.0
_RPC_TIMEOUT_S = 10.0


def _unavailable(reason: str) -> dict[str, Any]:
    """The structured signal every shim returns when the daemon can't answer."""
    return {"error": "cango_unavailable", "reason": reason}


async def _read_frame(reader: asyncio.StreamReader) -> Any:
    """Read one length-prefixed JSON frame. Raises on EOF or oversize frame."""
    header = await reader.readexactly(4)
    (length,) = struct.unpack(">I", header)
    if length > _MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {length} bytes")
    body = await reader.readexactly(length)
    return json.loads(body.decode("utf-8"))


def _encode_frame(payload: Any) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return struct.pack(">I", len(body)) + body


async def _cango_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """One-shot JSON-RPC call to the cango daemon over its Unix socket.

    Never raises for an unreachable daemon or an RPC-level error — those are
    expected operational states, returned as ``cango_unavailable`` so the MCP
    tool surface degrades gracefully. Only truly unexpected conditions would
    propagate, and even those are funneled into the same structured shape.

    `params` is filtered of ``None`` values so optional args the caller didn't
    set don't reach the daemon's zod schema as explicit nulls.
    """
    socket_path = config.settings.cango_socket
    clean_params = {k: v for k, v in params.items() if v is not None}
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": clean_params}

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(socket_path),
            timeout=_CONNECT_TIMEOUT_S,
        )
    except (FileNotFoundError, ConnectionRefusedError):
        return _unavailable(f"daemon socket not reachable at {socket_path}")
    except (asyncio.TimeoutError, OSError) as exc:
        return _unavailable(f"could not connect to {socket_path}: {exc}")

    try:
        writer.write(_encode_frame(request))
        await writer.drain()
        response = await asyncio.wait_for(_read_frame(reader), timeout=_RPC_TIMEOUT_S)
    except asyncio.TimeoutError:
        return _unavailable(f"timed out after {_RPC_TIMEOUT_S}s calling {method}")
    except (asyncio.IncompleteReadError, ConnectionError, ValueError) as exc:
        return _unavailable(f"malformed response from daemon: {exc}")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass

    if isinstance(response, dict) and response.get("error"):
        err = response["error"]
        reason = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        return _unavailable(f"daemon RPC error: {reason}")
    result = response.get("result") if isinstance(response, dict) else None
    if result is None:
        return _unavailable("daemon returned an empty result")
    return result


def register_cango_tools(mcp: FastMCP) -> None:
    """Register the calendar shims on the shared FastMCP instance.

    Mirrors `register_tools` in mcp_tools.py: each tool is an `@mcp.tool()`
    coroutine. No AppState is needed — the daemon owns all calendar state.
    """

    @mcp.tool()
    async def check_availability(
        start: str,
        end: str,
        people: list[str] | None = None,
    ) -> dict[str, Any]:
        """Can we go? Return a free / soft_conflict / hard_conflict verdict for
        a time window across the family's calendars.

        Use this when an invitation arrives and you need to know whether the
        relevant people are free. The verdict is computed by the cango daemon
        from every configured calendar feed; this tool surfaces it, it does not
        decide for the user.

        Args:
          start: ISO-8601 window start, e.g. "2026-06-14T09:00:00Z".
          end: ISO-8601 window end.
          people: optional list of person ids to restrict the check to. Omit
            to check every person in the family graph.

        Returns the daemon's verdict payload:
          {"verdict": "free"|"soft_conflict"|"hard_conflict",
           "conflicts": [{"person": {...}, "event": {...},
                          "overlap_minutes": int}],
           "degraded": bool, "stale_sources": [str]}
        `degraded`/`stale_sources` flag calendars served past their freshness
        window — a non-empty `stale_sources` means the verdict may be out of
        date. On daemon unavailability returns
          {"error": "cango_unavailable", "reason": str}.
        """
        return await _cango_rpc(
            "checkAvailability", {"start": start, "end": end, "people": people}
        )

    @mcp.tool()
    async def find_free_slot(
        duration_minutes: int,
        between_start: str,
        between_end: str,
        people: list[str] | None = None,
        working_hours: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Find candidate free windows of a given length within a range.

        Use this when the user wants to *schedule* something ("when can we all
        do a 90-minute outing next weekend?") rather than check a fixed time.

        Args:
          duration_minutes: required slot length, e.g. 90.
          between_start: ISO-8601 start of the range to search.
          between_end: ISO-8601 end of the range to search.
          people: optional list of person ids the slot must be free for. Omit
            for the whole family.
          working_hours: optional {"start": "09:00", "end": "17:00"} to confine
            candidate slots to a daily window (e.g. waking hours).

        Returns {"slots": [{"start": iso, "end": iso}], "degraded": bool,
        "stale_sources": [str]}, or
          {"error": "cango_unavailable", "reason": str}.
        """
        return await _cango_rpc(
            "findFreeSlot",
            {
                "duration_minutes": duration_minutes,
                "between": {"start": between_start, "end": between_end},
                "people": people,
                "working_hours": working_hours,
            },
        )

    @mcp.tool()
    async def list_events(
        start: str,
        end: str,
        people: list[str] | None = None,
        extended: bool = False,
        exclude_roles: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List calendar events in a window with their resolved roles.

        Use this to see *what's actually on the calendar* for a window — each
        event carries its resolved role (hard/soft/info). Good for explaining a
        verdict or eyeballing a day.

        Output is **compact by default** (a full-month window is one call): the
        large Exchange GUIDs (`id`, `series_id`) are omitted, and
        `resolved_by`/`resolved_reason` are dropped for plain source-default
        verdicts (kept whenever a rule/attendance/structural decision applied).
        A `day_span` field appears on multi-day events (e.g. 3 for a Fri→Sun
        event) so spans don't hide behind a single start date.

        Args:
          start: ISO-8601 window start.
          end: ISO-8601 window end.
          people: optional list of person ids to restrict to. Omit for all.
          extended: when True, include `id` and `series_id` and always include
            `resolved_by`/`resolved_reason`. Needed to get an event id for
            follow-up calls like `explain_event`.
          exclude_roles: optional roles to drop (e.g. ["soft"]). Default keeps
            everything so a soft-only event is never silently hidden.
          limit: max events returned (daemon default 1000 — high enough that a
            month is a single fetch). offset: events to skip, for paging.

        Returns {"events": [ {source_id, person_id, title, start, end, all_day,
        resolved_role, day_span?, ... (id/series_id only when extended)} ],
        "total": int, "returned": int, "truncated": bool, "degraded": bool,
        "stale_sources": [str]}, or {"error": "cango_unavailable", "reason": str}.
        """
        return await _cango_rpc(
            "listEvents",
            {
                "start": start,
                "end": end,
                "people": people,
                "extended": extended,
                "exclude_roles": exclude_roles,
                "limit": limit,
                "offset": offset,
            },
        )

    @mcp.tool()
    async def explain_event(event_id: str) -> dict[str, Any]:
        """Explain how a single event's role was resolved, layer by layer.

        Use this to answer "why is this counted as a hard conflict?" — the
        trace shows each resolution layer (structural / rule / source-default /
        mask) and its outcome, so the user can see whether to add or amend a
        rule (via `record_rule` / `amend_rule`) to change the verdict.

        Args:
          event_id: the `id` of an event, as returned by `list_events` or in a
            `check_availability` conflict entry.

        Returns {"resolved": {...resolved event...}, "trace": [{"layer": str,
        "outcome": str}], "degraded": bool, "stale_sources": [str]}. If the
        event id is unknown the daemon answers with an RPC error, surfaced as
          {"error": "cango_unavailable", "reason": str}.
        """
        return await _cango_rpc("explainEvent", {"event_id": event_id})

    @mcp.tool()
    async def list_series(source_id: str) -> dict[str, Any]:
        """List recent recurring series on a calendar source.

        Use this when the user wants to add an attendance rule ("the kid
        sometimes attends the Tuesday practice") — it surfaces the series ids
        and titles to pick from. Record the rule with `record_rule` using a
        `{person_id, series_id}` match.

        Args:
          source_id: the calendar source id to enumerate series for.

        Returns {"series": [{"series_id": str, "title": str, "last_start": iso,
        "count": int}], "degraded": bool, "stale_sources": [str]}, or
          {"error": "cango_unavailable", "reason": str}.
        """
        return await _cango_rpc("listSeries", {"source_id": source_id})

    @mcp.tool()
    async def create_event(
        source_id: str,
        title: str,
        start: str,
        end: str,
        all_day: bool = False,
        occupants: list[str | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a calendar event on a writable source. (Write tool.)

        Use this to actually add an event to a calendar — e.g. blocking out a
        trip so it shows up in future `check_availability` / `list_events`
        results. The daemon writes it to the underlying calendar (CalDAV today)
        and refreshes, so the returned event reflects what was stored.

        Only sources explicitly marked `writable: true` in family.yaml accept
        writes; ICS feeds and un-flagged sources are read-only. Writing to a
        non-writable or unknown source, or passing end <= start, is rejected by
        the daemon and surfaced as {"error": "cango_unavailable", "reason": str}.

        Treat `title` as potentially untrusted (it may come from a third-party
        invitation); the daemon escapes it safely before writing.

        Args:
          source_id: the writable calendar source id to create the event on.
          title: event summary/title.
          start: ISO-8601 start. For all_day events, the date component is used.
          end: ISO-8601 end. For all_day events this is *exclusive* — name the
            day after the last day (e.g. a 16–20 June trip ends 2026-06-21).
          all_day: whether this is a date-only, all-day event. Default false.
          occupants: optional person/group ids who attend this *specific* event
            beyond the calendar owner — e.g. ["wife"] for "add this to my
            calendar but include my wife". Each entry is either a bare id string
            (defaults to role `hard` = "attending") OR an object carrying a
            per-occupant role, `{"id": "<person/group id>", "role": "hard"|"soft"|"info"}`.
            Use `{"id": "wife", "role": "soft"}` for "might go" (surfaces as a
            soft conflict, not a hard block); `hard` is "attending" and is the
            bare-string default. The per-event role is written onto the calendar
            event itself, so it round-trips on the next fetch. Mixing shapes is
            fine, e.g. `[{"id": "wife", "role": "soft"}, "kid"]`. Each person
            with a known email gets an ATTENDEE line written into the event;
            ids without an email are still applied but come back in
            `unwritten_occupants` so you can add an email or a fan-out rule
            instead. For a *recurring* household pattern (e.g. "the whole family
            may attend Saras läger"), use `record_rule` with `effect="fanout"`
            rather than per-event occupants here — see that tool.

        Returns {"event": {id, source_id, title, start, end, all_day,
        resolved_role, occupants?, ...}, "unwritten_occupants"?: [str],
        "degraded": bool, "stale_sources": [str]}, or
          {"error": "cango_unavailable", "reason": str}.
        """
        return await _cango_rpc(
            "createEvent",
            {
                "source_id": source_id,
                "title": title,
                "start": start,
                "end": end,
                "all_day": all_day,
                "occupants": occupants,
            },
        )

    @mcp.tool()
    async def list_rules(include_retracted: bool = False) -> dict[str, Any]:
        """List the tiebreaker rules that adjust how events resolve. (Read.)

        Rules are the editable layer between structural checks and the source
        default: each maps a match to a role. Use this to see what's currently
        teaching cango ("standups are soft", "she never attends Thursday
        practice", "ignore my work calendar during vacation") before adding or
        amending one.

        Args:
          include_retracted: also return retracted (soft-deleted) rules as
            tombstones. Default False (active rules only).

        Returns {"rules": [{"id", "match", "role", "effect", "reason",
        "created_at", "updated_at", "retracted_at"?}], "degraded": bool,
        "stale_sources": [str]}, or {"error": "cango_unavailable", ...}.
        """
        return await _cango_rpc("listRules", {"include_retracted": include_retracted})

    @mcp.tool()
    async def record_rule(
        match: dict[str, Any],
        role: str,
        reason: str,
        effect: str = "self",
        occupants: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a tiebreaker rule that changes how matching events resolve.

        Use this to persist a scheduling judgement the calendar can't express on
        its own, so future verdicts reflect it without re-asking. Takes effect
        immediately. Three non-obvious shapes worth knowing:

        - **Attendance edge**: a rule whose `match` is `{person_id, series_id}`
          is exactly the old attendance edge — e.g. role `inherit` = "attends,
          treat as the calendar's normal weight", `soft` = "sometimes", `info` =
          "never attends". Use `list_series` to find the series_id.
        - **Out of office / vacation**: `effect="mask"` marks the matched event
          as an OOO block; while it is happening, every *other* event on the
          **same calendar (source)** is ignored (demoted to info). E.g.
          `match={"source_id": "src-work", "title_regex": "(?i)vacation"}`,
          `role="info"`, `effect="mask"` → "ignore my work calendar that week".
        - **Household fan-out**: `effect="fanout"` with `occupants` adjusts how a
          matched event occupies *named people*, so an event on one person's
          calendar shows up on (or drops off) the others' availability. This is
          the way to model "this is really a whole-family thing." A fanout rule
          can *raise*, *lower*, or *remove* a named occupant; when several fanout
          rules touch the same occupant on the same event, the **most specific
          rule (most match fields) wins per occupant** — so a broad family rule
          and a narrow per-person rule can disagree and the narrow one decides.
          Fan-out is the right home for *recurring / series-level* household
          patterns and for events on read-only feeds you can't write ATTENDEE
          onto. For a one-off event you are creating, prefer
          `create_event(occupants=...)` instead, so the occupancy lives on the
          event and self-cleans.
          Example — "the whole family may attend Saras läger; flag it, don't
          hard-block the kids":
            `match={"series_id": "lager-2026"}`, `role="soft"`,
            `effect="fanout"`, `occupants=["family"]`.
          Role on a fanout rule is the role the *named* occupants carry: `soft` =
          "might go" (surfaces as a soft conflict — the right default for
          uncertainty), `hard` = "definitely going" (raises them), `inherit` =
          same role the event already has, and `info` *removes* that occupant
          from the verdict — the "doesn't really attend" / demote-or-remove
          primitive (e.g. fan a camp onto the family but `info` the kid who's
          staying home). The calendar owner is protected from a group-driven
          demotion: a rule that fans onto a *group* won't lower or remove the
          owner unless the owner is named literally in `occupants`. The owner
          otherwise keeps their own (usually stronger) role, so a camp can be
          hard for the parent and soft for the kids.

        Args:
          match: any of `person_id`, `source_id`, `series_id`, `title_regex`
            (PCRE `(?i)` inline flags allowed), `organizer_is_self` (bool),
            `rsvp_status_in` (list). Must constrain at least one field. When
            several rules match an event the most specific (most fields) wins.
          role: one of `hard`, `soft`, `info`, or `inherit`
            (= fall through to the source default).
          reason: short human explanation, surfaced in `explain_event`.
          effect: `self` (default — set the matched event's own role), `mask`
            (the cross-event out-of-office behaviour above), or `fanout` (add
            `occupants` to the matched event).
          occupants: required for `effect="fanout"` (and rejected otherwise) —
            the person/group ids to add as occupants of the matched event(s).

        Returns {"rule": {...}, "degraded": bool, "stale_sources": [str]}, or
        {"error": "cango_unavailable", "reason": str} (e.g. unknown source/person,
        a bad title_regex, or occupants on a non-fanout rule).
        """
        return await _cango_rpc(
            "createRule",
            {
                "match": match,
                "role": role,
                "reason": reason,
                "effect": effect,
                "occupants": occupants,
            },
        )

    @mcp.tool()
    async def amend_rule(
        id: str,
        match: dict[str, Any] | None = None,
        role: str | None = None,
        reason: str | None = None,
        effect: str | None = None,
        occupants: list[str] | None = None,
    ) -> dict[str, Any]:
        """Change an existing rule in place, keeping its id stable.

        Use this to correct a rule rather than retiring and re-recording it
        ("actually make standups hard now"). Only the fields you pass change;
        omitted fields keep their current values. Amending a retracted rule is
        rejected.

        Args:
          id: the rule id from `record_rule` / `list_rules`.
          match: replacement match dict (same shape as `record_rule`), if changing.
          role / effect / reason: new values, if changing. On a `fanout` rule the
            `role` is what the named occupants carry — `hard`/`soft` raise or
            lower them, `info` removes them from the verdict, `inherit` matches
            the event (see `record_rule`'s fan-out notes; most-specific rule wins
            per occupant, and the owner is protected from a group-driven demote).
          occupants: for a `fanout` rule, the new occupant id list (replaces the
            old one). Pass [] to clear it. Only valid on a fanout-effect rule.

        Returns {"rule": {...}, "degraded": bool, "stale_sources": [str]}, or
        {"error": "cango_unavailable", "reason": str} (e.g. unknown/retracted id).
        """
        return await _cango_rpc(
            "amendRule",
            {
                "id": id,
                "match": match,
                "role": role,
                "reason": reason,
                "effect": effect,
                "occupants": occupants,
            },
        )

    @mcp.tool()
    async def forget_rule(id: str, reason: str | None = None) -> dict[str, Any]:
        """Retract a rule (soft delete). It stops applying but is kept as a
        tombstone for audit/undo, visible via `list_rules(include_retracted=True)`.

        Use this when a rule no longer reflects reality ("she's back on the
        Thursday squad — drop the never-attends rule").

        Args:
          id: the rule id from `record_rule` / `list_rules`.
          reason: optional note on why it was retracted.

        Returns {"rule": {...with retracted_at...}, "degraded": bool,
        "stale_sources": [str]}, or {"error": "cango_unavailable", "reason": str}.
        """
        return await _cango_rpc("retractRule", {"id": id, "reason": reason})
