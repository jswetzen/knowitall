"""Tiny MCP client for local smoke testing.

Usage:
    uv run python -m client.cli health
    uv run python -m client.cli record --kind decision \\
        --body "Use Kuzu over Neo4j" --project knowitall \\
        --anchor '{"kind":"file","repo":"knowitall","path":"PLAN.md"}'
    uv run python -m client.cli query "graph database choice" --project knowitall
    uv run python -m client.cli forget <id> --reason "duplicate"
    uv run python -m client.cli update-todo <id> --status done \\
        --anchor '{"kind":"commit","sha":"abc1234","repo":"knowitall"}'
    uv run python -m client.cli cypher "MATCH (p:Project) RETURN p.name"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _endpoint() -> str:
    return os.environ.get("KNOWITALL_URL", "http://127.0.0.1:8765/mcp")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ.get('KNOWITALL_TOKEN', 'devtoken')}"}


def _print_block(block) -> None:
    if getattr(block, "type", None) == "text":
        try:
            print(json.dumps(json.loads(block.text), indent=2, default=str))
        except json.JSONDecodeError:
            print(block.text)
    else:
        print(block)


async def _call(name: str, args: dict) -> None:
    async with streamablehttp_client(_endpoint(), headers=_headers()) as (
        read,
        write,
        _,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            for block in result.content:
                _print_block(block)


def _parse_anchors(values: list[str] | None) -> list[dict]:
    if not values:
        return []
    out: list[dict] = []
    for v in values:
        try:
            out.append(json.loads(v))
        except json.JSONDecodeError as e:
            raise SystemExit(f"--anchor must be JSON: {v} ({e})")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")

    r = sub.add_parser("record")
    r.add_argument("--kind", required=True)
    r.add_argument("--body", required=True)
    r.add_argument("--project", default=None)
    r.add_argument("--anchor", action="append", default=[],
                   help="JSON anchor object; repeatable")

    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--project", default=None)
    q.add_argument("-k", type=int, default=10)
    q.add_argument("--hops", type=int, default=1)
    q.add_argument("--include-retracted", action="store_true")
    q.add_argument("--node-type", action="append", default=None,
                   help="filter by node_type; repeatable")

    f = sub.add_parser("forget")
    f.add_argument("id")
    f.add_argument("--reason", required=True)

    u = sub.add_parser("update-todo")
    u.add_argument("id")
    u.add_argument("--status", required=True)
    u.add_argument("--anchor", action="append", default=[])

    c = sub.add_parser("cypher")
    c.add_argument("query")

    args = p.parse_args()

    if args.cmd == "health":
        base = _endpoint().rsplit("/mcp", 1)[0]
        rh = httpx.get(f"{base}/healthz", timeout=5.0)
        print(rh.text)
        return

    if args.cmd == "record":
        payload: dict = {"kind": args.kind, "body": args.body}
        if args.project:
            payload["project_hint"] = args.project
        anchors = _parse_anchors(args.anchor)
        if anchors:
            payload["anchors"] = anchors
        asyncio.run(_call("record", payload))
    elif args.cmd == "query":
        payload = {
            "query": args.text,
            "k": args.k,
            "expand_hops": args.hops,
            "include_retracted": args.include_retracted,
        }
        if args.project:
            payload["project_hint"] = args.project
        if args.node_type:
            payload["node_types"] = args.node_type
        asyncio.run(_call("query_memory", payload))
    elif args.cmd == "forget":
        asyncio.run(_call("forget", {"id": args.id, "reason": args.reason}))
    elif args.cmd == "update-todo":
        payload = {"id": args.id, "status": args.status}
        anchors = _parse_anchors(args.anchor)
        if anchors:
            payload["anchors"] = anchors
        asyncio.run(_call("update_todo", payload))
    elif args.cmd == "cypher":
        asyncio.run(_call("cypher", {"query": args.query}))
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
