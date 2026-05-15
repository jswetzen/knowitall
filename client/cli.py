"""Tiny MCP client for local smoke testing.

Usage:
    uv run python -m client.cli health
    uv run python -m client.cli store "some text" --kind note --project knowitall
    uv run python -m client.cli query "search words" --project knowitall
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
                if getattr(block, "type", None) == "text":
                    try:
                        print(json.dumps(json.loads(block.text), indent=2, default=str))
                    except json.JSONDecodeError:
                        print(block.text)
                else:
                    print(block)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")

    s = sub.add_parser("store")
    s.add_argument("text")
    s.add_argument("--kind", default="note")
    s.add_argument("--project", default=None)

    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--project", default=None)
    q.add_argument("-k", type=int, default=10)

    c = sub.add_parser("cypher")
    c.add_argument("query")

    args = p.parse_args()

    if args.cmd == "health":
        base = _endpoint().rsplit("/mcp", 1)[0]
        r = httpx.get(f"{base}/healthz", timeout=5.0)
        print(r.text)
        return

    if args.cmd == "store":
        payload = {"text": args.text, "kind": args.kind}
        if args.project:
            payload["project_hint"] = args.project
        asyncio.run(_call("store_episode", payload))
    elif args.cmd == "query":
        payload = {"query": args.text, "k": args.k}
        if args.project:
            payload["project_hint"] = args.project
        asyncio.run(_call("query_memory", payload))
    elif args.cmd == "cypher":
        asyncio.run(_call("cypher", {"query": args.query}))
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
