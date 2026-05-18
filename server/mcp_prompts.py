"""MCP prompts surfaced to Claude Code as /knowitall:*.

These return user-role messages that instruct Claude what to do next using
the `record`, `update_todo`, `forget`, `query_memory`, `cypher` tools. The
server does not actually execute the work — it ships a templated playbook
that Claude runs.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from server.deps import AppState


def register_prompts(mcp: FastMCP, state: AppState) -> None:
    @mcp.prompt(
        name="status",
        description="Compile a project status digest from knowitall memory.",
    )
    def status(project: str) -> str:
        return (
            f"Use the knowitall MCP tools to produce a markdown status digest "
            f"for project '{project}'.\n\n"
            "Steps:\n"
            f"1. Call cypher with: MATCH (d:Decision)-[:ANCHORED_TO]->"
            f"(p:Project {{name: '{project}'}}) WHERE d.retracted_at IS NULL "
            "RETURN d.id, d.body, d.decided_at ORDER BY d.decided_at DESC LIMIT 10\n"
            f"2. Call cypher with: MATCH (t:Task)-[:ANCHORED_TO]->"
            f"(p:Project {{name: '{project}'}}) WHERE t.retracted_at IS NULL "
            "AND t.status <> 'done' RETURN t.id, t.body, t.status ORDER BY "
            "t.created_at DESC LIMIT 20\n"
            f"3. Call query_memory with project_hint='{project}', "
            "node_types=['episode'], k=8 to surface recent narrative entries.\n"
            "4. Run `git log --oneline -20` in the current working directory "
            "for recent commits (out-of-band; not in the graph unless ingested).\n"
            "5. Emit a markdown digest with sections: Recent decisions · Open "
            "tasks · Recent episodes · Recent commits. Cite ids so the user "
            "can `forget` or `update_todo` them."
        )

    @mcp.prompt(
        name="capture",
        description="Propose a batch of records to commit at the end of a session.",
    )
    def capture(scope: str = "session") -> str:
        return (
            f"Review the conversation so far (scope='{scope}') and propose a "
            "batch of `record` calls capturing what is worth carrying into a "
            "future Claude Code session.\n\n"
            "Rules:\n"
            "- Group by kind: decision · task · idea · note · summary · "
            "blocker · fact.\n"
            "- For each proposed record, list: kind, one-line body, "
            "project_hint, anchors[]. Use commit/file/symbol/concept anchors "
            "where they make the record citable.\n"
            "- Skip transient debug output and anything already in git or "
            "code.\n"
            "- Do NOT call `record` yet. Present the batch as a numbered "
            "list and ask the user to approve, edit, or drop items.\n"
            "- On approval, call `record` for each surviving item in one go; "
            "don't ask per item."
        )

    @mcp.prompt(
        name="provenance",
        description="Find all records anchored to a given file/commit/concept and expand 2 hops.",
    )
    def provenance(anchor: str) -> str:
        return (
            f"Find every memory anchored to '{anchor}'. The anchor may be a "
            "file path, a commit sha, or a concept name — pick the matching "
            "node type by shape (slashes → File; hex → Commit; otherwise "
            "Concept).\n\n"
            "Steps:\n"
            "1. Call cypher to find the anchor node:\n"
            f"   - File: MATCH (f:File {{path: '{anchor}'}}) RETURN f.id LIMIT 1\n"
            f"   - Commit: MATCH (c:Commit {{sha: '{anchor}'}}) RETURN c.sha LIMIT 1\n"
            f"   - Concept: MATCH (k:Concept {{name: '{anchor}'}}) RETURN k.id LIMIT 1\n"
            "2. Call cypher: MATCH (m)-[:ANCHORED_TO]->(n {<pk>: $val}) "
            "WHERE m.retracted_at IS NULL RETURN label(m), m.id, m.body LIMIT 50\n"
            "3. For each m, expand one more hop: MATCH (m {id: $mid})-"
            "[:ANCHORED_TO]->(o) RETURN label(o), o LIMIT 10\n"
            "4. Render as markdown: anchor at top, then each record with its "
            "neighbors as nested bullets. Cite record ids."
        )

    @mcp.prompt(
        name="reflect",
        description="Propose a session summary from the last N episodes.",
    )
    def reflect(project: str, last_n: int = 20) -> str:
        return (
            f"Pull the last {last_n} episode-flavored entries for project "
            f"'{project}' and propose a single `summary` record that captures "
            "the throughline.\n\n"
            "Steps:\n"
            f"1. Call cypher: MATCH (e:Episode)-[:ANCHORED_TO]->"
            f"(p:Project {{name: '{project}'}}) WHERE e.retracted_at IS NULL "
            f"RETURN e.id, e.body, e.kind, e.created_at ORDER BY e.created_at "
            f"DESC LIMIT {last_n}\n"
            "2. Identify themes, recurring blockers, and net-new facts. "
            "Ignore one-off chatter.\n"
            "3. Draft a 3-6 sentence `summary` body. Show it to the user. "
            "Do NOT call `record` until the user endorses it.\n"
            "4. On approval, call record(kind='summary', body=<draft>, "
            f"project_hint='{project}', anchors=[...]) — anchors should cite "
            "the most load-bearing commits/files referenced in the summary."
        )
