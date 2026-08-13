# Kùzu DROP-column checkpoint hazard

## The bug

Kùzu 0.11.3 segfaults during checkpoint when a row created **before** an
`ALTER TABLE ... DROP <column>` is **updated in the same session that applied
the DROP**. The crash is in `ChunkedNodeGroup::scanCommitted`, reached from
`Database::~Database` → `StorageManager::checkpoint`, so it lands at database
close rather than at the write itself.

`schema/v4.cypher` drops `Idea.status` and `Idea.died_at`, and is the only
`DROP` in the schema history — so `Idea` is the only table exposed.

Measured, with no server code involved:

| scenario | result |
|---|---|
| `DROP` + `UPDATE` in one process | **SIGSEGV (139)** |
| `DROP`, close; `UPDATE` in a later process | clean |
| row created after the `DROP`, updated | clean |
| same writes, no `DROP` at all | clean |

The same-session qualifier is the whole story. An implicit checkpoint on close
already normalises the chunks, so **every session after the migrating one is
safe on its own.**

## What this means for the live database

Not much, as it turns out. v4 was applied on 2026-08-05 and the server has
restarted many times since, so the 8 pre-v4 Idea rows are past the window and
can be amended safely.

The residual exposure is one specific window: `build_state()` applies
migrations at startup and then serves requests for that process's entire
lifetime. A deploy that applies a `DROP` migration and then handles an
`amend()` of a pre-`DROP` row before shutdown would crash on close.

Since v4 is already applied and is the only `DROP`, that window is currently
closed. It reopens only if a future migration drops a column.

## The fix that didn't work

A `CHECKPOINT` statement immediately after the `DROP` cleanly defuses the
same-session crash — verified repeatedly against the segfault repro, where the
unpatched sequence reliably returns 139 and the checkpointed one returns 0.

It shipped briefly as `schema/v7.cypher` and **was reverted**, because it does
not survive contact with a real database:

- On the toy DBs the tests build (a handful of rows): completes instantly.
- On the production store (~300 MB, ~570 files): **hangs indefinitely.**
  Observed >60s under two independent harnesses, with and without a stale
  `kuzu.shadow` present, on both the live data directory and a clean restore
  from backup.

Because migrations run inside `build_state()` during startup, this hung the
server before uvicorn produced a single line of output, and systemd SIGKILLed
it at the ~80s start timeout — into a restart loop. The service was down until
v7 was removed.

The lesson generalises past this one statement: **a migration is executed on
startup against the real store, so anything whose cost scales with data size
can turn a deploy into an outage.** The test suite's small fixtures cannot
catch that class of problem.

## Current mitigation

Operational, not code:

1. Do not amend a pre-`DROP` `Idea` row in the same process that applied v4.
   Already satisfied — v4 is long applied.
2. If a future migration drops a column, restart the service once, with no
   writes in between, before serving traffic. The close-checkpoint normalises
   the chunks, which is the same effect `CHECKPOINT` was reaching for, without
   running it inline during startup.

## Upstream

There is no upstream fix and there will not be one: kuzudb/kuzu was archived
2025-10-10, with 0.11.3 (same day) as the final release. The issue tracker has
nothing matching; the nearest neighbour (#4777, drop-then-add breaking
insert/copy, fixed in #4786) is a different sequence and not a crash.

This is one of the arguments for eventually replacing Kùzu.

## Regression coverage

`tests/test_schema_migrate.py` keeps a strict xfail driving the repro in a
subprocess (pytest cannot trap SIGSEGV in-process), plus two passing controls
so it cannot pass for the wrong reason.
