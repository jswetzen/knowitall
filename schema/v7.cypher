// knowitall schema v7 — checkpoint after v4's DROP to defuse a live Kuzu
// segfault hazard.
//
// Kuzu 0.11.3 segfaults during checkpoint (Database close) when a row
// created BEFORE an `ALTER TABLE ... DROP <column>` is later UPDATED. The
// crash is in ChunkedNodeGroup::scanCommitted, reached from
// StorageManager::checkpoint at process shutdown — not at the write itself.
// v4 dropped Idea.status/died_at, so every Idea row that predates v4 is a
// live landmine: amending one crashes the server on close. v4 is the only
// DROP in the schema history, so Idea is the only table affected.
//
// Kuzu is archived (0.11.3, 2025-10-10, is the final release) — there is no
// upstream fix coming. But a plain CHECKPOINT run once, here, after the
// DROP, rewrites the on-disk chunk layout for the dropped column and clears
// the hazard permanently: verified against the real segfault repro in both
// a single-process reproduction and a realistic two-process one (migrate in
// one process/session, UPDATE the legacy row in a later, separate one — the
// shape a real deploy + a later amend() call actually takes). Both come
// back clean (exit 0) where the unpatched sequence reliably segfaults
// (SIGSEGV) on close.
//
// Safe to run repeatedly and on a DB with no legacy Idea rows (fresh
// installs): CHECKPOINT is a normal statement, not a DROP — it just forces
// Kuzu to flush and normalize whatever's currently in memory to disk.

CHECKPOINT;
