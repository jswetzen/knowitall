// knowitall schema v7 — checkpoint after v4's DROP to close a Kuzu segfault
// window.
//
// Kuzu 0.11.3 segfaults during checkpoint (ChunkedNodeGroup::scanCommitted,
// reached from StorageManager::checkpoint at Database close) when a row
// created BEFORE an `ALTER TABLE ... DROP <column>` is UPDATED **in the same
// session that applied the DROP**. v4 dropped Idea.status/died_at, so Idea
// is the only table exposed — v4 is the only DROP in the schema history.
//
// The same-session qualifier is the whole story, and it narrows this a lot:
//   drop + update in one process   -> SIGSEGV on close
//   drop, close; update later      -> clean, with or without this migration
// An implicit checkpoint on close already normalises the chunks, so any
// session after the migrating one is safe on its own. That means the live
// database — which applied v4 on 2026-08-05 and has restarted since — is
// already past the window, and this migration changes nothing for the 8
// Idea rows that predate v4.
//
// What it does buy: the migrating process itself. `build_state()` applies
// migrations at startup and then serves requests for that process's whole
// lifetime, so without this, a deploy that applies a DROP and then handles
// an amend() of a pre-DROP row before shutdown would crash on close. This
// forces the normalising checkpoint immediately, while the process is still
// only migrating. Any future migration containing a DROP should end with a
// CHECKPOINT for the same reason.
//
// Kuzu is archived (0.11.3, 2025-10-10 is the final release), so there is no
// upstream fix to wait for.
//
// Safe to re-run and safe on a fresh install: CHECKPOINT just flushes and
// normalises whatever is currently in memory.

CHECKPOINT;
