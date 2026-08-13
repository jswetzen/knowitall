// knowitall schema v6 — amend_reason, decoupling `reason` from retraction.
//
// v5 added retract_reason and `amend(id, retract=True, reason=...)`, but
// wired `reason` to retraction only: passing it on any other amend raised.
// That put the field out of reach on exactly the operation where it's worth
// most — a correcting amend, where "why did this body change?" is the single
// most useful thing to record. Worse, it raised rather than warned, so the
// whole write (body included) was discarded over a metadata coupling.
//
// amend_reason is the non-retraction counterpart to retract_reason, and pairs
// with v3's amended_at the way retract_reason pairs with retracted_at. Both
// are last-write-wins: they describe the most recent amend, not a history.
// A full revision log would need an edge table, not a column — deliberately
// out of scope here.

ALTER TABLE Decision ADD IF NOT EXISTS amend_reason STRING;
ALTER TABLE Task     ADD IF NOT EXISTS amend_reason STRING;
ALTER TABLE Idea     ADD IF NOT EXISTS amend_reason STRING;
ALTER TABLE Note     ADD IF NOT EXISTS amend_reason STRING;
ALTER TABLE Episode  ADD IF NOT EXISTS amend_reason STRING;
