// knowitall schema v5 — retract_reason, folding `forget` into `amend`.
//
// `forget(id, reason)` is retired as a standalone tool: retraction becomes
// `amend(id, retract=True, reason=...)`, which also gains working un-retract
// (amend(id, retract=False)) — previously only possible via raw cypher.
// `reason` used to be accepted by forget() and simply discarded (returned
// in the response, never persisted) — this column finally stores it.

ALTER TABLE Decision ADD IF NOT EXISTS retract_reason STRING;
ALTER TABLE Task     ADD IF NOT EXISTS retract_reason STRING;
ALTER TABLE Idea     ADD IF NOT EXISTS retract_reason STRING;
ALTER TABLE Note     ADD IF NOT EXISTS retract_reason STRING;
ALTER TABLE Episode  ADD IF NOT EXISTS retract_reason STRING;
