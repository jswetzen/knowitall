// knowitall schema v4 — retire Idea.status/died_at.
//
// status was written once at creation ('open') and never transitioned:
// no tool ever moved it through the documented incubating/graduated/
// dropped lifecycle, and the GRADUATED_TO/DROPPED edges that would
// represent those transitions are declared in v1 but never written by
// any tool. Rather than build unused lifecycle machinery, retire the
// dead fields. An idea being abandoned is now just `forget()`
// (retracted_at); a graduation, if it ever happens, is a `relates_to`
// edge or a manual cypher write, not a status enum nobody could flip.

ALTER TABLE Idea DROP status;
ALTER TABLE Idea DROP died_at;
