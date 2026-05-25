// knowitall schema v3 — typed mutation surface + enumeration support.
// Adds amended_at audit column to memory-bearing nodes, an optional
// summary column for cheap enumeration (Note already has `title`), and
// polymorphic memory→memory relation edges so renames/refinements/
// supersessions can be modeled in the graph rather than in prose.
//
// Existing Decision→Decision SUPERSEDES edge is intentionally left alone:
// Kuzu doesn't support ALTER REL TABLE to add FROM/TO pairs, so widening
// it would be a destructive DROP+CREATE. New polymorphic edges below
// (SUPERSEDES_MEMORY, REFINES, CONTRADICTS, RELATES_TO_MEMORY) cover the
// generalized cases without touching v1's edge.

// ---------- audit column ----------

ALTER TABLE Decision ADD IF NOT EXISTS amended_at TIMESTAMP;
ALTER TABLE Task     ADD IF NOT EXISTS amended_at TIMESTAMP;
ALTER TABLE Idea     ADD IF NOT EXISTS amended_at TIMESTAMP;
ALTER TABLE Note     ADD IF NOT EXISTS amended_at TIMESTAMP;
ALTER TABLE Episode  ADD IF NOT EXISTS amended_at TIMESTAMP;

// ---------- summary column ----------
// Optional title-shaped string for enumeration. Skipped on Note because
// Note.title already serves this purpose (a parallel `summary` column
// would just be confusion).

ALTER TABLE Decision ADD IF NOT EXISTS summary STRING;
ALTER TABLE Task     ADD IF NOT EXISTS summary STRING;
ALTER TABLE Idea     ADD IF NOT EXISTS summary STRING;
ALTER TABLE Episode  ADD IF NOT EXISTS summary STRING;

// ---------- polymorphic memory→memory edges ----------
// Same FROM/TO matrix as ANCHORED_TO's source side (Episode|Decision|
// Task|Idea|Note × Episode|Decision|Task|Idea|Note) but neither end is
// an anchor — both ends are memory nodes. Bi-temporal like every edge.

CREATE REL TABLE IF NOT EXISTS SUPERSEDES_MEMORY(
    FROM Episode  TO Episode,  FROM Episode  TO Decision, FROM Episode  TO Task,
    FROM Episode  TO Idea,     FROM Episode  TO Note,
    FROM Decision TO Episode,  FROM Decision TO Decision, FROM Decision TO Task,
    FROM Decision TO Idea,     FROM Decision TO Note,
    FROM Task     TO Episode,  FROM Task     TO Decision, FROM Task     TO Task,
    FROM Task     TO Idea,     FROM Task     TO Note,
    FROM Idea     TO Episode,  FROM Idea     TO Decision, FROM Idea     TO Task,
    FROM Idea     TO Idea,     FROM Idea     TO Note,
    FROM Note     TO Episode,  FROM Note     TO Decision, FROM Note     TO Task,
    FROM Note     TO Idea,     FROM Note     TO Note,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS REFINES(
    FROM Episode  TO Episode,  FROM Episode  TO Decision, FROM Episode  TO Task,
    FROM Episode  TO Idea,     FROM Episode  TO Note,
    FROM Decision TO Episode,  FROM Decision TO Decision, FROM Decision TO Task,
    FROM Decision TO Idea,     FROM Decision TO Note,
    FROM Task     TO Episode,  FROM Task     TO Decision, FROM Task     TO Task,
    FROM Task     TO Idea,     FROM Task     TO Note,
    FROM Idea     TO Episode,  FROM Idea     TO Decision, FROM Idea     TO Task,
    FROM Idea     TO Idea,     FROM Idea     TO Note,
    FROM Note     TO Episode,  FROM Note     TO Decision, FROM Note     TO Task,
    FROM Note     TO Idea,     FROM Note     TO Note,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS CONTRADICTS(
    FROM Episode  TO Episode,  FROM Episode  TO Decision, FROM Episode  TO Task,
    FROM Episode  TO Idea,     FROM Episode  TO Note,
    FROM Decision TO Episode,  FROM Decision TO Decision, FROM Decision TO Task,
    FROM Decision TO Idea,     FROM Decision TO Note,
    FROM Task     TO Episode,  FROM Task     TO Decision, FROM Task     TO Task,
    FROM Task     TO Idea,     FROM Task     TO Note,
    FROM Idea     TO Episode,  FROM Idea     TO Decision, FROM Idea     TO Task,
    FROM Idea     TO Idea,     FROM Idea     TO Note,
    FROM Note     TO Episode,  FROM Note     TO Decision, FROM Note     TO Task,
    FROM Note     TO Idea,     FROM Note     TO Note,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS RELATES_TO_MEMORY(
    FROM Episode  TO Episode,  FROM Episode  TO Decision, FROM Episode  TO Task,
    FROM Episode  TO Idea,     FROM Episode  TO Note,
    FROM Decision TO Episode,  FROM Decision TO Decision, FROM Decision TO Task,
    FROM Decision TO Idea,     FROM Decision TO Note,
    FROM Task     TO Episode,  FROM Task     TO Decision, FROM Task     TO Task,
    FROM Task     TO Idea,     FROM Task     TO Note,
    FROM Idea     TO Episode,  FROM Idea     TO Decision, FROM Idea     TO Task,
    FROM Idea     TO Idea,     FROM Idea     TO Note,
    FROM Note     TO Episode,  FROM Note     TO Decision, FROM Note     TO Task,
    FROM Note     TO Idea,     FROM Note     TO Note,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);
