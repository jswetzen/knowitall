// knowitall schema v2 — MCP surface refactor.
// Adds Episode as a real graph node, a generic ANCHORED_TO citation edge from
// memory-bearing types to graph-anchor types, and retracted_at columns on the
// types that get soft-deleted via the forget() tool. Project membership is
// expressed as ANCHORED_TO Project rather than a denormalized project_id.

// ---------- nodes ----------

CREATE NODE TABLE IF NOT EXISTS Episode(
    id STRING PRIMARY KEY,
    body STRING,
    kind STRING,
    created_at TIMESTAMP,
    model_version STRING,
    retracted_at TIMESTAMP
);

// retracted_at on the v1 types that get soft-deleted via forget().
ALTER TABLE Decision ADD IF NOT EXISTS retracted_at TIMESTAMP;
ALTER TABLE Task     ADD IF NOT EXISTS retracted_at TIMESTAMP;
ALTER TABLE Idea     ADD IF NOT EXISTS retracted_at TIMESTAMP;
ALTER TABLE Note     ADD IF NOT EXISTS retracted_at TIMESTAMP;

// ---------- edges ----------

// Generic citation edge from memory-bearing nodes (Episode/Decision/Task/Idea/Note)
// to graph-anchor types (Commit/File/Symbol/Project/Concept/Person).
// Bi-temporal, like every edge in the schema.
CREATE REL TABLE IF NOT EXISTS ANCHORED_TO(
    FROM Episode  TO Commit,  FROM Episode  TO File,    FROM Episode  TO Symbol,
    FROM Episode  TO Project, FROM Episode  TO Concept, FROM Episode  TO Person,
    FROM Decision TO Commit,  FROM Decision TO File,    FROM Decision TO Symbol,
    FROM Decision TO Project, FROM Decision TO Concept, FROM Decision TO Person,
    FROM Task     TO Commit,  FROM Task     TO File,    FROM Task     TO Symbol,
    FROM Task     TO Project, FROM Task     TO Concept, FROM Task     TO Person,
    FROM Idea     TO Commit,  FROM Idea     TO File,    FROM Idea     TO Symbol,
    FROM Idea     TO Project, FROM Idea     TO Concept, FROM Idea     TO Person,
    FROM Note     TO Commit,  FROM Note     TO File,    FROM Note     TO Symbol,
    FROM Note     TO Project, FROM Note     TO Concept, FROM Note     TO Person,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);
