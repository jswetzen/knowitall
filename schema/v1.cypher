// knowitall schema v1 — closes the schema per PLAN.md §3.
// Adds the remaining 9 node types and 16 edge types on top of v0.
// ConversationTurn (PLAN §3) is intentionally omitted; episodes are stored
// via store_episode only, no auto-firehose from PostToolUse hooks.

// ---------- nodes ----------

CREATE NODE TABLE IF NOT EXISTS File(
    id STRING PRIMARY KEY,
    repo_id STRING,
    path STRING
);

CREATE NODE TABLE IF NOT EXISTS Symbol(
    id STRING PRIMARY KEY,
    file_id STRING,
    name STRING,
    kind STRING,
    line INT64
);

CREATE NODE TABLE IF NOT EXISTS Note(
    id STRING PRIMARY KEY,
    path STRING,
    title STRING,
    created_at TIMESTAMP
);

CREATE NODE TABLE IF NOT EXISTS Conversation(
    session_uuid STRING PRIMARY KEY,
    project_id STRING,
    started_at TIMESTAMP
);

CREATE NODE TABLE IF NOT EXISTS Decision(
    id STRING PRIMARY KEY,
    body STRING,
    decided_at TIMESTAMP
);

CREATE NODE TABLE IF NOT EXISTS Concept(
    id STRING PRIMARY KEY,
    name STRING
);

CREATE NODE TABLE IF NOT EXISTS Person(
    id STRING PRIMARY KEY,
    name STRING,
    email STRING
);

CREATE NODE TABLE IF NOT EXISTS Task(
    id STRING PRIMARY KEY,
    body STRING,
    status STRING,
    created_at TIMESTAMP,
    closed_at TIMESTAMP
);

CREATE NODE TABLE IF NOT EXISTS Idea(
    id STRING PRIMARY KEY,
    body STRING,
    status STRING,
    created_at TIMESTAMP,
    died_at TIMESTAMP
);

// ---------- edges (all bi-temporal) ----------

CREATE REL TABLE IF NOT EXISTS AUTHORED(
    FROM Person TO Commit,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS MODIFIED(
    FROM Commit TO File,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS DEFINED_IN(
    FROM Symbol TO File,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS CALLS(
    FROM Symbol TO Symbol,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS IMPORTS(
    FROM File TO File,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS DEPENDS_ON(
    FROM Project TO Project,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS BLOCKS(
    FROM Task TO Task,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS SUPERSEDES(
    FROM Decision TO Decision,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS GRADUATED_TO(
    FROM Idea TO Project,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS ALIAS_OF(
    FROM Concept TO Concept,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS RELATES_TO(
    FROM Concept TO Concept,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS DECIDED_IN(
    FROM Decision TO Conversation,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS DROPPED(
    FROM Idea TO Conversation,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS CLOSED_BY(
    FROM Task TO Commit,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS TOUCHED_BY(
    FROM Concept TO Commit,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS BELONGS_TO(
    FROM File TO Project,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS MENTIONED_IN(
    FROM Concept TO Conversation,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);
