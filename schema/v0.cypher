// knowitall schema v0 — week-1 walking skeleton.
// Closed schema discipline (see PLAN.md §3). Bi-temporal fields on every edge
// from day one even though week-1 queries do not yet use them.

CREATE NODE TABLE IF NOT EXISTS Project(
    id STRING PRIMARY KEY,
    name STRING,
    status STRING,
    created_at TIMESTAMP
);

CREATE NODE TABLE IF NOT EXISTS Repo(
    id STRING PRIMARY KEY,
    path STRING,
    remote_url STRING,
    default_branch STRING
);

CREATE NODE TABLE IF NOT EXISTS Commit(
    sha STRING PRIMARY KEY,
    repo_id STRING,
    message STRING,
    authored_at TIMESTAMP
);

CREATE REL TABLE IF NOT EXISTS PART_OF(
    FROM Repo TO Project,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);

CREATE REL TABLE IF NOT EXISTS IN_REPO(
    FROM Commit TO Repo,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    recorded_at TIMESTAMP,
    source_extractor STRING,
    extractor_version STRING
);
