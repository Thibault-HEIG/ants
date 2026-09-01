-- Active: 1788208739513@@127.0.0.1@3306
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    notes      TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER NOT NULL REFERENCES runs(id),
    time                REAL NOT NULL,
    recent_code_changes INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_snapshots_run_time ON snapshots(run_id, time);

CREATE TABLE IF NOT EXISTS training_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id   INTEGER NOT NULL REFERENCES snapshots(id),
    species_name  TEXT NOT NULL,
    generation    INTEGER,
    metric        TEXT NOT NULL,
    best          REAL NOT NULL,
    avg           REAL NOT NULL,
    best_lifetime REAL NOT NULL,
    avg_lifetime  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_snapshot ON training_metrics(snapshot_id, species_name, metric);

CREATE TABLE IF NOT EXISTS live_stats (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id       INTEGER NOT NULL REFERENCES snapshots(id),
    species_name      TEXT NOT NULL,
    generation        INTEGER,
    alive             INTEGER NOT NULL,
    max_pop           INTEGER NOT NULL,
    fitness_best      REAL NOT NULL,
    fitness_avg       REAL NOT NULL,
    lifetime_best     REAL NOT NULL,
    lifetime_avg      REAL NOT NULL,
    food_best         REAL NOT NULL,
    food_avg          REAL NOT NULL,
    enemies_best      REAL NOT NULL,
    enemies_avg       REAL NOT NULL,
    tiles_best        REAL NOT NULL,
    tiles_avg         REAL NOT NULL,
    release_home_best INTEGER NOT NULL,
    release_home_avg  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_livestats_snapshot ON live_stats(snapshot_id, species_name);

CREATE TABLE IF NOT EXISTS metric_bounds (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    species_name TEXT NOT NULL,
    generation   INTEGER,
    metric       TEXT NOT NULL,
    max_observed REAL NOT NULL,
    bound        REAL NOT NULL,
    UNIQUE(run_id, species_name, metric)
);
CREATE INDEX IF NOT EXISTS idx_bounds_run ON metric_bounds(run_id, species_name, metric);

CREATE TABLE IF NOT EXISTS creatures (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                      INTEGER NOT NULL REFERENCES runs(id),
    species_name                TEXT NOT NULL,
    generation                  INTEGER,
    creature_uid                INTEGER NOT NULL,
    is_alive                    INTEGER NOT NULL,
    fitness                     REAL NOT NULL,
    lifetime                    REAL NOT NULL,
    food_eaten                  INTEGER NOT NULL,
    computed_food_eaten         REAL NOT NULL,
    times_eating_for_nothing    INTEGER NOT NULL,
    enemies_touched             INTEGER NOT NULL,
    computed_enemies_touched    REAL NOT NULL,
    times_attacking_for_nothing INTEGER NOT NULL,
    tiles_covered               INTEGER NOT NULL,
    release_at_home_count       INTEGER NOT NULL,
    walk_in_home_direction      REAL DEFAULT 0.0,
    walk_in_opposite_direction  REAL DEFAULT 0.0,
    UNIQUE(run_id, creature_uid)
);
CREATE INDEX IF NOT EXISTS idx_creatures_run ON creatures(run_id, species_name, creature_uid);

CREATE TABLE IF NOT EXISTS genomes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    species_name TEXT NOT NULL,
    generation   INTEGER NOT NULL,
    rank         INTEGER NOT NULL,
    fitness      REAL NOT NULL,
    brain        BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_genomes_run ON genomes(run_id, species_name, generation);