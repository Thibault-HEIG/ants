"""
tracking_db.py — SQLite Tracking Database
==========================================

Persists training metrics, live stats, metric bounds, per-creature snapshots,
and genome checkpoints to a local SQLite database. Written every STATS_INTERVAL
wall-clock seconds, independent of simulation speed or broadcast cadence.

Read concurrently by the HTTP endpoint handler thread (WAL mode).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any

from core.constants import GENERATION_DURATION, TRACKING_DB_PATH
from core.serialization import compute_species_stats, compute_metric_bounds
from core.utils import SpeciesStats


class TrackingDB:
    """Manages SQLite connection, schema, and read/write operations for tracking data."""

    def __init__(self, db_path: str = TRACKING_DB_PATH) -> None:
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create all tables and indexes if they don't exist."""
        with self.conn:
            self.conn.executescript("""
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    notes      TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS snapshots (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    time   REAL NOT NULL
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
    snapshot_id  INTEGER NOT NULL REFERENCES snapshots(id),
    species_name TEXT NOT NULL,
    generation   INTEGER,
    metric       TEXT NOT NULL,
    max_observed REAL NOT NULL,
    bound        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bounds_snapshot ON metric_bounds(snapshot_id, species_name, metric);

CREATE TABLE IF NOT EXISTS creature_snapshots (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id                 INTEGER NOT NULL REFERENCES snapshots(id),
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
    release_at_home_count       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_creature_snap ON creature_snapshots(snapshot_id, species_name, creature_uid);

CREATE TABLE IF NOT EXISTS genomes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id  INTEGER NOT NULL REFERENCES snapshots(id),
    species_name TEXT NOT NULL,
    generation   INTEGER NOT NULL,
    rank         INTEGER NOT NULL,
    fitness      REAL NOT NULL,
    brain        BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_genomes_snapshot ON genomes(snapshot_id, species_name, generation);
            """)

    def start_run(self, notes: str = "") -> int:
        """Insert a new run row and return its id."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO runs (started_at, notes) VALUES (?, ?)",
                (datetime.utcnow().isoformat(), notes),
            )
            return cursor.lastrowid

    def end_run(self, run_id: int) -> None:
        """Mark the given run as ended."""
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET ended_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), run_id),
            )

    def write_snapshot(self, world: Any, simulation: Any, run_id: int) -> None:
        """Batch-write all tracking data for the current tick in a single transaction.

        Writes training_metrics, live_stats, metric_bounds, and creature_snapshots
        for every active species.
        """
        # Lazy import to avoid circular dependency (simulation -> world -> species)
        from core.simulation import TRAINING_METRICS

        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO snapshots (run_id, time) VALUES (?, ?)",
                (run_id, float(world.round_time)),
            )
            snapshot_id = cursor.lastrowid

            for cls in world.active_species:
                species_name = getattr(cls, "species_name", cls.__name__)
                generation = int(world.round_time / GENERATION_DURATION) + 1
                living = world.creatures.get(cls, [])

                stats = compute_species_stats(world, cls)
                bounds = compute_metric_bounds(world, cls)

                # --- training_metrics ---
                metrics_list = TRAINING_METRICS.get(species_name, [])
                tm_rows = []
                for metric in metrics_list:
                    if living:
                        best_creature = max(living, key=lambda c, m=metric: float(getattr(c, m, 0.0)))
                        best_val = float(getattr(best_creature, metric, 0.0))
                        best_lifetime = float(getattr(best_creature, "survival_time", 0.0))
                        avg_val = sum(float(getattr(c, metric, 0.0)) for c in living) / len(living)
                        avg_lifetime = sum(float(getattr(c, "survival_time", 0.0)) for c in living) / len(living)
                    else:
                        best_val = best_lifetime = avg_val = avg_lifetime = 0.0
                    tm_rows.append((
                        snapshot_id, species_name, generation, metric,
                        best_val, avg_val, best_lifetime, avg_lifetime,
                    ))
                if tm_rows:
                    self.conn.executemany(
                        "INSERT INTO training_metrics "
                        "(snapshot_id, species_name, generation, metric, best, avg, best_lifetime, avg_lifetime) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        tm_rows,
                    )

                # --- live_stats ---
                self.conn.execute(
                    "INSERT INTO live_stats "
                    "(snapshot_id, species_name, generation, alive, max_pop, "
                    "fitness_best, fitness_avg, lifetime_best, lifetime_avg, "
                    "food_best, food_avg, enemies_best, enemies_avg, "
                    "tiles_best, tiles_avg, release_home_best, release_home_avg) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        snapshot_id, species_name, generation,
                        stats["alive"], stats["maxPop"],
                        stats["bestFitness"], stats["avgFitness"],
                        stats["bestLifetime"], stats["avgLifetime"],
                        stats["bestComputedFood"], stats["avgComputedFood"],
                        stats["bestComputedEnemies"], stats["avgComputedEnemies"],
                        stats["bestTilesCovered"], stats["avgTilesCovered"],
                        stats["bestReleaseAtHome"], stats["avgReleaseAtHome"],
                    ),
                )

                # --- metric_bounds ---
                mb_rows = []
                for metric_name, bound_data in bounds.items():
                    mb_rows.append((
                        snapshot_id, species_name, generation, metric_name,
                        bound_data["max"], bound_data["bound"],
                    ))
                if mb_rows:
                    self.conn.executemany(
                        "INSERT INTO metric_bounds "
                        "(snapshot_id, species_name, generation, metric, max_observed, bound) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        mb_rows,
                    )

                # --- creature_snapshots ---
                cs_rows = []
                for c in living:
                    fitness = float(c.compute_fitness())
                    cs_rows.append((
                        snapshot_id, species_name, generation,
                        c.creature_uid, 1, fitness,
                        float(getattr(c, "survival_time", 0.0)),
                        int(getattr(c, "food_eaten", 0)),
                        float(getattr(c, "computed_food_eaten", 0.0)),
                        int(getattr(c, "times_eating_for_nothing", 0)),
                        int(getattr(c, "enemies_touched", 0)),
                        float(getattr(c, "computed_enemies_touched", 0.0)),
                        int(getattr(c, "times_attacking_for_nothing", 0)),
                        int(getattr(c, "tiles_covered", 0)),
                        int(getattr(c, "release_at_home_count", 0)),
                    ))
                if cs_rows:
                    self.conn.executemany(
                        "INSERT INTO creature_snapshots "
                        "(snapshot_id, species_name, generation, creature_uid, is_alive, fitness, "
                        "lifetime, food_eaten, computed_food_eaten, times_eating_for_nothing, "
                        "enemies_touched, computed_enemies_touched, times_attacking_for_nothing, "
                        "tiles_covered, release_at_home_count) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        cs_rows,
                    )

    def write_genomes(self, world: Any, run_id: int, species_cls: type) -> None:
        """Write a genome checkpoint for a species, ranked by fitness descending."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO snapshots (run_id, time) VALUES (?, ?)",
                (run_id, float(world.round_time)),
            )
            snapshot_id = cursor.lastrowid

            species_name = getattr(species_cls, "species_name", species_cls.__name__)
            generation = int(world.round_time / GENERATION_DURATION) + 1
            living = world.creatures.get(species_cls, [])

            ranked = sorted(living, key=lambda c: c.compute_fitness(), reverse=True)
            g_rows = []
            for rank, c in enumerate(ranked):
                fitness = float(c.compute_fitness())
                brain_blob = c.genome.tobytes()
                g_rows.append((
                    snapshot_id, species_name, generation, rank, fitness, brain_blob,
                ))
            if g_rows:
                self.conn.executemany(
                    "INSERT INTO genomes "
                    "(snapshot_id, species_name, generation, rank, fitness, brain) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    g_rows,
                )

    def restore_species_stats(self, run_id: int) -> None:
        """Restore SpeciesStats accumulators from the latest snapshot of a run.

        Rebuilds in-memory peak values so fitness normalization works after a load.
        """
        cursor = self.conn.cursor()

        # Find latest snapshot for this run
        cursor.execute(
            "SELECT id FROM snapshots WHERE run_id = ? ORDER BY time DESC LIMIT 1",
            (run_id,),
        )
        row = cursor.fetchone()
        if not row:
            return
        latest_snap = row[0]

        # Restore from live_stats
        cursor.execute(
            "SELECT species_name, fitness_best, lifetime_best, food_best, enemies_best "
            "FROM live_stats WHERE snapshot_id = ?",
            (latest_snap,),
        )
        for r in cursor.fetchall():
            s_name = r[0]
            SpeciesStats.max_fitness[s_name] = max(
                SpeciesStats.max_fitness.get(s_name, 0.0), float(r[1])
            )
            SpeciesStats.max_lifetime[s_name] = max(
                SpeciesStats.max_lifetime.get(s_name, 0.0), float(r[2])
            )
            SpeciesStats.max_computed_food[s_name] = max(
                SpeciesStats.max_computed_food.get(s_name, 0.0), float(r[3])
            )
            SpeciesStats.max_computed_enemies[s_name] = max(
                SpeciesStats.max_computed_enemies.get(s_name, 0.0), float(r[4])
            )

        # Restore from metric_bounds
        cursor.execute(
            "SELECT species_name, metric, max_observed "
            "FROM metric_bounds WHERE snapshot_id = ?",
            (latest_snap,),
        )
        for r in cursor.fetchall():
            s_name, metric, max_obs = r[0], r[1], float(r[2])
            if s_name not in SpeciesStats.max_metrics:
                SpeciesStats.max_metrics[s_name] = {}
            SpeciesStats.max_metrics[s_name][metric] = max(
                SpeciesStats.max_metrics[s_name].get(metric, 0.0), max_obs
            )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
