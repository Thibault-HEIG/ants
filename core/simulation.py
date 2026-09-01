"""
simulation.py — Simulation Orchestrator and ACTIVE_SPECIES Engine
=================================================================

Manages the simulation lifecycle, multi-species continuous evolution,
speed multipliers, and the core ACTIVE_SPECIES list for dynamic isolation
or multi-species ecosystem testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.constants import RANDOM_SEED, SPEED_MULTIPLIERS, SPECIES_CONFIG
from species.ant import Ant
from species.spider import Spider
from world.world import World

# Per-ability metrics tracked at snapshot time (read-only, no per-frame cost).
# Spider lacks pheromone/carry behaviors so those fields stay zero — omitted.
TRAINING_METRICS: dict[str, list[str]] = {
    "Ant": [
        "times_eating_for_nothing", "computed_food_eaten",
        "computed_times_attacking_for_nothing", "computed_enemies_touched",
        "follow_pheromones", "released_pheromone_around_food_source",
        "walk_with_object_in_opposite_home_direction",
        "release_at_home_count", "walk_with_object_in_home_direction",
    ],
    "Spider": [
        "times_eating_for_nothing", "computed_food_eaten",
        "computed_times_attacking_for_nothing", "computed_enemies_touched",
    ],
}

# ---------------------------------------------------------------------------
# Resolve SPECIES_CONFIG name strings → class objects
# ---------------------------------------------------------------------------
_SPECIES_CLASSES: dict[str, type] = {"Ant": Ant, "Spider": Spider}

ACTIVE_SPECIES: dict[type, dict] = {
    _SPECIES_CLASSES[name]: cfg
    for name, cfg in SPECIES_CONFIG.items()
    if cfg["active"]
}


class Simulation:
    """High-level continuous simulation controller and evolutionary orchestrator.

    Parameters
    ----------
    rng : np.random.Generator or None
        Seeded random number generator.
    active_species : dict[type, dict] or list[type] or None
        Species configuration mapping species classes to their config dicts.
        Defaults to ACTIVE_SPECIES.
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        active_species: dict[type, dict] | list[type] | None = None,
    ) -> None:
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
        cfg = active_species if active_species is not None else ACTIVE_SPECIES
        if isinstance(cfg, dict):
            self.active_species: list[type] = list(cfg.keys())
            self.species_config: dict[type, dict] = dict(cfg)
        else:
            self.active_species: list[type] = list(cfg)
            self.species_config: dict[type, dict] = {
                cls: SPECIES_CONFIG.get(getattr(cls, "species_name", cls.__name__), {})
                for cls in self.active_species
            }

        for cls in self.active_species:
            cls.npc = self.is_npc(cls)

        self.world: World = World(self.rng, active_species=cfg)
        self.running: bool = True
        self.ultra_mode: bool = False
        self.speed_idx: int = 1  # Index 1 -> 1x speed in SPEED_MULTIPLIERS
        self.actual_multiplier: float = 1.0  # What the backend can actually sustain
        self.multiplier_capped: bool = False  # True when backend can't keep up
        self.loaded_genomes: dict[type, list[np.ndarray]] | None = None

        # Track historical peak populations spawned per species
        self.total_spawned: dict[type, int] = {
            cls: getattr(cls, "initial_count", 10) for cls in self.active_species
        }

    def is_npc(self, cls: type) -> bool:
        """Return True if species cls is configured as an NPC (non-evolving)."""
        cfg = self.species_config.get(cls, {})
        return bool(cfg.get("npc", getattr(cls, "npc", False)))

    def load_from_db(self, tracking_db: Any, run_id: int, reset_stats: bool = False) -> None:
        """Load genomes and state from the SQLite tracking database."""
        from evolution.genetics import mutate
        from core.constants import GENERATION_DURATION

        cursor = tracking_db.conn.cursor()
        
        # Get the most recent snapshot for this run
        cursor.execute("SELECT id, time FROM snapshots WHERE run_id = ? ORDER BY time DESC LIMIT 1", (run_id,))
        row = cursor.fetchone()
        if not row:
            print(f"[ERROR] No snapshots found for run {run_id}")
            return
            
        snapshot_id, round_time = row
        saved_round_time = float(round_time)
        saved_generation_counts = {}
        
        genomes_by_species = {cls: [] for cls in self.active_species}
        for cls in self.active_species:
            sp_name = getattr(cls, "species_name", cls.__name__)
            cursor.execute(
                "SELECT brain FROM genomes WHERE run_id = ? AND species_name = ? "
                "AND generation = (SELECT MAX(generation) FROM genomes WHERE run_id = ? AND species_name = ?) "
                "ORDER BY rank ASC", 
                (run_id, sp_name, run_id, sp_name)
            )
            for g_row in cursor.fetchall():
                try:
                    # Stored with tobytes() -> load with frombuffer()
                    arr = np.frombuffer(g_row[0], dtype=float)
                    genomes_by_species[cls].append(arr)
                except Exception:
                    pass
            
            # Fill genomes to match target population size via clones/mutation
            target = getattr(cls, "initial_count", 10)
            genomes = genomes_by_species[cls]
            if len(genomes) > 0 and len(genomes) < target:
                original_count = len(genomes)
                while len(genomes) < target:
                    parent_idx = len(genomes) % original_count
                    parent = genomes[parent_idx]
                    genomes.append(mutate(parent, self.rng))
            elif len(genomes) > target:
                genomes_by_species[cls] = genomes[:target]

            # Estimate generation counts
            saved_generation_counts[cls] = int(saved_round_time / GENERATION_DURATION) + 1

        self.loaded_genomes = genomes_by_species
        self.world.reset_with_genomes(genomes_by_species)

        # Restore timer and generations after reset (if not branching)
        if not reset_stats:
            self.world.round_time = saved_round_time
            for cls in self.active_species:
                self.world.generation_counts[cls] = saved_generation_counts[cls]
                self.world.generation_timers[cls] = saved_round_time % GENERATION_DURATION
        else:
            # If branching, reset time and generations
            for cls in self.active_species:
                self.world.generation_counts[cls] = 0
                self.world.generation_timers[cls] = 0.0
        
        if reset_stats:
            from core.utils import SpeciesStats
            SpeciesStats.reset()
        else:
            tracking_db.restore_species_stats(run_id)

        print(f"[LOAD] Successfully loaded run {run_id} from tracking DB.")

    def reset(self) -> None:
        """Reset simulation world to initial state or loaded save."""
        self.world.reset_with_genomes(self.loaded_genomes or {})

    @property
    def target_multiplier(self) -> float:
        """User-requested simulation speed multiplier."""
        return SPEED_MULTIPLIERS[self.speed_idx]

    @property
    def speed_multiplier(self) -> float:
        """Effective simulation speed (actual_multiplier). Backward compat."""
        return self.actual_multiplier

    def set_speed(self, idx: int) -> None:
        """Set target speed multiplier index, clamped to valid bounds."""
        self.speed_idx = max(0, min(idx, len(SPEED_MULTIPLIERS) - 1))

    def step(self, dt: float) -> None:
        """Advance simulation by one frame with continuous real-time reproduction."""
        if not self.running:
            return

        self.world.update(dt)

        # Update running peak stats
        for cls in self.active_species:
            current_total = len(self.world.creatures.get(cls, [])) + len(self.world.dead_creatures.get(cls, []))
            if current_total > self.total_spawned.get(cls, 0):
                self.total_spawned[cls] = current_total

    def get_total_spawned(self, cls: type) -> int:
        """Return historical peak population for a given species class."""
        if cls in self.total_spawned:
            return self.total_spawned[cls]
        return len(self.world.creatures.get(cls, []))

    # ------------------------------------------------------------------
    # Backwards compatibility properties for Ant and Spider totals
    # ------------------------------------------------------------------

    @property
    def total_ants_spawned(self) -> int:
        return self.get_total_spawned(Ant)

    @total_ants_spawned.setter
    def total_ants_spawned(self, value: int) -> None:
        self.total_spawned[Ant] = value

    @property
    def total_spiders_spawned(self) -> int:
        return self.get_total_spawned(Spider)

    @total_spiders_spawned.setter
    def total_spiders_spawned(self, value: int) -> None:
        self.total_spawned[Spider] = value
