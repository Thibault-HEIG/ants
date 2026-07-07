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

from core.constants import RANDOM_SEED, SPEED_MULTIPLIERS
from species.ant import Ant
from species.spider import Spider
from world.world import World

# ---------------------------------------------------------------------------
# Core ACTIVE_SPECIES Configuration
# Modify this list for seamless isolation testing (e.g. ACTIVE_SPECIES = [Ant])
# ---------------------------------------------------------------------------
ACTIVE_SPECIES: list[type] = [Ant, Spider]


class Simulation:
    """High-level continuous simulation controller and evolutionary orchestrator.

    Parameters
    ----------
    rng : np.random.Generator or None
        Seeded random number generator.
    active_species : list[type] or None
        List of species classes to simulate. Defaults to ACTIVE_SPECIES.
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        active_species: list[type] | None = None,
        load_path: str | None = None,
    ) -> None:
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
        self.active_species: list[type] = list(active_species if active_species is not None else ACTIVE_SPECIES)

        self.world: World = World(self.rng, active_species=self.active_species)
        self.running: bool = True
        self.speed_idx: int = 1  # Index 1 -> 1x speed in SPEED_MULTIPLIERS
        self.loaded_genomes: dict[type, list[np.ndarray]] | None = None

        # Track historical peak populations spawned per species
        self.total_spawned: dict[type, int] = {
            cls: getattr(cls, "initial_count", 10) for cls in self.active_species
        }

        if load_path is not None:
            self.load_from_save(load_path)

    def save_top_brains(self) -> str | None:
        """Save top 10% brains of each active species to a JSON file in saves/ directory."""
        import json
        import os
        from datetime import datetime

        os.makedirs("saves", exist_ok=True)
        # Format: mm.hh.dd.mm.yy (e.g. 21-23-06-07-26)
        filename = datetime.now().strftime("%M-%H-%d-%m-%y.json")
        filepath = os.path.join("saves", filename)

        save_data = []
        save_data.append({"notes_to_self": "notes"},)
        for cls in self.active_species:
            species_name = getattr(cls, "species_name", cls.__name__)
            pool = list(self.world.creatures.get(cls, [])) + list(self.world.dead_creatures.get(cls, []))
            if not pool:
                continue

            pool.sort(key=lambda c: c.compute_fitness(), reverse=True)
            top_count = max(1, int(len(pool) * 0.1))
            top_creatures = pool[:top_count]

            for c in top_creatures:
                save_data.append({
                    "species_name": species_name,
                    "fitness": float(c.compute_fitness()),
                    "genome": c.genome.tolist(),
                })

        if not save_data:
            print("[SAVE] No creatures available to save.")
            return None

        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"[SAVE] Saved top 10% brains ({len(save_data)} genomes) to {filepath}")
        return filepath

    def load_from_save(self, filepath: str) -> None:
        """Load genomes from a JSON save file and initialize world populations."""
        import json
        import os

        actual_path = filepath
        if not os.path.exists(actual_path) and not actual_path.endswith(".json"):
            actual_path += ".json"
        if not os.path.exists(actual_path):
            alt_path = actual_path.replace(".", "-") if "." in filepath else actual_path.replace("-", ".")
            if not alt_path.endswith(".json"):
                alt_path += ".json"
            if os.path.exists(alt_path):
                actual_path = alt_path

        if not os.path.exists(actual_path):
            print(f"[ERROR] Save file not found: {filepath}")
            return

        with open(actual_path, "r") as f:
            save_data = json.load(f)

        print(f"[LOAD] Loading saved genomes from {actual_path}...")

        species_map = {getattr(cls, "species_name", cls.__name__): cls for cls in self.active_species}
        for cls in self.active_species:
            species_map[cls.__name__] = cls

        genomes_by_species: dict[type, list[np.ndarray]] = {cls: [] for cls in self.active_species}

        for item in save_data:
            sp_name = item.get("species_name")
            gen_list = item.get("genome")
            if sp_name in species_map and gen_list is not None:
                cls = species_map[sp_name]
                genomes_by_species[cls].append(np.array(gen_list, dtype=float))

        self.loaded_genomes = genomes_by_species
        self.world.reset_with_genomes(genomes_by_species)
        print("[LOAD] Successfully started simulation from saved genomes.")

    def reset(self) -> None:
        """Reset simulation world to initial state or loaded save."""
        self.world.reset_with_genomes(self.loaded_genomes or {})

    @property
    def speed_multiplier(self) -> float:
        """Current simulation speed multiplier."""
        return SPEED_MULTIPLIERS[self.speed_idx]

    def set_speed(self, idx: int) -> None:
        """Set speed multiplier index, clamped to valid bounds."""
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
