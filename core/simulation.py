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
        load_path: str | None = None,
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
        self.loaded_genomes: dict[type, list[np.ndarray]] | None = None

        # Track historical peak populations spawned per species
        self.total_spawned: dict[type, int] = {
            cls: getattr(cls, "initial_count", 10) for cls in self.active_species
        }

        # Track average fitness curves over simulation time
        self.history_time: list[float] = []
        self.history_fitness: dict[str, list[float]] = {
            getattr(cls, "species_name", cls.__name__): [] for cls in self.active_species
        }
        self._plot_timer: float = 0.0
        self._record_fitness_stat()

        if load_path is not None:
            self.load_from_save(load_path)

    def is_npc(self, cls: type) -> bool:
        """Return True if species cls is configured as an NPC (non-evolving)."""
        cfg = self.species_config.get(cls, {})
        return bool(cfg.get("npc", getattr(cls, "npc", False)))

    def _build_constants_snapshot(self) -> dict:
        """Collect all uppercase constants from core and species modules."""
        import core.constants as const_mod
        from species import ant_constants as ant_mod
        from species import spider_constants as spider_mod

        const_snap = {}
        for mod in (const_mod, ant_mod, spider_mod):
            for k in dir(mod):
                if k.isupper() and not k.startswith("_"):
                    val = getattr(mod, k)
                    if isinstance(val, (int, float, str, bool)):
                        const_snap[k] = val
        return const_snap

    def _build_stats_snapshot(self) -> dict:
        """Collect all SpeciesStats accumulator fields for serialization."""
        from core.utils import SpeciesStats
        return {
            "total_dead_count": dict(SpeciesStats.total_dead_count),
            "sum_dead_fitness": dict(SpeciesStats.sum_dead_fitness),
            "sum_dead_food": dict(SpeciesStats.sum_dead_food),
            "sum_dead_computed_food": dict(SpeciesStats.sum_dead_computed_food),
            "sum_dead_enemies": dict(SpeciesStats.sum_dead_enemies),
            "sum_dead_computed_enemies": dict(SpeciesStats.sum_dead_computed_enemies),
            "sum_dead_lifetime": dict(SpeciesStats.sum_dead_lifetime),
            "max_fitness": dict(SpeciesStats.max_fitness),
            "max_foodeaten": dict(SpeciesStats.max_foodeaten),
            "max_enemies_touched": dict(SpeciesStats.max_enemies_touched),
            "max_computed_food": dict(SpeciesStats.max_computed_food),
            "max_computed_enemies": dict(SpeciesStats.max_computed_enemies),
            "max_lifetime": dict(SpeciesStats.max_lifetime),
            "max_metrics": {k: dict(v) for k, v in SpeciesStats.max_metrics.items()},
        }

    def save_full_state(self, filename: str | None = None, notes: str = "") -> str | None:
        """Save full simulation state to a JSON file in saves/ directory."""
        import json
        import os
        from datetime import datetime

        os.makedirs("saves", exist_ok=True)
        if filename is None:
            filename = datetime.now().strftime("%y-%m-%d-%H-%M")

        filepath = os.path.join("saves", f"{filename}.json" if not filename.endswith(".json") else filename)

        gen_counts = {}
        for cls in self.active_species:
            sp_name = getattr(cls, "species_name", cls.__name__)
            gen_counts[sp_name] = self.world.generation_counts.get(cls, 0) if hasattr(self.world, "generation_counts") else 0

        save_data = {
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
            "round_time": float(self.world.round_time),
            "generation_counts": gen_counts,
            "species": {},
            "history": {
                "time": list(self.history_time),
                "fitness": dict(self.history_fitness)
            },
            "constants_snapshot": self._build_constants_snapshot(),
            "stats": self._build_stats_snapshot(),
        }

        for cls in self.active_species:
            species_name = getattr(cls, "species_name", cls.__name__)
            living = self.world.creatures.get(cls, [])
            genomes = [c.genome.tolist() for c in living]
            cfg = self.species_config.get(cls, {})

            save_data["species"][species_name] = {
                "genomes": genomes,
                "config": cfg
            }

        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"[SAVE] Saved full state ({sum(len(s['genomes']) for s in save_data['species'].values())} genomes) to {filepath}")
        return filepath

    def load_from_save(self, filepath: str) -> dict | None:
        """Load genomes and state from a JSON save file.

        Expects the dict-based format produced by save_full_state().
        Missing fields default to 0 or empty rather than raising.
        If genome count < population target, fills with clones/mutated genomes.
        """
        import json
        import os
        from evolution.genetics import mutate
        from core.utils import SpeciesStats
        from core.constants import GENOME_SIZE

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

        try:
            with open(actual_path, "r") as f:
                save_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ERROR] Failed to read save file {actual_path}: {exc}")
            return

        if not isinstance(save_data, dict):
            print(f"[ERROR] Save file is not in the expected JSON dict format: {actual_path}")
            return

        print(f"[LOAD] Loading saved simulation from {actual_path}...")

        species_map = {getattr(cls, "species_name", cls.__name__): cls for cls in self.active_species}
        for cls in self.active_species:
            species_map[cls.__name__] = cls

        genomes_by_species: dict[type, list[np.ndarray]] = {cls: [] for cls in self.active_species}

        # Parse genomes from "species" block
        for sp_name, sp_data in save_data.get("species", {}).items():
            if sp_name not in species_map:
                continue
            cls = species_map[sp_name]
            for g in sp_data.get("genomes", []):
                try:
                    arr = np.array(g, dtype=float)
                    if arr.shape == (GENOME_SIZE,):
                        genomes_by_species[cls].append(arr)
                except (ValueError, TypeError):
                    continue

        # Validate: at least one valid genome per active species
        for cls in self.active_species:
            sp_name = getattr(cls, "species_name", cls.__name__)
            if len(genomes_by_species[cls]) == 0:
                print(f"[ERROR] No valid genome found for {sp_name}")

        # Reset round_time and gen_count (ignored on load)
        self.world.round_time = 0.0
        self.world.generation_counts[cls] = 0

        # Restore history (default to empty)
        hist = save_data.get("history", {})
        self.history_time = hist.get("time", [])
        self.history_fitness = hist.get("fitness",
            {getattr(cls, "species_name", cls.__name__): [] for cls in self.active_species})

        # Restore stats — every field defaults to empty dict when absent
        stats = save_data.get("stats", {})
        SpeciesStats.total_dead_count = stats.get("total_dead_count", {})
        SpeciesStats.sum_dead_fitness = stats.get("sum_dead_fitness", {})
        SpeciesStats.sum_dead_food = stats.get("sum_dead_food", {})
        SpeciesStats.sum_dead_computed_food = stats.get("sum_dead_computed_food", {})
        SpeciesStats.sum_dead_enemies = stats.get("sum_dead_enemies", {})
        SpeciesStats.sum_dead_computed_enemies = stats.get("sum_dead_computed_enemies", {})
        SpeciesStats.sum_dead_lifetime = stats.get("sum_dead_lifetime", {})
        SpeciesStats.max_fitness = stats.get("max_fitness", {})
        SpeciesStats.max_foodeaten = stats.get("max_foodeaten", {})
        SpeciesStats.max_enemies_touched = stats.get("max_enemies_touched", {})
        SpeciesStats.max_computed_food = stats.get("max_computed_food", {})
        SpeciesStats.max_computed_enemies = stats.get("max_computed_enemies", {})
        SpeciesStats.max_lifetime = stats.get("max_lifetime", {})
        SpeciesStats.max_metrics = stats.get("max_metrics", {})

        # Fill genomes to match target population size via clones/mutation
        for cls, genomes in genomes_by_species.items():
            target = getattr(cls, "initial_count", 10)
            if len(genomes) > 0 and len(genomes) < target:
                original_count = len(genomes)
                while len(genomes) < target:
                    parent_idx = len(genomes) % original_count
                    parent = genomes[parent_idx]
                    genomes.append(mutate(parent, self.rng))

        self.loaded_genomes = genomes_by_species
        self.world.reset_with_genomes(genomes_by_species)
        print("[LOAD] Successfully started simulation from saved state.")
        return save_data.get("constants_snapshot", {})



    def reset(self) -> None:
        """Reset simulation world to initial state or loaded save."""
        self.world.reset_with_genomes(self.loaded_genomes or {})
        self.history_time.clear()
        for series in self.history_fitness.values():
            series.clear()
        self._plot_timer = 0.0
        self._record_fitness_stat()

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

        self._plot_timer += dt
        if self._plot_timer >= 1.0:
            self._plot_timer -= 1.0
            self._record_fitness_stat()

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

    def _record_fitness_stat(self) -> None:
        """Record current average fitness for each active species."""
        self.history_time.append(round(self.world.round_time, 1))
        for cls in self.active_species:
            name = getattr(cls, "species_name", cls.__name__)
            living = self.world.creatures.get(cls, [])
            if living:
                avg_fit = sum(c.compute_fitness() for c in living) / len(living)
            else:
                avg_fit = 0.0
            if name not in self.history_fitness:
                self.history_fitness[name] = []
            self.history_fitness[name].append(round(avg_fit, 2))

    def plot_fitness_curves(self) -> None:
        """Draw historical average fitness curves for all species in the terminal using plotext."""
        can_plot = True
        try:
            import plotext as plt
        except ImportError:
            print("[WARN] plotext library not installed. Cannot display terminal plots.")
            can_plot = False

        if can_plot and (not self.history_time or len(self.history_time) < 2):
            print("[INFO] Not enough simulation history recorded to plot fitness curves.")
            can_plot = False

        if can_plot:
            plt.clear_figure()
            plt.title("Ants vs Spiders — Evolutionary Fitness Curves")
            plt.xlabel("Simulation Time (seconds)")
            plt.ylabel("Average Fitness Score")

            colors = {"Ant": "green", "Spider": "red"}
            for name, fitness_series in self.history_fitness.items():
                color = colors.get(name, "blue")
                plt.plot(self.history_time, fitness_series, label=f"{name} Avg Fitness", color=color)

            plt.plotsize(100, 25)
            print("\n" + "=" * 80)
            print(" " * 25 + "SIMULATION FITNESS CURVES")
            print("=" * 80)
            plt.show()
            print("=" * 80)

        self._print_metric_recap()

    def _print_metric_recap(self) -> None:
        """Print end-of-simulation metric recap for each active species."""
        from species.ant_constants import ANT_METRIC_BOUNDS
        from species.spider_constants import SPIDER_METRIC_BOUNDS
        from core.utils import SpeciesStats

        print("\n" + "=" * 80)
        print(" " * 26 + "SIMULATION METRIC RECAP")
        print("=" * 80)

        for cls in self.active_species:
            species_name = getattr(cls, "species_name", cls.__name__)
            bounds_table = getattr(cls, "metrics", {})
            if not bounds_table:
                if species_name == "Ant":
                    bounds_table = ANT_METRIC_BOUNDS
                elif species_name == "Spider":
                    bounds_table = SPIDER_METRIC_BOUNDS

            recap_metrics = list(bounds_table.keys()) if bounds_table else [
                "survival_time",
                "computed_food_eaten",
                "computed_enemies_touched",
                "times_eating_for_nothing",
                "times_attacking_for_nothing",
                "tiles_covered",
            ]

            peak_table = SpeciesStats.max_metrics.get(species_name, {})
            all_creatures = self.world.creatures.get(cls, []) + self.world.dead_creatures.get(cls, [])

            print(f"\n--- {species_name} Metric Recap ---")
            print(f"  {'Metric':<45} | {'Max Value':>10} | {'Bound':>10} | Status")
            print("  " + "-" * 78)
            for metric_name in recap_metrics:
                peak_val = peak_table.get(metric_name, 0.0)
                curr_max = max((float(getattr(c, metric_name, 0.0)) for c in all_creatures), default=0.0)
                val = max(peak_val, curr_max)
                bound_val = float(bounds_table.get(metric_name, 0.0))

                if metric_name == "survival_time" or val % 1 != 0:
                    val_str = f"{val:.1f}"
                else:
                    val_str = f"{int(val)}"

                bound_str = f"{bound_val:.0f}" if bound_val % 1 == 0 else f"{bound_val:.1f}"
                warning = "⚠️" if val > bound_val else ""

                print(f"  {metric_name:<45} | {val_str:>10} | {bound_str:>10} | {warning}")

        print("\n" + "=" * 80 + "\n")
