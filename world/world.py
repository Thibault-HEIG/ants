"""
world.py — The Simulation Arena and Entity Container
====================================================

Manages the lifecycle, spatial hashing, and collision physics for all active
creature species and passive entities. Fully dynamic and scalable to arbitrary
species lists without hardcoded class name dependencies in the main loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    MAX_FOOD,
    ZONE_BOUNDARY_X,
)
from core.utils import SpeciesStats
from evolution.genetics import mutate, select_parents
from species.ant import Ant
from species.spider import Spider
from world.food import Food, FoodSource
from world.kingdom import Kingdom
from world.obstacle import Lake
from world.physics import SpatialHash, resolve_combat, resolve_food_collisions, resolve_lake_collisions
from world.environment import EnvironmentSystem
from world.grid import TileGrid


class World:
    """The 2-D simulation arena managing dynamic species and passive objects.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random number generator.
    active_species : list[type] or None
        List of species classes to spawn and update in this world.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        active_species: dict[type, bool] | list[type] | None = None,
    ) -> None:
        SpeciesStats.reset()
        self.width: int = WORLD_WIDTH
        self.height: int = WORLD_HEIGHT
        self.rng: np.random.Generator = rng

        if active_species is None:
            from core.simulation import ACTIVE_SPECIES
            cfg = ACTIVE_SPECIES
        else:
            cfg = active_species

        if isinstance(cfg, dict):
            self.active_species: list[type] = list(cfg.keys())
            self.npc_species: dict[type, bool] = dict(cfg)
        else:
            self.active_species: list[type] = list(cfg)
            self.npc_species: dict[type, bool] = {
                cls: getattr(cls, "npc", False) for cls in self.active_species
            }

        for cls in self.active_species:
            cls.npc = self.is_npc(cls)

        # Dynamic containers keyed by species class
        self.creatures: dict[type, list[Any]] = {cls: [] for cls in self.active_species}
        self.dead_creatures: dict[type, list[Any]] = {cls: [] for cls in self.active_species}
        self.all_time_counts: dict[type, int] = {cls: 0 for cls in self.active_species}
        self.repro_timers: dict[type, float] = {cls: 0.0 for cls in self.active_species}
        self._parent_alternate_state: dict[type, bool] = {cls: False for cls in self.active_species}

        self.food_items: list[Food] = []
        self.spatial_hash: SpatialHash = SpatialHash(cell_size=100.0)
        self.environment: EnvironmentSystem = EnvironmentSystem(self)
        self.round_time: float = 0.0

        # Phase 2: Kingdoms
        self.kingdoms: dict[type, Kingdom] = {
            Ant: Kingdom("anthill", Ant, np.array([120.0, 120.0]), spawn_radius=60.0),
            Spider: Kingdom("toile", Spider, np.array([self.width - 120.0, self.height - 120.0]), spawn_radius=60.0),
        }

        # Phase 3: Lakes
        self.lakes: list[Lake] = [
            Lake(np.array([380.0, 420.0]), radius=50.0),
            Lake(np.array([600.0, 240.0]), radius=55.0),
            Lake(np.array([820.0, 560.0]), radius=50.0),
        ]

        # Phase 4: Spatial Tile Grid & Pheromones (10x10 tiles)
        self.tile_grid: TileGrid = TileGrid(float(self.width), float(self.height), cell_size=10.0)
        self.pheromone_cell_size: float = self.tile_grid.cell_size
        self.pheromone_grid: np.ndarray = self.tile_grid.create_float_grid()

        # Populate initial world state
        for cls in self.active_species:
            self._spawn_species(cls)

        # Kick-start with an initial food source so creatures have something to eat
        self.environment.source_cooldown = 0.0  # allow immediate first source

    def is_npc(self, cls: type) -> bool:
        """Return True if species cls is configured as an NPC (does not evolve)."""
        return bool(self.npc_species.get(cls, getattr(cls, "npc", False)))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def food_sources(self) -> list[FoodSource]:
        """Expose food sources from the environment system for rendering."""
        return self.environment.food_sources

    # ------------------------------------------------------------------
    # Backwards compatibility properties for Ant and Spider
    # ------------------------------------------------------------------

    @property
    def ants(self) -> list[Ant]:
        return self.creatures.get(Ant, [])

    @ants.setter
    def ants(self, value: list[Ant]) -> None:
        if Ant in self.creatures:
            self.creatures[Ant] = value

    @property
    def dead_ants(self) -> list[Ant]:
        return self.dead_creatures.get(Ant, [])

    @dead_ants.setter
    def dead_ants(self, value: list[Ant]) -> None:
        if Ant in self.dead_creatures:
            self.dead_creatures[Ant] = value

    @property
    def spiders(self) -> list[Spider]:
        return self.creatures.get(Spider, [])

    @spiders.setter
    def spiders(self, value: list[Spider]) -> None:
        if Spider in self.creatures:
            self.creatures[Spider] = value

    @property
    def dead_spiders(self) -> list[Spider]:
        return self.dead_creatures.get(Spider, [])

    @dead_spiders.setter
    def dead_spiders(self, value: list[Spider]) -> None:
        if Spider in self.dead_creatures:
            self.dead_creatures[Spider] = value

    # ------------------------------------------------------------------
    # Per-frame update loop
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance the simulation world by one time step without hardcoded type checks."""
        self.round_time += dt

        # Phase 0.5: Decay ant pheromone grid
        from species.ant_constants import PHEROMONE_STRENGTH, PHEROMONE_DURATION
        decay_amount = (PHEROMONE_STRENGTH / PHEROMONE_DURATION) * dt
        np.maximum(self.pheromone_grid - decay_amount, 0.0, out=self.pheromone_grid)

        # Phase 0: Environment update first — spawns food sources and food items
        self.environment.update(dt)

        # Populate spatial grid
        self.spatial_hash.clear()
        for pop in self.creatures.values():
            for c in pop:
                if getattr(c, "alive", False):
                    self.spatial_hash.insert(c)
        for f in self.food_items:
            if not getattr(f, "consumed", False):
                self.spatial_hash.insert(f)

        # Gather living populations per species
        living_by_species = {
            cls: [c for c in pop if getattr(c, "alive", False)]
            for cls, pop in self.creatures.items()
        }

        # Phase 1 & 2: Sensing and acting across all active species
        for cls, pop in self.creatures.items():
            allies = living_by_species.get(cls, [])
            enemies = [
                enemy
                for other_cls, other_pop in living_by_species.items()
                if other_cls != cls
                for enemy in other_pop
            ]
            for creature in pop:
                if not getattr(creature, "alive", False):
                    continue
                sensor_data = creature.sensors.perceive(
                    creature.position,
                    creature.direction,
                    self.food_items,
                    enemies,
                    allies,
                    float(self.width),
                    float(self.height),
                    spatial_hash=self.spatial_hash,
                    lakes=self.lakes,
                    pheromone_grid=self.pheromone_grid,
                    pheromone_cell_size=self.pheromone_cell_size,
                )
                creature.update(dt, sensor_data, world=self)

        # Phase 3 & 4: Physics and collision resolution (spatial hash accelerated)
        resolve_combat(self.creatures, spatial_hash=self.spatial_hash)
        resolve_food_collisions(self.creatures, self.food_items, spatial_hash=self.spatial_hash)
        resolve_lake_collisions(self.creatures, self.lakes)

        # Phase 5: Cleanup dead entities
        for cls in list(self.creatures.keys()):
            newly_dead = [c for c in self.creatures[cls] if not getattr(c, "alive", False)]
            self.dead_creatures[cls].extend(newly_dead)
            if len(self.dead_creatures[cls]) > 100:
                self.dead_creatures[cls].sort(key=lambda c: c.compute_fitness(), reverse=True)
                self.dead_creatures[cls] = self.dead_creatures[cls][:100]
            self.creatures[cls] = [c for c in self.creatures[cls] if getattr(c, "alive", False)]

        self.food_items = [f for f in self.food_items if not getattr(f, "consumed", False)]

        # Phase 5.5: Continuous Fitness-Based Reproduction & Extinction Recovery
        for cls in self.active_species:
            pop = self.creatures[cls]
            max_pop = getattr(cls, "max_population", 100)
            threshold = getattr(cls, "reproduction_threshold", 200.0)
            available = max_pop - len(pop)

            if len(pop) == 0:
                # Extinction recovery: repopulate starter batch from fittest dead ancestors (or fresh)
                dead_pool = self.dead_creatures.get(cls, [])
                init_count = getattr(cls, "initial_count", 10)
                if dead_pool:
                    from evolution.genetics import create_offspring_batch
                    new_genomes = create_offspring_batch(
                        dead_pool, init_count, self.rng, npc=self.is_npc(cls)
                    )
                    self._spawn_species(cls, new_genomes)
                    self.dead_creatures[cls].clear()
                else:
                    self._spawn_species(cls)
                self.repro_timers[cls] = 0.0
            elif available > 0 and len(pop) >= 1:
                self.repro_timers[cls] += dt * available
                while self.repro_timers[cls] >= threshold and len(self.creatures[cls]) < max_pop:
                    self.repro_timers[cls] -= threshold
                    parent = self._select_parent(self.creatures[cls], cls)
                    kingdom = self.kingdoms.get(cls)
                    if kingdom is not None:
                        child_pos = kingdom.sample_spawn_position(self.rng, float(self.width), float(self.height))
                    else:
                        offset = self.rng.uniform(-5.0, 5.0, size=2)
                        child_pos = np.clip(parent.position + offset, 0.0, [self.width, self.height])
                    child = cls(child_pos, self.rng)
                    if self.is_npc(cls):
                        child.genome = parent.genome.copy()
                    else:
                        child.genome = mutate(parent.genome, self.rng)
                    child.world = self
                    self.creatures[cls].append(child)
                    self.all_time_counts[cls] = self.all_time_counts.get(cls, 0) + 1
            else:
                self.repro_timers[cls] = 0.0

    def _select_parent(self, creatures: list[Any], cls: type | None = None) -> Any:
        """Select a parent alternating 1/2 best individual, 1/2 random from top 20% best parents."""
        if not creatures:
            return None

        scored = sorted(creatures, key=lambda c: c.compute_fitness(), reverse=True)
        use_best = not self._parent_alternate_state.get(cls, False)
        if cls is not None:
            self._parent_alternate_state[cls] = use_best

        if use_best:
            return scored[0]

        top_20_count = max(1, int(len(scored) * 0.20))
        top_pool = scored[:top_20_count]
        return top_pool[int(self.rng.integers(len(top_pool)))]

    def _spawn_species(self, cls: type, genomes: list[np.ndarray] | None = None) -> None:
        """Spawn initial population for a given species class."""
        target_count = getattr(cls, "initial_count", 10)
        count = max(len(genomes), target_count) if genomes is not None and len(genomes) > 0 else target_count
        self.creatures[cls] = []
        self.dead_creatures[cls] = []

        for i in range(count):
            kingdom = self.kingdoms.get(cls)
            if kingdom is not None:
                pos = kingdom.sample_spawn_position(self.rng, float(self.width), float(self.height))
            else:
                pos = self.rng.uniform([50, 50], [self.width - 50, self.height - 50])
            creature = cls(pos, self.rng)
            creature.world = self
            if genomes is not None and len(genomes) > 0:
                if i < len(genomes):
                    creature.genome = genomes[i]
                else:
                    parent_genome = genomes[i % len(genomes)]
                    if self.is_npc(cls):
                        creature.genome = parent_genome.copy()
                    else:
                        creature.genome = mutate(parent_genome, self.rng)
            self.creatures[cls].append(creature)
            self.all_time_counts[cls] = self.all_time_counts.get(cls, 0) + 1

    def reset_with_genomes(
        self,
        genomes_by_species: dict[type, list[np.ndarray]] | list[np.ndarray],
        spider_genomes: list[np.ndarray] | None = None,
    ) -> None:
        """Reset world and spawn initial populations."""
        if isinstance(genomes_by_species, dict):
            gen_map = genomes_by_species
        else:
            gen_map = {}
            if Ant in self.active_species:
                gen_map[Ant] = genomes_by_species
            if Spider in self.active_species and spider_genomes is not None:
                gen_map[Spider] = spider_genomes

        for cls in self.active_species:
            self._spawn_species(cls, gen_map.get(cls))

        self.food_items = []
        self.environment.food_sources.clear()
        self.environment.source_cooldown = 0.0
        self.repro_timers = {cls: 0.0 for cls in self.active_species}
        self.all_time_counts = {cls: 0 for cls in self.active_species}
        self.round_time = 0.0
        self.pheromone_grid.fill(0.0)

    def get_all_entities(self) -> dict:
        """Return all entities for rendering."""
        return {
            "ants": self.ants,
            "spiders": self.spiders,
            "creatures": self.creatures,
            "food_items": self.food_items,
            "food_sources": self.food_sources,
            "kingdoms": list(self.kingdoms.values()),
            "lakes": self.lakes,
            "pheromone_grid": self.pheromone_grid,
            "pheromone_cell_size": self.pheromone_cell_size,
        }
