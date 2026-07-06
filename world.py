"""
world.py — The Simulation World
=================================

The World is the **container** for all entities (ants, spiders, food).
It owns the per-frame update loop that:

  1. Lets each ant sense and act (ants see spiders as enemies).
  2. Lets each spider sense and act (spiders see ants as enemies).
  3. Resolves combat collisions (symmetric — both sides deal damage).
  4. Lets both species eat food.
  5. Cleans up dead/consumed entities.
  6. Spawns new food.

**Design note**: The World only manages entity lifetimes and physics.
It does NOT handle evolution (that's ``simulation.py``) or rendering
(that's ``renderer.py``).  This separation keeps each module focused.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ant_simulator.ant import Ant
from ant_simulator.predator import Predator
from ant_simulator.food import Food
from ant_simulator.constants import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    ANT_COUNT,
    SPIDER_COUNT,
    ANT_DAMAGE,
    SPIDER_DAMAGE,
    FOOD_SPAWN_RATE,
    MAX_FOOD,
    ROUND_TIME_LIMIT,
    ZONE_BOUNDARY_X,
    MAX_ANTS,
    MAX_SPIDERS,
    ANT_REPRODUCTION_THRESHOLD,
    SPIDER_REPRODUCTION_THRESHOLD,
)
from ant_simulator.genetics import mutate, select_parents
from ant_simulator.utils import distance, SpeciesStats


class World:
    """The 2-D arena where ants and spiders fight.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random generator for deterministic simulations.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        SpeciesStats.reset()
        self.width: int = WORLD_WIDTH
        self.height: int = WORLD_HEIGHT
        self.rng: np.random.Generator = rng

        # Entity lists
        self.ants: list[Ant] = []
        self.spiders: list[Predator] = []
        self.food_items: list[Food] = []

        # Food spawning timer
        self._food_timer: float = 0.0

        # Fitness-based reproduction timers
        # These accumulate dt * available_slots each frame.
        # When they cross the threshold, a new creature is spawned
        # using a mutated copy of a top-3 fittest genome.
        self._ant_repro_timer: float = 0.0
        self._spider_repro_timer: float = 0.0

        # Tracks dead entities for the genetic algorithm
        self.dead_ants: list[Ant] = []
        self.dead_spiders: list[Predator] = []

        # Round timer — prevents infinite rounds with random brains
        self.round_time: float = 0.0

        # Populate the world
        self._spawn_ants()
        self._spawn_spiders()
        self._spawn_food_batch(MAX_FOOD // 2)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance the entire world by one time step.

        The order matters — both species sense and act first, then we
        resolve all collisions simultaneously so neither side gets an
        unfair advantage.

        Parameters
        ----------
        dt : float
            Elapsed time in seconds.
        """
        self.round_time += dt
        # --- Phase 1: Ant sensing + acting ---
        # Ants see spiders as enemies, and other ants as allies.
        alive_spiders = [s for s in self.spiders if s.alive]
        alive_ants = [a for a in self.ants if a.alive]
        for ant in self.ants:
            if not ant.alive:
                continue
            sensor_data = ant.sensors.perceive(
                ant.position, ant.direction,
                self.food_items, alive_spiders, alive_ants,
                float(self.width), float(self.height),
            )
            ant.update(dt, sensor_data)

        # --- Phase 2: Spider sensing + acting ---
        # Spiders see ants as enemies, and other spiders as allies.
        for spider in self.spiders:
            if not spider.alive:
                continue
            sensor_data = spider.sensors.perceive(
                spider.position, spider.direction,
                self.food_items, alive_ants, alive_spiders,
                float(self.width), float(self.height),
            )
            spider.update(dt, sensor_data)

        # --- Phase 3: Combat resolution (strike reach vs hurtbox) ---
        # Instead of mutual collision damage, damage is only dealt when a creature
        # actively chooses to attack (is_attacking == True) and its target's
        # hurtbox is within its strike reach. This allows an attacker with superior
        # reach (or attacking an idle target) to deal damage without taking damage.
        for spider in self.spiders:
            if not spider.alive:
                continue
            for ant in self.ants:
                if not ant.alive:
                    continue
                dist = distance(spider.position, ant.position)

                # Ant attacks Spider
                if ant.is_attacking and dist < ant.strike_range + spider.radius:
                    spider.take_damage(ANT_DAMAGE)
                    ant.enemies_touched += 1

                # Spider attacks Ant
                if spider.is_attacking and dist < spider.strike_range + ant.radius:
                    ant.take_damage(SPIDER_DAMAGE)
                    spider.enemies_touched += 1

        # --- Phase 4: Food collisions (both species eat) ---
        for ant in self.ants:
            if not ant.alive:
                continue
            for food in self.food_items:
                if food.consumed:
                    continue
                if distance(ant.position, food.position) < ant.radius + food.radius:
                    ant.eat(food.nutrition_value)
                    food.consumed = True

        for spider in self.spiders:
            if not spider.alive:
                continue
            for food in self.food_items:
                if food.consumed:
                    continue
                if distance(spider.position, food.position) < spider.radius + food.radius:
                    spider.eat(food.nutrition_value)
                    food.consumed = True

        # --- Phase 5: Cleanup ---
        newly_dead_ants = [a for a in self.ants if not a.alive]
        self.dead_ants.extend(newly_dead_ants)
        self.ants = [a for a in self.ants if a.alive]

        newly_dead_spiders = [s for s in self.spiders if not s.alive]
        self.dead_spiders.extend(newly_dead_spiders)
        self.spiders = [s for s in self.spiders if s.alive]

        # Remove consumed food
        self.food_items = [f for f in self.food_items if not f.consumed]
        
        # --- Phase 5.5: Fitness-Based Reproduction ---
        # Instead of per-creature timers, the world itself spawns new
        # creatures on a global clock.  The spawn interval adapts to
        # population pressure:
        #
        #   The timer accumulates:  timer += dt × available_slots
        #   When timer >= THRESHOLD → spawn one creature, reset timer.
        #
        #   effective_interval = THRESHOLD / available_slots
        #
        # When population is low (many slots), spawning is fast.
        # When population nears the cap, spawning slows dramatically.
        #
        # The new creature's genome is a mutated copy of one of the
        # top 3 fittest currently alive creatures.

        # --- Ant reproduction ---
        ant_available = MAX_ANTS - len(self.ants)
        if ant_available > 0 and len(self.ants) >= 1:
            self._ant_repro_timer += dt * ant_available
            while self._ant_repro_timer >= ANT_REPRODUCTION_THRESHOLD and len(self.ants) < MAX_ANTS:
                self._ant_repro_timer -= ANT_REPRODUCTION_THRESHOLD
                # Pick a random parent from the fittest alive ants (truncation selection)
                parent = self._select_parent(self.ants)
                offset = self.rng.uniform(-5.0, 5.0, size=2)
                child_pos = np.clip(parent.position + offset, 0.0, [WORLD_WIDTH, WORLD_HEIGHT])
                child = Ant(child_pos, self.rng)
                child.genome = mutate(parent.genome, self.rng)
                self.ants.append(child)
        else:
            self._ant_repro_timer = 0.0

        # --- Spider reproduction ---
        spider_available = MAX_SPIDERS - len(self.spiders)
        if spider_available > 0 and len(self.spiders) >= 1:
            self._spider_repro_timer += dt * spider_available
            while self._spider_repro_timer >= SPIDER_REPRODUCTION_THRESHOLD and len(self.spiders) < MAX_SPIDERS:
                self._spider_repro_timer -= SPIDER_REPRODUCTION_THRESHOLD
                # Pick a random parent from the fittest alive spiders (truncation selection)
                parent = self._select_parent(self.spiders)
                offset = self.rng.uniform(-5.0, 5.0, size=2)
                child_pos = np.clip(parent.position + offset, 0.0, [WORLD_WIDTH, WORLD_HEIGHT])
                child = Predator(child_pos, self.rng)
                child.genome = mutate(parent.genome, self.rng)
                self.spiders.append(child)
        else:
            self._spider_repro_timer = 0.0

        # --- Phase 6: Spawn new food ---
        self._food_timer += dt * FOOD_SPAWN_RATE
        while self._food_timer >= 1.0 and len(self.food_items) < MAX_FOOD:
            self._food_timer -= 1.0
            self._spawn_food_batch(1)

    # ------------------------------------------------------------------
    # Reproduction helpers
    # ------------------------------------------------------------------

    def _select_parent(self, creatures: list) -> Any:
        """Select a random parent using truncation selection from genetics.py.

        Parameters
        ----------
        creatures : list
            Living creatures (ants or spiders).  Must have at least 1.

        Returns
        -------
        Creature
            One of the top-performing creatures, chosen at random.
        """
        parents = select_parents(creatures)
        return parents[int(self.rng.integers(len(parents)))]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_entities(self) -> dict:
        """Return all entities for the renderer to draw."""
        return {
            "ants": self.ants,
            "spiders": self.spiders,
            "food_items": self.food_items,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _spawn_ants(self, genomes: Optional[list[np.ndarray]] = None) -> None:
        """Create a fresh batch of ants."""
        count = len(genomes) if genomes else ANT_COUNT
        self.ants = []
        self.dead_ants = []

        for i in range(count):
            pos = self.rng.uniform(
                [50, 50], [self.width - 50, self.height - 50],
            )
            ant = Ant(pos, self.rng)
            if genomes is not None:
                ant.genome = genomes[i]
            self.ants.append(ant)

    def _spawn_spiders(self, genomes: Optional[list[np.ndarray]] = None) -> None:
        """Create a fresh batch of spiders."""
        count = len(genomes) if genomes else SPIDER_COUNT
        self.spiders = []
        self.dead_spiders = []

        for i in range(count):
            pos = self.rng.uniform(
                [50, 50], [self.width - 50, self.height - 50],
            )
            spider = Predator(pos, self.rng)
            if genomes is not None:
                spider.genome = genomes[i]
            self.spiders.append(spider)

    def _spawn_food_batch(self, count: int) -> None:
        """Add ``count`` new food items.
        
        80% of the food spawns in the Right Zone (High Food Zone).
        """
        for _ in range(count):
            if len(self.food_items) >= MAX_FOOD:
                break
            
            # Decide zone: 80% chance for right zone
            if self.rng.random() < 0.8:
                x = self.rng.uniform(ZONE_BOUNDARY_X, self.width - 10)
            else:
                x = self.rng.uniform(10, ZONE_BOUNDARY_X)
            
            y = self.rng.uniform(10, self.height - 10)
            
            self.food_items.append(Food(np.array([x, y])))

    def reset_with_genomes(
        self,
        ant_genomes: list[np.ndarray],
        spider_genomes: list[np.ndarray],
    ) -> None:
        """Reset the world for a new generation with evolved genomes.

        Parameters
        ----------
        ant_genomes : list[np.ndarray]
            Evolved ant genomes.
        spider_genomes : list[np.ndarray]
            Evolved spider genomes.
        """
        self._spawn_ants(ant_genomes)
        self._spawn_spiders(spider_genomes)
        self.food_items = []
        self._spawn_food_batch(MAX_FOOD // 2)
        self._food_timer = 0.0
        self._ant_repro_timer = 0.0
        self._spider_repro_timer = 0.0
        self.round_time = 0.0
