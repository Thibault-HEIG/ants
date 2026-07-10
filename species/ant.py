"""
ant.py — The Ant Species Implementation
=======================================

Inherits shared spatial, lifecycle, and neural mechanics from Creature.
Overrides species-specific constants and fitness evaluation logic.
"""

from __future__ import annotations

import numpy as np

from species.creature import Creature
from species.spider_constants import SPIDER_MAX_SPEED
from species.ant_constants import (
    ANT_COUNT,
    ANT_INITIAL_HEALTH,
    ANT_MAX_SPEED,
    ANT_RADIUS,
    ANT_STRIKE_RANGE,
    ANT_TURN_RATE,
    ANT_DAMAGE,
    ANT_ATTACK_COST,
    ANT_EATING_TIME,
    ANT_SENSOR_RANGE,
    ANT_SENSOR_ANGLE,
    ANT_REPRODUCTION_THRESHOLD,
    MAX_ANTS,
    DENSITY_RADIUS_ANT,
    FITNESS_SURVIVAL_WEIGHT,
    FITNESS_FOOD_WEIGHT,
    FITNESS_ENEMIES_TOUCHED_WEIGHT,
    FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT,
    FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT,
    PHEROMONE_STRENGTH,
    FITNESS_FOLLOW_PHEROMONES_WEIGHT,
    FITNESS_DISTANCE_WALKED_WEIGHT,
)


class Ant(Creature):
    """The Ant evolving agent.

    Parameters
    ----------
    position : np.ndarray
        Starting [x, y] coordinates.
    rng : np.random.Generator
        Seeded random number generator.
    """

    species_name: str = "Ant"
    initial_health: float = ANT_INITIAL_HEALTH
    max_speed: float = ANT_MAX_SPEED
    radius: float = float(ANT_RADIUS)
    strike_range: float = ANT_STRIKE_RANGE
    turn_rate: float = ANT_TURN_RATE
    damage: float = ANT_DAMAGE
    attack_cost: float = ANT_ATTACK_COST
    eating_time: float = ANT_EATING_TIME
    sensor_range: float = ANT_SENSOR_RANGE
    sensor_angle: float = ANT_SENSOR_ANGLE
    reproduction_threshold: float = ANT_REPRODUCTION_THRESHOLD
    max_population: int = MAX_ANTS
    initial_count: int = ANT_COUNT

    def __init__(self, position: np.ndarray, rng: np.random.Generator) -> None:
        super().__init__(
            position,
            rng,
            initial_health=ANT_INITIAL_HEALTH,
            max_speed=ANT_MAX_SPEED,
            radius=ANT_RADIUS,
            strike_range=ANT_STRIKE_RANGE,
            turn_rate=ANT_TURN_RATE,
            damage=ANT_DAMAGE,
            attack_cost=ANT_ATTACK_COST,
            eating_time=ANT_EATING_TIME,
            sensor_range=ANT_SENSOR_RANGE,
            sensor_angle=ANT_SENSOR_ANGLE,
            density_radius=DENSITY_RADIUS_ANT,
        )

    def get_effective_max_speed(self, zone: float) -> float:
        """Ants move at normal speed in Ants Zone (0.0), but are slowed down to SPIDER_MAX_SPEED in Spiders Zone (1.0)."""
        if zone >= 0.5:
            return SPIDER_MAX_SPEED
        return self._max_speed

    def update(self, dt: float, sensor_data: Any, world: Any | None = None) -> None:
        """Update ant state, check pheromone following reward, and deposit pheromone trail."""
        super().update(dt, sensor_data, world=world)
        if not self.alive or world is None or getattr(world, "pheromone_grid", None) is None:
            return

        gw, gh = world.pheromone_grid.shape
        cell_size = getattr(world, "pheromone_cell_size", 10.0)
        cx = int(max(0.0, min(float(gw - 1), self.position[0] / cell_size)))
        cy = int(max(0.0, min(float(gh - 1), self.position[1] / cell_size)))

        if self._last_tile != (cx, cy):
            current_strength = float(world.pheromone_grid[cx, cy])
            if self._last_tile is not None and current_strength > self._last_tile_strength:
                self.follow_pheromones += current_strength

            world.pheromone_grid[cx, cy] = min(float(world.pheromone_grid[cx, cy]) + PHEROMONE_STRENGTH, 2.0)
            self._last_tile = (cx, cy)
            self._last_tile_strength = float(world.pheromone_grid[cx, cy])

    def compute_fitness(self) -> float:
        """Calculate this ant's fitness score using static weights from ant_constants."""
        return (
            (self.survival_time / 20.0) * FITNESS_SURVIVAL_WEIGHT
            + self.food_eaten * FITNESS_FOOD_WEIGHT
            + self.enemies_touched * FITNESS_ENEMIES_TOUCHED_WEIGHT
            + self.times_eating_for_nothing * FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT
            + self.times_attacking_for_nothing * FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT
            + self.follow_pheromones * FITNESS_FOLLOW_PHEROMONES_WEIGHT
            + self.distance_walked * FITNESS_DISTANCE_WALKED_WEIGHT
        )
