"""
spider.py — The Spider Species Implementation
=============================================

Inherits shared spatial, lifecycle, and neural mechanics from Creature.
Overrides species-specific constants, zone speed behaviors, and fitness logic.
"""

from __future__ import annotations

import numpy as np

from species.creature import Creature
from species.spider_constants import (
    SPIDER_COUNT,
    SPIDER_INITIAL_HEALTH,
    SPIDER_MAX_SPEED,
    SPIDER_RADIUS,
    SPIDER_STRIKE_RANGE,
    SPIDER_TURN_RATE,
    SPIDER_DAMAGE,
    SPIDER_ATTACK_COST,
    SPIDER_EATING_TIME,
    SPIDER_SENSOR_RANGE,
    SPIDER_REPRODUCTION_THRESHOLD,
    MAX_SPIDERS,
    DENSITY_RADIUS_SPIDER,
    FITNESS_SURVIVAL_WEIGHT,
    FITNESS_FOOD_WEIGHT,
    FITNESS_ENEMIES_TOUCHED_WEIGHT,
    FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT,
    FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT,
)


class Spider(Creature):
    """The Spider evolving agent (co-evolving opponent to Ants).

    Parameters
    ----------
    position : np.ndarray
        Starting [x, y] coordinates.
    rng : np.random.Generator
        Seeded random number generator.
    """

    species_name: str = "Spider"
    initial_health: float = SPIDER_INITIAL_HEALTH
    max_speed: float = SPIDER_MAX_SPEED
    radius: float = float(SPIDER_RADIUS)
    strike_range: float = SPIDER_STRIKE_RANGE
    turn_rate: float = SPIDER_TURN_RATE
    damage: float = SPIDER_DAMAGE
    attack_cost: float = SPIDER_ATTACK_COST
    eating_time: float = SPIDER_EATING_TIME
    sensor_range: float = SPIDER_SENSOR_RANGE
    reproduction_threshold: float = SPIDER_REPRODUCTION_THRESHOLD
    max_population: int = MAX_SPIDERS
    initial_count: int = SPIDER_COUNT

    def __init__(self, position: np.ndarray, rng: np.random.Generator) -> None:
        super().__init__(
            position,
            rng,
            initial_health=SPIDER_INITIAL_HEALTH,
            max_speed=SPIDER_MAX_SPEED,
            radius=SPIDER_RADIUS,
            strike_range=SPIDER_STRIKE_RANGE,
            turn_rate=SPIDER_TURN_RATE,
            damage=SPIDER_DAMAGE,
            attack_cost=SPIDER_ATTACK_COST,
            eating_time=SPIDER_EATING_TIME,
            sensor_range=SPIDER_SENSOR_RANGE,
            density_radius=DENSITY_RADIUS_SPIDER,
        )

    def get_effective_max_speed(self, zone: float) -> float:
        """Spiders always move at their normal max speed across all zones."""
        return self._max_speed

    def compute_fitness(self) -> float:
        """Calculate this spider's fitness score using static weights from spider_constants."""
        return (
            (self.survival_time / 20.0) * FITNESS_SURVIVAL_WEIGHT
            + self.food_eaten * FITNESS_FOOD_WEIGHT
            + self.enemies_touched * FITNESS_ENEMIES_TOUCHED_WEIGHT
            + self.times_eating_for_nothing * FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT
            + self.times_attacking_for_nothing * FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT
        )
