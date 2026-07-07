"""
ant.py — The Ant Species Implementation
=======================================

Inherits shared spatial, lifecycle, and neural mechanics from Creature.
Overrides species-specific constants and fitness evaluation logic.
"""

from __future__ import annotations

import numpy as np

from species.creature import Creature
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
    ANT_REPRODUCTION_THRESHOLD,
    MAX_ANTS,
    DENSITY_RADIUS_ANT,
    FITNESS_SURVIVAL_WEIGHT,
    FITNESS_FOOD_WEIGHT,
    FITNESS_ENEMIES_TOUCHED_WEIGHT,
    FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT,
    FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT,
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
            density_radius=DENSITY_RADIUS_ANT,
        )

    def compute_fitness(self) -> float:
        """Calculate this ant's fitness score using static weights from ant_constants."""
        return (
            (self.survival_time / 20.0) * FITNESS_SURVIVAL_WEIGHT
            + self.food_eaten * FITNESS_FOOD_WEIGHT
            + self.enemies_touched * FITNESS_ENEMIES_TOUCHED_WEIGHT
            + self.times_eating_for_nothing * FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT
            + self.times_attacking_for_nothing * FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT
        )
