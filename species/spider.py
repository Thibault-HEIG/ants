"""
spider.py — The Spider Species Implementation
=============================================

Inherits shared spatial, lifecycle, and neural mechanics from Creature.
Overrides species-specific constants, zone speed behaviors, and fitness logic.
"""

from __future__ import annotations

from typing import Any
import math

import numpy as np

from species.creature import Creature
from species.spider_constants import (
    SPIDER_METRIC_BOUNDS,
    SPIDER_COUNT,
    SPIDER_STRIKE_RANGE,
    SPIDER_TURN_RATE,
    SPIDER_DAMAGE,
    SPIDER_ATTACK_COST,
    SPIDER_EATING_TIME,
    SPIDER_REPRODUCTION_THRESHOLD,
    MAX_SPIDERS,
    FITNESS_SURVIVAL_WEIGHT,
    FITNESS_FOOD_WEIGHT,
    FITNESS_ENEMIES_TOUCHED_WEIGHT,
    FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT,
    FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT,
    FITNESS_TILES_COVERED_WEIGHT,
    FITNESS_BRAIN_ORIGINALITY_WEIGHT,
    CAN_ATTACK,
    CAN_TAKE,
    CAN_MAKE,
    CAN_EAT,
    SPIDER_SPEED_MIN,
    SPIDER_SPEED_MAX,
    SPIDER_BASE_RADIUS,
    SPIDER_HP_MIN,
    SPIDER_HP_MAX,
    SPIDER_VISION_RANGE_MIN,
    SPIDER_VISION_RANGE_MAX,
    SPIDER_FOV_MIN_DEG,
    SPIDER_FOV_MAX_DEG,
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
    npc: bool = False
    metrics: dict[str, Any] = SPIDER_METRIC_BOUNDS
    strike_range: float = SPIDER_STRIKE_RANGE
    turn_rate: float = SPIDER_TURN_RATE
    damage: float = SPIDER_DAMAGE
    attack_cost: float = SPIDER_ATTACK_COST
    eating_time: float = SPIDER_EATING_TIME
    reproduction_threshold: float = SPIDER_REPRODUCTION_THRESHOLD
    max_population: int = MAX_SPIDERS
    initial_count: int = SPIDER_COUNT

    trait_bounds_config: dict[str, float] = {
        "hp_min": SPIDER_HP_MIN,
        "hp_max": SPIDER_HP_MAX,
        "speed_min": SPIDER_SPEED_MIN,
        "speed_max": SPIDER_SPEED_MAX,
        "vision_range_min": SPIDER_VISION_RANGE_MIN,
        "vision_range_max": SPIDER_VISION_RANGE_MAX,
        "fov_min": SPIDER_FOV_MIN_DEG,
        "fov_max": SPIDER_FOV_MAX_DEG,
        "base_radius": SPIDER_BASE_RADIUS,
    }

    def __init__(self, position: np.ndarray, rng: np.random.Generator) -> None:
        super().__init__(
            position,
            rng,
            strike_range=SPIDER_STRIKE_RANGE,
            turn_rate=SPIDER_TURN_RATE,
            damage=SPIDER_DAMAGE,
            attack_cost=SPIDER_ATTACK_COST,
            eating_time=SPIDER_EATING_TIME,
            can_attack=CAN_ATTACK,
            can_take=CAN_TAKE,
            can_make=CAN_MAKE,
            can_eat=CAN_EAT,
        )

    def get_effective_max_speed(self, zone: float) -> float:
        """Spiders always move at their normal max speed across all zones."""
        return self._max_speed

    def compute_fitness(self, force: bool = False) -> float:
        """Calculate this spider's fitness score using normalized metrics and brain originality."""
        cached = self._check_cached_fitness(force=force)
        if cached is not None:
            return cached

        self.brain_originality = self.compute_brain_originality()
        
        # Food
        food_eaten = self.normalize_metric("computed_food_eaten") * FITNESS_FOOD_WEIGHT
        eating_for_nothing = self.normalize_metric("times_eating_for_nothing") * FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT
        
        # Combat
        enemies_touched = self.normalize_metric("computed_enemies_touched") * FITNESS_ENEMIES_TOUCHED_WEIGHT
        attacking_for_nothing = self.normalize_metric("times_attacking_for_nothing") * FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT
        
        # Behavior
        survival_time = self.normalize_metric("survival_time") * FITNESS_SURVIVAL_WEIGHT
        tiles_covered = self.normalize_metric("tiles_covered") * FITNESS_TILES_COVERED_WEIGHT
        
        # Total fitness
        total = (food_eaten + eating_for_nothing + enemies_touched + attacking_for_nothing + survival_time + tiles_covered)
        
        sum_weights = (
            abs(FITNESS_FOOD_WEIGHT) + abs(FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT) +
            abs(FITNESS_ENEMIES_TOUCHED_WEIGHT) + abs(FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT) +
            abs(FITNESS_SURVIVAL_WEIGHT) + abs(FITNESS_TILES_COVERED_WEIGHT)
        )
        if sum_weights != 0:
            total = (total / sum_weights) * 100.0
        else:
            total = 0.0
        
        result = total * (1 - FITNESS_BRAIN_ORIGINALITY_WEIGHT) + (self.brain_originality * 100.0 * FITNESS_BRAIN_ORIGINALITY_WEIGHT)
        return self._store_cached_fitness(result)
