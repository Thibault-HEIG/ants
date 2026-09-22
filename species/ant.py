"""
ant.py — The Ant Species Implementation
=======================================

Inherits shared spatial, lifecycle, and neural mechanics from Creature.
Overrides species-specific constants and fitness evaluation logic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.constants import WORLD_WIDTH, WORLD_HEIGHT
from core.utils import normalize_angle
from species.creature import Creature
from species.spider_constants import SPIDER_MAX_SPEED
from species.ant_constants import (
    ANT_METRIC_BOUNDS,
    ANT_COUNT,
    ANT_STRIKE_RANGE,
    ANT_TURN_RATE,
    ANT_DAMAGE,
    ANT_ATTACK_COST,
    ANT_EATING_TIME,
    ANT_REPRODUCTION_THRESHOLD,
    MAX_ANTS,
    ANT_VISION_RANGE_MIN,
    ANT_VISION_RANGE_MAX,
    ANT_FOV_MIN_DEG,
    ANT_FOV_MAX_DEG,
    ANT_SPEED_MIN,
    ANT_SPEED_MAX,
    ANT_HP_MIN,
    ANT_HP_MAX,
    ANT_BASE_RADIUS,
    FITNESS_SURVIVAL_WEIGHT,
    FITNESS_FOOD_WEIGHT,
    FITNESS_ENEMIES_TOUCHED_WEIGHT,
    FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT,
    FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT,
    PHEROMONE_STRENGTH,
    PHEROMONE_MAX_STRENGTH,
    PHEROMONE_COOLDOWN,
    FITNESS_FOLLOW_PHEROMONES_WEIGHT,
    FITNESS_TILES_COVERED_WEIGHT,
    FITNESS_BRAIN_ORIGINALITY_WEIGHT,
    FITNESS_TAKEN_OBJECT_WEIGHT,
    FITNESS_WALKING_CARRYING_WEIGHT,
    FITNESS_RELEASE_ANYWHERE_WEIGHT,
    FITNESS_RELEASE_AT_HOME_WEIGHT,
    FITNESS_RELEASED_PHEROMONE_AROUND_FOOD_SOURCE_WEIGHT,
    CAN_ATTACK,
    CAN_TAKE,
    CAN_MAKE,
    CAN_EAT,
)
from evolution.sensors import Sensors

# Pre-computed normalisation constant for pheromone-near-food distance scoring
_PHEROMONE_DIST_MAX: float = math.sqrt(float(WORLD_WIDTH) ** 2 + float(WORLD_HEIGHT) ** 2) / 3.0


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
    npc: bool = False
    metrics: dict[str, Any] = ANT_METRIC_BOUNDS
    initial_health: float = (ANT_HP_MIN + ANT_HP_MAX) / 2.0
    max_speed: float = (ANT_SPEED_MIN + ANT_SPEED_MAX) / 2.0
    radius: float = ANT_BASE_RADIUS
    strike_range: float = ANT_STRIKE_RANGE
    turn_rate: float = ANT_TURN_RATE
    damage: float = ANT_DAMAGE
    attack_cost: float = ANT_ATTACK_COST
    eating_time: float = ANT_EATING_TIME
    sensor_range: float = (ANT_VISION_RANGE_MIN + ANT_VISION_RANGE_MAX) / 2.0
    reproduction_threshold: float = ANT_REPRODUCTION_THRESHOLD
    max_population: int = MAX_ANTS
    initial_count: int = ANT_COUNT

    def __init__(self, position: np.ndarray, rng: np.random.Generator) -> None:
        # Default mid-gene values for initial construction (overridden by apply_trait_genes)
        mid_health = (ANT_HP_MIN + ANT_HP_MAX) / 2.0
        mid_speed = (ANT_SPEED_MIN + ANT_SPEED_MAX) / 2.0
        mid_range = (ANT_VISION_RANGE_MIN + ANT_VISION_RANGE_MAX) / 2.0
        mid_fov_rad = math.radians((ANT_FOV_MIN_DEG + ANT_FOV_MAX_DEG) / 2.0) / 2.0

        super().__init__(
            position,
            rng,
            initial_health=mid_health,
            max_speed=mid_speed,
            radius=ANT_BASE_RADIUS,
            strike_range=ANT_STRIKE_RANGE,
            turn_rate=ANT_TURN_RATE,
            damage=ANT_DAMAGE,
            attack_cost=ANT_ATTACK_COST,
            eating_time=ANT_EATING_TIME,
            sensor_range=mid_range,
            sensor_angle=mid_fov_rad,
            density_radius=mid_range,
            can_attack=CAN_ATTACK,
            can_take=CAN_TAKE,
            can_make=CAN_MAKE,
            can_eat=CAN_EAT,
        )
        # Attributes for serialization (overridden when genome is installed)
        self.vision_range: float = mid_range
        self.fov: float = (ANT_FOV_MIN_DEG + ANT_FOV_MAX_DEG) / 2.0

    def apply_trait_genes(self) -> None:
        """Decode vision_gene and physique_gene, update all derived physical traits."""
        trait_genes = self.brain.trait_genes
        vision_gene = float(trait_genes[0])
        physique_gene = float(trait_genes[1])

        # Vision trade-off: high gene → long range, narrow FOV
        self.vision_range = ANT_VISION_RANGE_MIN + vision_gene * (ANT_VISION_RANGE_MAX - ANT_VISION_RANGE_MIN)
        self.fov = ANT_FOV_MAX_DEG - vision_gene * (ANT_FOV_MAX_DEG - ANT_FOV_MIN_DEG)

        # Physique trade-off: high gene → slow, tanky
        new_initial_health = ANT_HP_MIN + physique_gene * (ANT_HP_MAX - ANT_HP_MIN)
        self._max_speed = ANT_SPEED_MAX - physique_gene * (ANT_SPEED_MAX - ANT_SPEED_MIN)

        # Derived body size scales with HP
        hp_ratio = (new_initial_health - ANT_HP_MIN) / (ANT_HP_MAX - ANT_HP_MIN)
        self.radius = ANT_BASE_RADIUS * (0.75 + 0.5 * hp_ratio)

        # Apply HP values (fresh spawn starts at max)
        self.max_health = new_initial_health
        self.health = new_initial_health

        # Reconfigure sensors in-place with new FOV and range
        fov_half_rad = math.radians(self.fov) / 2.0
        self.sensors.reconfigure(
            sensor_range=self.vision_range,
            sensor_angle=fov_half_rad,
            density_radius=self.vision_range,
        )

    def get_effective_max_speed(self, zone: float) -> float:
        """Ants move at normal speed in Ants Zone (0.0), but are slowed down to SPIDER_MAX_SPEED in Spiders Zone (1.0)."""
        if zone >= 0.5:
            return SPIDER_MAX_SPEED
        return self._max_speed

    def update(self, dt: float, sensor_data: Any, world: Any | None = None) -> None:
        """Update ant state: pheromone following reward and brain-controlled pheromone release."""
        super().update(dt, sensor_data, world=world)
        if not self.alive or world is None or getattr(world, "pheromone_grid", None) is None:
            return

        cx, cy = self.current_tile

        # Pheromone following reward on tile transition
        if self._last_tile != (cx, cy):
            current_strength = float(world.pheromone_grid[cx, cy])
            if self._last_tile is not None and current_strength > self._last_tile_strength:
                self.follow_pheromones += current_strength
            self._last_tile = (cx, cy)
            self._last_tile_strength = float(world.pheromone_grid[cx, cy])

        # Brain-controlled pheromone release
        if self.make_signal and self._pheromone_cooldown_timer <= 0.0:
            world.pheromone_grid[cx, cy] = min(
                float(world.pheromone_grid[cx, cy]) + PHEROMONE_STRENGTH,
                PHEROMONE_MAX_STRENGTH,
            )
            self._pheromone_cooldown_timer = PHEROMONE_COOLDOWN

            score = self._compute_pheromone_placement_score(world)
            self.released_pheromone_around_food_source += score

    def _compute_pheromone_placement_score(self, world: Any) -> float:
        """Score a pheromone release based on proximity to the nearest food source
        and alignment with the anthill→source corridor.

        Returns a value in [0, 2]: distance_score [0,1] + is_between [-0.5,1].
        """
        food_sources = getattr(world, "food_sources", [])
        if not food_sources:
            return 0.0

        ax = float(self.position[0])
        ay = float(self.position[1])

        # Find nearest active food source (O(n) where n ≤ MAX_FOOD_SOURCES ≈ 10)
        best_dist_sq = float("inf")
        best_source = None
        for source in food_sources:
            sx = float(source.position[0])
            sy = float(source.position[1])
            d_sq = (ax - sx) ** 2 + (ay - sy) ** 2
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_source = source

        if best_source is None:
            return 0.0

        # Distance score: inverse distance normalised by world_diagonal / 3
        raw_dist = math.sqrt(best_dist_sq)
        distance_score = max(0.0, 1.0 - raw_dist / _PHEROMONE_DIST_MAX)

        # Is-between score: angle alignment + projection onto anthill→source segment
        kingdom = world.kingdoms.get(type(self)) if hasattr(world, "kingdoms") else None
        if kingdom is None:
            return distance_score

        hx = float(kingdom.position[0])
        hy = float(kingdom.position[1])
        sx = float(best_source.position[0])
        sy = float(best_source.position[1])

        # Vector from anthill to food source
        seg_dx = sx - hx
        seg_dy = sy - hy
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq < 1e-9:
            return distance_score

        # Distance from ant point (ax, ay) to the anthill→source line
        perp_dist = abs(seg_dx * (hy - ay) - (hx - ax) * seg_dy) / math.sqrt(seg_len_sq)

        # Projection of ant position onto the anthill→source segment, clamped to [0, 1]
        proj_t = ((ax - hx) * seg_dx + (ay - hy) * seg_dy) / seg_len_sq

        # Corridor is +- 50px wide and between the anthill and food source
        is_between = 1.0 if (perp_dist <= 50.0 and 0.0 <= proj_t <= 1.0) else -0.5

        return distance_score + is_between

    def compute_fitness(self, force: bool = False) -> float:
        """Calculate this ant's fitness score using normalized metrics and brain originality."""
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
        follow_pheromones = self.normalize_metric("follow_pheromones") * FITNESS_FOLLOW_PHEROMONES_WEIGHT
        tiles_covered = self.normalize_metric("tiles_covered") * FITNESS_TILES_COVERED_WEIGHT

        # Carry-object
        taken_object = self.normalize_metric("computed_taken_object") * FITNESS_TAKEN_OBJECT_WEIGHT
        # walking_carrying is already in [-1, 1], so we don't normalize it, just scale by weight
        walk_carry = self.walking_carrying * FITNESS_WALKING_CARRYING_WEIGHT
        release_anywhere = self.normalize_metric("computed_release_anywhere") * FITNESS_RELEASE_ANYWHERE_WEIGHT
        release_at_home = self.normalize_metric("release_at_home_count") * FITNESS_RELEASE_AT_HOME_WEIGHT

        # Pheromone placement
        pheromone_placement = self.normalize_metric("released_pheromone_around_food_source") * FITNESS_RELEASED_PHEROMONE_AROUND_FOOD_SOURCE_WEIGHT
        
        # Total fitness
        total = (food_eaten + eating_for_nothing + enemies_touched + attacking_for_nothing + follow_pheromones + survival_time + tiles_covered + taken_object + walk_carry + release_anywhere + release_at_home + pheromone_placement)
        
        sum_weights = (
            abs(FITNESS_FOOD_WEIGHT) + abs(FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT) +
            abs(FITNESS_ENEMIES_TOUCHED_WEIGHT) + abs(FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT) +
            abs(FITNESS_SURVIVAL_WEIGHT) + abs(FITNESS_FOLLOW_PHEROMONES_WEIGHT) +
            abs(FITNESS_TILES_COVERED_WEIGHT) + abs(FITNESS_TAKEN_OBJECT_WEIGHT) +
            abs(FITNESS_WALKING_CARRYING_WEIGHT) +
            abs(FITNESS_RELEASE_ANYWHERE_WEIGHT) + abs(FITNESS_RELEASE_AT_HOME_WEIGHT) +
            abs(FITNESS_RELEASED_PHEROMONE_AROUND_FOOD_SOURCE_WEIGHT)
        )
        if sum_weights != 0:
            total = (total / sum_weights) * 100.0
        else:
            total = 0.0
        
        result = total * (1 - FITNESS_BRAIN_ORIGINALITY_WEIGHT) + (self.brain_originality * 100.0 * FITNESS_BRAIN_ORIGINALITY_WEIGHT)
        return self._store_cached_fitness(result)