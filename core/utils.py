"""
utils.py — Pure Helper Functions and Global Statistics
======================================================

Stateless utility functions and global stats tracking used across the simulator.
"""

from __future__ import annotations

import math
import numpy as np


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Restrict *value* to the closed interval [min_val, max_val]."""
    return max(min_val, min(value, max_val))


def normalize_angle(angle: float) -> float:
    """Wrap *angle* into the range (-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D position vectors.

    Uses math.hypot on raw floats instead of np.linalg.norm to avoid
    NumPy dispatch overhead on scalar pairs.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_squared(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance (avoids the sqrt when only comparing)."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle (radians) from point *a* to point *b*."""
    delta = b - a
    return float(math.atan2(delta[1], delta[0]))


def point_in_rect(point: np.ndarray, width: float, height: float) -> bool:
    """Check whether *point* lies inside the rectangle (0, 0, width, height)."""
    x, y = float(point[0]), float(point[1])
    return 0 <= x <= width and 0 <= y <= height


class SpeciesStats:
    """Tracks simulation-wide historical maximums and cumulative totals dynamically per species."""
    max_lifetime: dict[str, float] = {}
    max_foodeaten: dict[str, int] = {}
    max_enemies_touched: dict[str, int] = {}
    max_metrics: dict[str, dict[str, float]] = {}

    # All-time historical maximums across all creatures ever spawned
    max_fitness: dict[str, float] = {}
    max_computed_food: dict[str, float] = {}
    max_computed_enemies: dict[str, float] = {}

    # Accumulators for all creatures that have died (to compute all-time averages combined with living)
    total_dead_count: dict[str, int] = {}
    sum_dead_fitness: dict[str, float] = {}
    sum_dead_food: dict[str, float] = {}
    sum_dead_computed_food: dict[str, float] = {}
    sum_dead_enemies: dict[str, float] = {}
    sum_dead_computed_enemies: dict[str, float] = {}
    sum_dead_lifetime: dict[str, float] = {}

    # Backwards compatibility attributes
    ant_max_lifetime: float = 1e-5
    ant_max_foodeaten: int = 1
    ant_max_enemies_touched: int = 1

    spider_max_lifetime: float = 1e-5
    spider_max_foodeaten: int = 1
    spider_max_enemies_touched: int = 1

    @classmethod
    def reset(cls) -> None:
        """Reset historical maximums when the simulation resets."""
        cls.max_lifetime.clear()
        cls.max_foodeaten.clear()
        cls.max_enemies_touched.clear()
        cls.max_metrics.clear()
        cls.max_fitness.clear()
        cls.max_computed_food.clear()
        cls.max_computed_enemies.clear()

        cls.total_dead_count.clear()
        cls.sum_dead_fitness.clear()
        cls.sum_dead_food.clear()
        cls.sum_dead_computed_food.clear()
        cls.sum_dead_enemies.clear()
        cls.sum_dead_computed_enemies.clear()
        cls.sum_dead_lifetime.clear()

        cls.ant_max_lifetime = 1e-5
        cls.ant_max_foodeaten = 1
        cls.ant_max_enemies_touched = 1

        cls.spider_max_lifetime = 1e-5
        cls.spider_max_foodeaten = 1
        cls.spider_max_enemies_touched = 1

    @classmethod
    def record_dead_creature(cls, creature: Any) -> None:
        """Record final stats of a creature upon death or generation completion for all-time averages."""
        species_name = getattr(creature, "species_name", getattr(type(creature), "species_name", type(creature).__name__))
        cls.total_dead_count[species_name] = cls.total_dead_count.get(species_name, 0) + 1
        try:
            fit = float(creature.compute_fitness(force=True))
        except TypeError:
            fit = float(creature.compute_fitness())
        cls.sum_dead_fitness[species_name] = cls.sum_dead_fitness.get(species_name, 0.0) + fit
        cls.sum_dead_food[species_name] = cls.sum_dead_food.get(species_name, 0.0) + float(getattr(creature, "food_eaten", 0))
        cls.sum_dead_computed_food[species_name] = cls.sum_dead_computed_food.get(species_name, 0.0) + float(getattr(creature, "computed_food_eaten", 0.0))
        cls.sum_dead_enemies[species_name] = cls.sum_dead_enemies.get(species_name, 0.0) + float(getattr(creature, "enemies_touched", 0))
        cls.sum_dead_computed_enemies[species_name] = cls.sum_dead_computed_enemies.get(species_name, 0.0) + float(getattr(creature, "computed_enemies_touched", 0.0))
        cls.sum_dead_lifetime[species_name] = cls.sum_dead_lifetime.get(species_name, 0.0) + float(getattr(creature, "survival_time", 0.0))
        cls.update_metrics(creature)
        cls.update(species_name, float(getattr(creature, "survival_time", 0.0)), int(getattr(creature, "food_eaten", 0)), int(getattr(creature, "enemies_touched", 0)))

    @classmethod
    def update_metrics(cls, creature: Any) -> None:
        species_name = getattr(creature, "species_name", "Unknown")
        if species_name not in cls.max_metrics:
            cls.max_metrics[species_name] = {}
        metric_bounds = getattr(creature, "metrics", {})
        for metric_name in metric_bounds:
            value = float(getattr(creature, metric_name, 0.0))
            if value > cls.max_metrics[species_name].get(metric_name, 0.0):
                cls.max_metrics[species_name][metric_name] = value

        # Track all-time peak fitness and computed scores
        try:
            try:
                fit = float(creature.compute_fitness(force=True))
            except TypeError:
                fit = float(creature.compute_fitness())
            if fit > cls.max_fitness.get(species_name, 0.0):
                cls.max_fitness[species_name] = fit
        except Exception:
            pass

        cf = float(getattr(creature, "computed_food_eaten", 0.0))
        if cf > cls.max_computed_food.get(species_name, 0.0):
            cls.max_computed_food[species_name] = cf

        ce = float(getattr(creature, "computed_enemies_touched", 0.0))
        if ce > cls.max_computed_enemies.get(species_name, 0.0):
            cls.max_computed_enemies[species_name] = ce

    @classmethod
    def update(cls, species_name: str, lifetime: float, foodeaten: int, enemies_touched: int) -> None:
        if lifetime > cls.max_lifetime.get(species_name, 1e-5):
            cls.max_lifetime[species_name] = lifetime
        if foodeaten > cls.max_foodeaten.get(species_name, 1):
            cls.max_foodeaten[species_name] = foodeaten
        if enemies_touched > cls.max_enemies_touched.get(species_name, 1):
            cls.max_enemies_touched[species_name] = enemies_touched

        if species_name == "Ant":
            cls.ant_max_lifetime = cls.max_lifetime.get("Ant", 1e-5)
            cls.ant_max_foodeaten = cls.max_foodeaten.get("Ant", 1)
            cls.ant_max_enemies_touched = cls.max_enemies_touched.get("Ant", 1)
        elif species_name == "Spider":
            cls.spider_max_lifetime = cls.max_lifetime.get("Spider", 1e-5)
            cls.spider_max_foodeaten = cls.max_foodeaten.get("Spider", 1)
            cls.spider_max_enemies_touched = cls.max_enemies_touched.get("Spider", 1)

    @classmethod
    def update_ant(cls, lifetime: float, foodeaten: int, enemies_touched: int) -> None:
        cls.update("Ant", lifetime, foodeaten, enemies_touched)

    @classmethod
    def update_spider(cls, lifetime: float, foodeaten: int, enemies_touched: int) -> None:
        cls.update("Spider", lifetime, foodeaten, enemies_touched)


GlobalStats = SpeciesStats

