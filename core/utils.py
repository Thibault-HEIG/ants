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
    """Euclidean distance between two 2-D position vectors."""
    return float(np.linalg.norm(a - b))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle (radians) from point *a* to point *b*."""
    delta = b - a
    return float(math.atan2(delta[1], delta[0]))


def point_in_rect(point: np.ndarray, width: float, height: float) -> bool:
    """Check whether *point* lies inside the rectangle (0, 0, width, height)."""
    x, y = float(point[0]), float(point[1])
    return 0 <= x <= width and 0 <= y <= height


class SpeciesStats:
    """Tracks simulation-wide historical maximums dynamically per species."""
    max_lifetime: dict[str, float] = {}
    max_foodeaten: dict[str, int] = {}
    max_enemies_touched: dict[str, int] = {}

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

        cls.ant_max_lifetime = 1e-5
        cls.ant_max_foodeaten = 1
        cls.ant_max_enemies_touched = 1

        cls.spider_max_lifetime = 1e-5
        cls.spider_max_foodeaten = 1
        cls.spider_max_enemies_touched = 1

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
