"""
utils.py — Pure Helper Functions
=================================

Stateless utility functions used across the simulator.  These have NO
dependency on game state — they are pure mathematical helpers.

Why a separate utils module?
  Keeping these functions isolated prevents circular imports and makes
  unit testing trivial (no pygame or world state needed).
"""

from __future__ import annotations

import math

import numpy as np


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Restrict *value* to the closed interval [min_val, max_val].

    This is the single most common guard in game math — it prevents
    health from going negative, positions from leaving the screen, etc.

    >>> clamp(150, 0, 100)
    100
    >>> clamp(-5, 0, 100)
    0
    """
    return max(min_val, min(value, max_val))


def normalize_angle(angle: float) -> float:
    """Wrap *angle* into the range (-π, π].

    Why (-π, π] instead of [0, 2π)?
      Signed angles make turning logic intuitive: negative = left,
      positive = right.  ``math.atan2`` already returns values in this
      range, so we stay consistent.

    >>> normalize_angle(math.pi + 0.1)  # just past π wraps to negative
    -3.041592653589793
    """
    # The modulo trick: shift into (0, 2π] then subtract π.
    return (angle + math.pi) % (2 * math.pi) - math.pi


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D position vectors.

    We use ``np.linalg.norm`` under the hood — it's readable and fast
    enough for our scale (a few hundred distance checks per frame).

    Parameters
    ----------
    a, b : np.ndarray of shape (2,)
        Position vectors [x, y].
    """
    return float(np.linalg.norm(a - b))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle (radians) from point *a* to point *b*.

    Result is in (-π, π] — compatible with our heading convention where
    0 = pointing right, π/2 = pointing down (screen coords, y-axis is
    inverted).

    Parameters
    ----------
    a, b : np.ndarray of shape (2,)
        Source and target position vectors.
    """
    delta = b - a
    return float(math.atan2(delta[1], delta[0]))


def point_in_rect(point: np.ndarray, width: float, height: float) -> bool:
    """Check whether *point* lies inside the rectangle (0, 0, width, height).

    Used to test if an ant or food item is still within the world bounds.

    Parameters
    ----------
    point : np.ndarray of shape (2,)
        The [x, y] coordinate to test.
    width, height : float
        Dimensions of the rectangle anchored at the origin.
    """
    x, y = float(point[0]), float(point[1])
    return 0 <= x <= width and 0 <= y <= height


class SpeciesStats:
    """Tracks simulation-wide historical maximums separately for Ants and Spiders."""
    # Ant maximums
    ant_max_lifetime: float = 1e-5
    ant_max_foodeaten: int = 1
    ant_max_enemies_touched: int = 1

    # Spider maximums
    spider_max_lifetime: float = 1e-5
    spider_max_foodeaten: int = 1
    spider_max_enemies_touched: int = 1

    @classmethod
    def reset(cls) -> None:
        """Reset historical maximums when the simulation resets."""
        cls.ant_max_lifetime = 1e-5
        cls.ant_max_foodeaten = 1
        cls.ant_max_enemies_touched = 1

        cls.spider_max_lifetime = 1e-5
        cls.spider_max_foodeaten = 1
        cls.spider_max_enemies_touched = 1

    @classmethod
    def update_ant(cls, lifetime: float, foodeaten: int, enemies_touched: int) -> None:
        if lifetime > cls.ant_max_lifetime:
            cls.ant_max_lifetime = lifetime
        if foodeaten > cls.ant_max_foodeaten:
            cls.ant_max_foodeaten = foodeaten
        if enemies_touched > cls.ant_max_enemies_touched:
            cls.ant_max_enemies_touched = enemies_touched

    @classmethod
    def update_spider(cls, lifetime: float, foodeaten: int, enemies_touched: int) -> None:
        if lifetime > cls.spider_max_lifetime:
            cls.spider_max_lifetime = lifetime
        if foodeaten > cls.spider_max_foodeaten:
            cls.spider_max_foodeaten = foodeaten
        if enemies_touched > cls.spider_max_enemies_touched:
            cls.spider_max_enemies_touched = enemies_touched


# Backwards compatibility alias
GlobalStats = SpeciesStats
