"""
food.py — Food Items and Food Sources
======================================

Provides typed food entities (Sugar, Seed) that inherit from Entity, and
FoodSource — a temporary spawner that generates food items around itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants import (
    SUGAR_NUTRITION,
    SEED_NUTRITION,
    SUGAR_WEIGHT,
    FOOD_RADIUS,
    MAX_FOOD,
    FOOD_SOURCE_RADIUS,
    FOOD_SOURCE_LIFETIME,
    FOOD_SOURCE_SPAWN_RATE,
    FOOD_SOURCE_SPAWN_RADIUS,
)
from world.entity import Entity


def is_in_lake(pos: np.ndarray, radius: float, lakes: list[Any] | None = None) -> bool:
    """Check if a circle at pos with given radius overlaps any lake obstacle."""
    if not lakes:
        return False
    px = float(pos[0])
    py = float(pos[1])
    for lake in lakes:
        lx = float(lake.position[0])
        ly = float(lake.position[1])
        lradius = float(getattr(lake, "radius", 50.0))
        dx = px - lx
        dy = py - ly
        min_dist = lradius + radius
        if dx * dx + dy * dy < min_dist * min_dist:
            return True
    return False


def is_in_home(pos: np.ndarray, radius: float, kingdoms: Any = None) -> bool:
    """Check if a circle at pos with given radius overlaps any kingdom (home) obstacle."""
    if not kingdoms:
        return False
    px = float(pos[0])
    py = float(pos[1])
    items = kingdoms.values() if isinstance(kingdoms, dict) else kingdoms
    for kingdom in items:
        kx = float(kingdom.position[0])
        ky = float(kingdom.position[1])
        kradius = float(getattr(kingdom, "spawn_radius", 60.0))
        dx = px - kx
        dy = py - ky
        min_dist = kradius + radius
        if dx * dx + dy * dy < min_dist * min_dist:
            return True
    return False


class Food(Entity):
    """Abstract base for consumable food items placed in the world.

    Parameters
    ----------
    position : np.ndarray
        [x, y] world coordinates.
    nutrition_value : float
        HP restored when consumed.
    food_type : str
        Identifier used by the renderer to select a sprite.
    """

    def __init__(self, position: np.ndarray, nutrition_value: float, food_type: str) -> None:
        super().__init__(position, radius=FOOD_RADIUS)
        self.nutrition_value: float = nutrition_value
        self.food_type: str = food_type
        self.being_carried: bool = False

    def on_consume(self) -> float:
        """Mark as consumed and return nutritional value."""
        self.consumed = True
        return self.nutrition_value

    def __repr__(self) -> str:
        if self.consumed:
            status = "eaten"
        elif self.being_carried:
            status = "carried"
        else:
            status = "available"
        return f"{self.food_type}(pos=[{self.position[0]:.0f}, {self.position[1]:.0f}], {status})"


class Sugar(Food):
    """High-nutrition food item (+60 HP by default)."""

    def __init__(self, position: np.ndarray) -> None:
        super().__init__(position, nutrition_value=SUGAR_NUTRITION, food_type="sugar")


class Seed(Food):
    """Standard-nutrition food item (+30 HP by default)."""

    def __init__(self, position: np.ndarray) -> None:
        super().__init__(position, nutrition_value=SEED_NUTRITION, food_type="seed")


class FoodSource:
    """A temporary spot on the map that generates food items around itself.

    Food sources last for a fixed duration and periodically scatter Sugar or
    Seed items within a spawn radius.  They are not Entity subclasses — they
    are lifecycle managers, not consumable objects.

    Parameters
    ----------
    position : np.ndarray
        [x, y] centre of the food source.
    rng : np.random.Generator
        Seeded RNG for food type selection and scatter positioning.
    """

    def __init__(self, position: np.ndarray, rng: np.random.Generator) -> None:
        self.position: np.ndarray = position.astype(float)
        self.radius: float = FOOD_SOURCE_RADIUS
        self.lifetime: float = FOOD_SOURCE_LIFETIME
        self.spawn_timer: float = 0.0
        self._rng: np.random.Generator = rng

    @property
    def expired(self) -> bool:
        """True when this source has outlived its duration."""
        return self.lifetime <= 0.0

    def update(
        self,
        dt: float,
        current_food_count: int,
        lakes: list[Any] | None = None,
        kingdoms: Any = None,
    ) -> list[Food]:
        """Advance the source by one tick and return any newly spawned food.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        current_food_count : int
            Current global food count (used to respect MAX_FOOD cap).
        lakes : list[Any] | None, optional
            Static lake obstacles to avoid when scattering food.
        kingdoms : Any, optional
            Species kingdoms (homes) to avoid when scattering food.

        Returns
        -------
        list[Food]
            Newly spawned food items (may be empty).
        """
        self.lifetime -= dt
        if self.lifetime <= 0.0:
            return []

        self.spawn_timer += dt * FOOD_SOURCE_SPAWN_RATE
        spawned: list[Food] = []

        while self.spawn_timer >= 1.0 and (current_food_count + len(spawned)) < MAX_FOOD:
            self.spawn_timer -= 1.0

            # Scatter position around the source centre, avoiding lakes and homes
            food_pos = None
            for _ in range(10):
                angle = self._rng.uniform(0, 2 * np.pi)
                dist = self._rng.uniform(0, FOOD_SOURCE_SPAWN_RADIUS)
                offset = np.array([np.cos(angle) * dist, np.sin(angle) * dist])
                candidate = self.position + offset
                if not is_in_lake(candidate, float(FOOD_RADIUS), lakes) and not is_in_home(candidate, float(FOOD_RADIUS), kingdoms):
                    food_pos = candidate
                    break

            if food_pos is None:
                continue

            # Choose sugar vs seed based on configured weights
            if self._rng.random() < SUGAR_WEIGHT:
                spawned.append(Sugar(food_pos))
            else:
                spawned.append(Seed(food_pos))

        return spawned

    def __repr__(self) -> str:
        return (
            f"FoodSource(pos=[{self.position[0]:.0f}, {self.position[1]:.0f}], "
            f"ttl={self.lifetime:.1f}s)"
        )
