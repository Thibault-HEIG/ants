"""
food.py — Food Items and Food Sources
======================================

Provides typed food entities (Sugar, Seed) that inherit from Entity, and
FoodSource — a temporary spawner that generates food items around itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    def on_consume(self) -> float:
        """Mark as consumed and return nutritional value."""
        self.consumed = True
        return self.nutrition_value

    def __repr__(self) -> str:
        status = "eaten" if self.consumed else "available"
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

    def update(self, dt: float, current_food_count: int) -> list[Food]:
        """Advance the source by one tick and return any newly spawned food.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        current_food_count : int
            Current global food count (used to respect MAX_FOOD cap).

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

            # Scatter position around the source centre
            angle = self._rng.uniform(0, 2 * np.pi)
            dist = self._rng.uniform(0, FOOD_SOURCE_SPAWN_RADIUS)
            offset = np.array([np.cos(angle) * dist, np.sin(angle) * dist])
            food_pos = self.position + offset

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
