"""
food.py — Food Items
====================

Inherits from Entity. Food items provide nutrition when consumed by creatures.
"""

from __future__ import annotations

import numpy as np

from core.constants import FOOD_NUTRITION, FOOD_RADIUS
from world.entity import Entity


class Food(Entity):
    """A single food item placed in the world.

    Parameters
    ----------
    position : np.ndarray
        [x, y] world coordinates.
    """

    def __init__(self, position: np.ndarray) -> None:
        super().__init__(position, radius=FOOD_RADIUS)
        self.nutrition_value: float = FOOD_NUTRITION

    def on_consume(self) -> float:
        """Mark as consumed and return nutritional value."""
        self.consumed = True
        return self.nutrition_value

    def __repr__(self) -> str:
        status = "eaten" if self.consumed else "available"
        return f"Food(pos=[{self.position[0]:.0f}, {self.position[1]:.0f}], {status})"
