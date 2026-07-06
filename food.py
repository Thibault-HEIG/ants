"""
food.py — Food Items
=====================

Food is the primary resource that keeps ants alive.  Each food item sits at
a fixed location in the world and is consumed when an ant gets close enough.

Design note: Food is intentionally simple — it has no behaviour of its own.
This makes it easy to extend later (e.g. different food types, decay over
time, or pheromone attraction).
"""

from __future__ import annotations

import numpy as np

from ant_simulator.constants import FOOD_NUTRITION, FOOD_RADIUS


class Food:
    """A single food item placed in the world.

    Attributes
    ----------
    position : np.ndarray
        [x, y] world coordinates.
    nutrition_value : float
        How much health the ant gains when eating this food.
    consumed : bool
        Set to True once an ant eats it — the World will clean it up.
    radius : float
        Collision radius (also used for rendering).
    """

    def __init__(self, position: np.ndarray) -> None:
        self.position: np.ndarray = position.astype(float)
        self.nutrition_value: float = FOOD_NUTRITION
        self.consumed: bool = False
        self.radius: float = float(FOOD_RADIUS)

    def __repr__(self) -> str:
        status = "eaten" if self.consumed else "available"
        return f"Food(pos=[{self.position[0]:.0f}, {self.position[1]:.0f}], {status})"
