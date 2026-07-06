"""
environment.py — Environmental Conditions and Dynamic Resource Spawning
=======================================================================

Provides the EnvironmentSystem hook for complex environmental features
such as dynamic weather conditions, seasonal cycles, and food spawning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from core.constants import FOOD_SPAWN_RATE, MAX_FOOD
from world.food import Food

if TYPE_CHECKING:
    from world.world import World


class EnvironmentSystem:
    """Manages environmental mechanics (weather, dynamic food spawning) in the arena.

    Parameters
    ----------
    world : World
        The simulation arena container.
    """

    def __init__(self, world: World) -> None:
        self.world: World = world
        self.food_timer: float = 0.0
        self.weather_condition: str = "clear"  # Hook for future weather modifiers

    def update(self, dt: float) -> None:
        """Advance environmental mechanics by one simulation step."""
        self._update_weather(dt)
        self._spawn_food(dt)

    def _update_weather(self, dt: float) -> None:
        """Hook for future dynamic weather simulation (rain, wind, temperature changes)."""
        pass

    def _spawn_food(self, dt: float) -> None:
        """Dynamically spawn food items into the arena."""
        self.food_timer += dt * FOOD_SPAWN_RATE
        while self.food_timer >= 1.0 and len(self.world.food_items) < MAX_FOOD:
            self.food_timer -= 1.0
            self.world.spawn_food_batch(1)
