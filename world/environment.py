"""
environment.py — Environmental Conditions and Dynamic Resource Spawning
=======================================================================

Manages FoodSource lifecycle: spawning new sources on a cooldown, updating
existing sources (which scatter Sugar/Seed items), and removing expired ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.constants import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    MAX_FOOD_SOURCES,
    FOOD_SOURCE_COOLDOWN,
    ZONE_BOUNDARY_X,
    FOOD_SOURCE_LEFT_ZONE_PROB,
    FOOD_SOURCE_RADIUS,
)
from world.food import FoodSource, is_in_lake

if TYPE_CHECKING:
    from world.world import World


class EnvironmentSystem:
    """Manages environmental mechanics (food sources, weather hooks) in the arena.

    Parameters
    ----------
    world : World
        The simulation arena container.
    """

    def __init__(self, world: World) -> None:
        self.world: World = world
        self.food_sources: list[FoodSource] = []
        self.source_cooldown: float = 0.0
        self.weather_condition: str = "clear"  # Hook for future weather modifiers

    def update(self, dt: float) -> None:
        """Advance environmental mechanics by one simulation step."""
        self._update_weather(dt)
        self._manage_food_sources(dt)

    def _update_weather(self, dt: float) -> None:
        """Hook for future dynamic weather simulation (rain, wind, temperature changes)."""
        pass

    def _manage_food_sources(self, dt: float) -> None:
        """Update existing food sources and spawn new ones on cooldown.

        Each active source generates food items around itself. When a source
        expires (lifetime reaches 0), it is removed. New sources are spawned
        when below the cap and the cooldown has elapsed.
        """
        rng = self.world.rng

        # --- Update existing sources and collect spawned food ---
        current_food_count = len(self.world.food_items)
        lakes = getattr(self.world, "lakes", [])
        surviving_sources: list[FoodSource] = []

        for source in self.food_sources:
            new_food = source.update(dt, current_food_count, lakes=lakes)
            self.world.food_items.extend(new_food)
            current_food_count += len(new_food)

            if not source.expired:
                surviving_sources.append(source)

        self.food_sources = surviving_sources

        # --- Spawn new sources on cooldown ---
        self.source_cooldown -= dt
        if self.source_cooldown <= 0.0 and len(self.food_sources) < MAX_FOOD_SOURCES:
            margin = 50.0
            for _ in range(20):
                if rng.random() < FOOD_SOURCE_LEFT_ZONE_PROB:
                    x = rng.uniform(margin, ZONE_BOUNDARY_X - margin)
                else:
                    x = rng.uniform(ZONE_BOUNDARY_X + margin, WORLD_WIDTH - margin)
                y = rng.uniform(margin, WORLD_HEIGHT - margin)
                pos = np.array([x, y])
                if not is_in_lake(pos, FOOD_SOURCE_RADIUS, lakes):
                    new_source = FoodSource(pos, rng)
                    self.food_sources.append(new_source)
                    self.source_cooldown = FOOD_SOURCE_COOLDOWN
                    break
