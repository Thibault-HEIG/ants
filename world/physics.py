"""
physics.py — Spatial Hashing and Collision Resolution
=====================================================

Provides spatial hashing for optimized proximity queries and resolves physical
interactions (combat strikes and food consumption) across arbitrary species.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from core.utils import distance

if TYPE_CHECKING:
    from world.food import Food


class SpatialHash:
    """2D spatial grid for fast proximity lookups.

    Parameters
    ----------
    cell_size : float
        Width and height of each grid cell in pixels.
    """

    def __init__(self, cell_size: float = 100.0) -> None:
        self.cell_size: float = cell_size
        self.grid: dict[tuple[int, int], list[Any]] = {}

    def clear(self) -> None:
        """Clear all entries from the grid."""
        self.grid.clear()

    def insert(self, entity: Any) -> None:
        """Insert an entity into its corresponding cell."""
        cell = self._get_cell(entity.position)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(entity)

    def query(self, position: np.ndarray, radius: float) -> list[Any]:
        """Return all entities in cells overlapping the bounding box of (position, radius)."""
        x, y = float(position[0]), float(position[1])
        min_cell_x = int((x - radius) // self.cell_size)
        max_cell_x = int((x + radius) // self.cell_size)
        min_cell_y = int((y - radius) // self.cell_size)
        max_cell_y = int((y + radius) // self.cell_size)

        results: list[Any] = []
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                if (cx, cy) in self.grid:
                    results.extend(self.grid[(cx, cy)])
        return results

    def _get_cell(self, position: np.ndarray) -> tuple[int, int]:
        return (int(position[0] // self.cell_size), int(position[1] // self.cell_size))


def resolve_combat(creatures_by_species: dict[type, list[Any]]) -> None:
    """Resolve combat strikes between different species without hardcoded type names.

    A creature deals damage only when actively attacking and the target is within
    strike reach. Enemies are any creatures belonging to a different species class.
    """
    living_by_species = {
        s_cls: [c for c in pop if getattr(c, "alive", False)]
        for s_cls, pop in creatures_by_species.items()
    }

    for attacker_cls, attackers in living_by_species.items():
        # Potential targets: living creatures of any other active species
        potential_targets = [
            target
            for target_cls, targets in living_by_species.items()
            if target_cls != attacker_cls
            for target in targets
            if getattr(target, "alive", False)
        ]
        if not potential_targets:
            continue

        for attacker in attackers:
            if not getattr(attacker, "alive", False) or not getattr(attacker, "is_attacking", False):
                continue

            for target in potential_targets:
                if not getattr(target, "alive", False):
                    continue

                dist = distance(attacker.position, target.position)
                if dist < attacker.strike_range + target.radius:
                    target.take_damage(attacker.damage)
                    attacker.enemies_touched += 1


def resolve_food_collisions(creatures_by_species: dict[type, list[Any]], food_items: list[Any]) -> None:
    """Check food collisions for all active species and feed creatures when touching."""
    available_food = [f for f in food_items if not getattr(f, "consumed", False)]
    if not available_food:
        return

    for pop in creatures_by_species.values():
        for creature in pop:
            if not getattr(creature, "alive", False):
                continue
            for food in available_food:
                if getattr(food, "consumed", False):
                    continue
                if distance(creature.position, food.position) < creature.radius + food.radius:
                    creature.eat(food.on_consume())
