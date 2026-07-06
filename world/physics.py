"""
physics.py — Spatial Hashing and Collision Resolution
=====================================================

Provides spatial hashing for optimized proximity queries and resolves physical
interactions (combat strikes and food consumption) across arbitrary species.

All distance comparisons use squared distances (math) to avoid sqrt overhead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

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
        self._inv_cell: float = 1.0 / cell_size
        self.grid: dict[tuple[int, int], list[Any]] = {}

    def clear(self) -> None:
        """Clear all entries from the grid."""
        self.grid.clear()

    def insert(self, entity: Any) -> None:
        """Insert an entity into its corresponding cell."""
        cx = int(entity.position[0] * self._inv_cell)
        cy = int(entity.position[1] * self._inv_cell)
        key = (cx, cy)
        try:
            self.grid[key].append(entity)
        except KeyError:
            self.grid[key] = [entity]

    def query(self, position: np.ndarray, radius: float) -> list[Any]:
        """Return all entities in cells overlapping the bounding box of (position, radius)."""
        inv = self._inv_cell
        x, y = float(position[0]), float(position[1])
        min_cx = int((x - radius) * inv)
        max_cx = int((x + radius) * inv)
        min_cy = int((y - radius) * inv)
        max_cy = int((y + radius) * inv)

        grid = self.grid
        results: list[Any] = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cell = grid.get((cx, cy))
                if cell is not None:
                    results.extend(cell)
        return results

    def _get_cell(self, position: np.ndarray) -> tuple[int, int]:
        return (int(position[0] * self._inv_cell), int(position[1] * self._inv_cell))


def resolve_combat(
    creatures_by_species: dict[type, list[Any]],
    spatial_hash: SpatialHash | None = None,
) -> None:
    """Resolve combat strikes between different species without hardcoded type names.

    When a spatial_hash is supplied, each attacker queries only its local
    neighbourhood instead of iterating the full global target list.
    """
    living_by_species = {
        s_cls: [c for c in pop if getattr(c, "alive", False)]
        for s_cls, pop in creatures_by_species.items()
    }

    # Pre-build a set of species class per entity id for O(1) species-check
    species_of: dict[int, type] = {}
    for s_cls, pop in living_by_species.items():
        for c in pop:
            species_of[id(c)] = s_cls

    for attacker_cls, attackers in living_by_species.items():
        for attacker in attackers:
            if not getattr(attacker, "alive", False) or not getattr(attacker, "is_attacking", False):
                continue

            ax = float(attacker.position[0])
            ay = float(attacker.position[1])
            reach = attacker.strike_range

            if spatial_hash is not None:
                candidates = spatial_hash.query(attacker.position, reach + 50.0)
            else:
                candidates = [
                    target
                    for target_cls, targets in living_by_species.items()
                    if target_cls != attacker_cls
                    for target in targets
                    if getattr(target, "alive", False)
                ]

            for target in candidates:
                if not getattr(target, "alive", False):
                    continue
                # Skip same-species and non-creature (food) entities
                tid = id(target)
                target_species = species_of.get(tid)
                if target_species is None or target_species == attacker_cls:
                    continue

                tx = float(target.position[0])
                ty = float(target.position[1])
                dx = ax - tx
                dy = ay - ty
                threshold = reach + target.radius
                if dx * dx + dy * dy < threshold * threshold:
                    target.take_damage(attacker.damage)
                    attacker.enemies_touched += 1


def resolve_food_collisions(
    creatures_by_species: dict[type, list[Any]],
    food_items: list[Any],
    spatial_hash: SpatialHash | None = None,
) -> None:
    """Check food collisions for all active species and feed creatures when touching.

    When a spatial_hash is supplied, each creature queries only nearby food
    instead of scanning the entire food list.
    """
    if not food_items:
        return

    # Quick check: if all food consumed already, bail out
    if all(getattr(f, "consumed", False) for f in food_items):
        return

    # Pre-build set of food ids for fast membership test when using spatial hash
    food_ids: set[int] | None = None
    if spatial_hash is not None:
        food_ids = set(id(f) for f in food_items if not getattr(f, "consumed", False))

    for pop in creatures_by_species.values():
        for creature in pop:
            if not getattr(creature, "alive", False):
                continue

            cx = float(creature.position[0])
            cy = float(creature.position[1])
            cr = creature.radius

            if spatial_hash is not None:
                candidates = spatial_hash.query(creature.position, cr + 20.0)
            else:
                candidates = food_items

            for food in candidates:
                if spatial_hash is not None and id(food) not in food_ids:
                    continue
                if getattr(food, "consumed", False):
                    continue

                fx = float(food.position[0])
                fy = float(food.position[1])
                dx = cx - fx
                dy = cy - fy
                threshold = cr + food.radius
                if dx * dx + dy * dy < threshold * threshold:
                    creature.eat(food.on_consume())
