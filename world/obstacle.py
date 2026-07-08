"""
obstacle.py — Static World Obstacles
====================================

Defines static environmental obstacles like circular Lakes that block
creature movement and obstruct vision raycasting.
"""

from __future__ import annotations

import numpy as np
from world.entity import Entity


class Lake(Entity):
    """A circular body of water that acts as an impassable obstacle to creatures.

    Parameters
    ----------
    position : np.ndarray
        [x, y] world coordinates of the center of the lake.
    radius : float
        Collision and rendering radius of the lake.
    """

    def __init__(self, position: np.ndarray, radius: float = 50.0) -> None:
        super().__init__(position, radius)

    def on_consume(self) -> float:
        """Lakes cannot be consumed."""
        return 0.0
