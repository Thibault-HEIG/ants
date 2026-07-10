"""
grid.py — Abstract Spatial Tile Grid System
===========================================

Provides a clean, reusable spatial grid abstraction for converting continuous
world coordinates (x, y) to discrete grid tile coordinates (gx, gy) and
managing 2D grids (such as pheromone grids, territory maps, or visited tile tracking).
"""

from __future__ import annotations

import numpy as np


class TileGrid:
    """An abstract 2D tile grid over a continuous rectangular world space.

    Parameters
    ----------
    width : float
        World width in pixels.
    height : float
        World height in pixels.
    cell_size : float, default 10.0
        Size of each square tile in pixels.
    """

    def __init__(self, width: float, height: float, cell_size: float = 10.0) -> None:
        self.width: float = float(width)
        self.height: float = float(height)
        self.cell_size: float = float(cell_size)

        self.grid_width: int = max(1, int(self.width / self.cell_size))
        self.grid_height: int = max(1, int(self.height / self.cell_size))

    def world_to_tile(self, x: float, y: float) -> tuple[int, int]:
        """Convert continuous world coordinates (x, y) to clamped grid tile coordinates (gx, gy)."""
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)

        gx = max(0, min(self.grid_width - 1, gx))
        gy = max(0, min(self.grid_height - 1, gy))
        return (gx, gy)

    def tile_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        """Return the world coordinate center of the given grid tile."""
        cx = (gx + 0.5) * self.cell_size
        cy = (gy + 0.5) * self.cell_size
        return (cx, cy)

    def create_float_grid(self, default: float = 0.0) -> np.ndarray:
        """Create a 2D float array of shape (grid_width, grid_height)."""
        if default == 0.0:
            return np.zeros((self.grid_width, self.grid_height), dtype=float)
        return np.full((self.grid_width, self.grid_height), default, dtype=float)

    def create_bool_grid(self, default: bool = False) -> np.ndarray:
        """Create a 2D bool array of shape (grid_width, grid_height)."""
        if not default:
            return np.zeros((self.grid_width, self.grid_height), dtype=bool)
        return np.ones((self.grid_width, self.grid_height), dtype=bool)
