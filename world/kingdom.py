"""
kingdom.py — Species Kingdom and Home Base Definition
=====================================================

Defines the Kingdom class representing the home territory and spawn origin
for each creature species (Anthill for Ants, Toile for Spiders).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Kingdom:
    """Represents a species' birth kingdom and home territory in the arena.

    Attributes
    ----------
    name : str
        The display name of the kingdom (e.g. 'anthill', 'toile').
    species : type
        The creature class associated with this kingdom.
    position : np.ndarray
        [x, y] center coordinates in the world.
    spawn_radius : float
        Radius around the kingdom center within which new creatures are born.
    """
    name: str
    species: type
    position: np.ndarray
    spawn_radius: float = 60.0

    def sample_spawn_position(self, rng: np.random.Generator, world_width: float, world_height: float) -> np.ndarray:
        """Sample a valid starting coordinate around the kingdom."""
        margin = 20.0
        r = rng.uniform(0.0, self.spawn_radius)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)
        pos = self.position + np.array([dx, dy])
        pos[0] = np.clip(pos[0], margin, world_width)
        pos[1] = np.clip(pos[1], margin, world_height)
        return pos
