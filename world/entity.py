"""
entity.py — Abstract Base Class for Passive World Objects
=========================================================

Provides spatial positioning and lifecycle properties for inanimate objects
in the simulation arena (such as Food or obstacles).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Entity(ABC):
    """Abstract base class for passive objects placed in the world.

    Parameters
    ----------
    position : np.ndarray
        [x, y] world coordinates.
    radius : float
        Collision and render radius.
    """

    def __init__(self, position: np.ndarray, radius: float) -> None:
        self.position: np.ndarray = position.astype(float)
        self.radius: float = float(radius)
        self.consumed: bool = False

    @abstractmethod
    def on_consume(self) -> float:
        """Called when a creature consumes or interacts with this entity.

        Returns
        -------
        float
            The value or effect produced by consuming this entity.
        """
        pass
