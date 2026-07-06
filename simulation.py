"""
simulation.py — Generation Orchestrator
=========================================

The Simulation class sits above the World and manages the big picture:

  - Running the per-frame update loop.
  - Detecting when a round is over (one side eliminated).
  - Triggering evolution for **both** species independently.
  - Tracking statistics across generations.

Both ants and spiders evolve their own brains — this co-evolutionary
arms race is the heart of the simulation.
"""

from __future__ import annotations

import numpy as np

from ant_simulator.world import World
from ant_simulator.utils import SpeciesStats
from ant_simulator.constants import RANDOM_SEED


class Simulation:
    """Top-level simulation controller.

    Manages the World and dual-species evolution cycle.  Call ``step()``
    once per frame from the main loop.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.  Set to ``None`` for
        non-deterministic behaviour.
    """

    def __init__(self, seed: int | None = RANDOM_SEED) -> None:
        self._seed = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self.world: World = World(self.rng)

        # Statistics
        self.total_ants_spawned: int = len(self.world.ants)
        self.total_spiders_spawned: int = len(self.world.spiders)

        self.running: bool = True

    # ------------------------------------------------------------------
    # Per-frame interface
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance the simulation by one time step.

        If one side has been eliminated, trigger evolution for both
        species and start the next generation.

        Parameters
        ----------
        dt : float
            Elapsed time in seconds (already multiplied by speed factor).
        """
        if not self.running:
            return

        self.world.update(dt)

        # Update running stats
        self.total_ants_spawned = max(self.total_ants_spawned, len(self.world.ants) + len(self.world.dead_ants))
        self.total_spiders_spawned = max(self.total_spiders_spawned, len(self.world.spiders) + len(self.world.dead_spiders))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a dictionary of current simulation statistics.

        Used by the renderer to display the HUD.

        Returns
        -------
        dict
            Stats for both species and the generation.
        """
        alive_ants = len(self.world.ants)
        alive_spiders = len(self.world.spiders)

        # Current generation fitness (alive + dead so far)
        all_ants = self.world.ants + self.world.dead_ants
        all_spiders = self.world.spiders + self.world.dead_spiders

        if all_ants:
            ant_fitnesses = [a.compute_fitness() for a in all_ants]
            avg_ant = sum(ant_fitnesses) / len(ant_fitnesses)
        else:
            avg_ant = 0.0

        if all_spiders:
            spider_fitnesses = [s.compute_fitness() for s in all_spiders]
            avg_spider = sum(spider_fitnesses) / len(spider_fitnesses)
        else:
            avg_spider = 0.0

        return {
            "elapsed_time": self.world.round_time,
            "alive_ants": alive_ants,
            "alive_spiders": alive_spiders,
            "total_ants": self.total_ants_spawned,
            "total_spiders": self.total_spiders_spawned,
            "avg_ant_fitness": avg_ant,
            "avg_spider_fitness": avg_spider,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restart the simulation from zero with a fresh seed."""
        SpeciesStats.reset()
        self.rng = np.random.default_rng(self._seed)
        self.world = World(self.rng)
        self.total_ants_spawned = len(self.world.ants)
        self.total_spiders_spawned = len(self.world.spiders)
        self.running = True
