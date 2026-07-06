"""
ant.py — The Ant Entity
========================

Each ant is an autonomous agent in the war against spiders.  It has:

  - A **position** and **heading** that change every frame.
  - **Health** that only changes through combat and eating.
  - A **brain** (neural network) that decides how to move.
  - **Sensors** that let the brain see nearby food, enemies, and walls.
  - A **genome** — the flat vector of all brain weights, used by evolution.

The ant does NOT know the global state of the world.  It can only react
to what its sensors detect.  All "intelligence" comes from the weights
in the neural network, which are discovered through evolution.
"""

from __future__ import annotations

import math

import numpy as np

from ant_simulator.brain import Brain
from ant_simulator.sensors import Sensors, SensorData
from ant_simulator.constants import (
    ANT_INITIAL_HEALTH,
    ANT_MAX_SPEED,
    ANT_RADIUS,
    ANT_STRIKE_RANGE,
    ANT_ATTACK_COST,
    ANT_TURN_RATE,
    ANT_SENSOR_RANGE,
    SENSOR_ANGLE,
    WORLD_WIDTH,
    WORLD_HEIGHT,
    HEALTH_DECAY_RATE,
    ZONE_BOUNDARY_X,
    DENSITY_RADIUS_ANT,
    ROUND_TIME_LIMIT,
)
from ant_simulator.utils import clamp, normalize_angle, SpeciesStats


class Ant:
    """A single ant in the colony.

    Parameters
    ----------
    position : np.ndarray
        Starting [x, y] coordinates.
    rng : np.random.Generator
        Seeded random generator (shared across the simulation for
        reproducibility).
    """

    def __init__(self, position: np.ndarray, rng: np.random.Generator) -> None:
        # --- Spatial state ---
        self.position: np.ndarray = position.astype(float)
        self.direction: float = rng.uniform(-math.pi, math.pi)

        # --- Vitals ---
        self.health: float = ANT_INITIAL_HEALTH
        self.alive: bool = True

        # --- Movement ---
        self.speed: float = 0.0  # current speed; set each frame by brain

        # --- AI components ---
        self.brain: Brain = Brain(rng)
        self.sensors: Sensors = Sensors(
            sensor_range=ANT_SENSOR_RANGE,
            sensor_angle=SENSOR_ANGLE,
            density_radius=DENSITY_RADIUS_ANT,
        )

        self._rng: np.random.Generator = rng

        # --- Fitness tracking ---
        # These values are accumulated throughout the ant's lifetime and
        # used by the genetic algorithm to rank ants after they die.
        self.food_eaten: int = 0
        self.survival_time: float = 0.0
        # Tracking how many times we touched/engaged an enemy (fitness)
        self.enemies_touched: int = 0


        self.radius: float = float(ANT_RADIUS)
        self.strike_range: float = float(ANT_STRIKE_RANGE)
        self.is_attacking: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hunger(self) -> float:
        """Normalised hunger: 0 = full HP, 1 = nearly dead."""
        return 1.0 - clamp(self.health / ANT_INITIAL_HEALTH, 0.0, 1.0)

    @property
    def genome(self) -> np.ndarray:
        """Shortcut to read the brain's genome."""
        return self.brain.get_genome()

    @genome.setter
    def genome(self, value: np.ndarray) -> None:
        """Install a new genome into the brain."""
        self.brain.set_genome(value)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, dt: float, sensor_data: SensorData) -> None:
        """Advance the ant by one simulation step.

        This is the core behaviour loop:
          1. Build the 22-element input vector from sensors + internal state.
          2. Run the neural network to get [turn, speed].
          3. Apply movement.
          4. Track survival time.

        No health decay — HP only changes through combat and eating.

        Parameters
        ----------
        dt : float
            Time elapsed since last frame, in seconds.
        sensor_data : SensorData
            What the ant's sensors detected this frame.
        """
        if not self.alive:
            return

        # --- Step 1: Prepare brain inputs ---
        hp_normalized = self.health / ANT_INITIAL_HEALTH
        zone = 1.0 if self.position[0] >= ZONE_BOUNDARY_X else 0.0
        speed_normalized = self.speed / ANT_MAX_SPEED if ANT_MAX_SPEED > 0 else 0.0
        age_normalized = min(1.0, self.survival_time / ROUND_TIME_LIMIT) if ROUND_TIME_LIMIT > 0 else 0.0
        inputs = sensor_data.to_array(hp_normalized, zone, speed_normalized, age_normalized)

        # --- Step 2: Think ---
        brain_output = self.brain.forward(inputs)
        turn_signal = brain_output[0]    # ∈ [-1, 1]
        speed_signal = brain_output[1]   # ∈ [0, 1]
        attack_signal = brain_output[2]  # ∈ {0.0, 1.0}
        self.is_attacking = bool(attack_signal > 0.5)

        # --- Step 3: Move ---
        self.direction += turn_signal * ANT_TURN_RATE * dt
        self.direction = normalize_angle(self.direction)

        self.speed = speed_signal * ANT_MAX_SPEED

        dx = math.cos(self.direction) * self.speed * dt
        dy = math.sin(self.direction) * self.speed * dt
        self.position[0] += dx
        self.position[1] += dy

        # Clamp to world boundaries
        self.position[0] = clamp(self.position[0], 0.0, float(WORLD_WIDTH))
        self.position[1] = clamp(self.position[1], 0.0, float(WORLD_HEIGHT))

        # --- Step 4: Health decay and survival ---
        self.health -= HEALTH_DECAY_RATE * dt
        if self.is_attacking:
            self.health -= ANT_ATTACK_COST * dt
        self.survival_time += dt

        # Update simulation-wide historical maximums for Ants
        SpeciesStats.update_ant(float(self.survival_time), int(self.food_eaten), int(self.enemies_touched))

        # --- Step 5: Death check ---
        if self.health <= 0:
            self.health = 0.0
            self.alive = False

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------

    def eat(self, food_value: float) -> None:
        """Consume food and restore health.

        Parameters
        ----------
        food_value : float
            Flat HP to restore (from the food's nutrition_value).
        """
        self.health = min(self.health + food_value, ANT_INITIAL_HEALTH)
        self.food_eaten += 1

    def take_damage(self, amount: float) -> None:
        """Receive damage from a spider.

        Parameters
        ----------
        amount : float
            HP to subtract.
        """
        self.health -= amount
        if self.health <= 0:
            self.health = 0.0
            self.alive = False

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def compute_fitness(self) -> float:
        """Calculate this ant's fitness score.

        The fitness function determines *what evolution optimises for*.
        In war mode we reward:
          - Surviving longer (staying alive in battle)
          - Eating food (sustaining yourself)
          - Killing enemies (the whole point of the war)

        Returns
        -------
        float
            A scalar fitness score (higher is better).
        """
        SpeciesStats.update_ant(float(self.survival_time), int(self.food_eaten), int(self.enemies_touched))

        return (
            (self.survival_time / 20) * 10.0
            + self.food_eaten * 30.0
            + self.enemies_touched * 50.0
        )

    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        return (
            f"Ant({status}, pos=[{self.position[0]:.0f},{self.position[1]:.0f}], "
            f"hp={self.health:.1f}, food={self.food_eaten}, touches={self.enemies_touched})"
        )
