"""
predator.py — The Spider Entity
=================================

Spiders are the ants' opponents in the co-evolutionary war.  Like ants,
each spider has a neural network brain that controls its movement, and
its genome evolves across generations.

**Key differences from ants:**
  - Fewer in number (10 vs 100) but much tougher (300 HP vs 100 HP).
  - Slower (80 px/s vs 150 px/s) but with better vision (250 px range).
  - Deal more damage on contact (100 vs 50).

Both species evolve independently — ants evolve to fight spiders, and
spiders evolve to fight ants.  This co-evolutionary arms race produces
interesting emergent strategies on both sides.
"""

from __future__ import annotations

import math

import numpy as np

from ant_simulator.brain import Brain
from ant_simulator.sensors import Sensors, SensorData
from ant_simulator.constants import (
    SPIDER_INITIAL_HEALTH,
    SPIDER_MAX_SPEED,
    SPIDER_RADIUS,
    SPIDER_STRIKE_RANGE,
    SPIDER_ATTACK_COST,
    SPIDER_TURN_RATE,
    SPIDER_SENSOR_RANGE,
    SENSOR_ANGLE,
    WORLD_WIDTH,
    WORLD_HEIGHT,
    HEALTH_DECAY_RATE,
    ZONE_BOUNDARY_X,
    ANT_MAX_SPEED,
    DENSITY_RADIUS_SPIDER,
    ROUND_TIME_LIMIT,
)
from ant_simulator.utils import clamp, normalize_angle, SpeciesStats


class Predator:
    """A spider — the ant colony's evolved opponent.

    The class is still named ``Predator`` to preserve import compatibility
    across the codebase, but conceptually these are co-evolving spiders.

    Parameters
    ----------
    position : np.ndarray
        Starting [x, y] coordinates.
    rng : np.random.Generator
        Seeded random generator.
    """

    def __init__(self, position: np.ndarray, rng: np.random.Generator) -> None:
        # --- Spatial state ---
        self.position: np.ndarray = position.astype(float)
        self.direction: float = rng.uniform(-math.pi, math.pi)

        # --- Vitals ---
        self.health: float = SPIDER_INITIAL_HEALTH
        self.alive: bool = True

        # --- Movement ---
        self.speed: float = 0.0

        # --- AI components ---
        # Spiders use the same brain architecture as ants (8→4→2) but
        # with different sensor range, giving them a different "world view".
        self.brain: Brain = Brain(rng)
        self.sensors: Sensors = Sensors(
            sensor_range=SPIDER_SENSOR_RANGE,
            sensor_angle=SENSOR_ANGLE,
            density_radius=DENSITY_RADIUS_SPIDER,
        )

        self._rng: np.random.Generator = rng

        self.food_eaten: int = 0
        self.survival_time: float = 0.0
        self.enemies_touched: int = 0


        self.radius: float = float(SPIDER_RADIUS)
        self.strike_range: float = float(SPIDER_STRIKE_RANGE)
        self.is_attacking: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hunger(self) -> float:
        """Normalised hunger: 0 = full HP, 1 = nearly dead."""
        return 1.0 - clamp(self.health / SPIDER_INITIAL_HEALTH, 0.0, 1.0)

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
        """Advance the spider by one simulation step.

        Identical structure to Ant.update() — sense → think → move.
        No health decay; HP only changes through combat and eating.

        Parameters
        ----------
        dt : float
            Time elapsed since last frame, in seconds.
        sensor_data : SensorData
            What the spider's sensors detected this frame.
        """
        if not self.alive:
            return

        # --- Step 1: Prepare brain inputs ---
        hp_normalized = self.health / SPIDER_INITIAL_HEALTH
        zone = 1.0 if self.position[0] >= ZONE_BOUNDARY_X else 0.0
        # Compute effective max speed for normalisation (zone-dependent)
        effective_max_speed = ANT_MAX_SPEED if self.position[0] < ZONE_BOUNDARY_X else SPIDER_MAX_SPEED
        speed_normalized = self.speed / effective_max_speed if effective_max_speed > 0 else 0.0
        age_normalized = min(1.0, self.survival_time / ROUND_TIME_LIMIT) if ROUND_TIME_LIMIT > 0 else 0.0
        inputs = sensor_data.to_array(hp_normalized, zone, speed_normalized, age_normalized)

        # --- Step 2: Think ---
        brain_output = self.brain.forward(inputs)
        turn_signal = brain_output[0]
        speed_signal = brain_output[1]
        attack_signal = brain_output[2]
        self.is_attacking = bool(attack_signal > 0.5)

        # --- Step 3: Move ---
        self.direction += turn_signal * SPIDER_TURN_RATE * dt
        self.direction = normalize_angle(self.direction)

        # Zone logic: Spiders are fast in the Left Zone (Danger Zone)
        if self.position[0] < ZONE_BOUNDARY_X:
            self.speed = speed_signal * ANT_MAX_SPEED
        else:
            self.speed = speed_signal * SPIDER_MAX_SPEED

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
            self.health -= SPIDER_ATTACK_COST * dt
        self.survival_time += dt

        # Update simulation-wide historical maximums for Spiders
        SpeciesStats.update_spider(float(self.survival_time), int(self.food_eaten), int(self.enemies_touched))

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
            Flat HP to restore.
        """
        self.health = min(self.health + food_value, SPIDER_INITIAL_HEALTH)
        self.food_eaten += 1

    def take_damage(self, amount: float) -> None:
        """Receive damage from an ant.

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
        """Calculate this spider's fitness score.

        Spiders get less reward per touch than ants because ants are
        more numerous and easier to catch.  The lower multiplier
        prevents spiders from evolving pure aggression — they still
        need to survive and eat to score well.

        Returns
        -------
        float
            A scalar fitness score (higher is better).
        """
        SpeciesStats.update_spider(float(self.survival_time), int(self.food_eaten), int(self.enemies_touched))

        return (
            (self.survival_time / 20) * 10.0
            + self.food_eaten * 10.0
            + self.enemies_touched * 50.0
        )

    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        return (
            f"Spider({status}, pos=[{self.position[0]:.0f},{self.position[1]:.0f}], "
            f"hp={self.health:.1f}, food={self.food_eaten}, touches={self.enemies_touched})"
        )
