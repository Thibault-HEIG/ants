"""
creature.py — Abstract Base Class for All Evolving Entities
===========================================================

Encapsulates spatial state, lifecycle and vitals, neural network hooks, and
inter-species communication stubs. Subclasses inherit all general simulation
physics and only implement species-specific behaviors and fitness evaluation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from core.constants import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    HEALTH_DECAY_RATE,
    MAX_AGE_NORMALIZATION,
    ZONE_BOUNDARY_X,
)
from core.utils import clamp, normalize_angle, SpeciesStats
from evolution.brain import Brain
from evolution.sensors import Sensors

if TYPE_CHECKING:
    from evolution.sensors import SensorData


class Creature(ABC):
    """Abstract base class representing an evolving autonomous agent.

    Parameters
    ----------
    position : np.ndarray
        Starting [x, y] coordinates.
    rng : np.random.Generator
        Seeded random number generator.
    """

    species_name: str = "Creature"
    initial_health: float = 100.0
    max_speed: float = 100.0
    radius: float = 10.0
    strike_range: float = 20.0
    turn_rate: float = 2.0
    damage: float = 10.0
    attack_cost: float = 5.0
    eating_time: float = 1.0
    sensor_range: float = 100.0
    reproduction_threshold: float = 200.0
    max_population: int = 100
    initial_count: int = 10

    def __init__(
        self,
        position: np.ndarray,
        rng: np.random.Generator,
        *,
        initial_health: float | None = None,
        max_speed: float | None = None,
        radius: float | None = None,
        strike_range: float | None = None,
        turn_rate: float | None = None,
        damage: float | None = None,
        attack_cost: float | None = None,
        eating_time: float | None = None,
        sensor_range: float | None = None,
        sensor_angle: float | None = None,
        density_radius: float | None = None,
    ) -> None:
        # --- Spatial state ---
        self.position: np.ndarray = position.astype(float)
        self.direction: float = rng.uniform(-math.pi, math.pi)
        self.speed: float = 0.0

        # --- Vitals ---
        self.health: float = initial_health if initial_health is not None else self.initial_health
        self.max_health: float = self.health
        self.alive: bool = True

        # --- Physical attributes ---
        self._max_speed: float = max_speed if max_speed is not None else self.max_speed
        self.radius: float = float(radius if radius is not None else self.radius)
        self.strike_range: float = float(strike_range if strike_range is not None else self.strike_range)
        self._turn_rate: float = turn_rate if turn_rate is not None else self.turn_rate
        self.damage: float = damage if damage is not None else self.damage
        self._attack_cost: float = attack_cost if attack_cost is not None else self.attack_cost
        self.eating_time: float = float(eating_time if eating_time is not None else self.eating_time)

        # --- AI & Sensory components ---
        s_range = sensor_range if sensor_range is not None else self.sensor_range
        s_angle = sensor_angle if sensor_angle is not None else math.radians(50)
        d_radius = density_radius if density_radius is not None else s_range

        self.brain: Brain = Brain(rng)
        self.sensors: Sensors = Sensors(
            sensor_range=s_range,
            sensor_angle=s_angle,
            density_radius=d_radius,
        )
        self._rng: np.random.Generator = rng

        # --- Fitness tracking ---
        self.food_eaten: int = 0
        self.survival_time: float = 0.0
        self.enemies_touched: int = 0
        self.is_attacking: bool = False

        # --- Eating state machine ---
        self.is_eating: bool = False
        self.eat_timer: float = 0.0

    @property
    def hunger(self) -> float:
        """Normalised hunger: 0 = full HP, 1 = nearly dead."""
        return 1.0 - clamp(self.health / self.max_health, 0.0, 1.0)

    @property
    def genome(self) -> np.ndarray:
        """Shortcut to read the brain's genome."""
        return self.brain.get_genome()

    @genome.setter
    def genome(self, value: np.ndarray) -> None:
        """Install a new genome into the brain."""
        self.brain.set_genome(value)

    def update(self, dt: float, sensor_data: SensorData) -> None:
        """Advance the creature by one simulation step: sense → think → move → decay.

        Eating state machine:
        - When the NN outputs eat=1 and the creature is not already eating,
          it enters the eating state (is_eating=True, eat_timer=self.eating_time).
        - While eating: creature is frozen (no movement, no turning, no attacking).
        - When eat_timer expires: physics.resolve_food_collisions picks up the
          food if any is nearby, then resets is_eating.
        """
        if not self.alive:
            return

        hp_normalized = self.health / self.max_health
        zone = 1.0 if self.position[0] >= ZONE_BOUNDARY_X else 0.0
        effective_max_speed = self.get_effective_max_speed(zone)
        speed_normalized = self.speed / effective_max_speed if effective_max_speed > 0 else 0.0
        age_normalized = min(1.0, self.survival_time / MAX_AGE_NORMALIZATION) if MAX_AGE_NORMALIZATION > 0 else 0.0

        inputs = sensor_data.to_array(hp_normalized, zone, speed_normalized, age_normalized)

        brain_output = self.brain.forward(inputs)
        turn_signal = brain_output[0]
        speed_signal = brain_output[1]
        attack_signal = brain_output[2]
        eat_signal = brain_output[3]

        # --- Eating state machine ---
        if self.is_eating:
            # Frozen while eating — no movement, no turning, no attacking
            self.eat_timer -= dt
            self.speed = 0.0
            self.is_attacking = False
            # eat_timer expiry is handled by physics.resolve_food_collisions
        else:
            # Check if creature wants to start eating
            if eat_signal > 0.5:
                self.is_eating = True
                self.eat_timer = self.eating_time
                self.speed = 0.0
                self.is_attacking = False
            else:
                # Normal movement and combat
                self.is_attacking = bool(attack_signal > 0.5)

                self.direction += turn_signal * self._turn_rate * dt
                self.direction = normalize_angle(self.direction)

                self.speed = speed_signal * effective_max_speed
                dx = math.cos(self.direction) * self.speed * dt
                dy = math.sin(self.direction) * self.speed * dt
                self.position[0] += dx
                self.position[1] += dy

                self.position[0] = clamp(self.position[0], 0.0, float(WORLD_WIDTH))
                self.position[1] = clamp(self.position[1], 0.0, float(WORLD_HEIGHT))

        # --- Health decay (always applies, even while eating) ---
        self.health -= HEALTH_DECAY_RATE * dt
        if self.is_attacking:
            self.health -= self._attack_cost * dt
        self.survival_time += dt

        SpeciesStats.update(self.species_name, float(self.survival_time), int(self.food_eaten), int(self.enemies_touched))

        if self.health <= 0:
            self.health = 0.0
            self.alive = False

    def get_effective_max_speed(self, zone: float) -> float:
        """Return maximum speed, allowing subclass overrides based on world zones."""
        return self._max_speed

    def eat(self, food_value: float) -> None:
        """Consume food and restore health up to maximum health."""
        self.health = min(self.health + food_value, self.max_health)
        self.food_eaten += 1

    def take_damage(self, amount: float) -> None:
        """Receive damage from an enemy attack."""
        self.health -= amount
        if self.health <= 0:
            self.health = 0.0
            self.alive = False

    # ------------------------------------------------------------------
    # Future-Proofing Hooks: Inter-Species Communication Stubs
    # ------------------------------------------------------------------

    def emit_signal(self, channel: int, intensity: float) -> None:
        """Stub for inter-species communication: broadcast a chemical or vocal signal."""
        pass

    def receive_signal(self, channel: int, intensity: float, source_direction: float) -> None:
        """Stub for inter-species communication: process an incoming signal from another entity."""
        pass

    @abstractmethod
    def compute_fitness(self) -> float:
        """Calculate this creature's fitness score. Must be implemented by species."""
        pass

    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        eating = " [EATING]" if self.is_eating else ""
        return (
            f"{self.__class__.__name__}({status}{eating}, pos=[{self.position[0]:.0f},{self.position[1]:.0f}], "
            f"hp={self.health:.1f}, food={self.food_eaten}, touches={self.enemies_touched})"
        )
