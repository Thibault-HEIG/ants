"""
sensors.py — Vision and Sensory Perception System
=================================================

Perception system supporting arbitrary species without hardcoded class references.
Creatures perceive the world through vision rays (left, right, forward),
omnidirectional sensing ("smell"), and density awareness ("teamwork").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants import MAX_DENSITY_COUNT
from core.utils import distance

if TYPE_CHECKING:
    from world.food import Food


@dataclass
class RayResult:
    """The raw distance measurements from a single ray.

    Values are normalised:
      0.0 = touching the sensor origin
      1.0 = nothing detected within max range
    """
    food_distance: float = 1.0
    enemy_distance: float = 1.0
    ally_distance: float = 1.0
    wall_distance: float = 1.0


@dataclass
class SensorData:
    """Combined readings from vision rays, omnidirectional, and density sensors."""
    # --- Ray-based vision (left / right / forward) ---
    food_left: float = 1.0
    food_right: float = 1.0
    food_fwd: float = 1.0
    enemy_left: float = 1.0
    enemy_right: float = 1.0
    enemy_fwd: float = 1.0
    ally_left: float = 1.0
    ally_right: float = 1.0
    ally_fwd: float = 1.0
    wall_left: float = 1.0
    wall_right: float = 1.0
    wall_fwd: float = 1.0

    # --- Omnidirectional sensing ("smell") ---
    nearest_food_dist: float = 1.0
    nearest_food_angle: float = 0.0   # [-1, 1] relative to heading
    nearest_enemy_dist: float = 1.0
    nearest_enemy_angle: float = 0.0  # [-1, 1] relative to heading

    # --- Density awareness (teamwork) ---
    ally_density: float = 0.0   # [0, 1] normalised count
    enemy_density: float = 0.0  # [0, 1] normalised count

    def to_array(self, hp: float, zone: float, speed: float, age: float) -> np.ndarray:
        """Convert sensor readings + internal state into a 22-element input vector."""
        return np.array([
            self.enemy_right,           #  0
            self.enemy_left,            #  1
            zone,                       #  2
            self.food_right,            #  3
            self.food_left,             #  4
            hp,                         #  5
            self.ally_left,             #  6
            self.ally_right,            #  7
            self.wall_left,             #  8
            self.wall_right,            #  9
            self.food_fwd,              # 10
            self.enemy_fwd,             # 11
            self.ally_fwd,              # 12
            self.wall_fwd,              # 13
            self.nearest_food_dist,     # 14
            self.nearest_food_angle,    # 15
            self.nearest_enemy_dist,    # 16
            self.nearest_enemy_angle,   # 17
            speed,                      # 18
            age,                        # 19
            self.ally_density,          # 20
            self.enemy_density,         # 21
        ], dtype=float)


class SensorRay:
    """A single vision ray cast from a creature's position.

    Parameters
    ----------
    angle_offset : float
        Angle relative to the creature's heading (radians).
    max_range : float
        Maximum detection distance in pixels.
    """

    def __init__(self, angle_offset: float, max_range: float) -> None:
        self.angle_offset: float = angle_offset
        self.max_range: float = max_range

    def cast(
        self,
        origin: np.ndarray,
        heading: float,
        food_items: list[Any],
        enemies: list[Any],
        allies: list[Any],
        world_width: float,
        world_height: float,
    ) -> RayResult:
        """Cast this ray and find the nearest food, enemy, ally, and wall."""
        ray_angle = heading + self.angle_offset
        ray_dir = np.array([math.cos(ray_angle), math.sin(ray_angle)])
        result = RayResult()

        # Detect nearest food along this ray
        result.food_distance = self._detect_nearest(
            origin, ray_dir, [(f.position, f.radius) for f in food_items if not getattr(f, 'consumed', False)]
        )

        # Detect nearest enemy along this ray
        result.enemy_distance = self._detect_nearest(
            origin, ray_dir, [(e.position, e.radius) for e in enemies if getattr(e, 'alive', True)]
        )

        # Detect nearest ally along this ray (excluding self)
        result.ally_distance = self._detect_nearest(
            origin, ray_dir, [(a.position, a.radius) for a in allies if getattr(a, 'alive', True) and np.linalg.norm(a.position - origin) > 0.1]
        )

        # Detect wall distance
        result.wall_distance = self._detect_wall(
            origin, ray_dir, world_width, world_height
        )

        return result

    def _detect_nearest(
        self,
        origin: np.ndarray,
        ray_dir: np.ndarray,
        targets: list[tuple[np.ndarray, float]],
    ) -> float:
        """Find the nearest target along the ray direction."""
        nearest = self.max_range

        for target_pos, target_radius in targets:
            to_target = target_pos - origin
            dist = float(np.linalg.norm(to_target))

            if dist < 1e-6:
                return 0.0

            if dist > self.max_range:
                continue

            dot_product = float(np.dot(to_target, ray_dir))

            if dot_product < 0:
                continue

            perp_dist = math.sqrt(max(0.0, dist * dist - dot_product * dot_product))
            hit_tolerance = target_radius + 15.0

            if perp_dist <= hit_tolerance and dot_product < nearest:
                nearest = dot_product

        return max(0.0, min(1.0, nearest / self.max_range))

    def _detect_wall(
        self,
        origin: np.ndarray,
        ray_dir: np.ndarray,
        world_width: float,
        world_height: float,
    ) -> float:
        """Calculate distance along the ray until it intersects a world boundary."""
        min_t = self.max_range

        if abs(ray_dir[0]) > 1e-9:
            t_left = -origin[0] / ray_dir[0]
            t_right = (world_width - origin[0]) / ray_dir[0]
            if t_left > 0:
                min_t = min(min_t, t_left)
            if t_right > 0:
                min_t = min(min_t, t_right)

        if abs(ray_dir[1]) > 1e-9:
            t_top = -origin[1] / ray_dir[1]
            t_bottom = (world_height - origin[1]) / ray_dir[1]
            if t_top > 0:
                min_t = min(min_t, t_top)
            if t_bottom > 0:
                min_t = min(min_t, t_bottom)

        return max(0.0, min(1.0, min_t / self.max_range))


class Sensors:
    """The complete sensor array: three vision rays plus omnidirectional and density sensors.

    Parameters
    ----------
    sensor_range : float
        Maximum detection distance in pixels.
    sensor_angle : float
        Angle offset for left/right rays (radians from heading).
    density_radius : float or None
        Radius for counting nearby allies/enemies.
    """

    def __init__(
        self,
        sensor_range: float,
        sensor_angle: float,
        density_radius: float | None = None,
    ) -> None:
        self.sensor_range: float = sensor_range
        self.left_ray = SensorRay(angle_offset=-sensor_angle, max_range=sensor_range)
        self.right_ray = SensorRay(angle_offset=+sensor_angle, max_range=sensor_range)
        self.forward_ray = SensorRay(angle_offset=0.0, max_range=sensor_range)
        self.density_radius: float = density_radius if density_radius is not None else sensor_range

    def perceive(
        self,
        position: np.ndarray,
        direction: float,
        food_items: list[Any],
        enemies: list[Any],
        allies: list[Any],
        world_width: float,
        world_height: float,
    ) -> SensorData:
        """Gather all sensor readings for one creature without hardcoded class dependencies."""
        data = SensorData()

        left_res = self.left_ray.cast(
            position, direction, food_items, enemies, allies, world_width, world_height
        )
        data.food_left = left_res.food_distance
        data.enemy_left = left_res.enemy_distance
        data.ally_left = left_res.ally_distance
        data.wall_left = left_res.wall_distance

        right_res = self.right_ray.cast(
            position, direction, food_items, enemies, allies, world_width, world_height
        )
        data.food_right = right_res.food_distance
        data.enemy_right = right_res.enemy_distance
        data.ally_right = right_res.ally_distance
        data.wall_right = right_res.wall_distance

        fwd_res = self.forward_ray.cast(
            position, direction, food_items, enemies, allies, world_width, world_height
        )
        data.food_fwd = fwd_res.food_distance
        data.enemy_fwd = fwd_res.enemy_distance
        data.ally_fwd = fwd_res.ally_distance
        data.wall_fwd = fwd_res.wall_distance

        # Omnidirectional sensing
        food_positions = [f.position for f in food_items if not getattr(f, 'consumed', False)]
        data.nearest_food_dist, data.nearest_food_angle = self._detect_omnidirectional(
            position, direction, food_positions
        )

        enemy_positions = [e.position for e in enemies if getattr(e, 'alive', True)]
        data.nearest_enemy_dist, data.nearest_enemy_angle = self._detect_omnidirectional(
            position, direction, enemy_positions
        )

        # Density awareness
        ally_positions = [a.position for a in allies if getattr(a, 'alive', True) and np.linalg.norm(a.position - position) > 0.1]
        data.ally_density = self._count_density(position, ally_positions)
        data.enemy_density = self._count_density(position, enemy_positions)

        return data

    def _detect_omnidirectional(
        self,
        origin: np.ndarray,
        heading: float,
        target_positions: list[np.ndarray],
    ) -> tuple[float, float]:
        """Find the nearest target in any direction (360°)."""
        if not target_positions:
            return 1.0, 0.0

        nearest_dist = float('inf')
        nearest_angle = 0.0

        for target_pos in target_positions:
            to_target = target_pos - origin
            dist = float(np.linalg.norm(to_target))

            if dist < nearest_dist:
                nearest_dist = dist

                if dist < 1e-6:
                    nearest_angle = 0.0
                else:
                    abs_angle = math.atan2(to_target[1], to_target[0])
                    rel_angle = abs_angle - heading
                    rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
                    nearest_angle = rel_angle / math.pi

        normalised_dist = max(0.0, min(1.0, nearest_dist / self.sensor_range))
        return normalised_dist, nearest_angle

    def _count_density(
        self,
        origin: np.ndarray,
        target_positions: list[np.ndarray],
    ) -> float:
        """Count how many targets are within the density radius."""
        count = 0
        for target_pos in target_positions:
            dist = float(np.linalg.norm(target_pos - origin))
            if dist <= self.density_radius:
                count += 1

        return min(1.0, count / MAX_DENSITY_COUNT)
