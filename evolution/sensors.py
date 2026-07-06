"""
sensors.py — Vision and Sensory Perception System
=================================================

Perception system supporting arbitrary species without hardcoded class references.
Creatures perceive the world through vision rays (left, right, forward),
omnidirectional sensing ("smell"), and density awareness ("teamwork").

Performance: all inner loops use scalar Python math (math.hypot, manual dot
products) to avoid NumPy dispatch overhead on tiny 2-element vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants import MAX_DENSITY_COUNT

if TYPE_CHECKING:
    from world.food import Food
    from world.physics import SpatialHash


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
        food_targets: list[tuple[float, float, float]],
        enemy_targets: list[tuple[float, float, float]],
        ally_targets: list[tuple[float, float, float]],
        world_width: float,
        world_height: float,
    ) -> RayResult:
        """Cast this ray and find the nearest food, enemy, ally, and wall.

        Targets are pre-extracted as (x, y, radius) tuples to avoid
        repeated attribute lookups inside the hot loop.
        """
        ray_angle = heading + self.angle_offset
        rdx = math.cos(ray_angle)
        rdy = math.sin(ray_angle)
        result = RayResult()

        ox = float(origin[0])
        oy = float(origin[1])

        result.food_distance = self._detect_nearest(ox, oy, rdx, rdy, food_targets)
        result.enemy_distance = self._detect_nearest(ox, oy, rdx, rdy, enemy_targets)
        result.ally_distance = self._detect_nearest(ox, oy, rdx, rdy, ally_targets)
        result.wall_distance = self._detect_wall(ox, oy, rdx, rdy, world_width, world_height)

        return result

    def _detect_nearest(
        self,
        ox: float, oy: float,
        rdx: float, rdy: float,
        targets: list[tuple[float, float, float]],
    ) -> float:
        """Find the nearest target along the ray direction.

        All arithmetic is scalar Python — no NumPy calls inside the loop.
        """
        max_range = self.max_range
        nearest = max_range

        for tx, ty, tradius in targets:
            # Vector from origin to target
            dx = tx - ox
            dy = ty - oy

            # Squared distance (skip sqrt where possible)
            dist_sq = dx * dx + dy * dy

            if dist_sq < 1e-12:
                return 0.0

            if dist_sq > max_range * max_range:
                continue

            # Dot product with ray direction
            dot = dx * rdx + dy * rdy

            if dot < 0.0:
                continue

            # Perpendicular distance squared
            perp_sq = dist_sq - dot * dot
            hit_tol = tradius + 15.0
            hit_tol_sq = hit_tol * hit_tol

            if perp_sq <= hit_tol_sq and dot < nearest:
                nearest = dot

        return max(0.0, min(1.0, nearest / max_range))

    def _detect_wall(
        self,
        ox: float, oy: float,
        rdx: float, rdy: float,
        world_width: float,
        world_height: float,
    ) -> float:
        """Calculate distance along the ray until it intersects a world boundary."""
        min_t = self.max_range

        if abs(rdx) > 1e-9:
            t_left = -ox / rdx
            t_right = (world_width - ox) / rdx
            if t_left > 0:
                min_t = min(min_t, t_left)
            if t_right > 0:
                min_t = min(min_t, t_right)

        if abs(rdy) > 1e-9:
            t_top = -oy / rdy
            t_bottom = (world_height - oy) / rdy
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
        spatial_hash: SpatialHash | None = None,
    ) -> SensorData:
        """Gather all sensor readings for one creature without hardcoded class dependencies.

        When a spatial_hash is provided, only entities within the sensor range
        are considered — reducing the inner-loop population from O(N) global
        to O(k) local where k << N.
        """
        data = SensorData()
        ox = float(position[0])
        oy = float(position[1])
        sr = self.sensor_range
        dr = self.density_radius

        # -----------------------------------------------------------------
        # Build candidate sets — either from spatial hash or full lists
        # -----------------------------------------------------------------
        if spatial_hash is not None:
            nearby = spatial_hash.query(position, max(sr, dr))
            nearby_set = set(id(n) for n in nearby)

            # Pre-extract (x, y, radius) tuples, filtering by category
            food_targets: list[tuple[float, float, float]] = []
            enemy_targets: list[tuple[float, float, float]] = []
            ally_targets: list[tuple[float, float, float]] = []

            enemy_ids = set(id(e) for e in enemies)
            ally_ids = set(id(a) for a in allies)

            for n in nearby:
                nid = id(n)
                px = float(n.position[0])
                py = float(n.position[1])

                if hasattr(n, 'consumed'):
                    # It's a food entity
                    if not n.consumed:
                        food_targets.append((px, py, float(n.radius)))
                elif nid in enemy_ids:
                    if getattr(n, 'alive', True):
                        enemy_targets.append((px, py, float(n.radius)))
                elif nid in ally_ids:
                    if getattr(n, 'alive', True):
                        # Exclude self
                        ddx = px - ox
                        ddy = py - oy
                        if ddx * ddx + ddy * ddy > 0.01:
                            ally_targets.append((px, py, float(n.radius)))
        else:
            # Fallback: full-list mode (no spatial hash available)
            food_targets = [
                (float(f.position[0]), float(f.position[1]), float(f.radius))
                for f in food_items if not getattr(f, 'consumed', False)
            ]
            enemy_targets = [
                (float(e.position[0]), float(e.position[1]), float(e.radius))
                for e in enemies if getattr(e, 'alive', True)
            ]
            ally_targets = [
                (float(a.position[0]), float(a.position[1]), float(a.radius))
                for a in allies if getattr(a, 'alive', True)
                and math.hypot(a.position[0] - ox, a.position[1] - oy) > 0.1
            ]

        # -----------------------------------------------------------------
        # Cast rays using the pre-extracted target tuples
        # -----------------------------------------------------------------
        left_res = self.left_ray.cast(
            position, direction, food_targets, enemy_targets, ally_targets,
            world_width, world_height
        )
        data.food_left = left_res.food_distance
        data.enemy_left = left_res.enemy_distance
        data.ally_left = left_res.ally_distance
        data.wall_left = left_res.wall_distance

        right_res = self.right_ray.cast(
            position, direction, food_targets, enemy_targets, ally_targets,
            world_width, world_height
        )
        data.food_right = right_res.food_distance
        data.enemy_right = right_res.enemy_distance
        data.ally_right = right_res.ally_distance
        data.wall_right = right_res.wall_distance

        fwd_res = self.forward_ray.cast(
            position, direction, food_targets, enemy_targets, ally_targets,
            world_width, world_height
        )
        data.food_fwd = fwd_res.food_distance
        data.enemy_fwd = fwd_res.enemy_distance
        data.ally_fwd = fwd_res.ally_distance
        data.wall_fwd = fwd_res.wall_distance

        # -----------------------------------------------------------------
        # Omnidirectional sensing — reuse the extracted tuples
        # -----------------------------------------------------------------
        food_positions = [(tx, ty) for tx, ty, _ in food_targets]
        data.nearest_food_dist, data.nearest_food_angle = self._detect_omnidirectional(
            ox, oy, direction, food_positions
        )

        enemy_positions = [(tx, ty) for tx, ty, _ in enemy_targets]
        data.nearest_enemy_dist, data.nearest_enemy_angle = self._detect_omnidirectional(
            ox, oy, direction, enemy_positions
        )

        # -----------------------------------------------------------------
        # Density awareness — reuse the extracted tuples
        # -----------------------------------------------------------------
        data.ally_density = self._count_density(ox, oy, ally_targets)
        data.enemy_density = self._count_density(ox, oy, enemy_targets)

        return data

    def _detect_omnidirectional(
        self,
        ox: float, oy: float,
        heading: float,
        target_positions: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """Find the nearest target in any direction (360°).

        Uses scalar math throughout — no NumPy calls.
        """
        if not target_positions:
            return 1.0, 0.0

        nearest_dist_sq = float('inf')
        nearest_angle = 0.0
        sr = self.sensor_range

        for tx, ty in target_positions:
            dx = tx - ox
            dy = ty - oy
            dist_sq = dx * dx + dy * dy

            if dist_sq < nearest_dist_sq:
                nearest_dist_sq = dist_sq

                if dist_sq < 1e-12:
                    nearest_angle = 0.0
                else:
                    abs_angle = math.atan2(dy, dx)
                    rel_angle = abs_angle - heading
                    rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
                    nearest_angle = rel_angle / math.pi

        nearest_dist = math.sqrt(nearest_dist_sq)
        normalised_dist = max(0.0, min(1.0, nearest_dist / sr))
        return normalised_dist, nearest_angle

    def _count_density(
        self,
        ox: float, oy: float,
        targets: list[tuple[float, float, float]],
    ) -> float:
        """Count how many targets are within the density radius.

        Compares squared distances to avoid sqrt calls.
        """
        dr_sq = self.density_radius * self.density_radius
        count = 0

        for tx, ty, _ in targets:
            dx = tx - ox
            dy = ty - oy
            if dx * dx + dy * dy <= dr_sq:
                count += 1

        return min(1.0, count / MAX_DENSITY_COUNT)
