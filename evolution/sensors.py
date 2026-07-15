"""
sensors.py — Vision and Sensory Perception System
=================================================

Perception system supporting arbitrary species without hardcoded class references.
Creatures perceive the world through 8 directional vision rays spanning a
configurable field of view, plus density awareness ("teamwork").

Distance convention (inverted):
  0.0 = nothing detected within range
  1.0 = touching the sensor origin

Performance: all inner loops use scalar Python math (math.hypot, manual dot
products) to avoid NumPy dispatch overhead on tiny 2-element vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from core.constants import MAX_DENSITY_COUNT, NN_NUM_SENSORS

if TYPE_CHECKING:
    from world.food import Food
    from world.physics import SpatialHash


@dataclass
class RayResult:
    """The raw distance and state measurements from a single ray.

    Values are normalised (inverted):
      0.0 = nothing detected within max range
      1.0 = touching the sensor origin
    """
    food_distance: float = 0.0
    enemy_distance: float = 0.0
    enemy_is_eating: float = 0.0
    enemy_is_attacking: float = 0.0
    ally_distance: float = 0.0
    ally_is_eating: float = 0.0
    ally_is_attacking: float = 0.0
    wall_distance: float = 0.0
    pheromone_distance: float = 0.0


@dataclass
class SectorReading:
    """Readings from a single directional sector."""
    enemy_distance: float = 0.0       # inverted: 0 = no enemy, 1 = touching
    enemy_is_eating: float = 0.0      # {0, 1}: 1 if seen enemy is eating
    enemy_is_attacking: float = 0.0   # {0, 1}: 1 if seen enemy is attacking
    ally_distance: float = 0.0        # inverted: 0 = no ally, 1 = touching
    ally_is_eating: float = 0.0       # {0, 1}: 1 if seen ally is eating
    ally_is_attacking: float = 0.0    # {0, 1}: 1 if seen ally is attacking
    food_distance: float = 0.0        # inverted: 0 = no food, 1 = touching
    wall_distance: float = 0.0        # inverted: 0 = no wall, 1 = touching
    pheromone_distance: float = 0.0   # inverted: 0 = no pheromone trail, 1 = touching trail


def _default_sectors() -> list[SectorReading]:
    """Create the default list of 8 empty sector readings."""
    return [SectorReading() for _ in range(NN_NUM_SENSORS)]


@dataclass
class SensorData:
    """Combined readings from 8-sector vision rays, density, and HP-gain sensors."""

    # --- 8-sector directional vision ---
    sectors: list[SectorReading] = field(default_factory=_default_sectors)

    # --- Density awareness (teamwork) ---
    ally_density: float = 0.0    # [0, 1] normalised count
    enemy_density: float = 0.0   # [0, 1] normalised count

    # --- HP gain feedback ---
    has_gained_hp: float = 0.0   # {0, 1}

    # --- Pheromone concentration under foot ---
    pheromone_strength: float = 0.0

    def to_array(self, hp: float, zone: float, speed: float, age: float) -> np.ndarray:
        """Convert sensor readings + internal state into an 80-element input vector.

        Layout:
          [0..71]  8 sectors × 9 features [enemy_dist, enemy_eat, enemy_atk, ally_dist, ally_eat, ally_atk, food_dist, wall_dist, pheromone_dist]
          [72..79] state inputs: [hp, zone, speed, age, ally_density, enemy_density, has_gained_hp, pheromone_strength]
        """
        arr = np.empty(NN_NUM_SENSORS * 9 + 8, dtype=float)

        # Pack sector readings: 8 sectors × 9 features
        idx = 0
        for sector in self.sectors:
            arr[idx] = sector.enemy_distance
            arr[idx + 1] = sector.enemy_is_eating
            arr[idx + 2] = sector.enemy_is_attacking
            arr[idx + 3] = sector.ally_distance
            arr[idx + 4] = sector.ally_is_eating
            arr[idx + 5] = sector.ally_is_attacking
            arr[idx + 6] = sector.food_distance
            arr[idx + 7] = sector.wall_distance
            arr[idx + 8] = sector.pheromone_distance
            idx += 9

        # Pack state inputs
        arr[idx] = hp
        arr[idx + 1] = zone
        arr[idx + 2] = speed
        arr[idx + 3] = age
        arr[idx + 4] = self.ally_density
        arr[idx + 5] = self.enemy_density
        arr[idx + 6] = self.has_gained_hp
        arr[idx + 7] = self.pheromone_strength

        return arr


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
        enemy_targets: list[tuple[float, float, float, bool, bool]],
        ally_targets: list[tuple[float, float, float, bool, bool]],
        world_width: float,
        world_height: float,
        lakes: list[Any] | None = None,
        pheromone_grid: np.ndarray | None = None,
        pheromone_cell_size: float = 10.0,
    ) -> RayResult:
        """Cast this ray and find the nearest food, enemy, ally, wall/lake, and pheromone trail."""
        ray_angle = heading + self.angle_offset
        rdx = math.cos(ray_angle)
        rdy = math.sin(ray_angle)
        result = RayResult()

        ox = float(origin[0])
        oy = float(origin[1])

        result.food_distance = self._detect_nearest(ox, oy, rdx, rdy, food_targets)
        result.enemy_distance, result.enemy_is_eating, result.enemy_is_attacking = self._detect_creature(ox, oy, rdx, rdy, enemy_targets)
        result.ally_distance, result.ally_is_eating, result.ally_is_attacking = self._detect_creature(ox, oy, rdx, rdy, ally_targets)
        result.wall_distance = self._detect_wall(ox, oy, rdx, rdy, world_width, world_height, lakes)
        result.pheromone_distance = self._detect_pheromone(ox, oy, rdx, rdy, pheromone_grid, pheromone_cell_size)

        return result

    def _detect_nearest(
        self,
        ox: float, oy: float,
        rdx: float, rdy: float,
        targets: list[tuple[float, float, float]],
    ) -> float:
        """Find the nearest target along the ray direction.

        Returns inverted distance: 0.0 = nothing detected, 1.0 = touching.
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
                return 1.0

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

        raw = nearest / max_range
        if raw >= 1.0:
            # Nothing detected within range
            return 0.0
        return max(0.0, 1.0 - raw)

    def _detect_creature(
        self,
        ox: float, oy: float,
        rdx: float, rdy: float,
        targets: list[tuple[float, float, float, bool, bool]],
    ) -> tuple[float, float, float]:
        """Find the nearest creature target along the ray direction and report its state.

        Returns (inverted_distance, is_eating, is_attacking):
          distance: 0.0 = nothing detected, 1.0 = touching.
          is_eating: 1.0 if eating, 0.0 otherwise.
          is_attacking: 1.0 if attacking, 0.0 otherwise.
        """
        max_range = self.max_range
        nearest = max_range
        nearest_eat = 0.0
        nearest_atk = 0.0

        for tx, ty, tradius, is_eat, is_atk in targets:
            dx = tx - ox
            dy = ty - oy
            dist_sq = dx * dx + dy * dy

            if dist_sq < 1e-12:
                return 1.0, 1.0 if is_eat else 0.0, 1.0 if is_atk else 0.0

            if dist_sq > max_range * max_range:
                continue

            dot = dx * rdx + dy * rdy
            if dot < 0.0:
                continue

            perp_sq = dist_sq - dot * dot
            hit_tol = tradius + 15.0
            hit_tol_sq = hit_tol * hit_tol

            if perp_sq <= hit_tol_sq and dot < nearest:
                nearest = dot
                nearest_eat = 1.0 if is_eat else 0.0
                nearest_atk = 1.0 if is_atk else 0.0

        raw = nearest / max_range
        if raw >= 1.0:
            return 0.0, 0.0, 0.0
        return max(0.0, 1.0 - raw), nearest_eat, nearest_atk

    def _detect_wall(
        self,
        ox: float, oy: float,
        rdx: float, rdy: float,
        world_width: float,
        world_height: float,
        lakes: list[Any] | None = None,
    ) -> float:
        """Calculate distance along the ray until it intersects a world boundary or static lake obstacle."""
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

        if lakes:
            for lake in lakes:
                lx = float(lake.position[0])
                ly = float(lake.position[1])
                lradius = float(getattr(lake, "radius", 50.0))

                dx = lx - ox
                dy = ly - oy
                dot = dx * rdx + dy * rdy
                if dot < 0.0:
                    continue
                dist_sq = dx * dx + dy * dy
                perp_sq = dist_sq - dot * dot
                if perp_sq <= lradius * lradius and dot < min_t:
                    min_t = dot

        raw = min_t / self.max_range
        return max(0.0, 1.0 - min(1.0, raw))

    def _detect_pheromone(
        self,
        ox: float, oy: float,
        rdx: float, rdy: float,
        pheromone_grid: np.ndarray | None,
        cell_size: float,
    ) -> float:
        """Detect the nearest pheromone trail cell along the ray direction."""
        if pheromone_grid is None:
            return 0.0

        gw, gh = pheromone_grid.shape
        step = 10.0
        dist = 5.0
        max_r = self.max_range
        while dist <= max_r:
            px = ox + rdx * dist
            py = oy + rdy * dist
            gx = int(px / cell_size)
            gy = int(py / cell_size)
            if 0 <= gx < gw and 0 <= gy < gh:
                if pheromone_grid[gx, gy] > 0.01:
                    return max(0.0, 1.0 - (dist / max_r))
            dist += step
        return 0.0


class Sensors:
    """The complete sensor array: 8 directional vision rays plus density sensors.

    Parameters
    ----------
    sensor_range : float
        Maximum detection distance in pixels.
    sensor_angle : float
        Half-FOV angle (radians from heading). Rays span from -sensor_angle
        to +sensor_angle, evenly distributed.
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
        self.density_radius: float = density_radius if density_radius is not None else sensor_range

        # Create 8 rays evenly spaced across the FOV [-sensor_angle, +sensor_angle]
        offsets = np.linspace(-sensor_angle, sensor_angle, NN_NUM_SENSORS)
        self.rays: list[SensorRay] = [
            SensorRay(angle_offset=float(offset), max_range=sensor_range)
            for offset in offsets
        ]

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
        lakes: list[Any] | None = None,
        pheromone_grid: np.ndarray | None = None,
        pheromone_cell_size: float = 10.0,
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

            # Pre-extract tuples, filtering by category
            food_targets: list[tuple[float, float, float]] = []
            enemy_targets: list[tuple[float, float, float, bool, bool]] = []
            ally_targets: list[tuple[float, float, float, bool, bool]] = []
            
            my_species = type(allies[0]) if allies else None

            for n in nearby:
                px = float(n.position[0])
                py = float(n.position[1])
                
                # Is it food?
                if hasattr(n, "consumed"):
                    if not n.consumed:
                        food_targets.append((px, py, float(n.radius)))
            
                # Is it a creature?
                elif n.alive:
                    # Is it my species?
                    if type(n) is my_species:
                        distance = math.hypot(px - ox, py - oy)
                        if distance > 0.1:  # Exclude self
                            ally_targets.append((px, py, float(n.radius), n.is_eating, n.is_attacking))
                    # It must be an enemy
                    else:
                        enemy_targets.append((px, py, float(n.radius), n.is_eating, n.is_attacking))
            
        else:
            # Fallback: full-list mode (no spatial hash available)
            food_targets = [
                (float(f.position[0]), float(f.position[1]), float(f.radius))
                for f in food_items if not getattr(f, 'consumed', False)
            ]
            enemy_targets = [
                (float(e.position[0]), float(e.position[1]), float(e.radius), bool(getattr(e, 'is_eating', False)), bool(getattr(e, 'is_attacking', False)))
                for e in enemies if getattr(e, 'alive', True)
            ]
            ally_targets = [
                (float(a.position[0]), float(a.position[1]), float(a.radius), bool(getattr(a, 'is_eating', False)), bool(getattr(a, 'is_attacking', False)))
                for a in allies if getattr(a, 'alive', True)
                and math.hypot(a.position[0] - ox, a.position[1] - oy) > 0.1
            ]

        # -----------------------------------------------------------------
        # Cast 8 rays using the pre-extracted target tuples
        # -----------------------------------------------------------------
        for i, ray in enumerate(self.rays):
            ray_res = ray.cast(
                position, direction, food_targets, enemy_targets, ally_targets,
                world_width, world_height,
                lakes=lakes,
                pheromone_grid=pheromone_grid,
                pheromone_cell_size=pheromone_cell_size,
            )
            data.sectors[i].enemy_distance = ray_res.enemy_distance
            data.sectors[i].enemy_is_eating = ray_res.enemy_is_eating
            data.sectors[i].enemy_is_attacking = ray_res.enemy_is_attacking
            data.sectors[i].ally_distance = ray_res.ally_distance
            data.sectors[i].ally_is_eating = ray_res.ally_is_eating
            data.sectors[i].ally_is_attacking = ray_res.ally_is_attacking
            data.sectors[i].food_distance = ray_res.food_distance
            data.sectors[i].wall_distance = ray_res.wall_distance
            data.sectors[i].pheromone_distance = ray_res.pheromone_distance

        # -----------------------------------------------------------------
        # Density awareness — reuse the extracted tuples
        # -----------------------------------------------------------------
        data.ally_density = self._count_density(ox, oy, ally_targets)
        data.enemy_density = self._count_density(ox, oy, enemy_targets)

        return data

    def _count_density(
        self,
        ox: float, oy: float,
        targets: list[Any],
    ) -> float:
        """Count how many targets are within the density radius.

        Compares squared distances to avoid sqrt calls.
        """
        dr_sq = self.density_radius * self.density_radius
        count = 0

        for target in targets:
            dx = target[0] - ox
            dy = target[1] - oy
            if dx * dx + dy * dy <= dr_sq:
                count += 1

        return min(1.0, count / MAX_DENSITY_COUNT)
