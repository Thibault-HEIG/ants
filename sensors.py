"""sensors.py — Vision System for Ants and Spiders
==================================================

Creatures perceive the world through three "vision rays" that extend
from their body at fixed angles relative to their heading direction:
  - **Left ray**: offset by -sensor_angle from heading
  - **Right ray**: offset by +sensor_angle from heading
  - **Forward ray**: straight ahead (0° offset)

Each ray detects the nearest:
  - food item
  - enemy (ants see spiders, spiders see ants)
  - ally (own species)
  - wall (world boundary)

In addition to ray-based vision, creatures have:
  - **Omnidirectional sensing** ("smell"): distance and angle to the
    nearest food and nearest enemy regardless of ray direction.
  - **Density awareness**: count of nearby allies and enemies within
    a configurable radius, enabling swarm/pack decisions.

Distances are normalised to [0, 1]:
  - ``0.0`` = object is touching
  - ``1.0`` = nothing detected within sensor range

Both species use the same Sensors class but with different range
parameters (ants: 150 px, spiders: 250 px).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ant_simulator.utils import distance
from ant_simulator.constants import MAX_DENSITY_COUNT

if TYPE_CHECKING:
    from ant_simulator.food import Food


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
    """Combined readings from all three vision rays plus omnidirectional
    and density sensors.

    This is the data structure passed to the creature's brain each frame.
    """
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
        """Convert sensor readings + internal state into the 22-element
        input vector expected by the brain.

        Parameters
        ----------
        hp : float
            Normalised current HP level (0 = dead, 1 = max hp).
        zone : float
            Current zone (0.0 = Danger Zone, 1.0 = Safe Zone).
        speed : float
            Normalised current speed (0 = stopped, 1 = max speed).
        age : float
            Normalised survival time (0 = just spawned, 1 = round limit).

        Returns
        -------
        np.ndarray, shape (22,)
            Full input vector for the neural network.
        """
        return np.array([
            # --- Original ray inputs (left / right) ---
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
            # --- Forward ray (B1) ---
            self.food_fwd,              # 10
            self.enemy_fwd,             # 11
            self.ally_fwd,              # 12
            self.wall_fwd,              # 13
            # --- Omnidirectional sensing (B2-B5) ---
            self.nearest_food_dist,     # 14
            self.nearest_food_angle,    # 15
            self.nearest_enemy_dist,    # 16
            self.nearest_enemy_angle,   # 17
            # --- Intelligence (A1-A2) ---
            speed,                      # 18
            age,                        # 19
            # --- Teamwork (C1, C4) ---
            self.ally_density,          # 20
            self.enemy_density,         # 21
        ], dtype=float)


class SensorRay:
    """A single vision ray cast from the ant's position.

    Parameters
    ----------
    angle_offset : float
        Angle relative to the ant's heading (radians).
        Negative = left, positive = right, zero = forward.
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
        food_items: list[Food],
        enemies: list[Any],
        allies: list[Any],
        world_width: float,
        world_height: float,
    ) -> RayResult:
        """Cast this ray and find the nearest food, enemy, and wall.

        Rather than stepping along the ray in small increments (slow), we
        compute exact distances to every candidate and keep the closest one
        that falls within the ray's cone.

        Parameters
        ----------
        origin : np.ndarray
            The creature's [x, y] position.
        heading : float
            The creature's current facing direction (radians).
        food_items : list[Food]
            All food in the world (we filter by range/angle).
        enemies : list
            All enemies in the world (ants see spiders, spiders see ants).
        allies : list
            All allies in the world (excluding self ideally, or distance check ignores 0).
        world_width, world_height : float
            World bounds for wall detection.

        Returns
        -------
        RayResult
            Normalised distances for this ray.
        """
        # The absolute angle this ray points in
        ray_angle = heading + self.angle_offset

        # Unit direction vector along the ray
        ray_dir = np.array([math.cos(ray_angle), math.sin(ray_angle)])

        result = RayResult()

        # --- Detect nearest food along this ray ---
        result.food_distance = self._detect_nearest(
            origin, ray_dir, [(f.position, f.radius) for f in food_items if not f.consumed]
        )

        # --- Detect nearest enemy along this ray ---
        result.enemy_distance = self._detect_nearest(
            origin, ray_dir, [(e.position, e.radius) for e in enemies if e.alive]
        )

        # --- Detect nearest ally along this ray ---
        # Exclude self by checking distance > 0.1
        result.ally_distance = self._detect_nearest(
            origin, ray_dir, [(a.position, a.radius) for a in allies if a.alive and np.linalg.norm(a.position - origin) > 0.1]
        )

        # --- Detect wall distance ---
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
        """Find the nearest target along the ray direction.

        We project each target onto the ray and check if it's close enough
        to the ray line to count as a "hit".

        Parameters
        ----------
        origin : np.ndarray
            Ray origin (ant position).
        ray_dir : np.ndarray
            Unit vector along the ray.
        targets : list of (position, radius)
            Candidate objects.

        Returns
        -------
        float
            Normalised distance (0 = touching, 1 = nothing detected).
        """
        nearest = self.max_range

        for target_pos, target_radius in targets:
            to_target = target_pos - origin
            dist = float(np.linalg.norm(to_target))

            if dist < 1e-6:
                # Target is right on top of us
                return 0.0

            if dist > self.max_range:
                continue

            # Project the target onto the ray direction.
            # dot_product tells us how far along the ray the target is.
            dot_product = float(np.dot(to_target, ray_dir))

            if dot_product < 0:
                # Target is behind the ray — ignore it.
                continue

            # Perpendicular distance from the target to the ray line.
            # If this is larger than the target's radius + a tolerance,
            # the ray "misses" the target.
            perp_dist = math.sqrt(max(0.0, dist * dist - dot_product * dot_product))

            # The hit tolerance scales with range — closer objects are
            # easier to detect.
            hit_tolerance = target_radius + 15.0  # pixels of leeway

            if perp_dist <= hit_tolerance and dot_product < nearest:
                nearest = dot_product

        # Normalise: 0 = touching, 1 = at max range or beyond
        return max(0.0, min(1.0, nearest / self.max_range))

    def _detect_wall(
        self,
        origin: np.ndarray,
        ray_dir: np.ndarray,
        world_width: float,
        world_height: float,
    ) -> float:
        """Calculate how far until the ray hits a world boundary.

        We compute the intersection with all four walls and return the
        nearest one.

        Returns
        -------
        float
            Normalised distance to the nearest wall (0 = at wall, 1 = far).
        """
        min_t = self.max_range

        # For each axis, compute how far along the ray until we hit
        # the near wall (coordinate = 0) or far wall (coordinate = width/height).
        # t = (wall_coord - origin_coord) / ray_dir_coord

        if abs(ray_dir[0]) > 1e-9:
            # Left wall (x = 0) and right wall (x = world_width)
            t_left = -origin[0] / ray_dir[0]
            t_right = (world_width - origin[0]) / ray_dir[0]
            if t_left > 0:
                min_t = min(min_t, t_left)
            if t_right > 0:
                min_t = min(min_t, t_right)

        if abs(ray_dir[1]) > 1e-9:
            # Top wall (y = 0) and bottom wall (y = world_height)
            t_top = -origin[1] / ray_dir[1]
            t_bottom = (world_height - origin[1]) / ray_dir[1]
            if t_top > 0:
                min_t = min(min_t, t_top)
            if t_bottom > 0:
                min_t = min(min_t, t_bottom)

        return max(0.0, min(1.0, min_t / self.max_range))


class Sensors:
    """The complete sensor array: three vision rays plus omnidirectional
    and density sensors.

    Used by both ants and spiders.  The sensor range is configurable
    per species (ants: 150 px, spiders: 250 px).

    Parameters
    ----------
    sensor_range : float
        Maximum detection distance in pixels.
    sensor_angle : float
        Angle offset for left/right rays (radians from heading).
    density_radius : float
        Radius for counting nearby allies/enemies (density inputs).
    """

    def __init__(self, sensor_range: float, sensor_angle: float,
                 density_radius: float | None = None) -> None:
        self.sensor_range: float = sensor_range
        self.left_ray = SensorRay(angle_offset=-sensor_angle, max_range=sensor_range)
        self.right_ray = SensorRay(angle_offset=+sensor_angle, max_range=sensor_range)
        self.forward_ray = SensorRay(angle_offset=0.0, max_range=sensor_range)
        self.density_radius: float = density_radius if density_radius is not None else sensor_range

    def perceive(
        self,
        position: np.ndarray,
        direction: float,
        food_items: list,
        enemies: list,
        allies: list,
        world_width: float,
        world_height: float,
    ) -> SensorData:
        """Gather all sensor readings for one creature.

        Parameters
        ----------
        position : np.ndarray
            Creature's [x, y] position.
        direction : float
            Creature's heading in radians.
        food_items : list[Food]
            All food currently in the world.
        enemies : list
             enemies : list
            Living enemies.
        allies : list
            Living allies.
        world_width, world_height : float
            World boundaries.

        Returns
        -------
        SensorData
            Complete sensor readings: left/right/forward rays,
            omnidirectional food/enemy, and ally/enemy density.
        """
        data = SensorData()
        
        # Cast left ray
        left_res = self.left_ray.cast(
            position, direction, food_items, enemies, allies, world_width, world_height
        )
        data.food_left = left_res.food_distance
        data.enemy_left = left_res.enemy_distance
        data.ally_left = left_res.ally_distance
        data.wall_left = left_res.wall_distance

        # Cast right ray
        right_res = self.right_ray.cast(
            position, direction, food_items, enemies, allies, world_width, world_height
        )
        data.food_right = right_res.food_distance
        data.enemy_right = right_res.enemy_distance
        data.ally_right = right_res.ally_distance
        data.wall_right = right_res.wall_distance

        # Cast forward ray
        fwd_res = self.forward_ray.cast(
            position, direction, food_items, enemies, allies, world_width, world_height
        )
        data.food_fwd = fwd_res.food_distance
        data.enemy_fwd = fwd_res.enemy_distance
        data.ally_fwd = fwd_res.ally_distance
        data.wall_fwd = fwd_res.wall_distance

        # --- Omnidirectional sensing (B2-B5) ---
        # Nearest food: distance + angle
        food_positions = [f.position for f in food_items if not f.consumed]
        data.nearest_food_dist, data.nearest_food_angle = self._detect_omnidirectional(
            position, direction, food_positions
        )

        # Nearest enemy: distance + angle
        enemy_positions = [e.position for e in enemies if e.alive]
        data.nearest_enemy_dist, data.nearest_enemy_angle = self._detect_omnidirectional(
            position, direction, enemy_positions
        )

        # --- Density awareness (C1, C4) ---
        ally_positions = [a.position for a in allies if a.alive and np.linalg.norm(a.position - position) > 0.1]
        data.ally_density = self._count_density(position, ally_positions)
        data.enemy_density = self._count_density(position, enemy_positions)

        return data

    def _detect_omnidirectional(
        self,
        origin: np.ndarray,
        heading: float,
        target_positions: list[np.ndarray],
    ) -> tuple[float, float]:
        """Find the nearest target in any direction (360°).

        Unlike ray-based detection, this considers ALL targets regardless
        of the creature's facing direction — like a "smell" sense.

        Parameters
        ----------
        origin : np.ndarray
            Creature's [x, y] position.
        heading : float
            Creature's heading in radians (used to compute relative angle).
        target_positions : list[np.ndarray]
            Positions of candidate targets.

        Returns
        -------
        tuple[float, float]
            (normalised_distance, relative_angle)
            - distance: 0 = touching, 1 = at/beyond sensor range
            - angle: [-1, 1] where -1 = directly left, +1 = directly right,
              0 = straight ahead or behind
        """
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
                    # Compute angle to target relative to heading
                    abs_angle = math.atan2(to_target[1], to_target[0])
                    rel_angle = abs_angle - heading

                    # Normalise to [-π, π]
                    rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))

                    # Map to [-1, 1]: -π → -1, 0 → 0, +π → +1
                    nearest_angle = rel_angle / math.pi

        # Normalise distance to [0, 1] using sensor range
        normalised_dist = max(0.0, min(1.0, nearest_dist / self.sensor_range))

        return normalised_dist, nearest_angle

    def _count_density(
        self,
        origin: np.ndarray,
        target_positions: list[np.ndarray],
    ) -> float:
        """Count how many targets are within the density radius.

        Returns a normalised value in [0, 1] where 1.0 means
        MAX_DENSITY_COUNT or more targets are nearby.

        Parameters
        ----------
        origin : np.ndarray
            Creature's [x, y] position.
        target_positions : list[np.ndarray]
            Positions of candidate targets (already filtered for alive).

        Returns
        -------
        float
            Normalised density in [0, 1].
        """
        count = 0
        for target_pos in target_positions:
            dist = float(np.linalg.norm(target_pos - origin))
            if dist <= self.density_radius:
                count += 1

        return min(1.0, count / MAX_DENSITY_COUNT)
