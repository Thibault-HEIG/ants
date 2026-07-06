"""
renderer.py — Pygame Visualization and Simulation Renderer
==========================================================

Renders the 2-D simulation arena, background zones, food items, living/dead
creature sprites, vision/density sensor rays, and UI HUD overlays.
Supports dynamic species scaling without hardcoded class name loops.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pygame

from core.constants import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    BG_COLOR,
    ZONE_DANGER_COLOR,
    ZONE_SAFE_COLOR,
    ZONE_BOUNDARY_X,
    HUD_FONT_SIZE,
    HUD_ACCENT_COLOR,
    SPIDER_ACCENT_COLOR,
)
from rendering.ui import draw_health_bar, draw_hud_panel

if TYPE_CHECKING:
    from world.world import World


class Renderer:
    """Handles all visual output using Pygame.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    """

    def __init__(self, width: int = WORLD_WIDTH, height: int = WORLD_HEIGHT) -> None:
        pygame.init()
        pygame.font.init()

        self.width: int = width
        self.height: int = height
        self.screen: pygame.Surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Ecosystem Simulation — Advanced OOP Engine")

        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.font: pygame.font.Font = pygame.font.SysFont("Consolas, Courier, monospace", HUD_FONT_SIZE, bold=True)
        self.show_sensors: bool = False

        # Load sprites dynamically from assets directory
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        self.sprites: dict[str, pygame.Surface] = {}

        self._load_sprite("Ant", os.path.join(assets_dir, "ant.png"), (24, 24))
        self._load_sprite("Spider", os.path.join(assets_dir, "spider.png"), (32, 32))

    def _load_sprite(self, species_name: str, filepath: str, size: tuple[int, int]) -> None:
        try:
            if os.path.exists(filepath):
                image = pygame.image.load(filepath).convert_alpha()
                self.sprites[species_name] = pygame.transform.smoothscale(image, size)
        except Exception:
            pass  # Fallback to geometric shapes if loading fails

    def toggle_sensors(self) -> None:
        """Toggle visualization of sensory rays."""
        self.show_sensors = not self.show_sensors

    def render(self, world: World, simulation: Any | None = None) -> None:
        """Render one full frame of the simulation."""
        # 1. Background and Zones
        self.screen.fill(BG_COLOR)

        left_zone = pygame.Rect(0, 0, int(ZONE_BOUNDARY_X), self.height)
        right_zone = pygame.Rect(int(ZONE_BOUNDARY_X), 0, self.width - int(ZONE_BOUNDARY_X), self.height)
        pygame.draw.rect(self.screen, ZONE_DANGER_COLOR, left_zone)
        pygame.draw.rect(self.screen, ZONE_SAFE_COLOR, right_zone)

        # Boundary dashed line
        dash_len = 10
        for y in range(0, self.height, dash_len * 2):
            pygame.draw.line(self.screen, (70, 80, 90), (ZONE_BOUNDARY_X, y), (ZONE_BOUNDARY_X, min(self.height, y + dash_len)), 1)

        # 2. Food Items
        for food in world.food_items:
            if getattr(food, "consumed", False):
                continue
            fx, fy = int(food.position[0]), int(food.position[1])
            pygame.draw.circle(self.screen, (80, 220, 100), (fx, fy), int(food.radius))
            pygame.draw.circle(self.screen, (150, 255, 170), (fx, fy), max(1, int(food.radius - 2)))

        # 3. Living Creatures
        for cls, alive_list in world.creatures.items():
            species_name = getattr(cls, "species_name", cls.__name__)
            for creature in alive_list:
                if not getattr(creature, "alive", False):
                    continue
                if self.show_sensors:
                    self._draw_sensors(creature)
                self._draw_creature(creature, species_name, is_dead=False)

        # 5. HUD Panel
        creature_stats = {}
        for cls in world.active_species:
            name = getattr(cls, "species_name", cls.__name__)
            alive = len(world.creatures.get(cls, []))
            total = simulation.get_total_spawned(cls) if simulation and hasattr(simulation, "get_total_spawned") else alive
            creature_stats[name] = (alive, total)

        speed = getattr(simulation, "speed_multiplier", 1.0) if simulation else 1.0

        draw_hud_panel(
            self.screen,
            self.font,
            self.width,
            round_time=world.round_time,
            sim_speed=speed,
            fps=self.clock.get_fps(),
            show_sensors=self.show_sensors,
            creature_stats=creature_stats,
        )

        pygame.display.flip()
        self.clock.tick(60)

    def _draw_creature(self, creature: Any, species_name: str, is_dead: bool) -> None:
        cx, cy = int(creature.position[0]), int(creature.position[1])
        angle_deg = -math.degrees(creature.direction) - 90.0

        sprite = self.sprites.get(species_name)
        if sprite is not None:
            if is_dead:
                # Tint dead sprite gray
                img = sprite.copy()
                img.fill((100, 100, 100, 128), special_flags=pygame.BLEND_RGBA_MULT)
            else:
                img = sprite
            rotated = pygame.transform.rotate(img, angle_deg)
            rect = rotated.get_rect(center=(cx, cy))
            self.screen.blit(rotated, rect)
        else:
            # Fallback geometric rendering
            color = HUD_ACCENT_COLOR if species_name == "Ant" else SPIDER_ACCENT_COLOR
            if is_dead:
                color = (100, 100, 100)
            pygame.draw.circle(self.screen, color, (cx, cy), int(creature.radius))
            # Heading indicator line
            hx = cx + int(math.cos(creature.direction) * creature.radius * 1.5)
            hy = cy + int(math.sin(creature.direction) * creature.radius * 1.5)
            pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (hx, hy), 2)

        if not is_dead:
            draw_health_bar(self.screen, float(cx), float(cy), creature.radius, creature.health, creature.max_health)

            # Attack visual slash
            if getattr(creature, "is_attacking", False):
                strike_r = int(getattr(creature, "strike_range", 20.0))
                pygame.draw.circle(self.screen, (255, 200, 50), (cx, cy), strike_r, 1)

    def _draw_sensors(self, creature: Any) -> None:
        cx, cy = float(creature.position[0]), float(creature.position[1])
        s_range = getattr(creature.sensors, "sensor_range", 100.0)

        # Draw vision rays
        for ray in [creature.sensors.left_ray, creature.sensors.right_ray, creature.sensors.forward_ray]:
            ray_angle = creature.direction + ray.angle_offset
            ex = cx + math.cos(ray_angle) * s_range
            ey = cy + math.sin(ray_angle) * s_range
            pygame.draw.line(self.screen, (100, 120, 140), (int(cx), int(cy)), (int(ex), int(ey)), 1)
