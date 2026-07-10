"""
renderer.py — Pygame Visualization and Simulation Renderer
==========================================================

Renders the 2-D simulation arena, background zones, food sources, typed food
items (sugar/seed sprites), living/dead creature sprites, vision/density sensor
rays, eating indicators, and UI HUD overlays.
Supports dynamic species scaling without hardcoded class name loops.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pygame
import pygame._sdl2.video as sdl2_video

from core.constants import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    BG_COLOR,
    ZONE_DANGER_COLOR,
    ZONE_SAFE_COLOR,
    ZONE_BOUNDARY_X,
    HUD_FONT_SIZE,
    HUD_ACCENT_COLOR,
    SPIDER_ACCENT_COLOR
)
from rendering.ui import (
    draw_health_bar,
    draw_fps_box,
    draw_window_b_panel,
    LiveFitnessChart,
)

if TYPE_CHECKING:
    from world.world import World


class Renderer:
    """Handles dual-window visual output using Pygame _sdl2.video.

    Parameters
    ----------
    width : int
        Simulation Window A width in pixels.
    height : int
        Simulation Window A height in pixels.
    """

    def __init__(self, width: int = WORLD_WIDTH, height: int = WORLD_HEIGHT) -> None:
        pygame.init()
        pygame.font.init()

        self.width: int = width
        self.height: int = height

        # Dual window SDL2 architecture
        self.win_a = sdl2_video.Window("Ants vs Spiders — Simulation View", size=(self.width, self.height))
        self.rend_a = sdl2_video.Renderer(self.win_a)
        self.surface_a = pygame.Surface((self.width, self.height))

        self.stats_width: int = 680
        self.stats_height: int = 720
        self.win_b = sdl2_video.Window("Ants vs Spiders — Stats & Chart Panel", size=(self.stats_width, self.stats_height))
        self.rend_b = sdl2_video.Renderer(self.win_b)
        self.surface_b = pygame.Surface((self.stats_width, self.stats_height))
        self.window_b_open: bool = True

        try:
            self.win_a.position = (50, 50)
            self.win_b.position = (50 + self.width + 10, 50)
        except Exception:
            pass

        self.chart = LiveFitnessChart(width=self.stats_width - 28, height=320)

        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.font: pygame.font.Font = pygame.font.SysFont("Consolas, Courier, monospace", HUD_FONT_SIZE, bold=True)
        self.show_sensors: bool = False

        # Load sprites dynamically from assets directory
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
        self.sprites: dict[str, pygame.Surface] = {}

        self._load_sprite("Ant", os.path.join(assets_dir, "ant.png"), (24, 24))
        self._load_sprite("Spider", os.path.join(assets_dir, "spider.png"), (32, 32))
        self._load_sprite("sugar", os.path.join(assets_dir, "sugar.png"), (14, 14))
        self._load_sprite("seed", os.path.join(assets_dir, "seed.png"), (12, 16))
        self._load_sprite("anthill", os.path.join(assets_dir, "anthill.png"), (200, 200))
        self._load_sprite("toile", os.path.join(assets_dir, "toile.png"), (200, 200))

        # Animation timer for general visual effects
        self._pulse_timer: float = 0.0

        # Cache for radial shadow surfaces to avoid per-frame allocation
        self._shadow_cache: dict[tuple[int, tuple[int, int, int]], pygame.Surface] = {}

    def _load_sprite(self, name: str, filepath: str, size: tuple[int, int]) -> None:
        try:
            if os.path.exists(filepath):
                image = pygame.image.load(filepath)
                try:
                    image = image.convert_alpha()
                except Exception:
                    pass
                self.sprites[name] = pygame.transform.smoothscale(image, size)
        except Exception:
            pass  # Fallback to geometric shapes if loading fails

    def _get_shadow_surface(self, radius: float, color: tuple[int, int, int]) -> pygame.Surface:
        """Get or generate a soft radial shadow surface of the specified radius and color."""
        r_shadow = max(4, int(radius * 1.35))
        key = (r_shadow, color)
        if key not in self._shadow_cache:
            shadow = pygame.Surface((r_shadow * 2, r_shadow * 2), pygame.SRCALPHA)
            pygame.draw.circle(shadow, (*color, 40), (r_shadow, r_shadow), r_shadow)
            self._shadow_cache[key] = shadow
        return self._shadow_cache[key]

    def toggle_sensors(self) -> None:
        """Toggle visualization of sensory rays."""
        self.show_sensors = not self.show_sensors

    def hide_window_b(self) -> None:
        """Hide Window B without destroying the SDL2 window object."""
        self.win_b.hide()
        self.window_b_open = False

    def toggle_window_b(self) -> None:
        """Toggle Window B visibility."""
        if self.window_b_open:
            self.win_b.hide()
            self.window_b_open = False
        else:
            self.win_b.show()
            self.window_b_open = True

    def render(self, world: World, simulation: Any | None = None) -> None:
        """Render one full frame across dual SDL2 windows."""
        self._pulse_timer += 1.0 / 60.0
        ultra_mode = getattr(simulation, "ultra_mode", False) if simulation else False

        # --- 1. Render Window A (Simulation View) ---
        if ultra_mode:
            # Static black background; skip all creature/food/pheromone drawing
            self.surface_a.fill((0, 0, 0))
        else:
            self.surface_a.fill(BG_COLOR)
            left_zone = pygame.Rect(0, 0, int(ZONE_BOUNDARY_X), self.height)
            right_zone = pygame.Rect(int(ZONE_BOUNDARY_X), 0, self.width - int(ZONE_BOUNDARY_X), self.height)
            pygame.draw.rect(self.surface_a, ZONE_DANGER_COLOR, left_zone)
            pygame.draw.rect(self.surface_a, ZONE_SAFE_COLOR, right_zone)

            dash_len = 10
            for y in range(0, self.height, dash_len * 2):
                pygame.draw.line(self.surface_a, (70, 80, 90), (ZONE_BOUNDARY_X, y), (ZONE_BOUNDARY_X, min(self.height, y + dash_len)), 1)

            if getattr(world, "pheromone_grid", None) is not None:
                cell_size = getattr(world, "pheromone_cell_size", 10.0)
                grid = world.pheromone_grid
                xs, ys = np.where(grid > 0.01)
                for x_idx, y_idx in zip(xs, ys):
                    strength = float(grid[x_idx, y_idx])
                    opacity = min(255, int((strength / 2.0) * 255))
                    if opacity > 5:
                        s = pygame.Surface((int(cell_size), int(cell_size)), pygame.SRCALPHA)
                        s.fill((255, 255, 150, opacity))
                        self.surface_a.blit(s, (int(x_idx * cell_size), int(y_idx * cell_size)))

                if self.show_sensors:
                    gw, gh = grid.shape
                    for gx in range(gw + 1):
                        x_pos = int(gx * cell_size)
                        pygame.draw.line(self.surface_a, (50, 55, 65), (x_pos, 0), (x_pos, self.height), 1)
                    for gy in range(gh + 1):
                        y_pos = int(gy * cell_size)
                        pygame.draw.line(self.surface_a, (50, 55, 65), (0, y_pos), (self.width, y_pos), 1)

            for lake in getattr(world, "lakes", []):
                lx, ly = int(lake.position[0]), int(lake.position[1])
                lrad = int(getattr(lake, "radius", 50.0))
                pygame.draw.circle(self.surface_a, (25, 60, 110), (lx, ly), lrad)
                pygame.draw.circle(self.surface_a, (45, 95, 165), (lx, ly), lrad, 3)

            kingdoms = world.kingdoms.values() if isinstance(getattr(world, "kingdoms", None), dict) else getattr(world, "kingdoms", [])
            for kingdom in kingdoms:
                kx, ky = int(kingdom.position[0]), int(kingdom.position[1])
                kname = getattr(kingdom, "name", "")
                sprite = self.sprites.get(kname)
                if sprite is not None:
                    rect = sprite.get_rect(center=(kx, ky))
                    self.surface_a.blit(sprite, rect)
                else:
                    pygame.draw.circle(self.surface_a, (100, 80, 60), (kx, ky), int(getattr(kingdom, "spawn_radius", 60.0)), 2)

            for food in world.food_items:
                if getattr(food, "consumed", False):
                    continue
                fx, fy = int(food.position[0]), int(food.position[1])
                food_type = getattr(food, "food_type", None)
                sprite = self.sprites.get(food_type) if food_type else None
                if sprite is not None:
                    rect = sprite.get_rect(center=(fx, fy))
                    self.surface_a.blit(sprite, rect)
                else:
                    pygame.draw.circle(self.surface_a, (80, 220, 100), (fx, fy), int(food.radius))
                    pygame.draw.circle(self.surface_a, (150, 255, 170), (fx, fy), max(1, int(food.radius - 2)))

            for cls, alive_list in world.creatures.items():
                species_name = getattr(cls, "species_name", cls.__name__)
                for creature in alive_list:
                    if not getattr(creature, "alive", False):
                        continue
                    if self.show_sensors:
                        self._draw_sensors(creature)
                    self._draw_creature(creature, species_name, is_dead=False)

        # Draw small FPS box top-right in Window A
        draw_fps_box(self.surface_a, self.font, self.clock.get_fps(), self.width, ultra_mode=ultra_mode)

        # Blit surface_a to sdl2 Renderer A
        tex_a = sdl2_video.Texture.from_surface(self.rend_a, self.surface_a)
        self.rend_a.clear()
        self.rend_a.blit(tex_a)
        self.rend_a.present()

        # --- 2. Render Window B (Stats & Chart Panel) ---
        if self.window_b_open:
            draw_window_b_panel(self.surface_b, self.font, world, simulation, self.chart)
            tex_b = sdl2_video.Texture.from_surface(self.rend_b, self.surface_b)
            self.rend_b.clear()
            self.rend_b.blit(tex_b)
            self.rend_b.present()

        # Update clock
        if ultra_mode:
            self.clock.tick()
        else:
            self.clock.tick(60)

    def _draw_creature(self, creature: Any, species_name: str, is_dead: bool) -> None:
        cx, cy = int(creature.position[0]), int(creature.position[1])
        angle_deg = -math.degrees(creature.direction) - 90.0

        if not is_dead:
            if getattr(creature, "is_attacking", False):
                shadow = self._get_shadow_surface(creature.radius, (240, 60, 60))
                self.surface_a.blit(shadow, shadow.get_rect(center=(cx, cy)))

            if getattr(creature, "is_eating", False):
                shadow = self._get_shadow_surface(creature.radius, (255, 255, 255))
                self.surface_a.blit(shadow, shadow.get_rect(center=(cx, cy)))

        sprite = self.sprites.get(species_name)
        if sprite is not None:
            if is_dead:
                img = sprite.copy()
                img.fill((100, 100, 100, 128), special_flags=pygame.BLEND_RGBA_MULT)
            else:
                img = sprite
            rotated = pygame.transform.rotate(img, angle_deg)
            rect = rotated.get_rect(center=(cx, cy))
            self.surface_a.blit(rotated, rect)
        else:
            color = HUD_ACCENT_COLOR if species_name == "Ant" else SPIDER_ACCENT_COLOR
            if is_dead:
                color = (100, 100, 100)
            pygame.draw.circle(self.surface_a, color, (cx, cy), int(creature.radius))
            hx = cx + int(math.cos(creature.direction) * creature.radius * 1.5)
            hy = cy + int(math.sin(creature.direction) * creature.radius * 1.5)
            pygame.draw.line(self.surface_a, (255, 255, 255), (cx, cy), (hx, hy), 2)

        if not is_dead:
            draw_health_bar(self.surface_a, float(cx), float(cy), creature.radius, creature.health, creature.max_health)

    def _draw_sensors(self, creature: Any) -> None:
        cx, cy = float(creature.position[0]), float(creature.position[1])
        s_range = getattr(creature.sensors, 'sensor_range', 100.0)

        for ray in creature.sensors.rays:
            ray_angle = creature.direction + ray.angle_offset
            ex = cx + math.cos(ray_angle) * s_range
            ey = cy + math.sin(ray_angle) * s_range
            pygame.draw.line(self.surface_a, (100, 120, 140), (int(cx), int(cy)), (int(ex), int(ey)), 1)
