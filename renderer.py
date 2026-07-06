"""
renderer.py — Pygame Rendering
================================

Draws the simulation state to screen.  This module has **read-only**
access to the World — it never modifies game state.

**Design principle**: Rendering is completely decoupled from simulation.
You could run the simulation headless (without a renderer) and nothing
would break.  This makes it easy to later add alternative renderers
(e.g. a web-based viewer) or run batch evolution without graphics.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import pygame
from ant_simulator.utils import SpeciesStats

from ant_simulator.constants import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    ZONE_DANGER_COLOR,
    ZONE_SAFE_COLOR,
    ZONE_BOUNDARY_X,
    HUD_TEXT_COLOR,
    HUD_BG_COLOR,
    HUD_ACCENT_COLOR,
    SPIDER_ACCENT_COLOR,
    HUD_FONT_SIZE,
    ANT_INITIAL_HEALTH,
    SPIDER_INITIAL_HEALTH,
    MAX_ANTS,
    MAX_SPIDERS,
)

if TYPE_CHECKING:
    from ant_simulator.world import World
    from ant_simulator.ant import Ant
    from ant_simulator.predator import Predator
    from ant_simulator.food import Food


class Renderer:
    """Pygame renderer for the ant vs spider simulation.

    Parameters
    ----------
    screen : pygame.Surface
        The main display surface to draw on.
    """

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen: pygame.Surface = screen

        # --- Load and scale sprites ---
        assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "assets"
        )

        self.ant_sprite: pygame.Surface = self._load_sprite(
            os.path.join(assets_dir, "ant.png"), target_size=20
        )
        self.spider_sprite: pygame.Surface = self._load_sprite(
            os.path.join(assets_dir, "predator.png"), target_size=40
        )
        self.food_sprite: pygame.Surface = self._load_sprite(
            os.path.join(assets_dir, "food.png"), target_size=14
        )

        # --- Font for HUD ---
        pygame.font.init()
        self.font: pygame.font.Font = pygame.font.SysFont(
            "monospace", HUD_FONT_SIZE
        )
        self.small_font: pygame.font.Font = pygame.font.SysFont(
            "monospace", 12
        )

    @staticmethod
    def _load_sprite(path: str, target_size: int) -> pygame.Surface:
        """Load a PNG sprite and scale it to a square of ``target_size``."""
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(image, (target_size, target_size))

    # ------------------------------------------------------------------
    # Main render call
    # ------------------------------------------------------------------

    def render(
        self,
        world: World,
        stats: dict,
        speed_multiplier: float = 1.0,
        paused: bool = False,
    ) -> None:
        """Draw one complete frame."""
        # Draw background zones
        left_rect = pygame.Rect(0, 0, int(ZONE_BOUNDARY_X), WORLD_HEIGHT)
        right_rect = pygame.Rect(int(ZONE_BOUNDARY_X), 0, WORLD_WIDTH - int(ZONE_BOUNDARY_X), WORLD_HEIGHT)
        pygame.draw.rect(self.screen, ZONE_DANGER_COLOR, left_rect)
        pygame.draw.rect(self.screen, ZONE_SAFE_COLOR, right_rect)

        # Draw in back-to-front order: food → ants → spiders → HUD
        self._draw_food(world.food_items)
        self._draw_ants(world.ants)
        self._draw_spiders(world.spiders)
        self._draw_hud(stats, speed_multiplier, paused)

    # ------------------------------------------------------------------
    # Entity drawing
    # ------------------------------------------------------------------

    def _draw_ants(self, ants: list[Ant]) -> None:
        """Draw all living ants with rotated sprites and health bars."""
        for ant in ants:
            if not ant.alive:
                continue

            angle_deg = -math.degrees(ant.direction)
            rotated = pygame.transform.rotate(self.ant_sprite, angle_deg)
            rect = rotated.get_rect(center=(int(ant.position[0]), int(ant.position[1])))
            self.screen.blit(rotated, rect)

            # Visual indicator when attacking (strike range)
            if ant.is_attacking:
                pygame.draw.circle(
                    self.screen,
                    (255, 100, 100),  # Red attack flash/circle
                    (int(ant.position[0]), int(ant.position[1])),
                    int(ant.strike_range),
                    1,
                )

            # Health bar
            health_frac = ant.health / ANT_INITIAL_HEALTH
            self._draw_health_bar(ant.position, health_frac)

    def _draw_spiders(self, spiders: list[Predator]) -> None:
        """Draw all living spiders with rotated sprites and health bars."""
        for spider in spiders:
            if not spider.alive:
                continue

            angle_deg = -math.degrees(spider.direction)
            rotated = pygame.transform.rotate(self.spider_sprite, angle_deg)
            rect = rotated.get_rect(center=(int(spider.position[0]), int(spider.position[1])))
            self.screen.blit(rotated, rect)

            # Visual indicator when attacking (strike range)
            if spider.is_attacking:
                pygame.draw.circle(
                    self.screen,
                    (255, 50, 200),  # Purple/pink attack circle
                    (int(spider.position[0]), int(spider.position[1])),
                    int(spider.strike_range),
                    2,
                )

            # Health bar — spiders get them too now
            health_frac = spider.health / SPIDER_INITIAL_HEALTH
            self._draw_health_bar(spider.position, health_frac, bar_width=22)

    def _draw_food(self, food_items: list[Food]) -> None:
        """Draw all unconsumed food items."""
        for food in food_items:
            if food.consumed:
                continue
            rect = self.food_sprite.get_rect(
                center=(int(food.position[0]), int(food.position[1]))
            )
            self.screen.blit(self.food_sprite, rect)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(
        self,
        stats: dict,
        speed_multiplier: float,
        paused: bool,
    ) -> None:
        """Draw the heads-up display panel showing both species stats."""
        minutes = int(stats['elapsed_time'] // 60)
        seconds = int(stats['elapsed_time'] % 60)
        
        lines = [
            (f"Time:        {minutes:02d}:{seconds:02d}", HUD_ACCENT_COLOR),
            (f"Speed:       {speed_multiplier:.1f}x", HUD_TEXT_COLOR),
            ("", HUD_TEXT_COLOR),  # spacer
            (f"Ants:        {stats['alive_ants']} / {MAX_ANTS}", HUD_ACCENT_COLOR),
            (f"  Total Bred: {stats['total_ants']}", HUD_TEXT_COLOR),
            (f"  Avg Fit:   {stats['avg_ant_fitness']:.1f}", HUD_TEXT_COLOR),
            ("", HUD_TEXT_COLOR),  # spacer
            (f"Spiders:     {stats['alive_spiders']} / {MAX_SPIDERS}", SPIDER_ACCENT_COLOR),
            (f"  Total Bred: {stats['total_spiders']}", HUD_TEXT_COLOR),
            (f"  Avg Fit:   {stats['avg_spider_fitness']:.1f}", HUD_TEXT_COLOR),
        ]
        if paused:
            lines.append(("** PAUSED **", (255, 200, 50)))

        # Semi-transparent background panel
        panel_width = 240
        panel_height = len(lines) * (HUD_FONT_SIZE + 4) + 16
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((*HUD_BG_COLOR, 200))
        self.screen.blit(panel, (8, 8))

        # Render each line
        y = 16
        for text, color in lines:
            if text:  # skip empty spacers
                text_surface = self.font.render(text, True, color)
                self.screen.blit(text_surface, (16, y))
            y += HUD_FONT_SIZE + 4

        self._draw_species_stats(8 + panel_height + 8)

    def _draw_species_stats(self, start_y: int) -> None:
        """Draw all-time species maximum records in a small font panel."""
        lines = [
            ("ALL-TIME SPECIES MAX RECORDS", HUD_ACCENT_COLOR),
            (f"Ants:    Life {SpeciesStats.ant_max_lifetime:.0f}s | Food {SpeciesStats.ant_max_foodeaten} | Touch {SpeciesStats.ant_max_enemies_touched}", HUD_TEXT_COLOR),
            (f"Spiders: Life {SpeciesStats.spider_max_lifetime:.0f}s | Food {SpeciesStats.spider_max_foodeaten} | Touch {SpeciesStats.spider_max_enemies_touched}", SPIDER_ACCENT_COLOR),
        ]

        panel_width = 380
        line_height = 16
        panel_height = len(lines) * line_height + 14
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((*HUD_BG_COLOR, 200))
        self.screen.blit(panel, (8, start_y))

        y = start_y + 7
        for text, color in lines:
            text_surface = self.small_font.render(text, True, color)
            self.screen.blit(text_surface, (16, y))
            y += line_height

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _draw_health_bar(
        self,
        position,
        health_fraction: float,
        bar_width: int = 16,
    ) -> None:
        """Draw a small health bar above an entity.

        Parameters
        ----------
        position : np.ndarray
            Entity's [x, y] position (bar is drawn above this).
        health_fraction : float
            0.0 = empty (red), 1.0 = full (green).
        bar_width : int
            Width of the health bar in pixels (wider for spiders).
        """
        bar_height = 3
        x = int(position[0]) - bar_width // 2
        y = int(position[1]) - 16

        # Background (dark)
        pygame.draw.rect(
            self.screen, (60, 60, 60),
            (x, y, bar_width, bar_height),
        )

        # Foreground — colour transitions from green to red
        fill_width = int(bar_width * max(0.0, min(1.0, health_fraction)))
        if health_fraction > 0.5:
            color = (100, 220, 100)
        elif health_fraction > 0.25:
            color = (220, 180, 50)
        else:
            color = (220, 60, 60)

        if fill_width > 0:
            pygame.draw.rect(
                self.screen, color,
                (x, y, fill_width, bar_height),
            )
