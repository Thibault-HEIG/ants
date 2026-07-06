"""
ui.py — User Interface Overlays and HUD Components
==================================================

Provides decoupled UI rendering utilities for health bars, text rendering,
and top-level simulation status panels.
"""

from __future__ import annotations

from typing import Any
import pygame

from core.constants import (
    HUD_TEXT_COLOR,
    HUD_BG_COLOR,
    HUD_ACCENT_COLOR,
    SPIDER_ACCENT_COLOR,
    ROUND_TIME_LIMIT,
)


def draw_health_bar(
    surface: pygame.Surface,
    x: float,
    y: float,
    radius: float,
    health: float,
    max_health: float,
) -> None:
    """Draw a miniature health bar above a creature."""
    if max_health <= 0 or health >= max_health:
        return

    bar_width = int(radius * 2.5)
    bar_height = 3
    bar_x = int(x - bar_width / 2)
    bar_y = int(y - radius - 6)

    # Background (red)
    pygame.draw.rect(surface, (180, 50, 50), (bar_x, bar_y, bar_width, bar_height))

    # Foreground (green)
    fill_width = int(bar_width * max(0.0, min(1.0, health / max_health)))
    if fill_width > 0:
        pygame.draw.rect(surface, (50, 180, 50), (bar_x, bar_y, fill_width, bar_height))


def draw_hud_panel(
    surface: pygame.Surface,
    font: pygame.font.Font,
    width: int,
    generation: int,
    round_time: float,
    sim_speed: float,
    fps: float,
    show_sensors: bool,
    creature_stats: dict[str, tuple[int, int]] | None = None,
    # Legacy arguments for backwards compatibility
    ant_count: int = 0,
    spider_count: int = 0,
    total_ants: int = 0,
    total_spiders: int = 0,
) -> None:
    """Draw the top HUD panel displaying simulation metrics and species counts."""
    panel_height = 40
    panel_rect = pygame.Rect(0, 0, width, panel_height)
    pygame.draw.rect(surface, HUD_BG_COLOR, panel_rect)
    pygame.draw.line(surface, HUD_ACCENT_COLOR, (0, panel_height - 1), (width, panel_height - 1), 2)

    # Format time string
    time_str = f"{round_time:.1f}s"
    if ROUND_TIME_LIMIT > 0:
        time_str += f" / {ROUND_TIME_LIMIT:.0f}s"

    # Build stat texts
    left_texts = [
        f"Gen: {generation}",
        f"Time: {time_str}",
        f"Speed: {sim_speed}x",
        f"FPS: {fps:.0f}",
    ]

    # Species counts
    species_texts = []
    if creature_stats is not None:
        for name, (alive, total) in creature_stats.items():
            species_texts.append((f"{name}s: {alive} (Peak: {total})", HUD_TEXT_COLOR))
    else:
        species_texts.append((f"Ants: {ant_count} (Peak: {total_ants})", HUD_ACCENT_COLOR))
        species_texts.append((f"Spiders: {spider_count} (Peak: {total_spiders})", SPIDER_ACCENT_COLOR))

    # Render left section
    x_offset = 15
    for text in left_texts:
        img = font.render(text, True, HUD_TEXT_COLOR)
        surface.blit(img, (x_offset, 10))
        x_offset += img.get_width() + 25

    # Render species counts in center/right
    for text, color in species_texts:
        img = font.render(text, True, color)
        surface.blit(img, (x_offset, 10))
        x_offset += img.get_width() + 25

    # Render sensor toggle status on far right
    sensor_text = "Sensors [S]: ON" if show_sensors else "Sensors [S]: OFF"
    img_sensor = font.render(sensor_text, True, (150, 200, 150) if show_sensors else (150, 150, 150))
    surface.blit(img_sensor, (width - img_sensor.get_width() - 15, 10))
