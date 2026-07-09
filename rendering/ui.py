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
)
from core.utils import SpeciesStats



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


def _draw_floating_box(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: list[tuple[str, tuple[int, int, int]]],
    pos_x: int,
    pos_y: int,
    align_right: bool = False,
    border_color: tuple[int, int, int, int] = (80, 95, 115, 150),
) -> None:
    """Draw a semi-transparent floating UI box with vertical text lines."""
    if not lines:
        return

    rendered_imgs = [font.render(text, True, color) for text, color in lines]
    padding_x = 14
    padding_y = 10
    line_spacing = 5

    max_w = max(img.get_width() for img in rendered_imgs)
    total_h = sum(img.get_height() for img in rendered_imgs) + line_spacing * (len(rendered_imgs) - 1)

    box_w = max_w + padding_x * 2
    box_h = total_h + padding_y * 2

    actual_x = pos_x - box_w if align_right else pos_x

    # Semi-transparent background (alpha ~75% -> 190)
    box_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    pygame.draw.rect(box_surf, (20, 25, 32, 190), box_surf.get_rect(), border_radius=10)
    pygame.draw.rect(box_surf, border_color, box_surf.get_rect(), width=1, border_radius=10)

    current_y = padding_y
    for img in rendered_imgs:
        box_surf.blit(img, (padding_x, current_y))
        current_y += img.get_height() + line_spacing

    surface.blit(box_surf, (actual_x, pos_y))


def draw_hud_panel(
    surface: pygame.Surface,
    font: pygame.font.Font,
    width: int,
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
    """Draw clean floating HUD panels: Top Left (Current Stats) and Top Right (History Stats)."""
    # 1. Top Left Panel: Current Stats
    left_lines: list[tuple[str, tuple[int, int, int]]] = [
        ("CURRENT STATS", (0, 220, 230)),
        (f"Sim Time    : {round_time:.1f}s", HUD_TEXT_COLOR),
        (f"Sim Speed   : {sim_speed}x", HUD_TEXT_COLOR),
        (f"FPS         : {fps:.0f}", HUD_TEXT_COLOR),
        (f"Sensors [S] : {'ON' if show_sensors else 'OFF'}", (150, 255, 150) if show_sensors else (180, 180, 180)),
        ("Curves  [F] : Terminal", (200, 220, 255)),
    ]

    if creature_stats is not None:
        for name, (alive, total) in creature_stats.items():
            color = HUD_ACCENT_COLOR if name == "Ant" else SPIDER_ACCENT_COLOR
            left_lines.append((f"{name}s Alive : {alive} (Peak: {total})", color))
    else:
        left_lines.append((f"Ants Alive   : {ant_count} (Peak: {total_ants})", HUD_ACCENT_COLOR))
        left_lines.append((f"Spiders Alive: {spider_count} (Peak: {total_spiders})", SPIDER_ACCENT_COLOR))

    _draw_floating_box(surface, font, left_lines, pos_x=15, pos_y=15, align_right=False, border_color=(0, 173, 181, 150))

    # 2. Top Right Panel: History Stats (All-Time Peak Records)
    right_lines: list[tuple[str, tuple[int, int, int]]] = [
        ("HISTORY STATS (ALL-TIME)", (255, 200, 80)),
    ]

    species_names = list(creature_stats.keys()) if creature_stats is not None else ["Ant", "Spider"]
    for name in species_names:
        color = HUD_ACCENT_COLOR if name == "Ant" else SPIDER_ACCENT_COLOR
        max_life = SpeciesStats.max_lifetime.get(name, 0.0)
        max_food = SpeciesStats.max_foodeaten.get(name, 0)
        max_touch = SpeciesStats.max_enemies_touched.get(name, 0)
        right_lines.append((f"--- {name} Records ---", color))
        right_lines.append((f"Max Lifetime : {max_life:.1f}s", HUD_TEXT_COLOR))
        right_lines.append((f"Max Food     : {max_food}", HUD_TEXT_COLOR))
        right_lines.append((f"Max Touches  : {max_touch}", HUD_TEXT_COLOR))

    _draw_floating_box(surface, font, right_lines, pos_x=width - 15, pos_y=15, align_right=True, border_color=(255, 180, 60, 150))
