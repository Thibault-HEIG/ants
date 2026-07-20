"""
ui.py — User Interface Overlays and HUD Components
==================================================

Provides decoupled UI rendering utilities for health bars, text rendering,
and top-level simulation status panels.
"""

from __future__ import annotations

import math
from typing import Any
import pygame

from core.constants import (
    HUD_TEXT_COLOR,
    HUD_ACCENT_COLOR,
    SPIDER_ACCENT_COLOR,
    GENERATION_DURATION,
)
from core.utils import SpeciesStats

_SYS_FONT_CACHE: dict[tuple[str, int, bool], pygame.font.Font] = {}


def get_cached_sysfont(name: str, size: int, bold: bool = False) -> pygame.font.Font:
    """Get a cached Pygame SysFont to avoid repeated system font directory scanning via BufferedReader."""
    key = (name, size, bold)
    if key not in _SYS_FONT_CACHE:
        _SYS_FONT_CACHE[key] = pygame.font.SysFont(name, size, bold=bold)
    return _SYS_FONT_CACHE[key]



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
) -> pygame.Rect:
    """Draw a semi-transparent floating UI box with vertical text lines."""
    if not lines:
        return pygame.Rect(pos_x, pos_y, 0, 0)

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

    rect = pygame.Rect(actual_x, pos_y, box_w, box_h)
    surface.blit(box_surf, rect.topleft)
    return rect


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
    generation = int(round_time / GENERATION_DURATION) + 1
    left_lines: list[tuple[str, tuple[int, int, int]]] = [
        ("CURRENT STATS", (0, 220, 230)),
        (f"Sim Time    : {round_time:.1f}s", HUD_TEXT_COLOR),
        (f"Generation  : {generation}", HUD_TEXT_COLOR),
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


def draw_commands_box(
    surface: pygame.Surface,
    font: pygame.font.Font,
    pos_x: int | None = None,
    pos_y: int = 12,
    align_right: bool = True,
) -> pygame.Rect:
    """Draw commands (keys) one per row in the empty space top right of Window B using a smaller font."""
    if pos_x is None:
        pos_x = surface.get_width() - 14
    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("COMMANDS (KEYS)", (0, 220, 230)),
        ("[SPACE] Pause / Resume", (210, 215, 225)),
        ("[U]     Toggle Ultra Mode", (210, 215, 225)),
        ("[S]     Toggle Sensors", (210, 215, 225)),
        ("[W]     Toggle Stats Panel", (210, 215, 225)),
        ("[F]     Plot Curves (Term)", (210, 215, 225)),
        ("[P]     Save Snapshot", (210, 215, 225)),
        ("[1-8]   Speed Multiplier", (210, 215, 225)),
        ("[ESC]   Exit Simulation", (210, 215, 225)),
    ]
    return _draw_floating_box(surface, font, lines, pos_x=pos_x, pos_y=pos_y, align_right=align_right, border_color=(0, 173, 181, 150))


def draw_fps_box(surface: pygame.Surface, font: pygame.font.Font, fps: float, width: int, ultra_mode: bool = False) -> None:
    """Draw a small FPS box in the top right of Window A."""
    mode_str = "ULTRA" if ultra_mode else "NORMAL"
    color = (255, 100, 100) if ultra_mode else (150, 255, 180)
    lines = [
        (f"FPS: {fps:.0f} [{mode_str}]", color),
    ]
    _draw_floating_box(surface, font, lines, pos_x=width - 15, pos_y=15, align_right=True, border_color=color)


def get_species_metrics(world: Any, cls: type) -> dict[str, Any]:
    """Compute living/dead population and all-time best/average vitals & fitness for a species."""
    species_name = getattr(cls, "species_name", cls.__name__)
    living = world.creatures.get(cls, [])
    dead = world.dead_creatures.get(cls, [])
    max_pop = getattr(cls, "max_population", 100)
    alive_count = len(living)
    dead_count = len(dead)
    all_time_count = getattr(world, "all_time_counts", {}).get(cls, alive_count + dead_count)
    all_creatures = living + dead

    # All-time best scores combining historical SpeciesStats records + currently living max
    best_fitness = max(SpeciesStats.max_fitness.get(species_name, 0.0), max((c.compute_fitness() for c in living), default=0.0))
    best_food = max(SpeciesStats.max_foodeaten.get(species_name, 0), max((getattr(c, "food_eaten", 0) for c in living), default=0))
    best_computed_food = max(SpeciesStats.max_computed_food.get(species_name, 0.0), max((getattr(c, "computed_food_eaten", 0.0) for c in living), default=0.0))
    best_enemies = max(SpeciesStats.max_enemies_touched.get(species_name, 0), max((getattr(c, "enemies_touched", 0) for c in living), default=0))
    best_computed_enemies = max(SpeciesStats.max_computed_enemies.get(species_name, 0.0), max((getattr(c, "computed_enemies_touched", 0.0) for c in living), default=0.0))
    best_lifetime = max(SpeciesStats.max_lifetime.get(species_name, 0.0), max((getattr(c, "survival_time", 0.0) for c in living), default=0.0))

    # All-time average scores combining dead accumulators + currently living
    total_dead_count = SpeciesStats.total_dead_count.get(species_name, 0)
    total_count = total_dead_count + len(living)

    if total_count > 0:
        avg_fitness = (SpeciesStats.sum_dead_fitness.get(species_name, 0.0) + sum(c.compute_fitness() for c in living)) / total_count
        avg_food = (SpeciesStats.sum_dead_food.get(species_name, 0.0) + sum(getattr(c, "food_eaten", 0) for c in living)) / total_count
        avg_computed_food = (SpeciesStats.sum_dead_computed_food.get(species_name, 0.0) + sum(getattr(c, "computed_food_eaten", 0.0) for c in living)) / total_count
        avg_enemies = (SpeciesStats.sum_dead_enemies.get(species_name, 0.0) + sum(getattr(c, "enemies_touched", 0) for c in living)) / total_count
        avg_computed_enemies = (SpeciesStats.sum_dead_computed_enemies.get(species_name, 0.0) + sum(getattr(c, "computed_enemies_touched", 0.0) for c in living)) / total_count
        avg_lifetime = (SpeciesStats.sum_dead_lifetime.get(species_name, 0.0) + sum(getattr(c, "survival_time", 0.0) for c in living)) / total_count
    elif all_creatures:
        avg_fitness = sum(c.compute_fitness() for c in all_creatures) / len(all_creatures)
        avg_food = sum(getattr(c, "food_eaten", 0) for c in all_creatures) / len(all_creatures)
        avg_computed_food = sum(getattr(c, "computed_food_eaten", 0.0) for c in all_creatures) / len(all_creatures)
        avg_enemies = sum(getattr(c, "enemies_touched", 0) for c in all_creatures) / len(all_creatures)
        avg_computed_enemies = sum(getattr(c, "computed_enemies_touched", 0.0) for c in all_creatures) / len(all_creatures)
        avg_lifetime = sum(getattr(c, "survival_time", 0.0) for c in all_creatures) / len(all_creatures)
    else:
        avg_fitness = 0.0
        avg_food = 0.0
        avg_computed_food = 0.0
        avg_enemies = 0.0
        avg_computed_enemies = 0.0
        avg_lifetime = 0.0

    bounds_table = getattr(cls, "metrics", {})
    metric_bounds: dict[str, tuple[float, float]] = {}
    peak_table = SpeciesStats.max_metrics.get(species_name, {})
    for k, bound_val in bounds_table.items():
        peak_val = peak_table.get(k, 0.0)
        curr_max = max((float(getattr(c, k, 0.0)) for c in all_creatures), default=0.0)
        metric_bounds[k] = (max(peak_val, curr_max), float(bound_val))

    return {
        "alive": alive_count,
        "max_pop": max_pop,
        "dead": dead_count,
        "all_time_count": all_time_count,
        "metric_bounds": metric_bounds,
        "best_fitness": float(best_fitness),
        "avg_fitness": float(avg_fitness),
        "best_food": int(best_food),
        "best_computed_food": float(best_computed_food),
        "avg_food": float(avg_food),
        "avg_computed_food": float(avg_computed_food),
        "best_enemies": int(best_enemies),
        "best_computed_enemies": float(best_computed_enemies),
        "avg_enemies": float(avg_enemies),
        "avg_computed_enemies": float(avg_computed_enemies),
        "best_lifetime": float(best_lifetime),
        "avg_lifetime": float(avg_lifetime),
    }


class LiveFitnessChart:
    """O(1) persistent static Surface live chart with 4 distinct series."""

    def __init__(self, width: int = 640, height: int = 300) -> None:
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.history: list[tuple[float, float, float, float, float]] = []
        self.last_update_time: float = -999.0
        self.max_time: float = 300.0
        self.min_fitness: float = 1.0
        self.max_fitness: float = 100.0
        self.margin_left = 55
        self.margin_bottom = 35
        self.margin_top = 35
        self.margin_right = 20
        self.plot_w = self.width - self.margin_left - self.margin_right
        self.plot_h = self.height - self.margin_top - self.margin_bottom
        self.font = get_cached_sysfont("Consolas, Courier, monospace", 11, bold=True)
        self._init_surface()

    def reset(self) -> None:
        """Clear chart history and reset persistent surface."""
        self.history.clear()
        self.last_update_time = -999.0
        self.max_time = 300.0
        self.min_fitness = 1.0
        self.max_fitness = 100.0
        self._init_surface()

    def resize(self, width: int, height: int) -> None:
        """Resize chart surface and recompute layout margins."""
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.plot_w = max(10, self.width - self.margin_left - self.margin_right)
        self.plot_h = max(10, self.height - self.margin_top - self.margin_bottom)
        self._redraw_all()

    def _init_surface(self) -> None:
        self.surface.fill((16, 20, 26))
        # Draw border
        pygame.draw.rect(self.surface, (70, 80, 95), self.surface.get_rect(), 1)
        # Draw plot area background
        plot_rect = pygame.Rect(self.margin_left, self.margin_top, self.plot_w, self.plot_h)
        pygame.draw.rect(self.surface, (22, 28, 36), plot_rect)
        pygame.draw.rect(self.surface, (90, 105, 125), plot_rect, 1)

        # Draw title & legend
        title_img = self.font.render("LIVE FITNESS CHART (Log Scale, Every 10s)", True, (230, 235, 245))
        self.surface.blit(title_img, (12, 10))

        legends = [
            ("ABest", (50, 240, 90)),
            ("AAvg", (20, 150, 55)),
            ("SBest", (255, 80, 80)),
            ("SAvg", (180, 35, 35)),
        ]
        lx = self.width - 340
        for label, color in legends:
            pygame.draw.line(self.surface, color, (lx, 16), (lx + 15, 16), 2)
            lbl_img = self.font.render(label, True, (210, 215, 225))
            self.surface.blit(lbl_img, (lx + 20, 10))
            lx += 80

        self._draw_grid_labels()

    def _draw_grid_labels(self) -> None:
        # Draw horizontal grid lines and Y-axis labels using log10 scale
        log_min = math.log10(self.min_fitness)
        log_max = math.log10(max(self.min_fitness * 10.0, self.max_fitness))
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            log_val = log_min + frac * (log_max - log_min)
            y_val = 10.0 ** log_val
            py = int(self.margin_top + self.plot_h * (1.0 - frac))
            if 0 < frac < 1.0:
                for x_dash in range(self.margin_left, self.margin_left + self.plot_w, 8):
                    pygame.draw.line(self.surface, (45, 55, 68), (x_dash, py), (min(self.margin_left + self.plot_w, x_dash + 4), py), 1)
            label_str = f"{y_val:.1f}" if y_val < 10 else f"{y_val:.0f}"
            lbl = self.font.render(label_str, True, (150, 160, 175))
            self.surface.blit(lbl, (self.margin_left - lbl.get_width() - 6, py - 6))

        # Draw X-axis labels
        for frac in [0.0, 0.5, 1.0]:
            x_val = self.max_time * frac
            px = int(self.margin_left + self.plot_w * frac)
            lbl = self.font.render(f"{x_val:.0f}s", True, (150, 160, 175))
            self.surface.blit(lbl, (px - lbl.get_width() // 2, self.margin_top + self.plot_h + 8))

    def _to_pixel(self, sim_time: float, value: float) -> tuple[int, int]:
        x_frac = max(0.0, min(1.0, sim_time / max(1e-5, self.max_time)))
        val_clamped = max(self.min_fitness, float(value))
        log_min = math.log10(self.min_fitness)
        log_max = math.log10(max(self.min_fitness * 10.0, self.max_fitness))
        y_frac = (math.log10(val_clamped) - log_min) / max(1e-5, (log_max - log_min))
        y_frac = max(0.0, min(1.0, y_frac))
        px = int(self.margin_left + self.plot_w * x_frac)
        py = int(self.margin_top + self.plot_h * (1.0 - y_frac))
        return (px, py)

    def _draw_segment(self, pt1: tuple[float, float, float, float, float], pt2: tuple[float, float, float, float, float]) -> None:
        t1, ab1, aa1, sb1, sa1 = pt1
        t2, ab2, aa2, sb2, sa2 = pt2
        colors = [(50, 240, 90), (20, 150, 55), (255, 80, 80), (180, 35, 35)]
        vals1 = [ab1, aa1, sb1, sa1]
        vals2 = [ab2, aa2, sb2, sa2]
        for c, v1, v2 in zip(colors, vals1, vals2):
            p1 = self._to_pixel(t1, v1)
            p2 = self._to_pixel(t2, v2)
            pygame.draw.line(self.surface, c, p1, p2, 2)

    def _redraw_all(self) -> None:
        self._init_surface()
        for i in range(1, len(self.history)):
            self._draw_segment(self.history[i - 1], self.history[i])

    def update(self, round_time: float, ant_best: float, ant_avg: float, spider_best: float, spider_avg: float) -> None:
        """Record point every 10 seconds of simulation time and incrementally update persistent surface."""
        if self.last_update_time < 0 or (round_time - self.last_update_time) >= 10.0:
            point = (round_time, ant_best, ant_avg, spider_best, spider_avg)
            self.history.append(point)
            self.last_update_time = round_time

            needs_redraw = False
            if round_time > self.max_time * 0.98:
                self.max_time = max(self.max_time * 2.0, round_time * 1.2)
                needs_redraw = True
            max_val = max(ant_best, ant_avg, spider_best, spider_avg)
            if max_val > self.max_fitness * 0.98:
                self.max_fitness = max(self.max_fitness * 1.5, max_val * 1.2)
                needs_redraw = True

            if needs_redraw or len(self.history) == 1:
                self._redraw_all()
            else:
                self._draw_segment(self.history[-2], self.history[-1])


_commands_font: pygame.font.Font | None = None
_stats_font: pygame.font.Font | None = None


def draw_window_b_panel(
    surface: pygame.Surface,
    font: pygame.font.Font,
    world: Any,
    simulation: Any,
    chart: LiveFitnessChart,
) -> None:
    """Render comprehensive stats panel and live chart on Window B surface."""
    surface.fill((16, 20, 26))

    global _stats_font
    if _stats_font is None:
        _stats_font = get_cached_sysfont("Consolas, Courier, monospace", 14, bold=True)

    # Import species classes dynamically to compute metrics if needed
    from species.ant import Ant
    from species.spider import Spider

    ant_m = get_species_metrics(world, Ant)
    spider_m = get_species_metrics(world, Spider)

    # Update live chart every interval
    chart.update(
        world.round_time,
        ant_m["best_fitness"],
        ant_m["avg_fitness"],
        spider_m["best_fitness"],
        spider_m["avg_fitness"],
    )

    ultra_mode = getattr(simulation, "ultra_mode", False)
    sim_speed = getattr(simulation, "speed_multiplier", 1.0)
    generation = int(world.round_time / GENERATION_DURATION) + 1

    top_lines: list[tuple[str, tuple[int, int, int]]] = [
        ("=== ECOSYSTEM STATS & EVOLUTION MONITOR ===", (0, 220, 230)),
        (f"Time: {world.round_time:.1f}s | Gen: {generation} | Speed: {sim_speed}x | Ultra [U]: {'ON' if ultra_mode else 'OFF'}", HUD_TEXT_COLOR),
        ("---------------------------------------------------------", (60, 75, 90)),
    ]

    active_species = getattr(world, "active_species", list(world.creatures.keys()))
    for cls in active_species:
        species_name = getattr(cls, "species_name", cls.__name__)
        m = get_species_metrics(world, cls)
        color = HUD_ACCENT_COLOR if species_name == "Ant" else (SPIDER_ACCENT_COLOR if species_name == "Spider" else (200, 220, 255))

        best_food_s = f"{m['best_food']:d} ({m['best_computed_food']:.1f})"
        avg_food_s = f"{m['avg_food']:.1f} ({m['avg_computed_food']:.1f})"
        best_enemies_s = f"{m['best_enemies']:d} ({m['best_computed_enemies']:.1f})"
        avg_enemies_s = f"{m['avg_enemies']:.1f} ({m['avg_computed_enemies']:.1f})"
        best_life_s = f"{m['best_lifetime']:.1f}s"
        avg_life_s = f"{m['avg_lifetime']:.1f}s"

        top_lines.append((f"[{species_name.upper()} STATS]      Alive: {m['alive']}/{m['max_pop']} (All-Time: {m['all_time_count']})", color))
        top_lines.append((f"  {'Fitness':<14} Best: {m['best_fitness']:>12.1f} | Avg: {m['avg_fitness']:>12.1f}", HUD_TEXT_COLOR))
        top_lines.append((f"  {'Food Eaten':<14} Best: {best_food_s:>12} | Avg: {avg_food_s:>12}", HUD_TEXT_COLOR))
        top_lines.append((f"  {'Enemies Touch':<14} Best: {best_enemies_s:>12} | Avg: {avg_enemies_s:>12}", HUD_TEXT_COLOR))
        top_lines.append((f"  {'Lifetime':<14} Best: {best_life_s:>12} | Avg: {avg_life_s:>12}", HUD_TEXT_COLOR))
        top_lines.append(("---------------------------------------------------------", (60, 75, 90)))

    stats_rect = _draw_floating_box(surface, _stats_font, top_lines, pos_x=14, pos_y=12, align_right=False, border_color=(0, 173, 181, 180))

    # Draw commands box at the top right of Window B (Stats Window)
    global _commands_font
    if _commands_font is None:
        _commands_font = get_cached_sysfont("Consolas, Courier, monospace", 11, bold=True)
    draw_commands_box(surface, _commands_font, pos_x=surface.get_width() - 14, pos_y=12, align_right=True)

    # Position graph right below the stats box to fill all remaining blank space down to bottom margin
    chart_y = stats_rect.bottom + 14 if stats_rect else 360
    chart_h = max(100, surface.get_height() - chart_y - 14)
    chart_w = max(100, surface.get_width() - 28)
    if chart.width != chart_w or chart.height != chart_h:
        chart.resize(chart_w, chart_h)

    # Blit live chart onto Window B
    chart_rect = chart.surface.get_rect(topleft=(14, chart_y))
    surface.blit(chart.surface, chart_rect)
