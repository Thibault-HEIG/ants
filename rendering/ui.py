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


def draw_fps_box(surface: pygame.Surface, font: pygame.font.Font, fps: float, width: int, ultra_mode: bool = False) -> None:
    """Draw a small FPS box in the top right of Window A."""
    mode_str = "ULTRA" if ultra_mode else "NORMAL"
    color = (255, 100, 100) if ultra_mode else (150, 255, 180)
    lines = [
        (f"FPS: {fps:.0f} [{mode_str}]", color),
    ]
    _draw_floating_box(surface, font, lines, pos_x=width - 15, pos_y=15, align_right=True, border_color=color)


def get_species_metrics(world: Any, cls: type) -> dict[str, Any]:
    """Compute living/dead population and best/average vitals & fitness for a species."""
    living = world.creatures.get(cls, [])
    dead = world.dead_creatures.get(cls, [])
    max_pop = getattr(cls, "max_population", 100)
    alive_count = len(living)
    dead_count = len(dead)
    all_time_count = getattr(world, "all_time_counts", {}).get(cls, alive_count + dead_count)
    all_creatures = living + dead

    if all_creatures:
        best_fitness = max(c.compute_fitness() for c in all_creatures)
        best_food = max(getattr(c, "food_eaten", 0) for c in all_creatures)
        best_enemies = max(getattr(c, "enemies_touched", 0) for c in all_creatures)
        best_lifetime = max(getattr(c, "survival_time", 0.0) for c in all_creatures)
    else:
        best_fitness = 0.0
        best_food = 0
        best_enemies = 0
        best_lifetime = 0.0

    if living:
        avg_fitness = sum(c.compute_fitness() for c in living) / len(living)
        avg_food = sum(getattr(c, "food_eaten", 0) for c in living) / len(living)
        avg_enemies = sum(getattr(c, "enemies_touched", 0) for c in living) / len(living)
        avg_lifetime = sum(getattr(c, "survival_time", 0.0) for c in living) / len(living)
    elif all_creatures:
        avg_fitness = sum(c.compute_fitness() for c in all_creatures) / len(all_creatures)
        avg_food = sum(getattr(c, "food_eaten", 0) for c in all_creatures) / len(all_creatures)
        avg_enemies = sum(getattr(c, "enemies_touched", 0) for c in all_creatures) / len(all_creatures)
        avg_lifetime = sum(getattr(c, "survival_time", 0.0) for c in all_creatures) / len(all_creatures)
    else:
        avg_fitness = 0.0
        avg_food = 0.0
        avg_enemies = 0.0
        avg_lifetime = 0.0

    species_name = getattr(cls, "species_name", cls.__name__)
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
        "avg_food": float(avg_food),
        "best_enemies": int(best_enemies),
        "avg_enemies": float(avg_enemies),
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
        self.font = pygame.font.SysFont("Consolas, Courier, monospace", 11, bold=True)
        self._init_surface()

    def reset(self) -> None:
        """Clear chart history and reset persistent surface."""
        self.history.clear()
        self.last_update_time = -999.0
        self.max_time = 300.0
        self.min_fitness = 1.0
        self.max_fitness = 100.0
        self._init_surface()

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
            ("Ant Best", (50, 240, 90)),
            ("Ant Avg", (20, 150, 55)),
            ("Spider Best", (255, 80, 80)),
            ("Spider Avg", (180, 35, 35)),
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
        _stats_font = pygame.font.SysFont("Consolas, Courier, monospace", 14, bold=True)

    # Import species classes dynamically to compute metrics
    from species.ant import Ant
    from species.spider import Spider

    ant_m = get_species_metrics(world, Ant)
    spider_m = get_species_metrics(world, Spider)

    # Update live chart every 30s
    chart.update(
        world.round_time,
        ant_m["best_fitness"],
        ant_m["avg_fitness"],
        spider_m["best_fitness"],
        spider_m["avg_fitness"],
    )

    ultra_mode = getattr(simulation, "ultra_mode", False)
    sim_speed = getattr(simulation, "speed_multiplier", 1.0)

    def _format_bound_pair(cur: float, bound: float, is_float: bool = False) -> str:
        b_str = f"{bound:.0f}" if bound % 1 == 0 else f"{bound:.1f}"
        c_str = f"{cur:.1f}" if is_float else f"{int(cur)}"
        return f"{c_str}/{b_str}"

    ab = ant_m.get("metric_bounds", {})
    sb = spider_m.get("metric_bounds", {})

    top_lines: list[tuple[str, tuple[int, int, int]]] = [
        ("=== ECOSYSTEM STATS & EVOLUTION MONITOR ===", (0, 220, 230)),
        (f"Time: {world.round_time:.1f}s | Speed: {sim_speed}x | Ultra Mode [U]: {'ON' if ultra_mode else 'OFF'}", HUD_TEXT_COLOR),
        ("---------------------------------------------------------", (60, 75, 90)),
        (f"[ANT STATS]      Alive: {ant_m['alive']}/{ant_m['max_pop']} (All-Time: {ant_m['all_time_count']})", HUD_ACCENT_COLOR),
        (f"  Fitness        Best: {ant_m['best_fitness']:6.1f} | Avg: {ant_m['avg_fitness']:6.1f}", HUD_TEXT_COLOR),
        (f"  Food Eaten     Best: {ant_m['best_food']:6d} | Avg: {ant_m['avg_food']:6.1f}", HUD_TEXT_COLOR),
        (f"  Enemies Touch  Best: {ant_m['best_enemies']:6d} | Avg: {ant_m['avg_enemies']:6.1f}", HUD_TEXT_COLOR),
        (f"  Lifetime       Best: {ant_m['best_lifetime']:6.1f}s| Avg: {ant_m['avg_lifetime']:6.1f}s", HUD_TEXT_COLOR),
        ("  -- ANT METRIC BOUNDS (Max Current / Expected Bound) --", (120, 230, 240)),
        (f"  Surv: {_format_bound_pair(*ab.get('survival_time', (0, 100)), True):>9} | Food   : {_format_bound_pair(*ab.get('food_eaten', (0, 30))):>7} | Touch: {_format_bound_pair(*ab.get('enemies_touched', (0, 50))):>7}", HUD_TEXT_COLOR),
        (f"  Tiles: {_format_bound_pair(*ab.get('tiles_covered', (0, 300))):>8} | Eat/Nth: {_format_bound_pair(*ab.get('times_eating_for_nothing', (0, 50))):>7} | Atk/Nth: {_format_bound_pair(*ab.get('times_attacking_for_nothing', (0, 50))):>5}", HUD_TEXT_COLOR),
        (f"  Pheromones : {_format_bound_pair(*ab.get('follow_pheromones', (0, 100)), True)}", HUD_TEXT_COLOR),
        ("---------------------------------------------------------", (60, 75, 90)),
        (f"[SPIDER STATS]   Alive: {spider_m['alive']}/{spider_m['max_pop']} (All-Time: {spider_m['all_time_count']})", SPIDER_ACCENT_COLOR),
        (f"  Fitness        Best: {spider_m['best_fitness']:6.1f} | Avg: {spider_m['avg_fitness']:6.1f}", HUD_TEXT_COLOR),
        (f"  Food Eaten     Best: {spider_m['best_food']:6d} | Avg: {spider_m['avg_food']:6.1f}", HUD_TEXT_COLOR),
        (f"  Enemies Touch  Best: {spider_m['best_enemies']:6d} | Avg: {spider_m['avg_enemies']:6.1f}", HUD_TEXT_COLOR),
        (f"  Lifetime       Best: {spider_m['best_lifetime']:6.1f}s| Avg: {spider_m['avg_lifetime']:6.1f}s", HUD_TEXT_COLOR),
        ("  -- SPIDER METRIC BOUNDS (Max Current / Expected Bound) --", (255, 150, 150)),
        (f"  Surv: {_format_bound_pair(*sb.get('survival_time', (0, 150)), True):>9} | Food   : {_format_bound_pair(*sb.get('food_eaten', (0, 30))):>7} | Touch: {_format_bound_pair(*sb.get('enemies_touched', (0, 50))):>7}", HUD_TEXT_COLOR),
        (f"  Tiles: {_format_bound_pair(*sb.get('tiles_covered', (0, 300))):>8} | Eat/Nth: {_format_bound_pair(*sb.get('times_eating_for_nothing', (0, 50))):>7} | Atk/Nth: {_format_bound_pair(*sb.get('times_attacking_for_nothing', (0, 50))):>5}", HUD_TEXT_COLOR),
        ("---------------------------------------------------------", (60, 75, 90)),
        ("[KEYS] [U] Ultra Mode  [SPACE] Pause  [R] Reset  [P] Save", (210, 215, 225)),
        ("       [S] Sensors     [F] Plot     [1-8] Spd  [W] WinB", (210, 215, 225)),
    ]

    _draw_floating_box(surface, _stats_font, top_lines, pos_x=14, pos_y=12, align_right=False, border_color=(0, 173, 181, 180))

    # Blit live chart onto bottom of Window B
    chart_rect = chart.surface.get_rect(topleft=(14, 495))
    surface.blit(chart.surface, chart_rect)
