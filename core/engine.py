"""
engine.py — Main Simulation Loop and Pygame Event Processing
============================================================

Contains the main application loop, handling user input events, stepping
the simulation orchestrator, and controlling visual output rendering.
"""

from __future__ import annotations

import argparse
import sys
import pygame

from core.constants import FPS
from core.simulation import Simulation
from rendering.renderer import Renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Ecosystem Simulation Engine")
    parser.add_argument(
        "-p", "--path", "--load",
        dest="load_path",
        type=str,
        default=None,
        help="Path to saved genomes JSON file to start the simulation from.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the simulation application."""
    args = parse_args()
    simulation = Simulation(load_path=args.load_path)
    renderer = Renderer()

    base_dt: float = 1.0 / float(FPS)
    paused: bool = False

    while simulation.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulation.running = False
                break
            elif event.type == getattr(pygame, "WINDOWCLOSE", 32787):
                win_attr = getattr(event, "window", None)
                win_id = getattr(win_attr, "id", win_attr)
                if win_id == renderer.win_a.id:
                    simulation.running = False
                    break
                elif win_id == renderer.win_b.id:
                    renderer.hide_window_b()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    simulation.running = False
                    break
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_t:
                    simulation.ultra_mode = not getattr(simulation, "ultra_mode", False)
                elif event.key == pygame.K_w or event.key == pygame.K_b:
                    renderer.toggle_window_b()
                elif event.key == pygame.K_r:
                    # Reset simulation world and spawn initial populations or reload save
                    simulation.reset()
                    renderer.chart.reset()
                elif event.key == pygame.K_p:
                    # Save top 10% brains to JSON file
                    simulation.save_top_brains()
                elif event.key == pygame.K_s:
                    renderer.toggle_sensors()
                elif event.key == pygame.K_f or event.key == pygame.K_c:
                    # Print terminal fitness curves immediately
                    simulation.plot_fitness_curves()
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    idx = event.key - pygame.K_1
                    simulation.set_speed(idx)
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_UP:
                    simulation.set_speed(simulation.speed_idx + 1)
                elif event.key == pygame.K_LEFT or event.key == pygame.K_DOWN:
                    simulation.set_speed(simulation.speed_idx - 1)

        if not paused and simulation.running:
            dt = base_dt * simulation.speed_multiplier
            max_step = 0.1
            remaining = dt
            while remaining > 0:
                if not simulation.running:
                    break
                step_dt = min(remaining, max_step)
                simulation.step(step_dt)
                remaining -= step_dt
            simulation.ticks_since_render += 1

        if simulation.running:
            if getattr(simulation, "ultra_mode", False):
                if simulation.ticks_since_render >= 10:
                    renderer.render(simulation.world, simulation)
                    simulation.ticks_since_render = 0
            else:
                renderer.render(simulation.world, simulation)
                simulation.ticks_since_render = 0

    # Always plot final fitness curves in terminal upon exiting
    simulation.plot_fitness_curves()
    try:
        renderer.win_a.destroy()
        renderer.win_b.destroy()
    except Exception:
        pass
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
