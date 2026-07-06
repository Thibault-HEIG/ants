"""
engine.py — Main Simulation Loop and Pygame Event Processing
============================================================

Contains the main application loop, handling user input events, stepping
the simulation orchestrator, and controlling visual output rendering.
"""

from __future__ import annotations

import sys
import pygame

from core.constants import FPS
from core.simulation import Simulation
from rendering.renderer import Renderer


def main() -> None:
    """Entry point for the simulation application."""
    simulation = Simulation()
    renderer = Renderer()

    base_dt: float = 1.0 / float(FPS)
    paused: bool = False

    while simulation.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulation.running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    simulation.running = False
                    break
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Reset current generation
                    simulation.world.reset_with_genomes({})
                elif event.key == pygame.K_s:
                    renderer.toggle_sensors()
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    idx = event.key - pygame.K_1
                    simulation.set_speed(idx)
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_UP:
                    simulation.set_speed(simulation.speed_idx + 1)
                elif event.key == pygame.K_LEFT or event.key == pygame.K_DOWN:
                    simulation.set_speed(simulation.speed_idx - 1)

        if not paused and simulation.running:
            dt = base_dt * simulation.speed_multiplier
            # Cap maximum single-step dt to prevent tunneling at very high speeds
            max_step = 0.1
            remaining = dt
            while remaining > 0:
                step_dt = min(remaining, max_step)
                simulation.step(step_dt)
                remaining -= step_dt

        if simulation.running:
            renderer.render(simulation.world, simulation)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
