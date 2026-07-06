"""
main.py — Entry Point
======================

This is the file you run to start the simulation::

    python -m ant_simulator.main

It sets up Pygame, creates the Simulation and Renderer, and runs the
main game loop.  All input handling (keyboard controls) lives here.

Controls:
  ↑ / ↓  — Increase / decrease simulation speed
  Space  — Pause / unpause
  R      — Restart from generation 1
  Escape — Quit
"""

from __future__ import annotations

import sys

import pygame

from ant_simulator.simulation import Simulation
from ant_simulator.renderer import Renderer
from ant_simulator.constants import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    FPS,
    SPEED_MULTIPLIERS,
)


def main() -> None:
    """Initialise Pygame and run the simulation loop."""

    # --- Pygame setup ---
    pygame.init()
    screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    pygame.display.set_caption("Ants vs Spiders — Evolution War")
    clock = pygame.time.Clock()

    # --- Create simulation and renderer ---
    simulation = Simulation()
    renderer = Renderer(screen)

    # --- Speed control ---
    speed_index: int = 1  # index into SPEED_MULTIPLIERS (default = 1x)
    paused: bool = False

    # --- Main loop ---
    running = True
    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    simulation.reset()
                    paused = False

                elif event.key == pygame.K_UP:
                    speed_index = min(
                        speed_index + 1, len(SPEED_MULTIPLIERS) - 1
                    )

                elif event.key == pygame.K_DOWN:
                    speed_index = max(speed_index - 1, 0)

        # --- Simulation step ---
        dt = clock.get_time() / 1000.0  # milliseconds → seconds

        if not paused:
            speed_multiplier = SPEED_MULTIPLIERS[speed_index]
            simulation.step(dt * speed_multiplier)
        else:
            speed_multiplier = SPEED_MULTIPLIERS[speed_index]

        # --- Render ---
        stats = simulation.get_stats()
        renderer.render(
            simulation.world,
            stats,
            speed_multiplier=speed_multiplier,
            paused=paused,
        )
        pygame.display.flip()

        # --- Frame rate cap ---
        clock.tick(FPS)

    # --- Clean shutdown ---
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
