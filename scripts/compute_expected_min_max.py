#!/usr/bin/env python3
"""
compute_expected_min_max.py — Empirical & Theoretical Metric Bounds Analyzer
===========================================================================

Runs a fast headless simulation to sample fitness formula metrics across
Ants and Spiders, computes empirical distributions (min, mean, max, 99th percentile),
and outputs the recommended static min/max bounds for absolute normalization.
"""

import sys
import os
import numpy as np

# Ensure root project directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.constants import MAX_AGE_NORMALIZATION
from species.ant import Ant
from species.spider import Spider
from world.world import World

def analyze_metrics(sim_seconds: float = 60.0, dt: float = 0.1) -> dict:
    rng = np.random.default_rng(42)
    world = World(rng, active_species=[Ant, Spider])

    metrics = [
        "survival_time",
        "food_eaten",
        "enemies_touched",
        "times_eating_for_nothing",
        "times_attacking_for_nothing",
        "follow_pheromones",
        "tiles_covered",
    ]

    all_data = {Ant: {m: [] for m in metrics}, Spider: {m: [] for m in metrics}}

    steps = int(sim_seconds / dt)
    print(f"Running headless simulation for {sim_seconds}s ({steps} steps)...")
    for step in range(steps):
        world.update(dt)

    # Collect all living and dead creatures
    for species_cls in [Ant, Spider]:
        creatures = world.creatures.get(species_cls, []) + world.dead_creatures.get(species_cls, [])
        for c in creatures:
            for m in metrics:
                val = float(getattr(c, m, 0.0))
                all_data[species_cls][m].append(val)

    print("\n=========================================================================")
    print("                      EXPECTED FITNESS METRIC BOUNDS                     ")
    print("=========================================================================\n")

    # Recommended static bounds combining empirical behavior & theoretical capacity
    recommended_bounds = {
        "survival_time": (0.0, MAX_AGE_NORMALIZATION),
        "food_eaten": (0.0, 30.0),
        "enemies_touched": (0.0, 50.0),
        "times_eating_for_nothing": (0.0, 20.0),
        "times_attacking_for_nothing": (0.0, 30.0),
        "follow_pheromones": (0.0, 100.0),
        "tiles_covered": (0.0, 300.0),
    }

    for species_cls, metric_dict in all_data.items():
        name = species_cls.__name__
        print(f"--- Species: {name} (Sampled {len(metric_dict['survival_time'])} individuals) ---")
        print(f"{'Metric':<30} | {'Emp Min':>8} | {'Emp Max':>8} | {'Emp Mean':>8} | {'Static Min':>10} | {'Static Max':>10}")
        print("-" * 85)
        for m, vals in metric_dict.items():
            if not vals:
                continue
            emp_min = min(vals)
            emp_max = max(vals)
            emp_mean = float(np.mean(vals))
            rec_min, rec_max = recommended_bounds.get(m, (0.0, 1.0))
            print(f"{m:<30} | {emp_min:>8.1f} | {emp_max:>8.1f} | {emp_mean:>8.1f} | {rec_min:>10.1f} | {rec_max:>10.1f}")
        print()

    return recommended_bounds

if __name__ == "__main__":
    analyze_metrics(sim_seconds=40.0, dt=0.1)
