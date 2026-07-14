"""
genetics.py — Evolutionary Algorithm and Breeding
=================================================

Implements truncation selection, Gaussian mutation, and continuous fitness-based
breeding for any creature species without hardcoded type dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.constants import CONTINUOUS_SELECTION_FRACTION, CONTINUOUS_MUTATION_RATE, CONTINUOUS_MUTATION_STRENGTH

Creature = Any


def select_parents(
    creatures: list[Creature],
    top_fraction: float = CONTINUOUS_SELECTION_FRACTION,
) -> list[Creature]:
    """Select the fittest creatures as parents for reproduction using truncation selection."""
    if not creatures:
        return []

    scored = [(c.compute_fitness(), c) for c in creatures]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    parent_count = max(min(len(scored), 3), int(len(scored) * top_fraction))
    parents = [c for _, c in scored[:parent_count]]
    return parents


def mutate(
    genome: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = CONTINUOUS_MUTATION_RATE,
    mutation_strength: float = CONTINUOUS_MUTATION_STRENGTH,
) -> np.ndarray:
    """Create a mutated copy of a genome by adding Gaussian noise to genes."""
    child = genome.copy()

    if mutation_rate >= 1.0:
        child += rng.normal(0.0, mutation_strength, size=child.shape)
    else:
        mask = rng.random(size=child.shape) < mutation_rate
        noise = rng.normal(0.0, mutation_strength, size=child.shape)
        child += mask * noise

    return child


def create_offspring_batch(
    creatures: list[Creature],
    population_size: int,
    rng: np.random.Generator,
    npc: bool = False,
) -> list[np.ndarray]:
    """Produce genomes for a population batch from top performing ancestors (unmutated if npc is True)."""
    parents = select_parents(creatures)

    if not parents:
        from evolution.brain import Brain
        return [Brain(rng).get_genome() for _ in range(population_size)]

    children: list[np.ndarray] = []
    best_parent = parents[0]

    for i in range(population_size):
        if npc or (i % 2 == 0):
            child_genome = best_parent.genome.copy()
        else:
            parent = parents[int(rng.integers(len(parents)))]
            child_genome = mutate(parent.genome, rng)

        children.append(child_genome)
    return children
