"""
genetics.py — Evolutionary Algorithm
======================================

This module implements the genetic algorithm that drives ant evolution.

**How evolution works in this simulator:**

1. Each ant has a **genome** — a flat vector of 46 floats representing
   every weight and bias in its neural network brain.

2. When all ants in a generation die, we compute a **fitness score** for
   each one based on how long it survived, how much food it ate, and
   how many predator hits it took.

3. **Selection**: We keep the top 10% (the "parents").  This is called
   **truncation selection** — the simplest selection strategy.

4. **Reproduction**: Each child genome is a copy of a random parent
   with small random changes (mutations) applied.

5. **Mutation**: We add Gaussian noise to every weight.  This is the
   *only* source of variation — there is no crossover (yet).

**Why no crossover?**
  Crossover (mixing genes from two parents) is powerful in biological
  evolution but can be tricky with neural networks because swapping
  arbitrary weight subsets often produces broken networks.  Mutation-only
  evolution is simpler to implement and understand, and works well for
  small genomes like ours (46 floats).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from ant_simulator.constants import (
    SELECTION_FRACTION,
    MUTATION_RATE,
    MUTATION_STRENGTH,
)

# Both Ant and Predator (spider) share the same interface:
#   .compute_fitness() -> float
#   .genome -> np.ndarray  (property)
# We use Any for simplicity since Python doesn't enforce structural typing.
Creature = Any


def select_parents(
    creatures: list[Creature],
    top_fraction: float = SELECTION_FRACTION,
) -> list[Creature]:
    """Select the fittest creatures as parents for the next generation.

    **Truncation selection**: sort all ants by fitness, keep the top N%.

    Why top 10%?
      - Too few parents (e.g. 1%) = low genetic diversity, population
        converges on a single strategy and gets stuck.
      - Too many parents (e.g. 50%) = weak selection pressure, evolution
        is slow because mediocre ants also reproduce.
      - 10% is a pragmatic middle ground.

    Parameters
    ----------
    creatures : list
        All creatures (ants or spiders) from the completed generation.
    top_fraction : float
        Fraction of ants to select (0.1 = top 10%).

    Returns
    -------
    list[Creature]
        Selected parent creature objects.
    """
    if not creatures:
        return []

    # Compute fitness for each creature
    scored = [(c.compute_fitness(), c) for c in creatures]

    # Sort by fitness (highest first)
    scored.sort(key=lambda pair: pair[0], reverse=True)

    # Keep the top fraction (at least 3 parents if population is large enough)
    # Why 3? For small populations (like 10 spiders), top 10% is just 1 parent.
    # If we only select 1 parent, we instantly destroy all genetic diversity and
    # they just clone the same (usually bad) behaviour forever.
    parent_count = max(min(len(scored), 3), int(len(scored) * top_fraction))
    parents = [c for _, c in scored[:parent_count]]

    return parents


def mutate(
    genome: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = MUTATION_RATE,
    mutation_strength: float = MUTATION_STRENGTH,
) -> np.ndarray:
    """Create a mutated copy of a genome.

    **Mutation** is the engine of evolution.  Without it, children would
    be exact copies of their parents and no new behaviours could emerge.

    We add Gaussian noise (mean=0, std=mutation_strength) to each gene.
    The ``mutation_rate`` controls *how many* genes are mutated:
      - 1.0 = mutate every gene (our default — simple and effective)
      - 0.5 = mutate ~half the genes at random

    Parameters
    ----------
    genome : np.ndarray
        The parent genome to mutate (not modified in place).
    rng : np.random.Generator
        Random number generator.
    mutation_rate : float
        Probability of mutating each individual gene.
    mutation_strength : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    np.ndarray
        A new genome with mutations applied.
    """
    child = genome.copy()

    if mutation_rate >= 1.0:
        # Fast path: mutate every gene
        child += rng.normal(0.0, mutation_strength, size=child.shape)
    else:
        # Stochastic path: randomly decide which genes to mutate
        mask = rng.random(size=child.shape) < mutation_rate
        noise = rng.normal(0.0, mutation_strength, size=child.shape)
        child += mask * noise

    return child


def create_next_generation(
    creatures: list[Creature],
    population_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Produce genomes for the entire next generation.

    Pipeline:
      1. Select the fittest parents (top 10%).
      2. Each parent produces an equal number of children (mutated clones).

    If no viable parents exist (e.g. first generation all died instantly),
    we fall back to random genomes.

    Parameters
    ----------
    creatures : list
        All creatures (ants or spiders) from the completed generation.
    population_size : int
        How many creatures to produce for the next generation.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    list[np.ndarray]
        ``population_size`` genomes ready to be installed.
    """
    parents = select_parents(creatures)

    if not parents:
        # No viable parents — start from scratch with random genomes.
        # This should only happen if the simulation is badly configured.
        from ant_simulator.brain import Brain
        return [Brain(rng).get_genome() for _ in range(population_size)]

    # Generate children — each parent produces an equal number of offspring.
    # Why equal shares instead of random selection?
    #   If we picked parents randomly, one lucky genome could dominate the
    #   entire next generation, collapsing genetic diversity (e.g. every
    #   ant inherits the same "spin in circles" behaviour).  Equal shares
    #   guarantee that all top performers contribute to the gene pool.
    children: list[np.ndarray] = []
    children_per_parent = population_size // len(parents)
    remainder = population_size % len(parents)

    for i, parent in enumerate(parents):
        # Each parent gets the same number of children, plus one extra
        # for the first 'remainder' parents to fill the population exactly.
        count = children_per_parent + (1 if i < remainder else 0)
        for _ in range(count):
            child_genome = mutate(parent.genome, rng)
            children.append(child_genome)

    return children
