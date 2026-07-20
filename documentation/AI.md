# AI.md — Development Guidelines (Ants vs Spiders)

Rules for anyone (human or agent) touching this codebase. These are conventions,
not a spec of the current architecture — the architecture itself evolves and
lives in code, not here.

---

## Performance Optimization

- Profile before optimizing. Don't guess — use `python3 -m cProfile -s tottime -m core.engine`
  (see `documentation/useful.md`) and optimize what the numbers actually show.
- Anything that runs per-frame per-creature (sensors, physics, forward pass) must not
  hide an O(n²) Python loop over the full population. Use `SpatialHash` for proximity
  queries instead of scanning every entity.
- Prefer squared-distance comparisons over `sqrt`/`np.linalg.norm` when only comparing
  magnitudes, not computing an actual distance value.
- Scalar Python math (`math.hypot`, manual dot products) is the right call for tiny
  2-element vector ops — NumPy dispatch overhead dominates at that size. Reach for
  vectorized NumPy only when operating on whole arrays/populations at once.
- If you add an O(n²) or worse loop, it needs a one-line justification comment — silence
  implies it was an oversight, not a decision.

## Reusing Functions

- Shared behavior lives in `Creature`/`Entity`. Species subclasses (`Ant`, `Spider`)
  override only species-specific values and `compute_fitness` — never duplicate
  movement, sensing, or lifecycle logic per species.
- `world/physics.py` and `evolution/sensors.py` stay species-agnostic: compare via
  class identity/category, never `isinstance(x, Ant)`.
- Reproduction, mutation, and selection have exactly one implementation:
  `evolution/genetics.py`. A second species needing "slightly different" reproduction
  is a config problem (`SPECIES_CONFIG`), not a new code path.
- Before adding a helper, check `core/utils.py` and the relevant module for an
  existing equivalent.

## Constants Isolation

- No magic numbers inline in logic files. Values belong in `core/constants.py` or
  `species/*_constants.py`.
- Fitness weights, metric bounds, and network dimensions are constants-file concerns,
  not literals inside `ant.py`/`spider.py`/`brain.py` logic.
- `SPECIES_CONFIG` is the single registration point for active species, reproduction
  mode, and NPC status — the update loop never branches on species name directly.

## Comment Relevance

- Comments describe current behavior and *why*, not history. Write as if the project
  shipped this way from day one.
- Never write "fixed X", "changed from Y to Z", "no longer does W", "this instead of
  that" — that's commit-message content, not code content.
- Skip comments that just restate the line below them (`# increment counter` /
  `count += 1`).
- Reserve comments for non-obvious rationale: why squared distance instead of sqrt,
  why static bounds instead of self-relative normalization, why a mutation is skipped
  for NPCs, etc.

## Don't

- Don't hardcode facts about the *evolving* architecture — network topology, genome
  size, fitness formula, metric bounds, reproduction parameters — anywhere outside
  their source of truth, including in this file. These change often and a stale copy
  is worse than no copy. Point to the source instead:
  - Network shape / genome size → `evolution/brain.py`, `core/constants.py` (`NN_*`, `GENOME_SIZE`)
  - Fitness formula → `species/ant.py::compute_fitness`, `species/spider.py::compute_fitness`
  - Metric bounds → `species/ant_constants.py`, `species/spider_constants.py`
  - Reproduction mode / NPC flags → `core/constants.py::SPECIES_CONFIG`
- If a doc or comment needs to reference a current value for context, write
  "see `<file>`" rather than copying the number.

## Architecture Invariants (these don't change)

- `Creature` / `Entity` abstraction boundary stays intact.
- `world/physics.py` and `evolution/sensors.py` never hardcode species types.
- `rendering/renderer.py` is read-only: reads state, never mutates simulation logic.
- `core/simulation.py` is the only place species get registered/activated.