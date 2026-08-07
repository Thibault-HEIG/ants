# Training Tab — Per-Ability Loss/Success Tracking

## Context
Follow `documentation/AI.md` guidelines.
Files affected: `web/index.html` `web/renderer.js`, `web/controls.js`, `server.py`, `core/serialization.py`.

Ask before assuming. Don't make any assumption.

## Goal
Track per-ability and per-specie training signal (loss/success) separately from overall fitness, reusing the existing 1.0s cadence from `Simulation._record_fitness_stat`.

## UI structure
Replace the "Live Analytics" `<h2>` with a tab bar: **Live Analytics** | **Ants Training** | **Spiders Training**. Live Analytics tab content is unchanged.

Training tabs are a 4 zones grid, each with:
- A loss chart and a success chart (Chart.js best+avg like the existing fitness chart)
- Updated exactly like the current population or fitness chart
- 2x2 grid at large viewport
- Per-ability history shouldn't survive the `--load` process-restart flow, it must reset like `history_fitness` already does on `Simulation.reset()`.

### Zone formulas (verify field names against `species/creature.py` / `species/ant.py` before implementing)

| Zone | Loss metric | Success metric |
|---|---|---|
| Eat | best+avg `times_eating_for_nothing` | best+avg `computed_food_eaten` |
| Attack | best+avg `computed_times_attacking_for_nothing` | best+avg `computed_enemies_touched` |
| Pheromone | *(no loss — hide the loss chart entirely for this zone, don't render an empty one)* | best+avg `follow_pheromones` |
| Carry | best+avg `walk_with_object_in_opposite_home_direction` | best+avg `release_at_home_count` **and** best+avg `walk_with_object_in_home_direction` — plot as two separate graphs, don't collapse into one |

## Backend / frontend contract
- Server maintains tracking state (mirrors `IS_PAUSED` global-style pattern already in `server.py`).
- Use the same communication method than for the other graphs.

### Performance
- Confirm the averaging itself is genuinely free: these are already-tracked per-creature attributes (`SpeciesStats.update_metrics` already runs every tick regardless). The real cost is serialization + Chart.js render, not just the client-side render.
- Do not add new per-frame work to `Creature.update()` / `World.update()` for this feature — everything needed is already accumulated on the creature; this is read-and-average at snapshot time only.

Slowing down a bit the front-end is way less important than slowing down back-end.