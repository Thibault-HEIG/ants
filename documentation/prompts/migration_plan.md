# Migration Prompt — Ants vs Spiders: Headless + Web Architecture

## Context
Follow `documentation/AI.md` guidelines

---

## Objective

Migrate the simulation from a Pygame-rendered desktop app to a **headless
Python/NumPy simulation core** that streams state over a **WebSocket** to a
**web-based frontend** (Canvas 2D rendering with pan/zoom, plus a live stats
panel), with a **bidirectional channel for hot-swapping constants at runtime**.

Rendering is no longer in the hot path. Simulation logic is the priority.
Pygame is removed from the simulation loop entirely — it is not kept as a
fallback.

## Non-negotiable constraints (carried over from existing architecture)

- `Creature` / `Entity` abstraction stays intact.
- `world/physics.py` and `evolution/sensors.py` remain species-agnostic — no
  hardcoded class checks.
- `SPECIES_CONFIG` remains the single source of species registration; the
  server/client layer must not require touching `world/world.py`'s update
  loop to add a species.
- Do not touch fitness formulas, weights, or normalization logic — out of
  scope for this migration (that's data-scientist territory, not this task).
- Do not introduce ML/web frameworks beyond what's specified below. Keep the
  dependency list minimal and explicit.

## New dependencies (explicit allowlist)

- `websockets` (Python) for the server.
- No frontend framework. Plain JS + Canvas 2D + a small charting approach
  (inline `<canvas>` chart or Chart.js via CDN — pick one, state which).
- No FastAPI unless you have a specific reason to prefer it over raw
  `websockets` — state the reason if you deviate.

---

## Deliverables, in strict order (produce diffs in this order, one section at a time)

### 1. `core/live_config.py` — hot-swappable constants registry

Design a `LiveConfig` class/module that wraps constants currently in
`core/constants.py`, `species/ant_constants.py`, `species/spider_constants.py`
with a **category tag** per constant:
- `SAFE` — applied instantly, read fresh every tick (e.g. fitness weights,
  damage, mutation strength).
- `DEFERRED` — applied but only affects future events (e.g.
  `GENERATION_DURATION`, `REPRODUCTION_THRESHOLD`).
- `RESTART_REQUIRED` — changes shape (genome size, network layer dims,
  `MAX_ANTS`/`MAX_SPIDERS`, metric bounds). These must be **rejected with an
  explicit error** back to the client unless the client sends an explicit
  `confirm_reset: true` flag, at which point the change is queued and
  applied at the next natural reset point (generation boundary or
  extinction event) — never mid-frame.

Use an observer pattern for constants changes. (Develop that point in the plan)

Do not silently mutate `ANT_METRIC_BOUNDS`/`SPIDER_METRIC_BOUNDS` mid-run —
this corrupts fitness comparability within a single run's chart, a problem
this project has already hit and fixed once (see project history on
normalization). Treat bound changes as `RESTART_REQUIRED`.

Existing code that does `from core.constants import X` needs a compatible
path forward — decide and state whether you're doing a full replace (all
constants now live in `LiveConfig`, `constants.py` becomes defaults only) or
a thinner wrapper. Prefer minimal disruption to existing imports; state your
choice before implementing.

### 2. Decouple simulation from Pygame

- Remove all `pygame` imports from `core/simulation.py`, `world/world.py`,
  `species/*`, `evolution/*`. These currently don't import pygame directly
  (verify), but confirm no indirect coupling exists (e.g. via
  `core.engine`).
- `core/engine.py` currently mixes the Pygame event loop with the sim step
  loop — split this. Produce a new headless runner (e.g. `core/headless.py`
  or repurpose `core/engine.py`) that runs `Simulation.step(dt)` in a loop
  with no rendering, no `pygame.time.Clock`, no window creation.
- Preserve the existing dt-stepping/sub-stepping logic (`max_step = 0.1`)
  and the ultra-mode speed multiplier logic from `SPEED_MULTIPLIERS`.
- `rendering/renderer.py` and `rendering/ui.py` (the Pygame dual-window
  code) should be deleted.

### 3. `core/serialization.py` — snapshot builder, format-agnostic

- One function, `build_snapshot(world, simulation) -> dict`, producing a
  plain Python dict/list structure: creature positions, direction, health,
  species, is_eating/is_attacking flags, food items, pheromone grid
  (consider downsampling — full grid every tick at high creature counts is
  wasteful), fitness stats per species.
- A separate `encode(snapshot: dict) -> str | bytes` function. **Start with
  `json.dumps`.** Structure this as a strategy/swap point (e.g. a module-level
  `ENCODER` variable or simple function reference) so swapping to
  `msgpack.packb` later is a one-line change, not a rewrite. Don't
  over-engineer this into a plugin system — a single swappable function is
  enough.
- Two snapshot tiers:
  - **Full** (used at ≤64x): includes per-creature spatial data.
  - **Aggregate-only** (used at 256x / ultra mode): fitness curves, counts,
    generation number — no per-creature data. Build this as a *separate,
    cheaper* function, not a filtered version of the full snapshot — the
    goal is to skip the expensive per-creature serialization work entirely
    at ultra speed, not just skip sending it.

### 4. WebSocket server

- `server.py` (or `core/server.py`) using the `websockets` library.
- Runs the headless simulation loop and the WebSocket server concurrently
  (use `asyncio` — the sim step loop and the broadcast loop should be
  separate tasks so broadcast rate is decoupled from simulation step rate,
  matching the existing `ticks_since_render` pattern conceptually).
- Broadcast full snapshots at a throttled rate (e.g. every N sim-seconds or
  every N ticks — pick one and make it configurable via `LiveConfig`), not
  every physics step.
- Handle incoming client messages:
  - `{"type": "set_constant", "key": ..., "value": ...}` → routed through
    `LiveConfig`, respecting the SAFE/DEFERRED/RESTART_REQUIRED handling
    from step 1.
  - `{"type": "set_speed", "value": ...}` → replaces the keyboard
    `set_speed` calls from `core/engine.py`.
  - Any pause/resume equivalent to the old `K_SPACE` handling.
- On `RESTART_REQUIRED` rejection, send back a structured error the client can display, don't just drop the message silently.

### 5. Web client

- Single-page app, plain HTML/JS, two full-width panels (simulation on top of stats panel):
  **Simulation view**: `<canvas>`, connects to the WebSocket, renders
  creatures/food/pheromones from the latest full snapshot. Implement pan
  (click-drag) and zoom (scroll wheel) via a camera transform
  (`worldToScreen(pos, zoom, panOffset)`) applied before every draw call —
  don't bake zoom into the snapshot data, keep it client-side only.
  The view must render the exact same elements as the current Pygame window. Use width: 100% and height: auto.
  At ultra-mode activated (aggregate-only snapshots), the sim view should show a clear
  "ultra mode — rendering disabled" state rather than trying to render
  stale/absent creature data.

  **Stats panel**: live fitness chart (best/avg per species, matching the
  existing `LiveFitnessChart` four-series design), current constants with
  editable fields wired to `set_constant` messages, generation/population
  counts. Use all the current WindowB stats + the terminal stats printing at `Key_f`. Nothing change in this section with ultra-mode.
  
  **Commands panel** : Create an action list in the bottom of the page matching the current command list
  - Keep ↑, ↓, space, S, U, P
  - Remove F and Escape

---

## Performance and Optimization
The project was built brick after brick. It might need a slight architecture change to maintain great performance.
- Run performance tests for rendering and computation to be able to scale.
- Look for dead codes or missed reused methods to get a cleaner architecture.


## Explicit non-goals for this pass (do not implement, just don't block them)
- Binary/msgpack encoding itself — only the swap *point*, per step 3.
- Save/load genome JSON flow (`save_top_brains`/`load_from_save`) — keep
  these working via the headless runner, but don't build comprehensive UI for them
  yet, just a quick button.

## Output format requested
#### Before coding, in the plan
- Complete Stats panel description with every single thing that will be displayed
- Clear implementation plan

#### With code
For each of the 5 deliverables above, produce a short (5-10 lines) summary for each. Don't stop between deliverables unless you have an urgent question. Flag anywhere the migration conflicts with an existing architecture rule instead of silently working around it.