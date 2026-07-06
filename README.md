# 🐜 Ants vs Spiders — Evolution War

A real-time co-evolutionary simulation where **ants** and **spiders** each evolve neural-network brains through natural selection. No behaviour is hand-coded — all intelligence emerges from evolution alone.

![Pygame](https://img.shields.io/badge/Pygame-2D%20Rendering-green) ![NumPy](https://img.shields.io/badge/NumPy-Neural%20Networks-blue) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow)

---

## Quick Start

```bash
# Install dependencies
pip install numpy pygame

# Run the simulation
python -m ant_simulator
```

### Controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Increase / decrease simulation speed (0.5×–8×) |
| `Space` | Pause / unpause |
| `R` | Restart simulation |
| `Escape` | Quit |

---

## How It Works

### The Core Loop

Each creature (ant or spider) runs this cycle **every frame**:

```
Sense → Think → Act → Evolve
```

1. **Sense** — Vision rays and omnidirectional sensors detect food, enemies, allies, and walls.
2. **Think** — A neural network processes 22 sensor inputs and outputs `[turn, speed]`.
3. **Act** — The creature moves according to the brain's decision.
4. **Evolve** — Creatures that survive long enough reproduce (continuous evolution). Their children inherit mutated copies of the parent's brain weights.

No fitness function drives generational resets — evolution is **continuous**. An ant that stays above 50% HP long enough spawns a mutated child nearby.

### The World

The 1200×800 arena is divided into two zones:

| Zone | Location | Properties |
|------|----------|------------|
| **Danger Zone** (red tint) | Left half | Spiders move at ant speed here. Less food spawns (20%). |
| **Safe Zone** (green tint) | Right half | Spiders are slow. Most food spawns here (80%). |

This asymmetry forces interesting strategic trade-offs: ants must venture into danger for territory, while spiders benefit from hunting in their fast zone.

### Species Comparison

| Attribute | 🐜 Ant | 🕷️ Spider |
|-----------|--------|-----------|
| Population | 100 (cap: 300) | 15 (cap: 100) |
| Health | 100 HP | 500 HP |
| Speed | 150 px/s | 80 px/s (150 in Danger Zone) |
| Damage | 50 | 80 |
| Attack Cost | 20 HP/s | 30 HP/s |
| Sensor Range | 150 px | 250 px |
| Strategy | Swarm, outnumber | Tank, ambush |

Combat uses **Reach vs Hurtbox** mechanics — damage is dealt when a creature actively decides to attack (`is_attacking == True` via its 3rd neural network output) and an enemy's collision hurtbox is within its strike reach. Because reach is larger than body radius, a creature can land a hit before the target is close enough to retaliate without taking any damage. Actively attacking drains stamina (20 HP/s for ants, 30 HP/s for spiders), forcing creatures to evolve disciplined timing rather than swinging blindly in empty space.

---

## Neural Network Architecture

Each creature has an identical `22 → 8 → 3` fully-connected neural network (tanh activations, no ML libraries — pure NumPy).

### 22 Inputs

| # | Input | Category |
|---|-------|----------|
| 0 | `enemy_right` | 👁️ Right ray |
| 1 | `enemy_left` | 👁️ Left ray |
| 2 | `zone` | 🧠 Internal (0 = Danger, 1 = Safe) |
| 3 | `food_right` | 👁️ Right ray |
| 4 | `food_left` | 👁️ Left ray |
| 5 | `hp` | 🧠 Internal (0–1) |
| 6 | `ally_left` | 👁️ Left ray |
| 7 | `ally_right` | 👁️ Right ray |
| 8 | `wall_left` | 👁️ Left ray |
| 9 | `wall_right` | 👁️ Right ray |
| 10 | `food_fwd` | 👁️ Forward ray |
| 11 | `enemy_fwd` | 👁️ Forward ray |
| 12 | `ally_fwd` | 👁️ Forward ray |
| 13 | `wall_fwd` | 👁️ Forward ray |
| 14 | `nearest_food_dist` | 👃 Omnidirectional |
| 15 | `nearest_food_angle` | 👃 Omnidirectional |
| 16 | `nearest_enemy_dist` | 👃 Omnidirectional |
| 17 | `nearest_enemy_angle` | 👃 Omnidirectional |
| 18 | `current_speed` | 🧠 Internal (0–1) |
| 19 | `age` | 🧠 Internal (0–1) |
| 20 | `ally_density` | 🐜 Teamwork (0–1) |
| 21 | `enemy_density` | 🐜 Teamwork (0–1) |

### 3 Outputs

| Output | Range | Meaning |
|--------|-------|---------|
| `turn` | [-1, 1] | Negative = left, positive = right |
| `speed` | [0, 1] | 0 = stop, 1 = full speed |
| `attack` | {0, 1} | 0 = hold fire, 1 = attack (strike reach enabled) |

### Genome

The genome is a flat vector of **211 floats** — every weight and bias concatenated:

```
[weights_input→hidden (176)] [bias_hidden (8)] [weights_hidden→output (24)] [bias_output (3)]
```

---

## Sensor System

Creatures perceive the world through three complementary systems:

### 1. Vision Rays (directional)

Three rays extend from the creature at fixed angles relative to its heading:
- **Left ray**: −50° offset
- **Right ray**: +50° offset
- **Forward ray**: 0° (straight ahead)

Each ray detects the nearest food, enemy, ally, and wall. Distances are normalised: `0.0` = touching, `1.0` = nothing within range.

### 2. Omnidirectional Sensing ("smell")

360° awareness of the nearest food and nearest enemy — returns both distance and relative angle ([-1, 1]). This lets the brain locate targets even outside the vision cone.

### 3. Density Awareness (teamwork)

Counts of nearby allies and enemies within sensor range, normalised by a cap of 10. This enables emergent swarming — an ant can "decide" to fight only when outnumbering the enemy.

---

## Evolution

### Continuous Evolution (primary mechanism)

No generational resets. Creatures reproduce **during the simulation** based on fitness ranking and population pressure:

- **Truncation Selection**: Whenever a species reproduces, the world selects a random parent from the **top N% fittest** living creatures of that species (using `select_parents` from `genetics.py`, which dynamically scales the parent pool while ensuring at least up to 3 parents when numbers are small). A mutated child is spawned near the parent.
- **Dynamic Spawn Rate**: The time interval between spawns scales dynamically based on available population slots:
  `interval = THRESHOLD / (max_population - current_population)`
  
  When a population is small (many available slots), spawning happens rapidly. As the population approaches its cap (300 for ants, 100 for spiders), available slots approach zero, causing the spawn interval to increase dramatically and naturally stabilizing population numbers without exponential blowup.

### Mutation

Every gene is mutated with Gaussian noise (mean=0, σ=0.1). There is no crossover — children are mutated clones of a single parent.

### Fitness Function

Used for ranking and statistics. Fitness is calculated directly from raw performance metrics (survival time, food consumed, and combat engagements). Because ants and spiders have different lifespans and roles, each species uses tailored weighting:

```python
# Ants
fitness = (survival_time / 20) * 10.0 + food_eaten * 30.0 + enemies_touched * 50.0

# Spiders
fitness = (survival_time / 20) * 10.0 + food_eaten * 10.0 + enemies_touched * 50.0
```

> [!NOTE]
> Historical maximum records (`SpeciesStats`) are tracked independently for both ants and spiders across the entire simulation run and displayed in real-time on the HUD in small font.

---

## Project Structure

```
ant/
├── assets/
│   ├── ant.png              # Ant sprite
│   ├── food.png             # Food sprite
│   └── predator.png         # Spider sprite
└── ant_simulator/
    ├── __init__.py           # Package docstring
    ├── __main__.py           # Entry point (python -m ant_simulator)
    ├── main.py               # Pygame loop, input handling
    ├── simulation.py         # Top-level orchestrator (stats, reset)
    ├── world.py              # Entity lifecycle, physics, combat, food spawning
    ├── ant.py                # Ant entity (position, health, brain, sensors)
    ├── predator.py           # Spider entity (same structure as ant)
    ├── brain.py              # Neural network (22→8→3, forward pass, genome encoding)
    ├── sensors.py            # Vision rays, omnidirectional sensing, density counting
    ├── genetics.py           # Selection, mutation, next-generation breeding
    ├── food.py               # Food entity (position, nutrition, consumed flag)
    ├── renderer.py           # Pygame rendering (sprites, health bars, HUD)
    ├── constants.py          # All tunable parameters (single source of truth)
    └── utils.py              # Pure math helpers (clamp, distance, angle)
```

### Module Dependency Graph

```
main.py
 ├── simulation.py
 │    └── world.py
 │         ├── ant.py ──────┐
 │         ├── predator.py ─┤
 │         ├── food.py      ├── brain.py ─── constants.py
 │         └── genetics.py  ├── sensors.py ─ constants.py, utils.py
 │                          └── constants.py
 └── renderer.py ─── constants.py
```

Key design decisions:
- **`renderer.py`** has read-only access to the world — it never modifies state. The simulation runs headless without it.
- **`constants.py`** is the single source of truth for all parameters. No magic numbers in the codebase.
- **`brain.py`** uses no ML libraries — the forward pass is manual NumPy for transparency.

---

## Key Constants

All tunable parameters live in [`constants.py`](ant_simulator/constants.py). The most impactful ones:

| Constant | Value | Impact |
|----------|-------|--------|
| `MUTATION_STRENGTH` | 0.1 | Higher = faster but noisier evolution |
| `HEALTH_DECAY_RATE` | 2.0 HP/s | Forces creatures to eat; too high = die before learning |
| `ANT_DUPLICATION_TIME` | 20s | Lower = faster population growth |
| `MAX_ANTS` / `MAX_SPIDERS` | 300 / 100 | Population caps; too high = slow frames |
| `SENSOR_ANGLE` | 50° | Narrower = more scanning needed, wider = less blind spots |
| `NN_HIDDEN` | 8 | More neurons = more capacity but slower evolution |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.20 | Neural network math, vector operations |
| `pygame` | ≥2.0 | Rendering, input handling, game loop |

No other dependencies. No ML frameworks.

---

## Extending the Simulator

Common extension points:

- **New inputs**: Add fields to `SensorData` in `sensors.py`, update `to_array()`, and increment `NN_INPUTS` in `constants.py`. The brain auto-adapts.
- **New outputs**: Add a 3rd output (e.g. `attack_intent`) by increasing `NN_OUTPUTS` and updating the post-processing in `brain.py`'s `forward()`.
- **New entities**: Follow the `Food` pattern — create a class with `position`, `radius`, and a consumed/alive flag. Register it in `world.py`.
- **Different evolution**: Modify `genetics.py` — add crossover, tournament selection, or speciation.
- **Headless mode**: Instantiate `Simulation` without `Renderer` and call `step()` in a loop. No Pygame needed.
