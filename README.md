# 🐜 Ants vs Spiders — Evolution War

A real-time co-evolutionary simulation where **ants** and **spiders** each evolve neural-network brains through natural selection. No behaviour is hand-coded — all intelligence emerges from evolution alone.

![Pygame](https://img.shields.io/badge/Pygame-2D%20Rendering-green) ![NumPy](https://img.shields.io/badge/NumPy-Neural%20Networks-blue) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow)

---

## Quick Start

```bash
# Install dependencies
pip install numpy pygame

# Run the simulation
python3 -m core.engine
```

### Controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Increase / decrease simulation speed (0.5×–256×) |
| `Space` | Pause / unpause |
| `S` | Toggle vision ray / sensor visualization |
| `U` | Activate ultra performance mode (no rendering) |
| `P` | Save top brains in a JSON file |
| `F` | Print the fitness curve in the terminal |
| `Escape` | Quit |

---

## How It Works

### The Core Loop

Each creature runs this cycle **every frame**:

```
Sense → Think → Act → Evolve
```

1. **Sense** — Eight directional vision rays and density sensors detect food, enemies, allies, and walls.
2. **Think** — A neural network processes 71 sensor inputs and outputs `[turn, speed, attack, eat]`.
3. **Act** — The creature moves and engages in combat according to the brain's decision.
4. **Evolve** — Depending on the species' configured evolution mode, reproduction happens either continuously (threshold-based spawning) or in fixed-length generational episodes (elitism-based replacement). Children inherit mutated copies of the parent's brain weights.

### The World

The 1000×900 arena is divided into two zones:

| Zone | Location | Properties |
|------|----------|------------|
| **Ants' Zone** (green) | Left half | More food spawns (66%). |
| **Spiders' Zone** (red) | Right half | Ants move slower here. Less food spawns (33%). |

### Species Comparison

| Attribute | 🐜 Ant | 🕷️ Spider |
|-----------|--------|-----------|
| Population | cap: 300 | cap: 100 |
| Initial Health | 100 HP | 300 HP |
| Speed | 150 px/s | 80 px/s |
| Damage | 50 | 80 |
| Attack Cost | 20 HP/s | 30 HP/s |
| Sensor Range | 100 px | 150 px |

Combat uses **Reach vs Hurtbox** mechanics — damage is dealt when a creature actively decides to attack (`is_attacking == True` via its 3rd neural network output) and an enemy's collision hurtbox is within its strike reach. Because reach is larger than body radius, a creature can land a hit before the target is close enough to retaliate without taking any damage. Actively attacking drains stamina (20 HP/s for ants, 30 HP/s for spiders), forcing creatures to evolve disciplined timing rather than swinging blindly in empty space.

---

## Neural Network Architecture

Each creature has an identical `71 → 16 → 8 → 4` fully-connected neural network (tanh activations, two hidden layers, no ML libraries — pure NumPy).

### 71 Inputs

**8-Sector Vision (64 inputs)** — 8 directional sensors evenly distributed across the ±50° FOV. Each sensor reports 8 features using inverted convention for distances: `0.0` = nothing detected, `1.0` = touching.

| Indices | Sensor | Angle Offset |
|---------|--------|--------------|
| 0–7 | Sensor 0 | −50.0° (leftmost) |
| 8–15 | Sensor 1 | −35.7° |
| 16–23 | Sensor 2 | −21.4° |
| 24–31 | Sensor 3 | −7.1° |
| 32–39 | Sensor 4 | +7.1° |
| 40–47 | Sensor 5 | +21.4° |
| 48–55 | Sensor 6 | +35.7° |
| 56–63 | Sensor 7 | +50.0° (rightmost) |

Each sensor outputs 8 values:

| Feature | Range | Description |
|---------|-------|-------------|
| `enemy_distance` | [0, 1] | 0 = no enemy, 1 = touching |
| `enemy_is_eating` | {0, 1} | 1 = seen enemy is eating |
| `enemy_is_attacking` | {0, 1} | 1 = seen enemy is attacking |
| `ally_distance` | [0, 1] | 0 = no ally, 1 = touching |
| `ally_is_eating` | {0, 1} | 1 = seen ally is eating |
| `ally_is_attacking` | {0, 1} | 1 = seen ally is attacking |
| `food_distance` | [0, 1] | 0 = no food, 1 = touching |
| `wall_distance` | [0, 1] | 0 = no wall in range, 1 = touching wall |

**Internal State (7 inputs)**

| # | Input | Range | Category |
|---|-------|-------|----------|
| 64 | `hp` | [0, 1] | 🧠 Internal |
| 65 | `zone` | {0, 1} | 🧠 Internal (0 = Spider, 1 = Ant) |
| 66 | `speed` | [0, 1] | 🧠 Internal |
| 67 | `age` | [0, 1] | 🧠 Internal |
| 68 | `ally_density` | [0, 1] | 🐜 Teamwork |
| 69 | `enemy_density` | [0, 1] | 🐜 Teamwork |
| 70 | `has_gained_hp` | {0, 1} | 🧠 Internal (HP increased in last second) |

### 4 Outputs

| Output | Range | Meaning |
|--------|-------|---------|
| `turn` | [-1, 1] | Negative = left, positive = right |
| `speed` | [0, 1] | 0 = stop, 1 = full speed |
| `attack` | {0, 1} | 0 = hold fire, 1 = attack (strike reach enabled) |
| `eat` | {0, 1} | 0 = don't eat, 1 = attempt to eat |

### Genome

The genome is a flat vector of **1468 floats** — every weight and bias concatenated:

```
[w_input→hidden1 (1136)] [b_hidden1 (16)] [w_hidden1→hidden2 (128)] [b_hidden2 (8)] [w_hidden2→output (32)] [b_output (4)]
```

---

## Sensor System

Creatures perceive the world through two complementary systems:

### 1. Directional Vision (8 sensors)

Eight vision rays extend from the creature, evenly distributed across a ±50° field of view (100° total). Each ray detects the nearest enemy, ally, food, and wall along its direction. When an enemy or ally is detected, the sensor also reports whether that creature is currently eating (`is_eating`) or attacking (`is_attacking`).

Distances use **inverted normalisation**: `0.0` = nothing within range, `1.0` = touching. A non-zero value implicitly indicates detection — no separate binary flag is needed.

### 2. Density Awareness (teamwork)

Counts of nearby allies and enemies within sensor range, normalised by a cap of 10. This enables emergent swarming — an ant can "decide" to fight only when outnumbering the enemy.

---

## Evolution

Each species can use one of two reproduction modes, configured per-species in `SPECIES_CONFIG` inside `core/constants.py`:

```python
SPECIES_CONFIG: dict[str, dict] = {
    "Ant":    {"active": True,  "reproduction_mode": "continuous",   "npc": False},
    "Spider": {"active": False, "reproduction_mode": "generational", "npc": False},
}
```

Setting `npc: True` disables mutation entirely — all offspring are exact genome copies (useful for static opponent baselines).

### Continuous Mode

Creatures reproduce **during the simulation** based on fitness ranking and population pressure:

- **Truncation Selection**: Whenever a species reproduces, the world selects a random parent from the **top N% fittest** living creatures of that species (using `select_parents` in `genetics.py`, which dynamically scales the parent pool). A mutated child is spawned at the anthill or spider web. Half of the time the best current specie is simply cloned.
- **Dynamic Spawn Rate**: The time interval between spawns scales dynamically based on available population slots:
  `interval = THRESHOLD / (max_population - current_population)`
  
  When a population is small (many available slots), spawning happens rapidly. As the population approaches its cap, available slots approach zero, causing the spawn interval to increase dramatically and naturally stabilizing population numbers without exponential blowup.
- **Extinction Recovery**: If the population drops to zero, the world repopulates from the fittest dead ancestors (or spawns fresh creatures if no dead pool exists).

### Generational Mode

Creatures live in fixed-length episodes (`GENERATION_DURATION`, default 60 s). At the end of each generation (or immediately if the population hits zero):

1. **Ranking** — All creatures (living + dead during that generation) are ranked by fitness.
2. **Elite Selection** — The top `GENERATIONAL_SELECTION_FRACTION` (default 20%) are selected as elites. Only their **genomes** are carried over — no HP, position, or stats.
3. **Refill** — The full `initial_count` population is respawned at the kingdom:
   - Elite genomes are re-used **unmutated**.
   - Remaining slots are filled with **mutated children** of elites (round-robin parent selection).
4. **Cleanup** — The dead pool is cleared, the generation timer resets, and the generation counter increments. Food, pheromones, lakes, and other species are untouched.

If total extinction occurs with no dead pool (first-generation edge case), fresh random genomes are spawned with boosted mutation (`EXTINCTION_MUTATION_RATE` / `EXTINCTION_MUTATION_STRENGTH`) to force exploration.

### Mutation

Every gene is mutated with Gaussian noise (mean, σ). There is no crossover — children are mutated clones of a single parent. Each mode has its own `MUTATION_RATE` and `MUTATION_STRENGTH` constants (`CONTINUOUS_*` / `GENERATIONAL_*`).

### Fitness Function (to optimize)

Used for ranking and parent selection. Fitness is calculated directly from raw performance metrics (survival time, food consumed, and combat engagements):

```python
# Ants
fitness = survival_time + food_eaten + enemies_touched + times_eating_for_nothing + times_attacking_for_nothing + follow_pheromones + distance_walked

# Spiders
fitness = survival_time + food_eaten + enemies_touched + times_eating_for_nothing + times_attacking_for_nothing + distance_walked
```

> [!NOTE]
> Each parameter has an attributed weight which is often changed to try to optimize intelligence manually.

---

## Modular OOP Project Structure

The codebase is engineered with strict Object-Oriented Programming (OOP) abstraction, modular package structures, and zero hardcoded class dependencies in the update loop:

```
ants-world/
├── assets/
│   ├── ant.png              # Ant sprite
│   ├── food.png             # Food sprite
│   └── spider.png           # Spider sprite
├── core/
│   ├── __init__.py
│   ├── engine.py            # Main application loop, Pygame input & event handling
│   ├── simulation.py        # Top-level orchestrator & ACTIVE_SPECIES configuration
│   ├── constants.py         # Global engine, window geometry & UI constants
│   ├── utils.py             # Math helpers & dynamic SpeciesStats tracker
│   └── main.py              # Convenience launcher alias
├── world/
│   ├── __init__.py
│   ├── world.py             # Dynamic arena container & entity lifecycle management
│   ├── entity.py            # Abstract Entity base class for passive objects
│   ├── food.py              # Concrete Food entity inheriting from Entity
│   ├── physics.py           # Optimized SpatialHash & generic collision resolution
│   └── environment.py       # EnvironmentSystem for dynamic food spawning & weather hooks
├── species/
│   ├── __init__.py
│   ├── creature.py          # Abstract Creature base class (spatial state, vitals, AI hooks)
│   ├── ant.py               # Ant species implementation inheriting from Creature
│   ├── ant_constants.py     # Dedicated Ant parameters & fitness weights
│   ├── spider.py            # Spider species implementation inheriting from Creature
│   └── spider_constants.py  # Dedicated Spider parameters & fitness weights
├── evolution/
│   ├── __init__.py
│   ├── brain.py             # Neural network wrapper (71→16→8→4, genome decoding/encoding)
│   ├── network.py           # Generic two-hidden-layer feedforward NeuralNetwork architecture
│   ├── sensors.py           # 8-sector directional vision & density perception (species agnostic)
│   └── genetics.py          # Truncation parent selection & Gaussian mutation algorithms
├── rendering/
│   ├── __init__.py
│   ├── renderer.py          # Pygame visualization & dynamic sprite loader
│   └── ui.py                # Top HUD overlays & miniature creature health bars
├── main.py                  # Root entry point script
├── __main__.py              # Package entry point (python -m ants_world or python .)
└── README.md                # Project documentation
```

### Module Dependency Graph

```
main.py / core.engine
 ├── core.simulation
 │    ├── world.world
 │    │    ├── world.physics (SpatialHash, combat & collision resolution)
 │    │    ├── world.environment (Dynamic food spawning & weather hooks)
 │    │    ├── species.ant ───────> species.creature ──> evolution.brain & evolution.sensors
 │    │    ├── species.spider ────> species.creature ──> evolution.brain & evolution.sensors
 │    │    ├── world.food ────────> world.entity
 │    │    └── evolution.genetics
 │    └── core.constants & core.utils
 └── rendering.renderer
      └── rendering.ui
```

### Architectural Design Decisions
- **`Creature` & `Entity` Abstractions**: All active agents inherit from `Creature`, while passive objects inherit from `Entity`. Common physics, vitals, neural hooks, and communication stubs live in the base classes.
- **`SPECIES_CONFIG` Isolation Engine**: In `core/constants.py`, the `SPECIES_CONFIG` dict defines which species are active, their reproduction mode (`"continuous"` or `"generational"`), and whether they are NPCs. You can toggle a species on/off or switch its evolution mode without changing any loop logic.
- **Generic Sensing & Collision Resolution**: `world/physics.py` and `evolution/sensors.py` never hardcode class names. Combat and vision checks identify opponents dynamically by comparing class types.
- **Decoupled Renderer**: `rendering/renderer.py` operates in read-only mode and loads sprites dynamically based on `species_name`. If a PNG is missing, it falls back cleanly to geometric rendering.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.20 | Neural network math, vector operations |
| `pygame` | ≥2.0 | Rendering, input handling, game loop |

No other dependencies. No ML frameworks.
