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

1. **Sense** — Eight directional vision rays, density sensors, and navigation/interaction sensors detect food, enemies, allies, walls, pheromone trails, item availability, and home coordinates.
2. **Think** — A neural network processes 87 sensor inputs and outputs `[turn, speed, attack, eat, take, release]`.
3. **Act** — The creature moves, engages in combat, consumes food, or takes/carries/releases items according to the brain's decision.
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

Each creature has an identical `87 → 16 → 8 → 6` fully-connected neural network (tanh activations, two hidden layers, no ML libraries — pure NumPy).

### 87 Inputs

**8-Sector Vision (72 inputs)** — 8 directional sensors evenly distributed across the ±80° FOV (160° total). Each sensor reports 9 features using inverted convention for distances: `0.0` = nothing detected, `1.0` = touching.

| Indices | Sensor | Angle Offset |
|---------|--------|--------------|
| 0–8 | Sensor 0 | −80.0° (leftmost) |
| 9–17 | Sensor 1 | −57.1° |
| 18–26 | Sensor 2 | −34.3° |
| 27–35 | Sensor 3 | −11.4° |
| 36–44 | Sensor 4 | +11.4° |
| 45–53 | Sensor 5 | +34.3° |
| 54–62 | Sensor 6 | +57.1° |
| 63–71 | Sensor 7 | +80.0° (rightmost) |

Each sensor outputs 9 values:

| Feature | Range | Description |
|---------|-------|-------------|
| `enemy_distance` | [0, 1] | 0 = no enemy, 1 = touching |
| `enemy_is_eating` | {0, 1} | 1 = seen enemy is eating |
| `enemy_is_attacking` | {0, 1} | 1 = seen enemy is attacking |
| `ally_distance` | [0, 1] | 0 = no ally, 1 = touching |
| `ally_is_eating` | {0, 1} | 1 = seen ally is eating |
| `ally_is_attacking` | {0, 1} | 1 = seen ally is attacking |
| `food_distance` | [0, 1] | 0 = no food, 1 = touching |
| `pheromone_strength` | [0, 1) | Weighted sum of pheromone intensity along ray squashed via tanh (`math.tanh(sum * 0.5)`) |

**Internal State & Navigation (15 inputs)**

| # | Input | Range | Category |
|---|-------|-------|----------|
| 72 | `hp` | [0, 1] | 🧠 Internal |
| 73 | `zone` | {0, 1} | 🧠 Internal (0 = Spider, 1 = Ant) |
| 74 | `speed` | [0, 1] | 🧠 Internal |
| 75 | `age` | [0, 1] | 🧠 Internal |
| 76 | `ally_density` | [0, 1] | 🐜 Teamwork |
| 77 | `enemy_density` | [0, 1] | 🐜 Teamwork |
| 78 | `has_gained_hp` | {0, 1} | 🧠 Internal (HP increased in last second) |
| 79 | `pheromone_strength` | [0, 2] | 🐜 Navigation (pheromone concentration under foot) |
| 80 | `is_carrying` | {0, 1} | 🎒 Carry State (1 = currently carrying an object) |
| 81 | `can_eat` | {0, 1} | 🍽️ Interaction (1 = food within pickup radius) |
| 82 | `can_take` | {0, 1} | 🎒 Carry State (1 = takeable object within radius & not carrying) |
| 83 | `can_touch` | {0, 1} | 🍽️ Interaction (1 = object within pickup radius) |
| 84 | `home_distance` | [0, 1] | 🏰 Navigation (normalised distance to kingdom/home) |
| 85 | `home_angle` | [-1, 1] | 🏰 Navigation (relative angle to kingdom/home ÷ π) |
| 86 | `is_at_home` | {0, 1} | 🏰 Navigation (1 = within kingdom spawn radius) |

### 6 Outputs

| Output | Range | Meaning |
|--------|-------|---------|
| `turn` | [-1, 1] | Negative = left, positive = right |
| `speed` | [0, 1] | 0 = stop, 1 = full speed |
| `attack` | {0, 1} | 0 = hold fire, 1 = attack (strike reach enabled) |
| `eat` | {0, 1} | 0 = don't eat, 1 = attempt to eat |
| `take` | {0, 1} | 0 = don't take, 1 = attempt to pick up an item within range |
| `release` | {0, 1} | 0 = don't release, 1 = attempt to drop/deliver carried item |

### Genome

The genome is a flat vector of **1598 floats** — every weight and bias concatenated:

```
[w_input→hidden1 (1392)] [b_hidden1 (16)] [w_hidden1→hidden2 (128)] [b_hidden2 (8)] [w_hidden2→output (48)] [b_output (6)]
```

---

## Sensor System

Creatures perceive the world through two complementary systems:

### 1. Directional Vision (8 sensors)

Eight vision rays extend from the creature, evenly distributed across a ±80° field of view (160° total). Each ray detects the nearest enemy, ally, food, and wall along its direction, as well as the pheromone intensity along the ray (`pheromone_strength`). When an enemy or ally is detected, the sensor also reports whether that creature is currently eating (`is_eating`) or attacking (`is_attacking`).

For entities and walls, distances use **inverted normalisation**: `0.0` = nothing within range, `1.0` = touching. For pheromone trails, each ray computes the weighted sum of intensity across all intersected grid cells (`sum(pheromone_intensity * distance_weight)`, where `distance_weight = max(0.0, 1.0 - dist / max_range)`) and squashes the raw sum into the `[0, 1)` range using `math.tanh(sum * 0.5)`. A non-zero value implicitly indicates detection — no separate binary flag is needed.

### 2. Density Awareness (teamwork)

Counts of nearby allies and enemies within sensor range, normalised by a cap of 10. This enables emergent swarming — an ant can "decide" to fight only when outnumbering the enemy.

### 3. Navigation & Item Perception

Creatures perceive their relationship to their home kingdom (`home_distance`, `home_angle`, `is_at_home`), enabling navigation back to base after exploring or gathering resources. They also receive immediate feedback on item interactions (`can_eat`, `can_take`, `can_touch`) and their current carrying status (`is_carrying`).

---

## Item Taking & Releasing Mechanics

Creatures can pick up items (such as unconsumed sugar or seed food items), transport them across the arena, and release or deliver them to their home kingdom.

### Action State Machine & Priorities
At every frame, the creature's action commands are evaluated with strict priority:
```
Attack > Eat > Take > Release
```

- **While Carrying (`is_carrying == True`)**:
  - The creature is physically carrying an object (`carried_object`).
  - **Restrictions**: It cannot `attack`, `eat`, or `take` another item (`take_signal = False`, `is_attacking = False`, `is_eating = False`).
  - **Movement Penalty**: Carrying an object reduces effective movement speed by 20% (`CARRY_SPEED_MULTIPLIER = 0.8`).
  - **Navigation & Tracking**: Every step taken toward the home kingdom while carrying an object contributes to `walk_with_object_in_home_direction`, whereas moving away increments `walk_with_object_in_opposite_home_direction`. This enables precise fitness shaping for foraging and resource retrieval.

- **Taking (`take > 0.5`)**:
  - If the creature is not carrying an object and is within pickup range (`EAT_PICKUP_RADIUS = 20.0`) of an unconsumed, uncarried food item, it picks up the item and enters the carrying state.

- **Releasing & Home Delivery (`release > 0.5`)**:
  - If the creature activates the `release` output while carrying an object, it drops the item at its current position.
  - **Home Delivery**: If the item is dropped inside the creature's home kingdom (`is_at_home == True`, within the kingdom's `spawn_radius`), the item is consumed (`consumed = True`), the species' home delivery count increments, and the creature earns a significant `release_at_home` fitness bonus.
  - **Field Drop**: If released outside the kingdom, the item remains on the ground (`release_anywhere`), allowing other creatures to pick it up or consume it later.

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
fitness = (food_eaten + eating_for_nothing + enemies_touched + attacking_for_nothing + follow_pheromones + survival_time + tiles_covered + taken_object + walk_home + walk_opposite + release_anywhere + release_at_home) * originality_multiplier

# Spiders
fitness = (food_eaten + eating_for_nothing + enemies_touched + attacking_for_nothing + survival_time + tiles_covered) * originality_multiplier
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
│   ├── brain.py             # Neural network wrapper (87→16→8→6, genome decoding/encoding)
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
