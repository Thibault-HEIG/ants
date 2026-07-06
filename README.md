# 🐜 Ants vs Spiders — Evolution War

A real-time co-evolutionary simulation where **ants** and **spiders** each evolve neural-network brains through natural selection. No behaviour is hand-coded — all intelligence emerges from evolution alone.

![Pygame](https://img.shields.io/badge/Pygame-2D%20Rendering-green) ![NumPy](https://img.shields.io/badge/NumPy-Neural%20Networks-blue) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow)

---

## Quick Start

```bash
# Install dependencies
pip install numpy pygame

# Run the simulation from the project root
python3 main.py

# Or execute the engine module directly
python3 -m core.engine
```

### Controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Increase / decrease simulation speed (0.5×–64×) |
| `Space` | Pause / unpause |
| `S` | Toggle vision ray / sensor visualization |
| `R` | Restart simulation |
| `Escape` | Quit |

---

## How It Works

### The Core Loop

Each creature runs this cycle **every frame**:

```
Sense → Think → Act → Evolve
```

1. **Sense** — Vision rays and omnidirectional sensors detect food, enemies, allies, and walls.
2. **Think** — A neural network processes 22 sensor inputs and outputs `[turn, speed, attack]`.
3. **Act** — The creature moves and engages in combat according to the brain's decision.
4. **Evolve** — Creatures that survive long enough reproduce (continuous evolution). Their children inherit mutated copies of the parent's brain weights.

No arbitrary fitness function drives generational resets — evolution is **continuous**. A creature that survives above its reproduction threshold long enough spawns a mutated child nearby.

### The World

The 1200×800 arena is divided into two zones:

| Zone | Location | Properties |
|------|----------|------------|
| **Danger Zone** (red tint) | Left half | Spiders move at ant speed here. Less food spawns (20%). |
| **Safe Zone** (green tint) | Right half | Spiders move at normal speed. Most food spawns here (80%). |

This asymmetry forces interesting strategic trade-offs: ants must venture into danger for territory and food, while spiders benefit from hunting in their fast zone.

### Species Comparison

| Attribute | 🐜 Ant | 🕷️ Spider |
|-----------|--------|-----------|
| Population | 100 (cap: 300) | 5 (cap: 100) |
| Initial Health | 100 HP | 300 HP |
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

### Continuous Evolution

Creatures reproduce **during the simulation** based on fitness ranking and population pressure:

- **Truncation Selection**: Whenever a species reproduces, the world selects a random parent from the **top N% fittest** living creatures of that species (using `select_parents` in `genetics.py`, which dynamically scales the parent pool). A mutated child is spawned near the parent.
- **Dynamic Spawn Rate**: The time interval between spawns scales dynamically based on available population slots:
  `interval = THRESHOLD / (max_population - current_population)`
  
  When a population is small (many available slots), spawning happens rapidly. As the population approaches its cap, available slots approach zero, causing the spawn interval to increase dramatically and naturally stabilizing population numbers without exponential blowup.

### Mutation

Every gene is mutated with Gaussian noise (mean=0, σ=0.5). There is no crossover — children are mutated clones of a single parent.

### Fitness Function

Used for ranking and parent selection. Fitness is calculated directly from raw performance metrics (survival time, food consumed, and combat engagements):

```python
# Ants
fitness = (survival_time / 20.0) * 10.0 + food_eaten * 30.0 + enemies_touched * 50.0

# Spiders
fitness = (survival_time / 20.0) * 10.0 + food_eaten * 10.0 + enemies_touched * 50.0
```

> [!NOTE]
> Historical maximum records (`SpeciesStats`) are tracked independently across the entire simulation run and displayed in real-time on the HUD.

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
│   ├── brain.py             # Neural network wrapper (22→8→3, genome decoding/encoding)
│   ├── network.py           # Generic feedforward NeuralNetwork architecture
│   ├── sensors.py           # Generic vision rays & density perception (species agnostic)
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
- **`ACTIVE_SPECIES` Isolation Engine**: In `core/simulation.py`, the `ACTIVE_SPECIES = [Ant, Spider]` list dictates which species exist. You can drop new species into this list or test a single species in isolation (`ACTIVE_SPECIES = [Ant]`) without changing any loop logic.
- **Generic Sensing & Collision Resolution**: `world/physics.py` and `evolution/sensors.py` never hardcode class names. Combat and vision checks identify opponents dynamically by comparing class types.
- **Decoupled Renderer**: `rendering/renderer.py` operates in read-only mode and loads sprites dynamically based on `species_name`. If a PNG is missing, it falls back cleanly to geometric rendering.

---

## Key Constants

All tunable parameters are decoupled into logical modules:

| Constant | Location | Impact |
|----------|----------|--------|
| `ACTIVE_SPECIES` | `core/simulation.py` | Defines active co-evolving species in the arena |
| `MUTATION_STRENGTH` | `core/constants.py` | Higher = faster but noisier evolutionary jumps |
| `HEALTH_DECAY_RATE` | `core/constants.py` | Forces creatures to forage; too high = die before learning |
| `MAX_ANTS` / `MAX_SPIDERS` | `species/*_constants.py` | Population caps for dynamic spawn scaling |
| `SENSOR_ANGLE` | `core/constants.py` | Narrower = more scanning needed, wider = fewer blind spots |
| `NN_HIDDEN` | `core/constants.py` | Number of hidden neurons in the brain architecture |

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

- **New Species**: Create `species/beetle.py` inheriting from `Creature`, define its parameters in `beetle_constants.py`, add a sprite to `assets/beetle.png`, and add `Beetle` to `ACTIVE_SPECIES` in `core/simulation.py`.
- **New Inputs**: Add fields to `SensorData` in `evolution/sensors.py`, update `to_array()`, and increment `NN_INPUTS` in `core/constants.py`. The brain auto-adapts.
- **New Outputs**: Add an output by increasing `NN_OUTPUTS` in `core/constants.py` and updating post-processing in `evolution/brain.py`.
- **Environmental Mechanics**: Implement custom weather, seasonal cycles, or obstacles inside `world/environment.py`.
- **Headless Mode**: Instantiate `Simulation` without `Renderer` and call `step(dt)` in a script loop for high-speed evolutionary training.
