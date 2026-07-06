"""
constants.py — Central Configuration for the Ant Colony Simulator
=================================================================

Every tunable parameter lives here so we have ONE source of truth.
Changing a value here propagates everywhere — no magic numbers scattered
across the codebase.

Design note: we keep these as plain module-level constants (UPPER_SNAKE_CASE)
rather than a config object because they are read-only at runtime and
importing them is dead simple:  ``from ant_simulator.constants import ANT_COUNT``
"""

import math

# ---------------------------------------------------------------------------
# World geometry
# ---------------------------------------------------------------------------
WORLD_WIDTH: int = 1200   # pixels — wide enough for interesting foraging
WORLD_HEIGHT: int = 800   # pixels — 3:2 aspect ratio fits most monitors

# ---------------------------------------------------------------------------
# Continuous Evolution Settings
# ---------------------------------------------------------------------------
# Fitness-based reproduction: every N seconds the world spawns a new
# creature using a mutated copy of a top-3 fittest genome.  The spawn
# interval adapts to population pressure:
#
#   interval = THRESHOLD / max(1, max_pop - current_pop)
#
# When population is low, empty slots are many → spawning is fast.
# When population is near cap, slots are few → spawning slows down.
ANT_REPRODUCTION_THRESHOLD: float = 200.0    # tuning knob for ant spawn rate
SPIDER_REPRODUCTION_THRESHOLD: float = 350.0 # spiders spawn slower

MAX_ANTS: int = 300      # hard cap to prevent exponential explosion
MAX_SPIDERS: int = 100   # hard cap to prevent exponential explosion

# ---------------------------------------------------------------------------
# Ant parameters
# ---------------------------------------------------------------------------
ANT_COUNT: int = 50                  # population size per generation
ANT_INITIAL_HEALTH: float = 100.0     # starting HP for each ant
HEALTH_DECAY_RATE: float = 2.0        # HP lost per second (both species)
ANT_MAX_SPEED: float = 150.0          # pixels per second — ants are fast
ANT_RADIUS: int = 8                   # collision circle radius in pixels
ANT_STRIKE_RANGE: float = 20.0        # reach is larger than hurtbox radius (8px)
ANT_TURN_RATE: float = 3.0            # max radians/sec the ant can steer
ANT_DAMAGE: float = 50.0              # HP dealt to a spider on contact
ANT_ATTACK_COST: float = 20.0         # HP cost per second while actively attacking
ANT_SENSOR_RANGE: float = 150.0       # how far each ray can "see" (pixels)

# ---------------------------------------------------------------------------
# Spider parameters
# ---------------------------------------------------------------------------
SPIDER_COUNT: int = 5                # fewer but tougher
SPIDER_INITIAL_HEALTH: float = 300.0  # tanky — survives multiple ant hits
SPIDER_MAX_SPEED: float = 80.0        # pixels per second — slow but deadly
SPIDER_RADIUS: int = 12               # larger collision circle
SPIDER_STRIKE_RANGE: float = 35.0     # reach is larger than hurtbox radius (12px)
SPIDER_TURN_RATE: float = 1.0         # less agile than ants
SPIDER_DAMAGE: float = 80.0          # HP dealt to an ant on contact
SPIDER_ATTACK_COST: float = 30.0      # HP cost per second while actively attacking
SPIDER_SENSOR_RANGE: float = 250.0    # better vision than ants

# ---------------------------------------------------------------------------
# Food parameters
# ---------------------------------------------------------------------------
FOOD_NUTRITION: float = 30.0          # flat HP restored (same for both species)
FOOD_SPAWN_RATE: float = 1.5          # new food items per second (on average)
MAX_FOOD: int = 80                    # cap to avoid flooding the world
FOOD_RADIUS: int = 6                  # collision/render radius in pixels

# ---------------------------------------------------------------------------
# Sensor system
# ---------------------------------------------------------------------------
SENSOR_ANGLE: float = math.radians(50)  # ±50° from heading → 100° total FOV
# Why 100° FOV?  Narrower than human vision (~120°) to force creatures to
# turn and actively scan their environment rather than passively drifting.
# Note: sensor RANGE is per-species (ANT_SENSOR_RANGE, SPIDER_SENSOR_RANGE).

# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
# 22 inputs:
#   Vision rays (L/R/Fwd × 4 types)  : 12  (food, enemy, ally, wall × 3 rays)
#   Omnidirectional sensing           :  4  (food dist/angle, enemy dist/angle)
#   Internal state                    :  4  (hp, zone, speed, age)
#   Teamwork                          :  2  (ally density, enemy density)
NN_INPUTS: int = 22
NN_HIDDEN: int = 8     # scaled up from 4 to handle 22 inputs
NN_OUTPUTS: int = 3    # output 0 → turn, output 1 → speed, output 2 → attack boolean

# Genome size = total number of floating-point weights + biases:
#   input→hidden weights : 22 × 8 = 176
#   hidden biases         :          8
#   hidden→output weights:  8 × 3 = 24
#   output biases         :          3
#                           TOTAL = 211
GENOME_SIZE: int = (NN_INPUTS * NN_HIDDEN) + NN_HIDDEN + \
                   (NN_HIDDEN * NN_OUTPUTS) + NN_OUTPUTS  # = 211

# ---------------------------------------------------------------------------
# Density sensing (teamwork)
# ---------------------------------------------------------------------------
# Radius within which allies/enemies are counted for density inputs.
# Uses sensor range by default so density awareness matches vision range.
DENSITY_RADIUS_ANT: float = ANT_SENSOR_RANGE
DENSITY_RADIUS_SPIDER: float = SPIDER_SENSOR_RANGE
MAX_DENSITY_COUNT: int = 10   # normalisation cap: 10+ nearby → 1.0

# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
MUTATION_RATE: float = 0.5            # probability each gene is mutated (1.0 = always)
MUTATION_STRENGTH: float = 0.5        # std-dev of Gaussian noise added to genes
SELECTION_FRACTION: float = 0.1       # top 10 % survive to breed
ROUND_TIME_LIMIT: float = 150.0       # seconds — round ends if neither side is eliminated
# Why a time limit?  Without health decay, early generations with random
# brains may never collide.  The cap ensures evolution keeps progressing.

# ---------------------------------------------------------------------------
# Rendering / UI
# ---------------------------------------------------------------------------
FPS: int = 60                         # target frames per second

# Colours (R, G, B)
BG_COLOR: tuple[int, int, int] = (34, 40, 49)        # dark charcoal background
ZONE_DANGER_COLOR: tuple[int, int, int] = (45, 30, 30) # faint red tint (Left Zone)
ZONE_SAFE_COLOR: tuple[int, int, int] = (30, 45, 30)   # faint green tint (Right Zone)
ZONE_BOUNDARY_X: float = WORLD_WIDTH / 2.0
HUD_TEXT_COLOR: tuple[int, int, int] = (238, 238, 238)  # near-white for readability
HUD_BG_COLOR: tuple[int, int, int] = (50, 56, 65)    # slightly lighter panel
HUD_ACCENT_COLOR: tuple[int, int, int] = (0, 173, 181)  # teal accent for highlights
SPIDER_ACCENT_COLOR: tuple[int, int, int] = (220, 80, 80)  # red accent for spiders
HUD_FONT_SIZE: int = 18

# Simulation speed presets — cycled with ↑/↓ arrow keys
# 0.5× for studying individual behaviour,  8× for fast-forwarding evolution
SPEED_MULTIPLIERS: list[float] = [0.5, 1, 2, 4, 8, 16, 32, 64]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42  # seed for numpy RNG — set to None for non-deterministic runs
