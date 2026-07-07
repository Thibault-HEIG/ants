"""
constants.py — Global Engine and World Configuration
======================================================

Contains engine, world, and architectural constants shared across the simulation.
Species-specific parameters and fitness variables live in their respective
modules within the /species package.
"""

import math

# ---------------------------------------------------------------------------
# World geometry
# ---------------------------------------------------------------------------
WORLD_WIDTH: int = 1200   # pixels
WORLD_HEIGHT: int = 800   # pixels


# ---------------------------------------------------------------------------
# General Lifecycle & World Settings
# ---------------------------------------------------------------------------
HEALTH_DECAY_RATE: float = 2.0        # HP lost per second
MAX_AGE_NORMALIZATION: float = 100.0  # seconds — used to normalize age for neural network inputs

# ---------------------------------------------------------------------------
# Food parameters
# ---------------------------------------------------------------------------
SUGAR_NUTRITION: float = 60.0         # HP restored by sugar
SEED_NUTRITION: float = 30.0          # HP restored by seed
SUGAR_WEIGHT: float = 0.3             # 30% chance a source spawns sugar
SEED_WEIGHT: float = 0.7              # 70% chance a source spawns seed
FOOD_RADIUS: int = 6                  # collision/render radius in pixels
MAX_FOOD: int = 80                    # global cap on food items

# ---------------------------------------------------------------------------
# Food source parameters
# ---------------------------------------------------------------------------
FOOD_SOURCE_RADIUS: float = 20.0           # visual/collision radius of the source spot
FOOD_SOURCE_LIFETIME: float = 180.0        # 3 minutes before a source expires
FOOD_SOURCE_SPAWN_RATE: float = 2.0        # food items generated per second per source
FOOD_SOURCE_SPAWN_RADIUS: float = 60.0     # scatter radius around the source
MAX_FOOD_SOURCES: int = 5                  # max simultaneous active sources
FOOD_SOURCE_COOLDOWN: float = 15.0         # seconds between new source spawns

# ---------------------------------------------------------------------------
# Eating mechanic
# ---------------------------------------------------------------------------
EAT_DURATION: float = 2.0                  # seconds a creature freezes while eating
EAT_PICKUP_RADIUS: float = 15.0            # must be within this radius of food to consume

# ---------------------------------------------------------------------------
# Sensor system
# ---------------------------------------------------------------------------
SENSOR_ANGLE: float = math.radians(50)  # ±50° from heading → 100° total FOV
MAX_DENSITY_COUNT: int = 10             # normalisation cap: 10+ nearby → 1.0

# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
NN_INPUTS: int = 22
NN_HIDDEN: int = 8
NN_OUTPUTS: int = 4    # output 0 → turn, output 1 → speed, output 2 → attack, output 3 → eat

GENOME_SIZE: int = (NN_INPUTS * NN_HIDDEN) + NN_HIDDEN + \
                   (NN_HIDDEN * NN_OUTPUTS) + NN_OUTPUTS  # = 219

# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
MUTATION_RATE: float = 0.5            # probability each gene is mutated
MUTATION_STRENGTH: float = 0.5        # std-dev of Gaussian noise added to genes
SELECTION_FRACTION: float = 0.1       # top 10% survive to breed

# ---------------------------------------------------------------------------
# Rendering / UI
# ---------------------------------------------------------------------------
FPS: int = 60                         # target frames per second

# Colours (R, G, B)
BG_COLOR: tuple[int, int, int] = (34, 40, 49)          # dark charcoal background
ZONE_DANGER_COLOR: tuple[int, int, int] = (45, 30, 30) # faint red tint (Left Zone)
ZONE_SAFE_COLOR: tuple[int, int, int] = (30, 45, 30)   # faint green tint (Right Zone)
ZONE_BOUNDARY_X: float = WORLD_WIDTH / 2.0
HUD_TEXT_COLOR: tuple[int, int, int] = (238, 238, 238) # near-white for readability
HUD_BG_COLOR: tuple[int, int, int] = (50, 56, 65)      # slightly lighter panel
HUD_ACCENT_COLOR: tuple[int, int, int] = (0, 173, 181) # teal accent for highlights
SPIDER_ACCENT_COLOR: tuple[int, int, int] = (220, 80, 80) # red accent for spiders
HUD_FONT_SIZE: int = 18

SPEED_MULTIPLIERS: list[float] = [0.5, 1, 2, 4, 8, 16, 32, 64]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
