"""
constants.py — Global Engine and World Configuration
======================================================

Contains engine, world, and architectural constants shared across the simulation.
Species-specific parameters and fitness variables live in their respective
modules within the /species package.
"""

import math

# ---------------------------------------------------------------------------
# Continuous mode evolution parameters (threshold-based reproduction)
# ---------------------------------------------------------------------------
CONTINUOUS_MUTATION_RATE: float = 0.1
CONTINUOUS_MUTATION_STRENGTH: float = 0.5
CONTINUOUS_SELECTION_FRACTION: float = 0.1

# ---------------------------------------------------------------------------
# Generational mode evolution parameters (elitism, fixed-time episodes)
# ---------------------------------------------------------------------------
GENERATIONAL_MUTATION_RATE: float = 0.1
GENERATIONAL_MUTATION_STRENGTH: float = 0.5
EXTINCTION_MUTATION_RATE: float = GENERATIONAL_MUTATION_RATE + 0.3 # if the specie extincts
EXTINCTION_MUTATION_STRENGTH: float = GENERATIONAL_MUTATION_STRENGTH + 0.5 # if the specie extincts
GENERATIONAL_SELECTION_FRACTION: float = 0.2
GENERATION_DURATION: float = 140.0               # seconds per generation

# ---------------------------------------------------------------------------
# Species configuration
# SPECIES_CONFIG: dict[str, dict] = {
#     "Ant":    {"active": True,  "reproduction_mode": "continuous",   "npc": False},
#     "Spider": {"active": False, "reproduction_mode": "generational", "npc": False},
# }
# ---------------------------------------------------------------------------
SPECIES_CONFIG: dict[str, dict] = {
    "Ant":    {"active": True, "reproduction_mode": "generational", "npc": False},
    "Spider": {"active": True, "reproduction_mode": "generational", "npc": False},
}

# ---------------------------------------------------------------------------
# World geometry
# ---------------------------------------------------------------------------
WORLD_WIDTH: int = 1000   # pixels
WORLD_HEIGHT: int = 900   # pixels


# ---------------------------------------------------------------------------
# General Lifecycle & World Settings
# ---------------------------------------------------------------------------
HEALTH_DECAY_RATE: float = 2.0        # HP lost per second
MAX_AGE_NORMALIZATION: float = 500.0  # seconds — used to normalize age for neural network inputs

# ---------------------------------------------------------------------------
# Food parameters
# ---------------------------------------------------------------------------
SUGAR_NUTRITION: float = 60.0         # HP restored by sugar
SEED_NUTRITION: float = 30.0          # HP restored by seed
SUGAR_WEIGHT: float = 0.3             # 30% chance a source spawns sugar
SEED_WEIGHT: float = 0.7              # 70% chance a source spawns seed
FOOD_RADIUS: int = 6                  # collision/render radius in pixels
MAX_FOOD: int = 150                    # global cap on food items

# ---------------------------------------------------------------------------
# Food source parameters
# ---------------------------------------------------------------------------
FOOD_SOURCE_RADIUS: float = 20.0           # visual/collision radius of the source spot
FOOD_SOURCE_LIFETIME: float = 180.0        # 3 minutes before a source expires
FOOD_SOURCE_SPAWN_RATE: float = 0.5        # food items generated per second per source
FOOD_SOURCE_SPAWN_RADIUS: float = 80.0     # scatter radius around the source
MAX_FOOD_SOURCES: int = 5                  # max simultaneous active sources
FOOD_SOURCE_COOLDOWN: float = 30.0         # seconds between new source spawns
FOOD_SOURCE_LEFT_ZONE_PROB: float = 0.66   # 66% chance a food source spawns in Ants Zone (Left)

# ---------------------------------------------------------------------------
# Eating mechanic
# ---------------------------------------------------------------------------
EAT_PICKUP_RADIUS: float = 20.0            # must be within this radius of food to consume

# ---------------------------------------------------------------------------
# Sensor system
# ---------------------------------------------------------------------------
SENSOR_ANGLE: float = math.radians(80)  # ±80° from heading → 160° total FOV
MAX_DENSITY_COUNT: int = 10             # normalisation cap: 10+ nearby → 1.0

# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
NN_NUM_SENSORS: int = 8         # directional sensors evenly spanning the FOV
NN_INPUTS: int = 80             # 8 sensors × 9 features + 8 state inputs
NN_HIDDEN_1: int = 16           # first hidden layer
NN_HIDDEN_2: int = 8            # second hidden layer
NN_OUTPUTS: int = 4             # turn, speed, attack, eat

GENOME_SIZE: int = (
    (NN_INPUTS * NN_HIDDEN_1) + NN_HIDDEN_1 +
    (NN_HIDDEN_1 * NN_HIDDEN_2) + NN_HIDDEN_2 +
    (NN_HIDDEN_2 * NN_OUTPUTS) + NN_OUTPUTS
)  # = 1468

# ---------------------------------------------------------------------------
# Rendering / UI
# ---------------------------------------------------------------------------
FPS: int = 60                         # target frames per second

# Colours (R, G, B)
BG_COLOR: tuple[int, int, int] = (34, 40, 49)          # dark charcoal background
ZONE_DANGER_COLOR: tuple[int, int, int] = (30, 45, 30) # faint green tint (Left Zone / Ants Zone)
ZONE_SAFE_COLOR: tuple[int, int, int] = (45, 30, 30)   # faint red tint (Right Zone / Spiders Zone)
ZONE_BOUNDARY_X: float = WORLD_WIDTH / 2.0
HUD_TEXT_COLOR: tuple[int, int, int] = (238, 238, 238) # near-white for readability
HUD_BG_COLOR: tuple[int, int, int] = (50, 56, 65)      # slightly lighter panel
HUD_ACCENT_COLOR: tuple[int, int, int] = (0, 173, 181) # teal accent for highlights
SPIDER_ACCENT_COLOR: tuple[int, int, int] = (220, 80, 80) # red accent for spiders
HUD_FONT_SIZE: int = 18

SPEED_MULTIPLIERS: list[float] = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 41
