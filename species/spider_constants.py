"""
spider_constants.py — Dedicated Parameters and Fitness Weights for Spiders
==========================================================================

Stores all species-specific configuration, physical limitations, sensory ranges,
and fitness weights for the Spider species.
"""
from core.constants import SENSOR_ANGLE, WORLD_HEIGHT, WORLD_WIDTH, GENERATION_DURATION

# Population and Lifecycle
SPIDER_COUNT: int = 15
SPIDER_INITIAL_HEALTH: float = 300.0
SPIDER_REPRODUCTION_THRESHOLD: float = 600.0
MAX_SPIDERS: int = 100

# Physical and Combat Attributes
SPIDER_MAX_SPEED: float = 80.0
SPIDER_RADIUS: int = 12
SPIDER_STRIKE_RANGE: float = 35.0
SPIDER_TURN_RATE: float = 1.0
SPIDER_DAMAGE: float = 80.0
SPIDER_ATTACK_COST: float = 30.0
SPIDER_EATING_TIME: float = 2.0

# Sensory Capacities
SPIDER_SENSOR_RANGE: float = 300.0
SPIDER_SENSOR_ANGLE: float = SENSOR_ANGLE
DENSITY_RADIUS_SPIDER: float = SPIDER_SENSOR_RANGE

# ---------------------------------------------------------------------------
# Bounds for fitness normalization
# ---------------------------------------------------------------------------
SPIDER_METRIC_BOUNDS: dict[str, float] = {
    "survival_time": GENERATION_DURATION, # generation: GENERATION_DURATION - max: 500.0
    "computed_food_eaten": 20.0,
    "computed_enemies_touched": 60.0,
    "times_eating_for_nothing": 30.0,
    "times_attacking_for_nothing": 120.0,
    "tiles_covered": (WORLD_HEIGHT * WORLD_WIDTH) / 900, # Realitic coverage is 1/900
}

# ---------------------------------------------------------------------------
# Fitness Evaluation Weights
# ---------------------------------------------------------------------------
# Behavior
FITNESS_SURVIVAL_WEIGHT: float = 0.0 # 0.0
FITNESS_TILES_COVERED_WEIGHT: float = 10.0 # 10.0

FITNESS_BRAIN_ORIGINALITY_WEIGHT: float = 0.05 # 0.05%

# Performance
FITNESS_FOOD_WEIGHT: float = 30.0 # 20.0
FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT: float = -10.0 # -10.0

FITNESS_ENEMIES_TOUCHED_WEIGHT: float = 50.0 # 50.0
FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT: float = -10.0 # -5.0 So they attack more.