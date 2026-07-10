"""
spider_constants.py — Dedicated Parameters and Fitness Weights for Spiders
==========================================================================

Stores all species-specific configuration, physical limitations, sensory ranges,
and fitness weights for the Spider species.
"""
from core.constants import SENSOR_ANGLE

# Population and Lifecycle
SPIDER_COUNT: int = 5
SPIDER_INITIAL_HEALTH: float = 300.0
SPIDER_REPRODUCTION_THRESHOLD: float = 350.0
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
SPIDER_SENSOR_RANGE: float = 150.0
SPIDER_SENSOR_ANGLE: float = SENSOR_ANGLE
DENSITY_RADIUS_SPIDER: float = SPIDER_SENSOR_RANGE

# Fitness Evaluation Weights
FITNESS_SURVIVAL_WEIGHT: float = 0.0
FITNESS_FOOD_WEIGHT: float = 20.0
FITNESS_ENEMIES_TOUCHED_WEIGHT: float = 50.0
FITNESS_TILES_COVERED_WEIGHT: float = 15.0
FITNESS_DISTANCE_WALKED_WEIGHT: float = FITNESS_TILES_COVERED_WEIGHT
FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT: float = -5.0
FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT: float = -5.0
FITNESS_BRAIN_ORIGINALITY_WEIGHT: float = 30.0