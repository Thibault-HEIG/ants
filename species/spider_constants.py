"""
spider_constants.py — Dedicated Parameters and Fitness Weights for Spiders
==========================================================================

Stores all species-specific configuration, physical limitations, sensory ranges,
and fitness weights for the Spider (Predator) species.
"""

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

# Sensory Capacities
SPIDER_SENSOR_RANGE: float = 250.0
DENSITY_RADIUS_SPIDER: float = SPIDER_SENSOR_RANGE

# Fitness Evaluation Weights
FITNESS_SURVIVAL_WEIGHT: float = 10.0
FITNESS_FOOD_WEIGHT: float = 10.0
FITNESS_ENEMIES_TOUCHED_WEIGHT: float = 50.0
