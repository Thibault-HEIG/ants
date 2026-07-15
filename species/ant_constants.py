"""
ant_constants.py — Dedicated Parameters and Fitness Weights for Ants
====================================================================

Stores all species-specific configuration, physical limitations, sensory ranges,
and fitness weights for the Ant species.
"""
from core.constants import SENSOR_ANGLE, WORLD_HEIGHT, WORLD_WIDTH, GENERATION_DURATION

# Population and Lifecycle
ANT_COUNT: int = 80
ANT_INITIAL_HEALTH: float = 100.0
ANT_REPRODUCTION_THRESHOLD: float = 200.0
MAX_ANTS: int = 300

# Physical and Combat Attributes
ANT_MAX_SPEED: float = 150.0
ANT_RADIUS: int = 8
ANT_STRIKE_RANGE: float = 20.0
ANT_TURN_RATE: float = 3.0
ANT_DAMAGE: float = 50.0
ANT_ATTACK_COST: float = 20.0
ANT_EATING_TIME: float = 1.0

# Sensory Capacities
ANT_SENSOR_RANGE: float = 300.0
ANT_SENSOR_ANGLE: float = SENSOR_ANGLE
DENSITY_RADIUS_ANT: float = ANT_SENSOR_RANGE

# Pheromone Trail Parameters
PHEROMONE_STRENGTH: float = 0.1
PHEROMONE_DURATION: float = 10.0

# ---------------------------------------------------------------------------
# Bounds for fitness normalization
# ---------------------------------------------------------------------------
ANT_METRIC_BOUNDS: dict[str, float] = {
    "survival_time": GENERATION_DURATION, # generation: GENERATION_DURATION - max: 300.0
    "computed_food_eaten": 30.0,
    "computed_enemies_touched": 30.0,
    "times_eating_for_nothing": 80.0,
    "times_attacking_for_nothing": 200.0,
    "follow_pheromones": 100.0,
    "tiles_covered": (WORLD_HEIGHT * WORLD_WIDTH) / 900, # Realitic coverage is 1/900
}

# ---------------------------------------------------------------------------
# Fitness Evaluation Weights
# ---------------------------------------------------------------------------
# Behavior
FITNESS_SURVIVAL_WEIGHT: float = 10.0 # 0.0
FITNESS_TILES_COVERED_WEIGHT: float = 10.0 # 10.0
FITNESS_FOLLOW_PHEROMONES_WEIGHT: float = 10.0 # 10.0

FITNESS_BRAIN_ORIGINALITY_WEIGHT: float = 0.05 # 5%

# Peformance
FITNESS_FOOD_WEIGHT: float = 30.0 # 30.0
FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT: float = -10.0 # -10.0

FITNESS_ENEMIES_TOUCHED_WEIGHT: float = 50.0 # 50.0
FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT: float = -5.0 # -5.0 So they attack more.