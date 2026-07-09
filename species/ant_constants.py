"""
ant_constants.py — Dedicated Parameters and Fitness Weights for Ants
====================================================================

Stores all species-specific configuration, physical limitations, sensory ranges,
and fitness weights for the Ant species.
"""

# Population and Lifecycle
ANT_COUNT: int = 50
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
ANT_SENSOR_RANGE: float = 150.0
DENSITY_RADIUS_ANT: float = ANT_SENSOR_RANGE

# Fitness Evaluation Weights
FITNESS_SURVIVAL_WEIGHT: float = 10.0
FITNESS_FOOD_WEIGHT: float = 30.0
FITNESS_ENEMIES_TOUCHED_WEIGHT: float = 50.0
FITNESS_FOLLOW_PHEROMONES_WEIGHT: float = 0.01 # Sinon trop récompensé
FITNESS_DISTANCE_WALKED_WEIGHT: float = 0.01

FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT: float = -5.0
FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT: float = -5.0

# Pheromone Trail Parameters
PHEROMONE_STRENGTH: float = 0.1
PHEROMONE_DURATION: float = 10.0