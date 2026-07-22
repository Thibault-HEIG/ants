"""
live_config.py — Hot-Swappable Constants Registry
=================================================

Provides a singleton configuration registry that allows updating constants
at runtime via the web client.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

import core.constants as const_mod
import species.ant_constants as ant_mod
import species.spider_constants as spider_mod

logger = logging.getLogger(__name__)


class ConstantCategory(Enum):
    SAFE = "safe"               # Applied instantly, read fresh every tick
    DEFERRED = "deferred"       # Applied but only affects future events
    RESTART_REQUIRED = "restart" # Rejected unless confirm_reset=True


class LiveConfig:
    _instance: LiveConfig | None = None

    def __new__(cls) -> LiveConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        
        self._values: dict[str, Any] = {}
        self._categories: dict[str, ConstantCategory] = {}
        self._types: dict[str, type] = {}
        self._observers: dict[str, list[Callable]] = {}
        self._pending_restart: dict[str, Any] = {}

        self._register_all()

    def _register(self, key: str, default: Any, category: ConstantCategory, source_module: Any, module_key: str | None = None) -> None:
        mod_key = module_key if module_key is not None else key
        self._values[key] = default
        self._categories[key] = category
        self._types[key] = type(default)
        if key not in self._observers:
            self._observers[key] = []

        # Default observer that updates the source module
        def update_module(k: str, old_value: Any, new_value: Any) -> None:
            setattr(source_module, mod_key, new_value)

        self.subscribe(key, update_module)

    def _register_all(self) -> None:
        # --- SAFE ---
        # Custom Broadcast config
        self._register("BROADCAST_INTERVAL", 0.05, ConstantCategory.SAFE, const_mod)
        setattr(const_mod, "BROADCAST_INTERVAL", 0.05)

        safe_core = [
            "HEALTH_DECAY_RATE", "MAX_AGE_NORMALIZATION",
            "CONTINUOUS_MUTATION_RATE", "CONTINUOUS_MUTATION_STRENGTH",
            "GENERATIONAL_MUTATION_RATE", "GENERATIONAL_MUTATION_STRENGTH", "GENERATIONAL_SELECTION_FRACTION",
            "EXTINCTION_MUTATION_RATE", "EXTINCTION_MUTATION_STRENGTH",
            "SUGAR_NUTRITION", "SEED_NUTRITION", "SUGAR_WEIGHT", "SEED_WEIGHT",
            "MAX_FOOD", "FOOD_SOURCE_SPAWN_RATE", "FOOD_SOURCE_SPAWN_RADIUS", "FOOD_SOURCE_LIFETIME", "MAX_FOOD_SOURCES", "FOOD_SOURCE_COOLDOWN",
            "EAT_PICKUP_RADIUS", "ATTACK_DURATION", "CARRY_SPEED_MULTIPLIER",
        ]
        for k in safe_core:
            self._register(k, getattr(const_mod, k), ConstantCategory.SAFE, const_mod)

        safe_ant = [
            "ANT_INITIAL_HEALTH", "ANT_MAX_SPEED", "ANT_DAMAGE", "ANT_ATTACK_COST", "ANT_EATING_TIME", "ANT_TURN_RATE",
            "PHEROMONE_STRENGTH", "PHEROMONE_DURATION"
        ]
        for k in safe_ant:
            self._register(k, getattr(ant_mod, k), ConstantCategory.SAFE, ant_mod)

        safe_spider = [
            "SPIDER_INITIAL_HEALTH", "SPIDER_MAX_SPEED", "SPIDER_DAMAGE", "SPIDER_ATTACK_COST", "SPIDER_EATING_TIME", "SPIDER_TURN_RATE",
        ]
        for k in safe_spider:
            self._register(k, getattr(spider_mod, k), ConstantCategory.SAFE, spider_mod)

        # Fitness weights
        ant_fitness = [
            "FITNESS_SURVIVAL_WEIGHT", "FITNESS_TILES_COVERED_WEIGHT", "FITNESS_FOLLOW_PHEROMONES_WEIGHT", 
            "FITNESS_BRAIN_ORIGINALITY_WEIGHT", "FITNESS_FOOD_WEIGHT", "FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT", 
            "FITNESS_ENEMIES_TOUCHED_WEIGHT", "FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT",
            "FITNESS_TAKEN_OBJECT_WEIGHT", "FITNESS_WALK_HOME_DIRECTION_WEIGHT", "FITNESS_WALK_OPPOSITE_HOME_WEIGHT", 
            "FITNESS_RELEASE_ANYWHERE_WEIGHT", "FITNESS_RELEASE_AT_HOME_WEIGHT"
        ]
        for k in ant_fitness:
            self._register(f"ANT_{k}", getattr(ant_mod, k), ConstantCategory.SAFE, ant_mod, k)

        spider_fitness = [
            "FITNESS_SURVIVAL_WEIGHT", "FITNESS_TILES_COVERED_WEIGHT", "FITNESS_BRAIN_ORIGINALITY_WEIGHT",
            "FITNESS_FOOD_WEIGHT", "FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT", "FITNESS_ENEMIES_TOUCHED_WEIGHT", 
            "FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT"
        ]
        for k in spider_fitness:
            self._register(f"SPIDER_{k}", getattr(spider_mod, k), ConstantCategory.SAFE, spider_mod, k)

        # --- DEFERRED ---
        deferred_core = ["GENERATION_DURATION", "CONTINUOUS_SELECTION_FRACTION"]
        for k in deferred_core:
            self._register(k, getattr(const_mod, k), ConstantCategory.DEFERRED, const_mod)
        
        self._register("ANT_REPRODUCTION_THRESHOLD", getattr(ant_mod, "ANT_REPRODUCTION_THRESHOLD"), ConstantCategory.DEFERRED, ant_mod)
        self._register("SPIDER_REPRODUCTION_THRESHOLD", getattr(spider_mod, "SPIDER_REPRODUCTION_THRESHOLD"), ConstantCategory.DEFERRED, spider_mod)

        # --- RESTART_REQUIRED ---
        restart_core = [
            "WORLD_WIDTH", "WORLD_HEIGHT", 
            "NN_INPUTS", "NN_HIDDEN_1", "NN_HIDDEN_2", "NN_OUTPUTS", "GENOME_SIZE", "NN_NUM_SENSORS", "STATE_INPUTS",
            "SENSOR_ANGLE", "MAX_DENSITY_COUNT",
            "RANDOM_SEED", "SPECIES_CONFIG"
        ]
        for k in restart_core:
            self._register(k, getattr(const_mod, k), ConstantCategory.RESTART_REQUIRED, const_mod)

        restart_ant = ["MAX_ANTS", "ANT_COUNT", "ANT_METRIC_BOUNDS"]
        for k in restart_ant:
            self._register(k, getattr(ant_mod, k), ConstantCategory.RESTART_REQUIRED, ant_mod)

        restart_spider = ["MAX_SPIDERS", "SPIDER_COUNT", "SPIDER_METRIC_BOUNDS"]
        for k in restart_spider:
            self._register(k, getattr(spider_mod, k), ConstantCategory.RESTART_REQUIRED, spider_mod)


    def get(self, key: str) -> Any:
        if key in self._pending_restart:
            return self._pending_restart[key]
        return self._values.get(key)

    def set(self, key: str, value: Any, confirm_reset: bool = False) -> dict:
        if key not in self._values:
            return {"ok": False, "error": "unknown_key", "key": key, "message": f"Key {key} is not registered."}

        # Type conversion & validation
        expected_type = self._types[key]
        try:
            if value is not None and not isinstance(value, expected_type):
                if expected_type is float and isinstance(value, int):
                    value = float(value)
                elif expected_type is int and isinstance(value, float):
                    value = int(value)
                elif expected_type in (int, float, str, bool) and isinstance(value, (int, float, str, bool)):
                    value = expected_type(value)
                else:
                    return {"ok": False, "error": "type_mismatch", "key": key, "message": f"Expected {expected_type.__name__}, got {type(value).__name__}"}
        except (ValueError, TypeError):
             return {"ok": False, "error": "type_mismatch", "key": key, "message": f"Could not convert {value} to {expected_type.__name__}"}

        category = self._categories[key]
        
        if category in (ConstantCategory.SAFE, ConstantCategory.DEFERRED):
            old_value = self._values[key]
            self._values[key] = value
            for cb in self._observers.get(key, []):
                cb(key, old_value, value)
            return {"ok": True, "key": key, "value": value, "category": category.value}

        elif category == ConstantCategory.RESTART_REQUIRED:
            if not confirm_reset:
                return {
                    "ok": False, 
                    "error": "restart_required", 
                    "key": key, 
                    "message": f"{key} changes genome/network shape and requires a simulation reset. Send confirm_reset=true to queue this change."
                }
            self._pending_restart[key] = value
            return {"ok": True, "key": key, "value": value, "category": "restart", "queued": True}

        return {"ok": False, "error": "unknown_category"}

    def apply_pending_restart(self) -> list[str]:
        applied = []
        for key, value in self._pending_restart.items():
            old_value = self._values[key]
            self._values[key] = value
            for cb in self._observers.get(key, []):
                cb(key, old_value, value)
            applied.append(key)
        self._pending_restart.clear()
        return applied

    def subscribe(self, key: str, callback: Callable) -> None:
        if key not in self._observers:
            self._observers[key] = []
        self._observers[key].append(callback)

    def get_all(self) -> dict[str, dict]:
        return {
            key: {
                "value": self._values[key],
                "category": self._categories[key].value,
                "type": self._types[key].__name__
            }
            for key in self._values
        }

    def get_category(self, key: str) -> ConstantCategory | None:
        return self._categories.get(key)
