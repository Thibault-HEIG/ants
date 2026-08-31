"""
serialization.py — Snapshot Builder
===================================

Builds full and aggregate simulation snapshots for the rendering channel
(WebSocket, BROADCAST_INTERVAL cadence). Contains only creature positions,
food, pheromones, kingdoms, lakes, and lightweight population counts.

Stats, training metrics, and metric bounds are written to SQLite by
core/tracking_db.py on a separate wall-clock timer.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from core.constants import FRAMES_PER_DT, GENERATION_DURATION, WORLD_HEIGHT, WORLD_WIDTH
from core.utils import SpeciesStats


def _json_encode(snapshot: dict[str, Any]) -> str:
    """Encode a snapshot dictionary to a JSON string."""
    return json.dumps(snapshot, separators=(',', ':'))


# Swap point: change this single reference to switch encoding (e.g., to msgpack)
encode = _json_encode


def compute_species_stats(world: Any, cls: type) -> dict[str, Any]:
    """Compute living/dead population and all-time best/average vitals & fitness for a species.

    Shared by the rendering snapshot (lightweight population subset) and
    tracking_db (full stats write).
    """
    species_name = getattr(cls, "species_name", cls.__name__)
    living = world.creatures.get(cls, [])
    dead = world.dead_creatures.get(cls, [])
    max_pop = getattr(cls, "max_population", 100)
    alive_count = len(living)
    dead_count = len(dead)
    all_time_count = getattr(world, "all_time_counts", {}).get(cls, alive_count + dead_count)
    all_creatures = living + dead

    best_fitness = max(SpeciesStats.max_fitness.get(species_name, 0.0), max((c.compute_fitness() for c in living), default=0.0))
    best_food = max(SpeciesStats.max_foodeaten.get(species_name, 0), max((getattr(c, "food_eaten", 0) for c in living), default=0))
    best_computed_food = max(SpeciesStats.max_computed_food.get(species_name, 0.0), max((getattr(c, "computed_food_eaten", 0.0) for c in living), default=0.0))
    best_enemies = max(SpeciesStats.max_enemies_touched.get(species_name, 0), max((getattr(c, "enemies_touched", 0) for c in living), default=0))
    best_computed_enemies = max(SpeciesStats.max_computed_enemies.get(species_name, 0.0), max((getattr(c, "computed_enemies_touched", 0.0) for c in living), default=0.0))
    best_lifetime = max(SpeciesStats.max_lifetime.get(species_name, 0.0), max((getattr(c, "survival_time", 0.0) for c in living), default=0.0))

    total_dead_count = SpeciesStats.total_dead_count.get(species_name, 0)
    total_count = total_dead_count + len(living)

    if total_count > 0:
        avg_fitness = (SpeciesStats.sum_dead_fitness.get(species_name, 0.0) + sum(c.compute_fitness() for c in living)) / total_count
        avg_food = (SpeciesStats.sum_dead_food.get(species_name, 0.0) + sum(getattr(c, "food_eaten", 0) for c in living)) / total_count
        avg_computed_food = (SpeciesStats.sum_dead_computed_food.get(species_name, 0.0) + sum(getattr(c, "computed_food_eaten", 0.0) for c in living)) / total_count
        avg_enemies = (SpeciesStats.sum_dead_enemies.get(species_name, 0.0) + sum(getattr(c, "enemies_touched", 0) for c in living)) / total_count
        avg_computed_enemies = (SpeciesStats.sum_dead_computed_enemies.get(species_name, 0.0) + sum(getattr(c, "computed_enemies_touched", 0.0) for c in living)) / total_count
        avg_lifetime = (SpeciesStats.sum_dead_lifetime.get(species_name, 0.0) + sum(getattr(c, "survival_time", 0.0) for c in living)) / total_count
    elif all_creatures:
        avg_fitness = sum(c.compute_fitness() for c in all_creatures) / len(all_creatures)
        avg_food = sum(getattr(c, "food_eaten", 0) for c in all_creatures) / len(all_creatures)
        avg_computed_food = sum(getattr(c, "computed_food_eaten", 0.0) for c in all_creatures) / len(all_creatures)
        avg_enemies = sum(getattr(c, "enemies_touched", 0) for c in all_creatures) / len(all_creatures)
        avg_computed_enemies = sum(getattr(c, "computed_enemies_touched", 0.0) for c in all_creatures) / len(all_creatures)
        avg_lifetime = sum(getattr(c, "survival_time", 0.0) for c in all_creatures) / len(all_creatures)
    else:
        avg_fitness = 0.0
        avg_food = 0.0
        avg_computed_food = 0.0
        avg_enemies = 0.0
        avg_computed_enemies = 0.0
        avg_lifetime = 0.0

    best_tiles = max(SpeciesStats.max_metrics.get(species_name, {}).get("tiles_covered", 0.0), max((getattr(c, "tiles_covered", 0.0) for c in living), default=0.0))
    best_release = max(SpeciesStats.max_metrics.get(species_name, {}).get("release_at_home_count", 0), max((getattr(c, "release_at_home_count", 0) for c in living), default=0))

    if alive_count > 0:
        avg_tiles = sum(getattr(c, "tiles_covered", 0.0) for c in living) / alive_count
        avg_release = sum(getattr(c, "release_at_home_count", 0) for c in living) / alive_count
    else:
        avg_tiles = 0.0
        avg_release = 0.0
        
    from core.constants import SPECIES_CONFIG
    reproduction_mode = SPECIES_CONFIG.get(species_name, {}).get("reproduction_mode", "continuous").capitalize()

    return {
        "alive": alive_count,
        "maxPop": max_pop,
        "allTimeCount": all_time_count,
        "evolutionMode": reproduction_mode,
        "bestFitness": float(best_fitness),
        "avgFitness": float(avg_fitness),
        "bestFood": int(best_food),
        "bestComputedFood": float(best_computed_food),
        "avgFood": float(avg_food),
        "avgComputedFood": float(avg_computed_food),
        "bestEnemies": int(best_enemies),
        "bestComputedEnemies": float(best_computed_enemies),
        "avgEnemies": float(avg_enemies),
        "avgComputedEnemies": float(avg_computed_enemies),
        "bestLifetime": float(best_lifetime),
        "avgLifetime": float(avg_lifetime),
        "bestTilesCovered": float(best_tiles),
        "avgTilesCovered": float(avg_tiles),
        "bestReleaseAtHome": int(best_release),
        "avgReleaseAtHome": float(avg_release),
    }

def compute_metric_bounds(world: Any, cls: type) -> dict[str, dict[str, float]]:
    """Compute the actual max vs bound values for species metrics."""
    species_name = getattr(cls, "species_name", cls.__name__)
    bounds_table = getattr(cls, "metrics", {})
    peak_table = SpeciesStats.max_metrics.get(species_name, {})
    living = world.creatures.get(cls, [])
    dead = world.dead_creatures.get(cls, [])
    all_creatures = living + dead

    res = {}
    for k, bound_val in bounds_table.items():
        peak_val = peak_table.get(k, 0.0)
        curr_max = max((float(getattr(c, k, 0.0)) for c in all_creatures), default=0.0)
        res[k] = {
            "max": float(max(peak_val, curr_max)),
            "bound": float(bound_val)
        }
    return res


def build_full_snapshot(world: Any, simulation: Any, paused: bool) -> dict[str, Any]:
    """Build a full snapshot for rendering at normal speeds.

    Contains creature positions/direction/hp/action-flags, food items,
    pheromone grid, kingdoms, lakes, and lightweight population counts.
    Stats, training metrics, and metric bounds are served via HTTP from SQLite.
    """
    creatures_dict = {}
    top_fit_dict = {}
    population_dict = {}

    for cls in world.active_species:
        species_name = getattr(cls, "species_name", cls.__name__)
        living = world.creatures.get(cls, [])
        c_list = []
        for c in living:
            food_type = None
            carried_obj = getattr(c, "carried_object", None)
            if carried_obj is not None:
                food_type = getattr(carried_obj, "food_type", None)
            
            c_list.append({
                "x": round(float(c.position[0]), 2),
                "y": round(float(c.position[1]), 2),
                "dir": round(float(c.direction), 2),
                "hp": round(float(getattr(c, "health", 0.0)), 2),
                "maxHp": round(float(getattr(c, "max_health", 1.0)), 2),
                "alive": bool(getattr(c, "alive", False)),
                "eating": bool(getattr(c, "is_eating", False)),
                "attacking": bool(getattr(c, "is_attacking", False)),
                "carrying": carried_obj is not None,
                "carriedType": food_type,
                "radius": float(getattr(c, "radius", 2.0)),
            })
        creatures_dict[species_name] = c_list

        fit_scores = [(c.compute_fitness(), idx) for idx, c in enumerate(living)]
        fit_scores.sort(key=lambda x: x[0], reverse=True)
        top_fit_dict[species_name] = [idx for _, idx in fit_scores[:3]]

        # Lightweight population counts duplicated from SQLite for real-time display
        population_dict[species_name] = {
            "alive": len(living),
            "maxPop": getattr(cls, "max_population", 100),
        }

    food_list = []
    for f in world.food_items:
        if getattr(f, "consumed", False):
            continue
        carried = bool(getattr(f, "being_carried", False))
        food_list.append({
            "x": round(float(f.position[0]), 2),
            "y": round(float(f.position[1]), 2),
            "type": getattr(f, "food_type", "sugar"),
            "carried": carried,
        })

    food_sources_list = []
    for fs in getattr(world, "food_sources", []):
        food_sources_list.append({
            "x": round(float(fs.position[0]), 2),
            "y": round(float(fs.position[1]), 2),
            "radius": float(getattr(fs, "radius", 0.0)),
        })

    kingdoms_list = []
    for k_cls, k in world.kingdoms.items():
        k_sp_name = getattr(k_cls, "species_name", k_cls.__name__)
        kingdoms_list.append({
            "x": round(float(k.position[0]), 2),
            "y": round(float(k.position[1]), 2),
            "name": k.name,
            "species": k_sp_name,
            "spawnRadius": float(getattr(k, "spawn_radius", 0.0)),
        })

    lakes_list = []
    for l in world.lakes:
        lakes_list.append({
            "x": round(float(l.position[0]), 2),
            "y": round(float(l.position[1]), 2),
            "radius": float(getattr(l, "radius", 0.0)),
        })

    grid = world.pheromone_grid
    active_indices = np.where(grid > 0.02)
    ph_data = []
    for x, y in zip(active_indices[0], active_indices[1]):
        strength = float(grid[x, y])
        ph_data.append([int(x), int(y), round(strength, 2)])

    return {
        "type": "full",
        "time": world.round_time,
        "generation": int(world.round_time / GENERATION_DURATION) + 1,
        "speed": getattr(simulation, "actual_multiplier", 1.0),
        "targetMultiplier": getattr(simulation, "target_multiplier", 1.0),
        "actualMultiplier": getattr(simulation, "actual_multiplier", 1.0),
        "framesPerDt": FRAMES_PER_DT,
        "multiplierCapped": getattr(simulation, "multiplier_capped", False),
        "ultra": getattr(simulation, "ultra_mode", False),
        "paused": paused,
        "world": {"width": WORLD_WIDTH, "height": WORLD_HEIGHT},
        "creatures": creatures_dict,
        "food": food_list,
        "foodSources": food_sources_list,
        "kingdoms": kingdoms_list,
        "lakes": lakes_list,
        "pheromones": {
            "cellSize": float(getattr(world, "pheromone_cell_size", 10.0)),
            "width": grid.shape[0],
            "height": grid.shape[1],
            "data": ph_data,
        },
        "population": population_dict,
        "topFit": top_fit_dict,
    }


def build_aggregate_snapshot(world: Any, simulation: Any, paused: bool) -> dict[str, Any]:
    """Build an aggregate snapshot for fast simulation (ultra mode).

    Contains only metadata and population counts — no creature positions,
    food, or pheromones. Stats and training data are in SQLite.
    """
    population_dict = {}
    for cls in world.active_species:
        species_name = getattr(cls, "species_name", cls.__name__)
        living = world.creatures.get(cls, [])
        population_dict[species_name] = {
            "alive": len(living),
            "maxPop": getattr(cls, "max_population", 100),
        }

    return {
        "type": "aggregate",
        "time": world.round_time,
        "generation": int(world.round_time / GENERATION_DURATION) + 1,
        "speed": getattr(simulation, "actual_multiplier", 1.0),
        "targetMultiplier": getattr(simulation, "target_multiplier", 1.0),
        "actualMultiplier": getattr(simulation, "actual_multiplier", 1.0),
        "framesPerDt": FRAMES_PER_DT,
        "multiplierCapped": getattr(simulation, "multiplier_capped", False),
        "ultra": True,
        "paused": paused,
        "population": population_dict,
    }
