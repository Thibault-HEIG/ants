"""
serialization.py — Snapshot Builder
===================================

Builds full and aggregate simulation snapshots to decouple game logic
from the rendering loop and network transmission.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from core.constants import GENERATION_DURATION, WORLD_HEIGHT, WORLD_WIDTH
from core.utils import SpeciesStats


def _json_encode(snapshot: dict[str, Any]) -> str:
    """Encode a snapshot dictionary to a JSON string."""
    return json.dumps(snapshot, separators=(',', ':'))


# Swap point: change this single reference to switch encoding (e.g., to msgpack)
encode = _json_encode


def _compute_species_stats(world: Any, cls: type) -> dict[str, Any]:
    """Compute living/dead population and all-time best/average vitals & fitness for a species."""
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
        
    return {
        "alive": alive_count,
        "maxPop": max_pop,
        "allTimeCount": all_time_count,
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
    }


def _compute_metric_bounds(world: Any, cls: type) -> dict[str, dict[str, float]]:
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
    """Build a full snapshot for rendering at normal speeds."""
    creatures_dict = {}
    top_fit_dict = {}
    stats_dict = {}
    bounds_dict = {}

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
                "fitness": round(float(c.compute_fitness()), 4),
                "radius": float(getattr(c, "radius", 2.0)),
            })
        creatures_dict[species_name] = c_list

        fit_scores = [(c_data["fitness"], idx) for idx, c_data in enumerate(c_list)]
        fit_scores.sort(key=lambda x: x[0], reverse=True)
        top_fit_dict[species_name] = [idx for _, idx in fit_scores[:3]]

        stats_dict[species_name] = _compute_species_stats(world, cls)
        bounds_dict[species_name] = _compute_metric_bounds(world, cls)

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
    active_indices = np.where(grid > 0.01)
    ph_data = []
    for x, y in zip(active_indices[0], active_indices[1]):
        strength = float(grid[x, y])
        ph_data.append([int(x), int(y), round(strength, 2)])
    ph_data.sort(key=lambda x: x[2], reverse=True)
    ph_data = ph_data[:500]

    return {
        "type": "full",
        "time": world.round_time,
        "generation": int(world.round_time / GENERATION_DURATION) + 1,
        "speed": getattr(simulation, "speed_multiplier", 1.0),
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
        "stats": stats_dict,
        "topFit": top_fit_dict,
        "metricBounds": bounds_dict,
    }


def build_aggregate_snapshot(world: Any, simulation: Any, paused: bool) -> dict[str, Any]:
    """Build an aggregate snapshot for fast simulation (ultra mode)."""
    stats_dict = {}
    bounds_dict = {}

    for cls in world.active_species:
        species_name = getattr(cls, "species_name", cls.__name__)
        stats_dict[species_name] = _compute_species_stats(world, cls)
        bounds_dict[species_name] = _compute_metric_bounds(world, cls)

    ant_best = stats_dict.get("Ant", {}).get("bestFitness", 0.0)
    ant_avg = stats_dict.get("Ant", {}).get("avgFitness", 0.0)
    spider_best = stats_dict.get("Spider", {}).get("bestFitness", 0.0)
    spider_avg = stats_dict.get("Spider", {}).get("avgFitness", 0.0)

    return {
        "type": "aggregate",
        "time": world.round_time,
        "generation": int(world.round_time / GENERATION_DURATION) + 1,
        "speed": getattr(simulation, "speed_multiplier", 1.0),
        "ultra": True,
        "paused": paused,
        "stats": stats_dict,
        "metricBounds": bounds_dict,
        "chart": {
            "time": world.round_time,
            "antBest": ant_best,
            "antAvg": ant_avg,
            "spiderBest": spider_best,
            "spiderAvg": spider_avg,
        },
    }
