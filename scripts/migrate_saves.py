#!/usr/bin/env python3
"""
migrate_saves.py — Migrate save files from 6-output to 7-output brain architecture
====================================================================================

Adds the 7th output neuron ("make") weights and bias to all genomes in save files.
The new weights are initialised to small random values so the new "make" action
starts near-neutral.

Uses standard library only.
"""

import json
import os
import sys
import glob
import shutil
import random

OLD_NN_OUTPUTS = 6
NEW_NN_OUTPUTS = 7
NN_HIDDEN_2 = 8
OLD_GENOME_SIZE = 1598
NEW_GENOME_SIZE = 1607

# Position where the output layer starts in the flat genome
OUTPUT_LAYER_START = OLD_GENOME_SIZE - (NN_HIDDEN_2 * OLD_NN_OUTPUTS + OLD_NN_OUTPUTS)  # = 1544


def migrate_genome(genome: list[float]) -> list[float]:
    """Migrate a single genome from 6 outputs to 7 outputs."""
    if len(genome) == NEW_GENOME_SIZE:
        return genome  # Already migrated
    if len(genome) != OLD_GENOME_SIZE:
        print(f"  [WARN] Unexpected genome size {len(genome)}, skipping")
        return genome

    # Extract parts
    prefix = genome[:OUTPUT_LAYER_START]  # Everything before output layer

    old_weights = genome[OUTPUT_LAYER_START:OUTPUT_LAYER_START + NN_HIDDEN_2 * OLD_NN_OUTPUTS]
    old_biases = genome[OUTPUT_LAYER_START + NN_HIDDEN_2 * OLD_NN_OUTPUTS:]

    # Reconstruct the hidden2->output weight matrix and add the 7th output
    new_weights = []
    # old_weights is flat, representing an (8 x 6) matrix.
    # We want to add a 7th column, so it becomes (8 x 7).
    for i in range(NN_HIDDEN_2):
        row = old_weights[i * OLD_NN_OUTPUTS : (i + 1) * OLD_NN_OUTPUTS]
        # Append random weight for the 7th output (mean=0.0, std=0.1 approx)
        row.append(random.gauss(0.0, 0.1))
        new_weights.extend(row)

    new_biases = list(old_biases)
    new_biases.append(random.gauss(0.0, 0.1))

    # Reassemble
    new_genome = prefix + new_weights + new_biases
    assert len(new_genome) == NEW_GENOME_SIZE, f"Expected {NEW_GENOME_SIZE}, got {len(new_genome)}"

    return new_genome


def migrate_save_file(filepath: str) -> bool:
    """Migrate a single save file in-place (with backup)."""
    print(f"\n[MIGRATE] Processing: {filepath}")

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [ERROR] Failed to read: {exc}")
        return False

    if not isinstance(data, dict) or "species" not in data:
        print(f"  [SKIP] Not a valid save file (no 'species' key)")
        return False

    migrated_any = False
    for sp_name, sp_data in data["species"].items():
        genomes = sp_data.get("genomes", [])
        if not genomes:
            continue

        # Check if already migrated
        first_size = len(genomes[0]) if genomes else 0
        if first_size == NEW_GENOME_SIZE:
            print(f"  [OK] {sp_name}: already at new genome size ({NEW_GENOME_SIZE})")
            continue
        if first_size != OLD_GENOME_SIZE:
            print(f"  [WARN] {sp_name}: unexpected genome size {first_size}, skipping")
            continue

        print(f"  [MIGRATING] {sp_name}: {len(genomes)} genomes ({OLD_GENOME_SIZE} → {NEW_GENOME_SIZE})")
        new_genomes = [migrate_genome(g) for g in genomes]
        sp_data["genomes"] = new_genomes
        migrated_any = True

    if not migrated_any:
        print(f"  [SKIP] No genomes needed migration")
        return False

    # Create backup
    backup_path = filepath + ".bak"
    shutil.copy2(filepath, backup_path)
    print(f"  [BACKUP] Created: {backup_path}")

    # Write migrated file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [DONE] Migrated successfully")
    return True


def main():
    saves_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saves")
    if not os.path.isdir(saves_dir):
        print(f"[ERROR] saves/ directory not found at {saves_dir}")
        sys.exit(1)

    save_files = sorted(glob.glob(os.path.join(saves_dir, "*.json")))
    if not save_files:
        print("[INFO] No save files found")
        sys.exit(0)

    print(f"Found {len(save_files)} save file(s) in {saves_dir}")
    print(f"Migration: {OLD_GENOME_SIZE} → {NEW_GENOME_SIZE} (+{NEW_GENOME_SIZE - OLD_GENOME_SIZE} weights for 7th output)")

    random.seed(42)
    migrated = 0
    for filepath in save_files:
        if migrate_save_file(filepath):
            migrated += 1

    print(f"\n{'='*60}")
    print(f"Migration complete: {migrated}/{len(save_files)} files migrated")
    print(f"Backups saved as *.json.bak")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
