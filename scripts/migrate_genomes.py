#!/usr/bin/env python3
"""
migrate_genomes.py — Genome Save Architecture Migrator
======================================================

Converts existing .json save files from the old neural network architecture
(1468 weights/biases: 80 inputs -> 16 hidden -> 8 hidden -> 4 outputs)
to the new carry-object architecture:
(1598 weights/biases: 87 inputs -> 16 hidden -> 8 hidden -> 6 outputs).

For existing weights/biases, the original values are preserved verbatim.
For the new input connections (7 carry/home sensor inputs) and new output
connections (2 motor outputs: take and release), new weights and biases are
initialized with random values (`standard_normal * 0.5`), matching the original
network initialization distribution.
"""

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Any

import numpy as np

# Ensure root project directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from core.constants import GENOME_SIZE as NEW_GENOME_SIZE
except ImportError:
    NEW_GENOME_SIZE = 1598

OLD_GENOME_SIZE = 1468

# Old network dimensions
OLD_INPUTS = 80
OLD_HIDDEN_1 = 16
OLD_HIDDEN_2 = 8
OLD_OUTPUTS = 4

# New network dimensions
NEW_INPUTS = 87
NEW_HIDDEN_1 = 16
NEW_HIDDEN_2 = 8
NEW_OUTPUTS = 6


def upgrade_genome(old_genome: list[float] | np.ndarray, rng: np.random.Generator) -> list[float]:
    """Convert a 1468-float old genome vector into a 1598-float new genome vector."""
    arr = np.array(old_genome, dtype=float)
    if arr.shape != (OLD_GENOME_SIZE,):
        raise ValueError(f"Expected genome of length {OLD_GENOME_SIZE}, got {arr.shape}")

    idx = 0

    # 1. Input -> Hidden 1
    size_ih1_old = OLD_INPUTS * OLD_HIDDEN_1
    old_w_ih1 = arr[idx:idx + size_ih1_old].reshape(OLD_INPUTS, OLD_HIDDEN_1)
    idx += size_ih1_old

    old_b_h1 = arr[idx:idx + OLD_HIDDEN_1].copy()
    idx += OLD_HIDDEN_1

    # 2. Hidden 1 -> Hidden 2
    size_h1h2 = OLD_HIDDEN_1 * OLD_HIDDEN_2
    old_w_h1h2 = arr[idx:idx + size_h1h2].reshape(OLD_HIDDEN_1, OLD_HIDDEN_2)
    idx += size_h1h2

    old_b_h2 = arr[idx:idx + OLD_HIDDEN_2].copy()
    idx += OLD_HIDDEN_2

    # 3. Hidden 2 -> Output
    size_h2o_old = OLD_HIDDEN_2 * OLD_OUTPUTS
    old_w_h2o = arr[idx:idx + size_h2o_old].reshape(OLD_HIDDEN_2, OLD_OUTPUTS)
    idx += size_h2o_old

    old_b_o = arr[idx:idx + OLD_OUTPUTS].copy()
    idx += OLD_OUTPUTS

    # --- Construct new weight matrices and biases ---

    # New Input -> Hidden 1 (87 x 16)
    new_w_ih1 = np.empty((NEW_INPUTS, NEW_HIDDEN_1), dtype=float)
    new_w_ih1[:OLD_INPUTS, :] = old_w_ih1
    new_w_ih1[OLD_INPUTS:, :] = rng.standard_normal((NEW_INPUTS - OLD_INPUTS, NEW_HIDDEN_1)) * 0.5

    # New Hidden 1 biases (16) — kept exactly intact
    new_b_h1 = old_b_h1

    # New Hidden 1 -> Hidden 2 (16 x 8) — kept exactly intact
    new_w_h1h2 = old_w_h1h2

    # New Hidden 2 biases (8) — kept exactly intact
    new_b_h2 = old_b_h2

    # New Hidden 2 -> Output (8 x 6)
    new_w_h2o = np.empty((NEW_HIDDEN_2, NEW_OUTPUTS), dtype=float)
    new_w_h2o[:, :OLD_OUTPUTS] = old_w_h2o
    new_w_h2o[:, OLD_OUTPUTS:] = rng.standard_normal((NEW_HIDDEN_2, NEW_OUTPUTS - OLD_OUTPUTS)) * 0.5

    # New Output biases (6)
    new_b_o = np.empty(NEW_OUTPUTS, dtype=float)
    new_b_o[:OLD_OUTPUTS] = old_b_o
    new_b_o[OLD_OUTPUTS:] = rng.standard_normal(NEW_OUTPUTS - OLD_OUTPUTS) * 0.5

    # Flatten all parts
    new_genome = np.concatenate([
        new_w_ih1.flatten(),
        new_b_h1.flatten(),
        new_w_h1h2.flatten(),
        new_b_h2.flatten(),
        new_w_h2o.flatten(),
        new_b_o.flatten(),
    ])

    assert len(new_genome) == NEW_GENOME_SIZE, f"Migration produced length {len(new_genome)}, expected {NEW_GENOME_SIZE}"
    return [float(x) for x in new_genome]


def migrate_file(filepath: str, rng: np.random.Generator) -> tuple[int, int]:
    """Migrate all genomes inside a single save file.

    Returns (upgraded_count, skipped_count).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    upgraded_count = 0
    skipped_count = 0

    for item in data:
        if not isinstance(item, dict) or "genome" not in item:
            continue

        genome = item["genome"]
        if not isinstance(genome, list):
            continue

        if len(genome) == OLD_GENOME_SIZE:
            item["genome"] = upgrade_genome(genome, rng)
            upgraded_count += 1
        elif len(genome) == NEW_GENOME_SIZE:
            skipped_count += 1
        else:
            print(f"  [WARNING] Unknown genome size {len(genome)} in {filepath}, skipping item.")

    if upgraded_count > 0:

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return upgraded_count, skipped_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate .json genome save files from old 1468-float architecture to new 1598-float architecture."
    )
    parser.add_argument(
        "--saves-dir",
        type=str,
        default="saves",
        help="Path to directory containing .json saves (default: saves/)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process a specific .json save file instead of a directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible weight/bias generation",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup files when modifying .json saves",
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.saves_dir, "*.json")))

    if not files:
        print(f"No .json files found in {args.saves_dir}")
        return

    print(f"Checking {len(files)} save file(s)...")
    total_upgraded = 0
    total_skipped = 0

    for filepath in files:
        print(f"Processing: {filepath}")
        try:
            upgraded, skipped = migrate_file(filepath, rng, backup=not args.no_backup)
            total_upgraded += upgraded
            total_skipped += skipped
            if upgraded > 0:
                print(f"  SUCCESS: Upgraded {upgraded} genome(s) ({skipped} already up to date).")
            elif skipped > 0:
                print(f"  INFO: All {skipped} genome(s) already up to date ({NEW_GENOME_SIZE} floats).")
            else:
                print("  INFO: No genomes found in file.")
        except Exception as e:
            print(f"  [ERROR] Failed to process {filepath}: {e}")

    print("\n--- Migration Summary ---")
    print(f"Total genomes upgraded : {total_upgraded}")
    print(f"Total genomes skipped  : {total_skipped}")


if __name__ == "__main__":
    main()
