"""Delete a run and its associated data from the tracking database."""

import sqlite3
import os
import sys

# ---- Set the run ID to delete here ----
run_id = 42
# ----------------------------------------

db_path = os.path.join(os.path.dirname(__file__), "..", "data", "tracking.db")
db_path = os.path.normpath(db_path)

if not os.path.exists(db_path):
    print(f"Database not found: {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

tables = ["runs", "snapshots", "creatures", "metric_bounds", "genomes"]
for table in tables:
    cursor.execute(f"DELETE FROM {table} WHERE {'id' if table == 'runs' else 'run_id'} = ?", (run_id,))
    print(f"  {table}: {cursor.rowcount} rows deleted")

# Orphan cleanup
for table in ["training_metrics", "live_stats"]:
    cursor.execute(f"DELETE FROM {table} WHERE snapshot_id NOT IN (SELECT id FROM snapshots)")
    print(f"  {table} (orphans): {cursor.rowcount} rows deleted")

conn.commit()
conn.close()
print(f"\nRun {run_id} deleted successfully.")
