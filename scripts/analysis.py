import sqlite3
import pandas as pd
import seaborn # Moteur visuel basé sur matplotlib
import matplotlib.pyplot as plt

conn = sqlite3.connect("data/tracking.db") # Créer une connexion

statement = '''SELECT fitness, release_at_home_count, walk_in_home_direction, walk_in_opposite_direction FROM creatures WHERE run_id = 14 ORDER BY fitness DESC;'''
df = pd.read_sql_query(statement, conn)

plt.figure() # Obligatory in for loops

seaborn.scatterplot(x='fitness', y='walk_in_opposite_direction', data=df)
plt.savefig(f"data/graphs/walking2.png", dpi=300, bbox_inches="tight")

plt.close() # Obligatory in for loops

conn.close()