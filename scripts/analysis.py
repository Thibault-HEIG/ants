import sqlite3
import pandas as pd
import seaborn # Moteur visuel basé sur matplotlib
import matplotlib.pyplot as plt

conn = sqlite3.connect("data/tracking.db") # Créer une connexion

statement = '''SELECT fitness, release_at_home_count, walking_carrying FROM creatures WHERE run_id = 14 ORDER BY fitness DESC;'''
df = pd.read_sql_query(statement, conn)

plt.figure() # Obligatory in for loops

seaborn.scatterplot(x='fitness', y='walking_carrying', data=df)
plt.savefig(f"data/graphs/walking_carrying.png", dpi=300, bbox_inches="tight")

plt.close() # Obligatory in for loops

conn.close()