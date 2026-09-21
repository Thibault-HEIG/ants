import sqlite3
import pandas as pd
import seaborn # Moteur visuel basé sur matplotlib
import matplotlib.pyplot as plt

conn = sqlite3.connect("data/tracking.db") # Créer une connexion

statement = '''SELECT fitness, walking_carrying FROM creatures WHERE run_id = 27 ORDER BY fitness DESC;'''
df = pd.read_sql_query(statement, conn)

seaborn.scatterplot(x='fitness', y='walking_carrying', data=df)
plt.savefig(f"data/graphs/walking_carrying_27.png", dpi=300, bbox_inches="tight")

conn.close()