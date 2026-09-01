import sqlite3
import pandas as pd
import seaborn # Moteur visuel basé sur matplotlib
import matplotlib.pyplot as plt

conn = sqlite3.connect("data/tracking.db") # Créer une connexion

statement = '''SELECT fitness, lifetime, food_eaten, computed_food_eaten, times_eating_for_nothing, tiles_covered FROM creatures ORDER BY fitness DESC;'''
df = pd.read_sql_query(statement, conn)

plots = [
    ("fitness", "food_eaten"),
    ("fitness", "times_eating_for_nothing"),
    ("fitness", "tiles_covered"),
    ("fitness", "lifetime")
]

for i, (x, y) in enumerate(plots):

    plt.figure()
    seaborn.scatterplot(x=x, y=y, data=df)

    plt.savefig(f"data/graphs/plot-{i}.png", dpi=300, bbox_inches="tight")

    plt.close()

conn.close()