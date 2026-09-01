# Basic graph creation
```python

import sqlite3
import pandas as pd
import seaborn # Moteur visuel basé sur matplotlib
import matplotlib.pyplot as plt

conn = sqlite3.connect("data/tracking.db") # Créer une connexion

statement = '''SELECT fitness, lifetime, food_eaten, computed_food_eaten, times_eating_for_nothing, tiles_covered FROM creatures ORDER BY fitness DESC;'''
df = pd.read_sql_query(statement, conn)

plt.figure() # Obligatory in for loops

seaborn.scatterplot(x='col-name', y='col-name', data=df)
plt.savefig(f"data/graphs/file-name.png", dpi=300, bbox_inches="tight")

plt.close() # Obligatory in for loops

conn.close()

```