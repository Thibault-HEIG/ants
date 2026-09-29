import sqlite3
import pandas as pd
import scipy.stats as stats

conn = sqlite3.connect("data/tracking.db") # Créer une connexion

statement = '''SELECT * FROM creatures WHERE run_id = 21 ORDER BY fitness DESC;'''
df = pd.read_sql_query(statement, conn)

df.drop(columns=['run_id', 'species_name', 'enemies_touched', 'computed_enemies_touched', 'times_attacking_for_nothing'], inplace=True)

for column in df.columns:
    pearson_coef, p_value = stats.pearsonr(df[column], df['fitness'])
    print(f"{column} : coef = {pearson_coef:.4f}, p-value = {p_value:.4f}")
    print("-"*50)
    
conn.close()