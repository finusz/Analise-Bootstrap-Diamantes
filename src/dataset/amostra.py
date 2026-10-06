from data import df_diamonds

import pandas as pd

from sklearn.model_selection import train_test_split

# O parâmetro stratify mantém a proporção da coluna informada
_, df_amostra = train_test_split(
    df_diamonds, test_size=5000, stratify=df_diamonds['lapidação'], random_state=42
)

# Proporção no dataset original (50 mil)
print("--- Original ---")
print(df_diamonds['lapidação'].value_counts(normalize=True) * 100)

# Proporção na amostra (5 mil)
print("\n--- Amostra ---")
print(df_amostra['lapidação'].value_counts(normalize=True) * 100)