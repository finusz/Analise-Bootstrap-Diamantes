import pandas as pd
import numpy as np
from seaborn import load_dataset

df_diamonds = load_dataset('diamonds')

df_diamonds = df_diamonds.rename(columns={'carat': 'quilate'})
df_diamonds = df_diamonds.rename(columns={'cut': 'lapidação'})
df_diamonds = df_diamonds.rename(columns={'color': 'cor'})
df_diamonds = df_diamonds.rename(columns={'clarity': 'pureza'})
df_diamonds = df_diamonds.rename(columns={'depth': 'profundidade'})
df_diamonds = df_diamonds.rename(columns={'table': 'largura do topo em relação à base'})
df_diamonds = df_diamonds.rename(columns={'price': 'preço'})

condicoes = [df_diamonds['lapidação'] == item for item in ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']]
df_diamonds['lapidação'] = np.select(condicoes, ['Ideal', 'Premium', 'Muito Boa', 'Boa', 'Regular'], default=df_diamonds['lapidação'])
