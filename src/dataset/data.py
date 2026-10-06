import pandas as pd
import numpy as np

from seaborn import load_dataset

from sklearn.model_selection import train_test_split


df_amostra = load_dataset('diamonds')

#TRADUÇÃO

df_amostra = df_amostra.rename(columns={'carat': 'quilate'})
df_amostra = df_amostra.rename(columns={'cut': 'lapidação'})
df_amostra = df_amostra.rename(columns={'color': 'cor'})
df_amostra = df_amostra.rename(columns={'clarity': 'pureza'})
df_amostra = df_amostra.rename(columns={'depth': 'profundidade'})
df_amostra = df_amostra.rename(columns={'table': 'largura do topo em relação à base'})
df_amostra = df_amostra.rename(columns={'price': 'preço'})

condicoes = [df_amostra['lapidação'] == item for item in ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']]
df_amostra['lapidação'] = np.select(condicoes, ['Ideal', 'Premium', 'Muito Boa', 'Boa', 'Regular'], default=df_amostra['lapidação'])

#FEATURES

epsilon = 1e-5
df_amostra['volume'] = df_amostra['x'] * df_amostra['y'] * df_amostra['z']
df_amostra['simetria'] = df_amostra['x'] / (df_amostra['y'] + epsilon)
df_amostra['proporcao_altura'] = df_amostra['z'] / (df_amostra['x'] + epsilon)

#AMOSTRA

_, df_amostra = train_test_split(
    df_amostra, test_size=5000, stratify=df_amostra['lapidação'], random_state=42
)