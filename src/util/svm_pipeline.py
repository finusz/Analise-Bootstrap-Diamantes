from src.dataset.data import df_amostra

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

recursos_numericos = ['quilate', 'profundidade', 'largura do topo em relação à base', 'preço', 'x', 'y', 'z', 'volume', 'simetria', 'proporcao_altura']
recursos_categoricos = ['cor', 'pureza']

# --- GARANTIA DE TIPOS DE DADOS (Resolve o erro da String de vez) ---
df_processado = df_amostra.copy()

# Força todas as colunas numéricas a serem estritamente floats (remove qualquer texto residual)
for col in recursos_numericos:
    df_processado[col] = pd.to_numeric(df_processado[col], errors='coerce')

# Força as colunas categóricas a serem estritamente strings
for col in recursos_categoricos:
    df_processado[col] = df_processado[col].astype(str)

# Remove linhas com valores nulos caso a conversão force algum erro ('coerce')
df_processado = df_processado.dropna(subset=recursos_numericos + recursos_categoricos + ['lapidação'])
# ---------------------------------------------------------------------

X = df_processado[recursos_numericos + recursos_categoricos]
y = df_processado['lapidação']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

processamento = ColumnTransformer(transformers=[
    ('num', StandardScaler(), recursos_numericos),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), recursos_categoricos)
])

base_pipeline = Pipeline([
    ('prepro', processamento),
    ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=42))
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.01, 0.1]
}

print("Executando o grid search com dados convertidos com segurança...")
grid_search = GridSearchCV(base_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(X_test)

print("\n=== RESULTADOS DO MODELO SVM ===")
print(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")
print(f"Acurácia final obtida no teste: {accuracy_score(y_test, y_pred):.4f}")

print("\nRelatório de Classificação Completo:")
print(classification_report(y_test, y_pred))
