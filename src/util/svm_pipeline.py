from src.dataset.data import df_amostra

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

recursos_numericos = ['quilate', 'profundidade', 'largura do topo em relação à base', 'preço', 'x', 'y', 'z', 'volume', 'simetria', 'proporcao_altura']
recursos_categoricos = ['cor', 'pureza']
ordem_cores = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
ordem_purezas = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

df_processado = df_amostra.copy()

for col in recursos_numericos:
    df_processado[col] = pd.to_numeric(df_processado[col], errors='coerce')

for col in recursos_categoricos:
    df_processado[col] = df_processado[col].astype(str)

df_processado = df_processado.dropna(subset=recursos_numericos + recursos_categoricos + ['lapidação'])


X = df_processado[recursos_numericos + recursos_categoricos]
y = df_processado['lapidação']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


processamento = ColumnTransformer(transformers=[
    ('num', StandardScaler(), recursos_numericos),
    ('cat', OrdinalEncoder(categories=[ordem_cores, ordem_purezas]), recursos_categoricos)
])


base_pipeline = Pipeline([
    ('prepro', processamento),
    ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=42))
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.01, 0.1]
}

grid_search = GridSearchCV(base_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(X_test)

print("\n=== RESULTADOS DO MODELO SVM ===")
print(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")
print(f"Acurácia final obtida no teste: {accuracy_score(y_test, y_pred):.4f}")

print("\nRelatório de Classificação Completo:")
print(classification_report(y_test, y_pred))
