import streamlit as st

import pandas as pd
import numpy as np
from src.dataset.data import df_amostra

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt

st.title("Classificação SVM - Dataset Diamonds")

# Cache para não recarregar os dados a cada clique no Streamlit

# Pegando as opções únicas disponíveis na coluna lapidação
opcoes_disponiveis = df_amostra['lapidação'].unique()

# Componente de seleção para o usuário
selecao = st.multiselect(
    "Escolha exatamente duas opções de lapidação para comparar:",
    options=opcoes_disponiveis,
    max_selections=2
)

# Validação: o modelo só roda se houver exatamente 2 opções escolhidas
if len(selecao) == 2:
    
    # 1. Filtra o dataset com base nas escolhas do usuário
    df_filtrado = df_amostra[df_amostra['lapidação'].isin(selecao)]
    
    # 2. Amostragem Proporcional (Ex: 500 de cada tipo, ou o máximo possível se for menor)
    tamanho_minimo = df_filtrado['lapidação'].value_counts().min()
    n_amostras = min(500, tamanho_minimo)
    
    df_filtrado = df_filtrado.groupby('lapidação').sample(n=n_amostras, random_state=42)
        
    # 3. Usando as variáveis geométricas para ver o SVM separar bem as classes
    X = df_filtrado[['profundidade', 'largura do topo em relação à base']]
    y = df_filtrado['lapidação']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Treinamento com hiperparâmetros ajustados (C=10.0, gamma=1.0)
    with st.spinner('A treinar o modelo SVM...'):
        modelo_svm = SVC(kernel='rbf', C=10.0, gamma=1.0, random_state=42)
        modelo_svm.fit(X_train_scaled, y_train)

    y_pred = modelo_svm.predict(X_test_scaled)
    acuracia = accuracy_score(y_test, y_pred)
    
    st.success(f"**Acurácia do modelo:** {acuracia:.4f}")

    # 5. Plotagem do gráfico
    fig, ax = plt.subplots(figsize=(8,6))

    DecisionBoundaryDisplay.from_estimator(
        modelo_svm,
        X_test_scaled,
        response_method='predict',
        plot_method='contour',
        colors='black',
        linewidths=0.3,
        ax=ax
    )

    # Pontos mais visíveis e com transparência
    scatter = ax.scatter(
        X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, 
        cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.8, s=35
    )

    ax.set_xlabel('Profundidade')
    ax.set_ylabel('Largura do Topo em Relação à Base')
    ax.set_title(f"Fronteira de Decisão do SVM ({selecao[0]} vs {selecao[1]})")

    handles, _ = scatter.legend_elements()
    ax.legend(handles, le.classes_, title='Lapidação')


    # Exibe o gráfico no Streamlit
    st.pyplot(fig)

elif len(selecao) > 2:
    st.warning("Por favor, selecione apenas duas opções para a visualização da fronteira de decisão.")
else:
    st.info("A aguardar a seleção de duas categorias de lapidação...")


# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import balanced_accuracy_score, classification_report, ConfusionMatrixDisplay

# @st.cache_resource
# def treinar(selecao_tuple, n_max=3000):
#     df_f = df_diamonds[df_diamonds['lapidação'].isin(selecao_tuple)]
#     X = df_f[['profundidade', 'largura do topo em relação à base']]
#     y = LabelEncoder().fit(df_f['lapidação'])
#     le = y; y = le.transform(df_f['lapidação'])

#     # 1) Divide ANTES de qualquer balanceamento, preservando a proporção real
#     X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#     # 2) Reduz só o treino (por custo), mantendo a estratificação
#     if len(X_tr) > n_max:
#         X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=n_max, stratify=y_tr, random_state=42)

#     # 3) Pipeline evita vazamento do scaler; class_weight trata o desbalanceamento
#     modelo = make_pipeline(StandardScaler(),
#                            SVC(kernel='rbf', C=10.0, gamma=1.0, class_weight='balanced'))
#     modelo.fit(X_tr, y_tr)
#     return modelo, le, X_te, y_te

# modelo, le, X_te, y_te = treinar(tuple(sorted(selecao)))
# y_pred = modelo.predict(X_te)
# st.metric("Acurácia balanceada", f"{balanced_accuracy_score(y_te, y_pred):.3f}")
# st.text(classification_report(y_te, y_pred, target_names=le.classes_))