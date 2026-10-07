from src.dataset.data import df_amostra
from src.util.graficos import hists
from src.util.calculos import bootstrap, confidence_interval
from src.models.svm_pipeline import treinar_svm

import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

import streamlit as st

import pandas as pd
import numpy as np

FEATURES = ['profundidade', 'largura do topo em relação à base']
PARAM_GRID = {'svm__C': [10.0], 'svm__gamma': [1.0]}   # mesmos valores fixos de antes


st.title('Comparação entre lapidações diferentes', text_alignment='center')
st.divider()

st.markdown('**Para as comparações está sendo utilizada uma amostra de 10% da população, o que representa cerca de 5.000 registros.**')

opcoes = st.multiselect("Selecione quais lapidações você deseja comparar:", df_amostra['lapidação'].unique(),max_selections=2)
filtro = df_amostra['lapidação'].isin(opcoes)

st.markdown(
    """
    <style>
    /* 1. Texto branco nos blocos de código (fundo marrom vem do config.toml) */
        pre code, pre code span {
                color: white !important;
            }

    /* 2. Removemos completamente qualquer cor de fundo padrão e filtros das camadas internas */
    div[data-testid="stAlert"] > div,
    div[data-testid="stAlert"] [role="alert"] {
        background-color: transparent !important;
        background-image: none !important;
    }

    /* 3. Aplicamos a cor pura diretamente no container principal */
    div[data-testid="stAlert"] {
        background-color: #6d5b4f !important; /* Insira aqui a cor EXATA do seu multiselect */
        border: none !important;
        border-radius: 18px !important;
    }
    
    /* 4. Mantém o texto e o ícone na cor branca */
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] svg {
        color: white !important;
        fill: white !important;
        font-weight: bold !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


if len(opcoes) < 2:
    st.info("É preciso selecionar exatamente 2 lapidações para exibir a comparação.")
    st.stop()

#BOOTSTRAP
st.divider()
st.subheader('Reamostragem: Bootstrap',text_alignment='center')

prices1 = df_amostra[df_amostra['lapidação'] == opcoes[0]]['preço']
prices2 = df_amostra[df_amostra['lapidação'] == opcoes[1]]['preço']

diff, mean1, mean2 = bootstrap(prices1, prices2)

fig = hists(diff, mean1, mean2,
            f'Proporção da diferença média:\nLapidação {opcoes[0]} x {opcoes[1]}',
            f'Proporção das médias:\nLapidação {opcoes[0]} x {opcoes[1]}',
            f'Diferença ({opcoes[0]} - {opcoes[1]})', [f'{opcoes[0]}', f'{opcoes[1]}'])

st.pyplot(fig)


means_prices = [mean1, mean2]
limit_inf, limit_sup = confidence_interval(means_prices)

if limit_inf > 0:
    st.markdown(f"Com 95% de confiança, a lapidação **{opcoes[0]}** é, em média, entre {limit_inf:.2f} reais e {limit_sup:.2f} reais **mais cara** que a {opcoes[1]}.")
elif limit_sup < 0:
    st.markdown(f"Com 95% de confiança, a lapidação **{opcoes[0]}** é, em média, entre {abs(limit_sup):.2f} reais e {abs(limit_inf):.2f} reais **mais barata** que a {opcoes[1]}.")
else:
    st.markdown(f"Não há diferença estatisticamente significativa entre os preços médios das lapidações **{opcoes[0]}** e **{opcoes[1]}** (o intervalo inclui zero).")

#SVM
st.divider()
st.subheader('Support Vector Machine: Classificação',text_alignment='center')

@st.cache_resource(show_spinner="Treinando o modelo SVM...")
def obter_resultado(df):          # classes como tupla (hashable)
    return treinar_svm(
        df,
        recursos_numericos=FEATURES,
        recursos_categoricos=[],       # só 2 variáveis para poder plotar em 2D
        param_grid=PARAM_GRID,
        n_por_classe=500,
    )

res = obter_resultado(df_amostra[filtro])

modelo = res['modelo']
X_test = res['X_test']
y_test = res['y_test']

st.success(f"**Acurácia do modelo:** {(res['acuracia'])*100:.2f}%")

# --- Gráfico ---
fig, ax = plt.subplots(figsize=(8, 6))

DecisionBoundaryDisplay.from_estimator(
    modelo,
    X_test,
    response_method='predict',
    plot_method='contour',
    colors='black',
    linewidths=0.3,
    ax=ax,
)

scatter = ax.scatter(
    X_test.iloc[:, 0], X_test.iloc[:, 1],
    c=(y_test == modelo.classes_[1]).astype(int),
    cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.8, s=35,
)

ax.set_xlabel('Profundidade')
ax.set_ylabel('Largura do Topo em Relação à Base')
ax.set_title(f"Fronteira de Decisão do SVM ({modelo.classes_[0]} vs {modelo.classes_[1]})")

handles, _ = scatter.legend_elements()
ax.legend(handles, modelo.classes_, title='Lapidação')

st.pyplot(fig)

with st.expander("Ver relatório de classificação"):
    st.code(res['relatorio'], language="text")