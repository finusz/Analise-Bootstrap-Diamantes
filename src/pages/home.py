import streamlit as st
from io import StringIO

import seaborn as sns
import matplotlib.pyplot as plt

from src.models.svm_pipeline import treinar_svm
from src.dataset.data import df_amostra
from src.util.graficos import *

from scipy import stats
import pandas as pd
import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay

FEATURES = ['profundidade', 'largura do topo em relação à base']
PARAM_GRID = {'svm__C': [10.0], 'svm__gamma': [1.0]}

st.title('Analise Exploratória de Diamantes', text_alignment='center')
st.subheader('Felipe Nunes e Heloísa Azevedo', text_alignment='center')
st.divider()
st.markdown('A base de dados utilizada é nativa da biblioteca do Seaborn. A análise tem como objetivo estimar a média de preço de diamentes com diferentes lapidações, afim de entender qual é a mais lucrativa de ser feita.')

st.divider()
st.subheader('Base de Dados de Diamantes', text_alignment='center')
st.dataframe(df_amostra)

st.divider()
buffer = StringIO()
df_amostra.info(buf=buffer)
st.subheader("Informações do dataset", text_alignment='center')
st.markdown(
    """
    <style>
    pre code {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.code(buffer.getvalue(), language="text")
st.markdown('A base contém ao todo 53940 registros e 10 característica, não possuindo valores nulos. Os tipos dos dados se dividem em: categórico, inteiros e flutuantes.')

st.divider()
st.subheader('Descrição dos dados', text_alignment='center')
st.dataframe(df_amostra.describe())

st.divider()
st.subheader('Correlação entre os dados', text_alignment='center')
corr = df_amostra.corr(numeric_only=True)
fig_corr, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)
st.markdown('Como esperado, o preço do diamante está muito ligado ao quilate dele.')

st.divider()
st.subheader('Comparação do preço entre os tipos de lapidações', text_alignment='center')
fig_bar = bar_diamonts()
st.pyplot(fig_bar)

st.divider()
st.subheader('Proporção de cada tipo de lapidação existente', text_alignment='center')
fig_pizza = proportion_category_diamonds()
st.pyplot(fig_pizza)
st.markdown(''' 
Dos diamantes apresentados, são dividios em 5 lapidações diferentes: 
  * **Ideal:** 21551 (40.0%);
  * **Premium:** 13791 (25.6%);
  * **Muito Boa:** 12082 (22.4%);
  * **Boa:** 4906 (9.1%);
  * **Regular:** 1610 (3.0%).
  ''')

st.divider()
st.subheader('Proporção dos dados',text_alignment='center')
fig_hist, ax = plt.subplots()
df_amostra.hist(ax=ax) 
st.pyplot(fig_hist)
st.write('')
st.subheader('Distribuição dos preços entre as lapidações', text_alignment='center')
fig_boxplot, ax = plt.subplots()
sns.boxplot(data=df_amostra, x='lapidação', y='preço', hue='lapidação', palette=["#6d5b4f", "#8d7b6f", "#ad9b8f", "#cdbbb0", "#ede0d4"], legend=False, ax=ax)
st.pyplot(fig_boxplot)

st.divider()
st.subheader('Boostrap inicial dos diamantes em relação ao preço', text_alignment='center')
st.markdown('Foi realizado o boostrap de 5000 médias do preço dos diamante')
fig_histMean, price_means = hist_means()
st.pyplot(fig_histMean)

st.markdown(r'h0: média geral do preço = 3900')
st.markdown('h1: média geral do preço != 3900')
st.markdown(r'Tendo em vista 95% de confiança')

h0_value = 3900
t_statistic, p_value = stats.ttest_1samp(price_means, h0_value)
st.markdown(f'O p-valor foi de: {p_value}\n')
if p_value > 0.05:
   st.markdown(f'Logo, aceita H0, com 95% de confiança podemos dizer que a média dos preços seja igual a {h0_value}')
else:
   st.markdown(f'Logo, rejeita H0, com 95% de confiança não podemos dizer que a média dos preços seja igual a {h0_value}')

st.divider()
st.subheader('Classificação geral utilizando SVM', text_alignment='center')

@st.cache_resource(show_spinner="Treinando o modelo SVM...")
def obter_resultado():          # classes como tupla (hashable)
    return treinar_svm(
        df_amostra,
        recursos_numericos=FEATURES,
        recursos_categoricos=[],       # só 2 variáveis para poder plotar em 2D
        param_grid=PARAM_GRID,
        n_classes=df_amostra['lapidação'].nunique()
    )

res = obter_resultado()

modelo = res['modelo']
X_test = res['X_test']
y_test = res['y_test']

st.success(f"**Acurácia do modelo:** {(res['acuracia'])*100:.2f}%")

X_plot = res.get(
    'X', X_test
) 
y_plot = res.get('y', y_test)

y_encoded = y_plot.astype('category').cat.codes

# --- Gráfico ---
fig, ax = plt.subplots(figsize=(8, 6))

DecisionBoundaryDisplay.from_estimator(
    modelo,
    X_plot,
    response_method='predict',
    plot_method='contourf', 
    cmap=plt.cm.coolwarm,
    alpha=0.3,
    ax=ax,
)

scatter = ax.scatter(
    X_plot.iloc[:, 0],
    X_plot.iloc[:, 1],
    c=y_encoded,  
    cmap=plt.cm.coolwarm,
    edgecolors='k',
    linewidths=0.5,
    alpha=0.8,
    s=30,
)

ax.set_xlabel('Profundidade')
ax.set_ylabel('Largura do Topo em Relação à Base')
ax.set_title('Fronteira de Decisão do SVM (Todas as Amostras)')

handles, _ = scatter.legend_elements()
ax.legend(handles, modelo.classes_, title='Lapidação', loc='best')

st.pyplot(fig)

with st.expander("Ver relatório de classificação"):
    st.code(res['relatorio'], language="text")
