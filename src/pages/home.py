import streamlit as st
from io import StringIO

import seaborn as sns
import matplotlib.pyplot as plt

from src.dataset.data import df_diamonds
from src.util.graficos import hist_means, bar_diamonts

from scipy import stats

st.title('Bootstrap do Preço dos Diamantes', text_alignment='center')
st.subheader('Felipe Nunes e Heloísa Azevedo', text_alignment='center')
st.divider()
st.markdown('A base de dados utilizada é nativa da biblioteca do Seaborn. A análise tem como objetivo estimar a média de preço de diamentes com diferentes lapidações, afim de entender qual é a mais lucrativa de ser feita.')

st.divider()
st.subheader('Base de Dados de Diamantes', text_alignment='center')
st.dataframe(df_diamonds)

st.divider()
buffer = StringIO()
df_diamonds.info(buf=buffer)
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
st.dataframe(df_diamonds.describe())

st.divider()
st.subheader('Correlação entre os dados', text_alignment='center')
corr = df_diamonds.corr(numeric_only=True)
fig_corr, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)
st.markdown('Como esperado, o preço do diamante está muito ligado ao quilate dele.')

st.divider()
st.subheader('Comparação do preço entre os tipos de lapidações', text_alignment='center')
fig_bar = bar_diamonts()
st.pyplot(fig_bar)

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