import streamlit as st
from io import StringIO

import seaborn as sns
import matplotlib.pyplot as plt

from src.dataset.data import df_diamonds
from src.dataset.graficos import hist_means

from scipy import stats

st.title('Bootstrap do Preço dos Diamantes', text_alignment='center')
st.subheader('Felipe Nunes e Heloísa Azevedo', text_alignment='center')
st.divider()
st.markdown('A base de dados utilizada é nativa da biblioteca do Seaborn. A análise foi realizada utilizando o método de reamostragem bootstrap para estimar a média do preço dos diamantes e comparar as médias entre diferentes tipos de corte afim de entender qual é vale mais a pena de ser feito.')

st.divider()
st.subheader('Base de Dados de Diamantes', text_alignment='center')
st.dataframe(df_diamonds)

st.divider()
buffer = StringIO()
df_diamonds.info(buf=buffer)
st.subheader("Informações do dataset", text_alignment='center')
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
st.subheader('Boostrap inicial dos diamantes em relação ao preço', text_alignment='center')
st.markdown('Foi realizado o boostrap de 5000 médias do preço dos diamante')
fig_histMean, price_means = hist_means()
st.pyplot(fig_histMean)

st.markdown(r'h0: média de preço = 3900')
st.markdown('h1: média de prçeo != 3900')
st.markdown(r'Tendo em vista 95% de confiança')

h0_value = 3900
t_statistic, p_value = stats.ttest_1samp(price_means, h0_value)
st.markdown(f'O p-valor foi de: {p_value}\n')
if p_value > 0.05:
   st.markdown(f'Logo, aceita H0, com 95% de confiança podemos dizer que a média dos preços seja igual a {h0_value}')
else:
   st.markdown(f'Logo, rejeita H0, com 95% de confiança não podemos dizer que a média dos preços seja igual a {h0_value}')