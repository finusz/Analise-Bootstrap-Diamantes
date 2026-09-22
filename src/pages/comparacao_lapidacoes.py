from src.dataset.data import df_diamonds
from src.dataset.graficos import hists
from src.dataset.calculos import bootstrap, confidence_interval

import streamlit as st
import pandas as pd
import numpy as np

st.title('Comparação entre lapidações diferentes', text_alignment='center')
st.divider()

opcoes = st.multiselect("Selecione quais lapidações você deseja comparar:", df_diamonds['lapidação'].unique(),max_selections=2)

if len(opcoes) < 2:
    st.info("É preciso elecionar exatamente 2 lapidações para exibir a comparação.")
    st.stop()

prices1 = df_diamonds[df_diamonds['lapidação'] == opcoes[0]]['preço']
prices2 = df_diamonds[df_diamonds['lapidação'] == opcoes[1]]['preço']

diff, mean1, mean2 = bootstrap(prices1, prices2)

fig = hists(diff, mean1, mean2,
            f'Proporção da diferença média:\nLapidação {opcoes[0]} x {opcoes[1]}',
            f'Proporção das médias:\nLapidação {opcoes[0]} x {opcoes[1]}',
            f'Diferença ({opcoes[0]} - {opcoes[1]})', [f'{opcoes[0]}', f'{opcoes[1]}'])

st.pyplot(fig)


means_prices = [mean1, mean2]
limitInf, limitSup = confidence_interval(means_prices)
st.markdown(f'Podemos afirmar com 95% de confiança que a média de preço dos diamantes com as lapidações {opcoes[0]} e {opcoes[1]} está na faixa de valor que vai de {limitInf:.2f} até {limitSup:.2f}.\n')