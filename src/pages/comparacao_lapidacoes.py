from src.dataset.data import df_amostra
from src.util.graficos import hists
from src.util.calculos import bootstrap, confidence_interval

import streamlit as st

import pandas as pd
import numpy as np

st.title('Comparação entre lapidações diferentes', text_alignment='center')
st.divider()

st.markdown('**Para as comparações está sendo utilizada uma amostra de 10% da população, o que representa cerca de 5.000 registros.**')

opcoes = st.multiselect("Selecione quais lapidações você deseja comparar:", df_amostra['lapidação'].unique(),max_selections=2)

st.markdown(
    """
    <style>
    /* 1. Removemos completamente qualquer cor de fundo padrão e filtros das camadas internas */
    div[data-testid="stAlert"] > div,
    div[data-testid="stAlert"] [role="alert"] {
        background-color: transparent !important;
        background-image: none !important;
    }

    /* 2. Aplicamos a cor pura diretamente no container principal */
    div[data-testid="stAlert"] {
        background-color: #6d5b4f !important; /* Insira aqui a cor EXATA do seu multiselect */
        border: none !important;
        border-radius: 18px !important;
    }
    
    /* 3. Mantém o texto e o ícone na cor branca */
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
