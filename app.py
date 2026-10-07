import streamlit as st

main_page = st.Page("src/pages/home.py", title="Analise Exploratória")
second_page = st.Page('src/pages/comparacao_lapidacoes.py', title="Comparações")
about_page = st.Page('src/pages/sobre.py', title="Sobre")

# navegação entre as paginas
pg = st.navigation([main_page, second_page, about_page])


pg.run()
#python -m streamlit run app.py