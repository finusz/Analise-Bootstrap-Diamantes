import streamlit as st

main_page = st.Page("src/pages/home.py", title="HOME")
second_page = st.Page('src/pages/comparacao_lapidacoes.py', title="COMPARAÇÕES")
third_page = st.Page('src/pages/SVM.py', title="SVM")

# navegação entre as paginas
pg = st.navigation([main_page, second_page, third_page])


pg.run()
#python -m streamlit run app.py