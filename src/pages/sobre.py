import streamlit as st

st.markdown(
    "<h1 style='text-align: center;'>Sobre</h1>", unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 1.1rem;'>"
    "Esse app foi desenvolvido para uma atividade da Faculdade de Tecnologia de Cotia (FATEC-COTIA). "
    "</p>",
    unsafe_allow_html=True,
)

st.write("")

st.markdown(
    """
<div style='text-align: center; font-size: 1.05rem;'>
    <p><b>Felipe Nunes</b> — 
        <a href='https://github.com/finusz' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/finusz/' target='_blank'>LinkedIn</a>
    </p>
    <p><b>Heloísa Azevedo</b> — 
        <a href='https://github.com/heloisa-azevedo' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/helo%C3%ADsa-azevedo/' target='_blank'>LinkedIn</a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)