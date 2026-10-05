import streamlit as st
import requests

import streamlit as st
import requests

# 1. Configuração da página (opcional, para dar mais espaço aos cards)
st.set_page_config(layout="wide")

# 1. Função com CACHE para buscar os dados de perfil do usuário
@st.cache_data(ttl=3600)  # Guarda na memória por 1 hora (3600 segundos)
def buscar_dados_perfil(username):
    url = f"https://github.com{username}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

# 2. Função com CACHE para buscar o conteúdo do README
@st.cache_data(ttl=3600)
def buscar_readme(username):
    # Tenta na branch main
    url = f"https://githubusercontent.com{username}/{username}/main/README.md"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.text
        
        # Se falhar, tenta na branch master
        url = f"https://githubusercontent.com{username}/{username}/master/README.md"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.text
    except Exception:
        pass
    return None

# 3. Função principal que monta a interface visual
def renderizar_criador(username, container):
    dados = buscar_dados_perfil(username)
    readme_text = buscar_readme(username)

    # Se ambas as funções falharem por conta do bloqueio temporário atual do seu IP:
    if not dados:
        container.warning(f"O GitHub limitou as requisições para @{username} temporariamente. Aguarde um momento.")
        return

    avatar_url = dados.get("avatar_url")
    nome_exibicao = dados.get("name") or username
    bio = dados.get("bio", "")

    # Monta o card visual dentro do container/coluna
    with container.container(border=True):
        col_foto, col_info = st.columns([1, 2]) # 1 parte para foto, 2 partes para texto
        
        with col_foto:
            if avatar_url:
                st.image(avatar_url, use_container_width=True)
        with col_info:
            st.subheader(nome_exibicao)
            st.caption(f"[@{username}](https://github.com{username})")
            if bio:
                st.markdown(f"*{bio}*")
        
        st.divider()
        
        if readme_text:
            st.markdown(readme_text)
        else:
            st.info("README não disponível ou não configurado.")

# --- ÁREA DE CONFIGURAÇÃO ---
# Dicionário com os criadores (Chave: Nome de exibição no app, Valor: Username do GitHub)
criadores = {
    "Felipe Nunes": "finusz",
    "Heloísa Azevedo": "heloisa-azevedo"
}

# --- INTERFACE DO APP ---
st.title("Sobre os Criadores")
st.write("Conheça os desenvolvedores responsáveis por este projeto:")

# Cria duas colunas na tela para colocar os cards lado a lado
col1, col2 = st.columns(2)

# Transforma o dicionário em uma lista para acessar por índice nas colunas
lista_criadores = list(criadores.items())

# Roda o primeiro criador na coluna 1 e o segundo na coluna 2
renderizar_criador(lista_criadores[0][1], col1)
renderizar_criador(lista_criadores[1][1], col2)

# v1
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