import streamlit as st
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

from src.dataset.data import df_amostra
from src.models.svm_pipeline import treinar_svm

FEATURES = ['profundidade', 'largura do topo em relação à base']
PARAM_GRID = {'svm__C': [10.0], 'svm__gamma': [1.0]}   # mesmos valores fixos de antes

# Texto branco nos blocos de código (fundo marrom vem do config.toml)
st.markdown(
    """
    <style>
    pre code, pre code span {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Treinando o modelo SVM...")
def obter_resultado(classes):          # classes como tupla (hashable)
    return treinar_svm(
        df_amostra,
        classes=classes,
        recursos_numericos=FEATURES,
        recursos_categoricos=[],       # só 2 variáveis para poder plotar em 2D
        param_grid=PARAM_GRID,
        n_por_classe=500,
    )


st.title("Classificação SVM - Dataset Diamonds")

opcoes_disponiveis = sorted(df_amostra['lapidação'].dropna().unique())

selecao = st.multiselect(
    "Escolha exatamente duas opções de lapidação para comparar:",
    options=opcoes_disponiveis,
    max_selections=2,
)

if len(selecao) == 2:
    try:
        res = obter_resultado(tuple(sorted(selecao)))
    except ValueError as e:
        st.error(str(e))
        st.stop()

    modelo = res['modelo']
    X_test = res['X_test']
    y_test = res['y_test']

    st.success(f"**Acurácia do modelo:** {res['acuracia']:.4f}")

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

elif len(selecao) > 2:
    st.warning("Por favor, selecione apenas duas opções para a visualização da fronteira de decisão.")
else:
    st.info("Aguardando a seleção de duas categorias de lapidação...")