import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.dataset.data import df_amostra

CORES = ["#6d5b4f", "#8d7b6f", "#ad9b8f", "#cdbbb0", "#b5bfe0"]


# ---------- Funções com cache ----------
@st.cache_data(show_spinner="Calculando o método do cotovelo...")
def cotovelo(X, k_max=10):
    ks = list(range(1, k_max + 1))
    inercias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]
    return ks, inercias


@st.cache_data(show_spinner="Calculando a estatística gap...")
def gap_statistic(X, k_max=10, n_refs=10, random_state=42):
    X = np.asarray(X)
    rng = np.random.default_rng(random_state)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    ks = list(range(1, k_max + 1))
    gaps, sks = [], []
    for k in ks:
        log_wk = np.log(KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X).inertia_)
        # Conjuntos de referência uniformes no mesmo "retângulo" dos dados
        log_ref = []
        for _ in range(n_refs):
            ref = rng.uniform(mins, maxs, size=X.shape)
            log_ref.append(np.log(KMeans(n_clusters=k, n_init=3, random_state=random_state).fit(ref).inertia_))
        log_ref = np.array(log_ref)
        gaps.append(log_ref.mean() - log_wk)
        sks.append(log_ref.std() * np.sqrt(1 + 1 / n_refs))
    # Critério de Tibshirani: menor k tal que Gap(k) >= Gap(k+1) - s(k+1)
    k_otimo = ks[-1]
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            k_otimo = ks[i]
            break
    return ks, gaps, sks, k_otimo


@st.cache_data
def k_distancias(X, k):
    dist, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return np.sort(dist[:, -1])


def render_clusterizacao():
    """Seção de clusterização (Cotovelo, Gap e DBSCAN) com todas as lapidações."""
    st.subheader('Clusterização com todas as lapidações', text_alignment='center')

    # ---------- Filtro de lapidações (todas por padrão) ----------
    todas = list(df_amostra['lapidação'].unique())
    lapidacoes = st.multiselect(
        "Lapidações incluídas na clusterização (todas por padrão):",
        todas,
        default=todas,
        key="cluster_lapidacoes",
    )
    if not lapidacoes:
        st.info("Selecione pelo menos uma lapidação.")
        return

    df = df_amostra[df_amostra['lapidação'].isin(lapidacoes)].copy()
    st.markdown(f"**{len(df)} registros** de {len(lapidacoes)} lapidações: {', '.join(lapidacoes)}.")

    # ---------- Variáveis ----------
    colunas_numericas = df.select_dtypes(include='number').columns.tolist()
    padrao = [c for c in ['quilate', 'profundidade', 'largura do topo em relação à base'] if c in colunas_numericas]
    colunas = st.multiselect(
        "Escolha 3 variáveis para a clusterização",
        colunas_numericas,
        key="cluster_colunas",
        default=padrao,
        max_selections=3,
    )
    if len(colunas) != 3:
        st.warning("Selecione exatamente 3 variáveis para o gráfico 3D.")
        return

    # 'simetria' e 'proporcao_altura' podem gerar inf quando x ou y = 0
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=colunas).reset_index(drop=True)

    normalizado = pd.DataFrame(StandardScaler().fit_transform(df[colunas]), columns=colunas)


    # ---------- Método do cotovelo e gap ----------
    st.divider()
    st.subheader('Número de clusters: cotovelo e gap', text_alignment='center')

    k_max = st.slider("k máximo testado", 3, 15, 10, key="cluster_kmax")

    if st.checkbox("Mostrar método do Cotovelo", key="cluster_cotovelo"):
        ks, inercias = cotovelo(normalizado, k_max)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, inercias, marker='o', color=CORES[0])
        ax.set_xlabel('Número de clusters (k)')
        ax.set_ylabel('Inércia (WCSS)')
        ax.set_title(f'Método do Cotovelo — {len(lapidacoes)} lapidações')
        ax.set_xticks(ks)
        st.pyplot(fig)

    if st.checkbox("Mostrar estatística Gap", key="cluster_gap"):
        ks, gaps, sks, k_otimo = gap_statistic(normalizado, k_max)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(ks, gaps, yerr=sks, marker='o', capsize=4, color=CORES[0])
        ax.axvline(k_otimo, linestyle='--', color=CORES[4], label=f'k sugerido = {k_otimo}')
        ax.set_xlabel('Número de clusters (k)')
        ax.set_ylabel('Gap(k)')
        ax.set_title(f'Estatística Gap — {len(lapidacoes)} lapidações')
        ax.set_xticks(ks)
        ax.legend()
        st.pyplot(fig)
        st.markdown(f"Pelo critério de Tibshirani, o número sugerido de clusters é **{k_otimo}**.")

    # ---------- DBSCAN ----------
    st.divider()
    st.subheader('DBSCAN', text_alignment='center')

    eps = st.slider("eps (raio da vizinhança)", 0.1, 2.0, 0.5, 0.05, key="cluster_eps")
    min_samples = st.slider("min_samples", 2, 30, 5, key="cluster_min_samples")

    if st.checkbox("Mostrar gráfico k-distância (ajuda a escolher o eps)", key="cluster_kdist"):
        dists = k_distancias(normalizado, min_samples)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dists, color=CORES[0])
        ax.axhline(eps, linestyle='--', color=CORES[4], label=f'eps atual = {eps}')
        ax.set_xlabel('Pontos ordenados')
        ax.set_ylabel(f'Distância ao {min_samples}º vizinho')
        ax.legend()
        st.pyplot(fig)

    y_pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(normalizado)
    cluster = pd.Series(y_pred.astype(str), name='Cluster')

    st.caption(
        f"{len(set(y_pred)) - (1 if -1 in y_pred else 0)} clusters encontrados, "
        f"{(y_pred == -1).sum()} pontos classificados como ruído"
    )

    colorir_por = st.radio("Colorir pontos por:", ['Cluster', 'Lapidação'], horizontal=True, key="cluster_cor")
    cor = cluster if colorir_por == 'Cluster' else df['lapidação']

    fig_original = px.scatter_3d(
        df, x=colunas[0], y=colunas[1], z=colunas[2],
        color=cor, labels={'color': colorir_por},
        title='Dados originais',
    )
    st.plotly_chart(fig_original, width='stretch')

    st.markdown('**Distribuição das lapidações em cada cluster (-1 = ruído)**')
    st.dataframe(pd.crosstab(cluster, df['lapidação'], margins=True, margins_name='Total'))
