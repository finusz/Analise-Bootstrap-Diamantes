import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

RECURSOS_NUMERICOS = [
    'quilate', 'profundidade', 'largura do topo em relação à base',
    'preço', 'x', 'y', 'z', 'volume', 'simetria', 'proporcao_altura',
]
RECURSOS_CATEGORICOS = ['cor', 'pureza']
ORDEM_CORES = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
ORDEM_PUREZAS = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

PARAM_GRID_PADRAO = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.01, 0.1],
}


def treinar_svm(
    df,
    alvo='lapidação',
    recursos_numericos=None,
    recursos_categoricos=None,
    param_grid=None,
    n_por_classe=2500,          # ex.: 500 -> amostra balanceada por classe
    test_size=0.2,
    cv=3,
    random_state=42,
    n_jobs=1,
    n_classes=2
):
    """Treina um SVM (RBF) entre duas classes de lapidação."""
    if recursos_numericos is None:
        recursos_numericos = RECURSOS_NUMERICOS
    if recursos_categoricos is None:
        recursos_categoricos = RECURSOS_CATEGORICOS
    if param_grid is None:
        param_grid = PARAM_GRID_PADRAO

    # --- Limpeza ---
    df_proc = df.copy()
    for col in recursos_numericos:
        df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
    for col in recursos_categoricos:
        df_proc[col] = df_proc[col].astype(str)
    df_proc = df_proc.dropna(subset=recursos_numericos + recursos_categoricos + [alvo])

    # --- Filtra as duas classes ---
    df_proc = df_proc[df_proc[alvo].isin(df[alvo].unique())]
    contagem = df_proc[alvo].value_counts()

    if n_classes == 2:    
        if len(contagem) < 2 or contagem.min() < max(cv, 2):
            raise ValueError(
                "Poucos dados para treinar com essas lapidações "
                f"(amostras por classe: {contagem.to_dict()})."
            )

    # --- Amostragem balanceada (opcional) ---
    if n_por_classe:
        n = min(n_por_classe, contagem.min())
        df_proc = df_proc.groupby(alvo).sample(n=n, random_state=random_state)
    else:
        df_proc = df_proc.groupby(alvo).sample(n=5, random_state=random_state)

    X = df_proc[recursos_numericos + recursos_categoricos]
    y = df_proc[alvo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- Pipeline ---
    transformers = [('num', StandardScaler(), recursos_numericos)]
    if recursos_categoricos:
        transformers.append((
            'cat',
            OrdinalEncoder(
                categories=[ORDEM_CORES, ORDEM_PUREZAS],
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            ),
            recursos_categoricos,
        ))

    pipeline = Pipeline([
        ('prepro', ColumnTransformer(transformers=transformers)),
        ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=random_state)),
    ])

    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    grid.fit(X_train, y_train)

    modelo = grid.best_estimator_
    y_pred = modelo.predict(X_test)

    return {
        'modelo': modelo,
        'classes': df[alvo].unique(),
        'melhores_params': grid.best_params_,
        'acuracia': accuracy_score(y_test, y_pred),
        'relatorio': classification_report(y_test, y_pred),
        'relatorio_dict': classification_report(y_test, y_pred, output_dict=True),
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
    }