#IMPORTAÇÃO DAS BIBLIOTECAS PARA A CRIAÇÂO DOS GRÁFICOS
import seaborn as sns
import matplotlib.pyplot as plt

#IMPORTAÇÃO DO DATASET
from .data import df_diamonds


def hist_means():
    price_means = []
    for i in range(5000):
        bootstrap = df_diamonds['preço'].sample(50, replace=True)
        diamonds_means = bootstrap.mean()
        price_means.append(diamonds_means)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=price_means,alpha=0.5, kde=True)

    ax.set_title('Proporção da média do preço de diamentes')
    ax.set_xlabel('Preço (US$)')
    ax.set_ylabel('Frequência')
    
    return fig, price_means

def hists(diff, mean1, mean2, title1:str, title2:str, lbl1:str, lbl2:list[str]):
    fig, axs = plt.subplots(2, 1, figsize=(20, 25))

    titles = [
        title1,
        title2
    ]

    datasets = [
        [diff],
        [mean1, mean2]
    ]

    labels = [
        [lbl1],
        lbl2
    ]

    for ax, title, data_group, label_group in zip(axs.flat, titles, datasets, labels):
        ax.set_title(title, fontsize=20, pad=20)
        ax.set_xlabel('Preço (US$)', fontsize=18, labelpad=10)
        ax.set_ylabel('Frequência', fontsize=18, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=12)

        for data, lbl in zip(data_group, label_group):
            sns.histplot(data=data, ax=ax, alpha=0.5, kde=True, label=lbl)

        if len(data_group) > 1:
            ax.legend()

    return fig