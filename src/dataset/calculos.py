import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from .data import df_diamonds


def bootstrap(series1, series2, n_iterations=5000, sample_size=50):
    diff_means = []
    means1 = []
    means2 = []

    for _ in range(n_iterations):
        # Já calculamos a média direto, economizando linhas
        mean1 = series1.sample(sample_size, replace=True).mean()
        mean2 = series2.sample(sample_size, replace=True).mean()

        diff_means.append(mean1 - mean2)
        means1.append(mean1)
        means2.append(mean2)

    return diff_means, means1, means2

def confidence_interval(means:list):
  a = np.array(means)
  limit_inf = np.percentile(a, 2.5) #percentil 2.5​
  limit_sup = np.percentile(a, 97.5) #percentil 97.5
  return limit_inf, limit_sup