from scipy.stats import rankdata
import numpy as np


def rank_based_transform(x, k=0.5):
    num_samp = np.sum(~np.isnan(x))
    ranks = (rankdata(x, method='ordinal').astype(float) - k) / (num_samp - 2 * k + 1)
    return np.log(ranks / (1 - ranks))


def data_standardization(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / arr.std()
