import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import umap


def pca_features(data: np.ndarray, n_components: int = 10) -> np.ndarray:
    flattened_data = np.array([img.flatten() for img in data])
    model = PCA(n_components=n_components)
    data_processed = model.fit_transform(flattened_data)
    # print(sum(model.explained_variance_ratio_))
    return data_processed


def t_sne_features(data: np.ndarray, n_components: int = 10):
    flattened_data = np.array([img.flatten() for img in data])
    data_embeded = TSNE(n_components=n_components,
                        learning_rate='auto',
                        init='random',
                        method='exact',
                        perplexity=3).fit_transform(flattened_data)
    print(data_embeded.shape)
    return data_embeded


def umap_features(data: np.ndarray, n_components: int = 10):
    flattened_data = np.array([img.flatten() for img in data])
    data_processed = umap.UMAP().fit_transform(flattened_data)
    return data_processed
