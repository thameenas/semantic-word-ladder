import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .neighbours import EMBEDDINGS, WORD_TO_IDX


def visualize_path(path):
    """
    Visualize a semantic path in 2D using PCA.
    """

    # Get embeddings for words in path
    vectors = np.array([EMBEDDINGS[WORD_TO_IDX[w]] for w in path])

    # Reduce to 2D
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(vectors)

    # Plot
    plt.figure()
    plt.plot(points_2d[:, 0], points_2d[:, 1], marker="o")
    
    for i, word in enumerate(path):
        plt.text(points_2d[i, 0], points_2d[i, 1], word)

    plt.title("Semantic Ladder in Vector Space")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.grid(True)
    plt.show()
