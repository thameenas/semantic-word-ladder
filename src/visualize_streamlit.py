import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .neighbours import EMBEDDINGS, WORD_TO_IDX


def plot_path(path, method="umap", dim=3):
    """
    Plot semantic ladder in 2D or 3D using UMAP or PCA.
    """

    vectors = np.array([EMBEDDINGS[WORD_TO_IDX[w]] for w in path])

    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=dim,
            n_neighbors=min(10, len(path) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        points = reducer.fit_transform(vectors)
        title = f"Semantic Ladder (UMAP {dim}D)"

    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=dim)
        points = reducer.fit_transform(vectors)
        title = f"Semantic Ladder (PCA {dim}D)"

    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(points[:, 0], points[:, 1], points[:, 2], marker="o")

        for i, word in enumerate(path):
            ax.text(points[i, 0], points[i, 1], points[i, 2], word)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    else:
        fig, ax = plt.subplots()
        ax.plot(points[:, 0], points[:, 1], marker="o")

        for i, word in enumerate(path):
            ax.text(points[i, 0], points[i, 1], word)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    ax.set_title(title)
    return fig
