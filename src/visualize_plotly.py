import numpy as np
import plotly.graph_objects as go

from .neighbours import EMBEDDINGS, WORD_TO_IDX


def plot_path_3d(path, method="umap"):
    """
    Interactive 3D plot of semantic ladder using Plotly.
    """

    vectors = np.array([EMBEDDINGS[WORD_TO_IDX[w]] for w in path])

    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(10, len(path) - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        points = reducer.fit_transform(vectors)
        title = "Semantic Ladder (UMAP 3D)"
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3)
        points = reducer.fit_transform(vectors)
        title = "Semantic Ladder (PCA 3D)"

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    fig = go.Figure()

    # Path line
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+markers+text",
            text=path,
            textposition="top center",
            marker=dict(size=6),
            line=dict(width=4),
            hovertext=path,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
    )

    return fig
