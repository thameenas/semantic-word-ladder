import faiss
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

def load_faiss_index():
    # Load embeddings
    embeddings = np.load(DATA_DIR / "embeddings.npy")

    dim = embeddings.shape[1]

    # Inner Product index (cosine similarity because vectors are normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index
