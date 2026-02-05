import json
import numpy as np
from pathlib import Path
from .index import load_faiss_index

DATA_DIR = Path("data")

# Load once at import time
with open(DATA_DIR / "word_to_idx.json") as f:
    WORD_TO_IDX = json.load(f)

IDX_TO_WORD = {i: w for w, i in WORD_TO_IDX.items()}

INDEX = load_faiss_index()
EMBEDDINGS = np.load(DATA_DIR / "embeddings.npy")


def get_neighbours(word: str, k: int =5):
    """
    Returns top-k nearest neighbors of a word.
    Output: list of (neighbor_word, cosine_similarity)
    """
    if word not in WORD_TO_IDX:
        raise ValueError(f"Word '{word}' not in vocabulary")
    
    idx = WORD_TO_IDX[word]
    query_vector = EMBEDDINGS[idx].reshape(1, -1)

    similarities, indices = INDEX.search(query_vector, k + 1)

    results = []
    for similarity, i in zip(similarities[0], indices[0]):
        neighbour = IDX_TO_WORD[i]
        if neighbour != word:  # skip self
            results.append((neighbour, float(similarity)))      

    return results[:k]


if __name__ == "__main__":
    neighbors = get_neighbours("king", k=5)
    for w, s in neighbors:
        print(w, round(s, 3))