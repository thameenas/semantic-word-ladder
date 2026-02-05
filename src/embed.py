from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
import faiss

DATA_DIR = Path("data")

def main():
    # Load words
    words = [w.strip() for w in (DATA_DIR / "words.txt").read_text().splitlines() if w.strip()]

    print(f"Embedding {len(words)} words...")

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode
    embeddings = model.encode(
        words,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save embeddings
    np.save(DATA_DIR / "embeddings.npy", embeddings)

    # Save word → index mapping
    word_to_idx = {word: i for i, word in enumerate(words)}
    with open(DATA_DIR / "word_to_idx.json", "w") as f:
        json.dump(word_to_idx, f)

    print("Saved:")
    print(" - data/embeddings.npy")
    print(" - data/word_to_idx.json")

    # Sanity check (optional but nice)
    idx_king = word_to_idx["king"]
    idx_queen = word_to_idx["queen"]
    sim = float(np.dot(embeddings[idx_king], embeddings[idx_queen]))
    print(f"cosine(king, queen) = {sim:.3f}")

if __name__ == "__main__":
    main()
