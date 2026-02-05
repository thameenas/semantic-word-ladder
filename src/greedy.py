from .neighbours import get_neighbours, EMBEDDINGS, WORD_TO_IDX
import numpy as np

def cosine_sim(word1: str, word2: str) -> float:
    """
    Cosine similarity between two words.
    Embeddings are already L2-normalized, so dot product == cosine similarity.
    """
    v1 = EMBEDDINGS[WORD_TO_IDX[word1]]
    v2 = EMBEDDINGS[WORD_TO_IDX[word2]]
    return float(np.dot(v1, v2))

def greedy_walk(start: str, target:str, max_steps:int = 10):
    path = [start]
    current = start

    for _ in range(max_steps):
        if current == target:
            return path
        neighbours = get_neighbours(current, k=5)

        best_score = -1
        best_word = None
        # for each neighbour, get similarity score to target and pick the neighbours with highest score
        for neighbour,_ in neighbours:
            if neighbour in path:
                continue
            score = cosine_sim(neighbour, target)
            if score > best_score:
                best_score = score
                best_word = neighbour
        
        if best_word is None:
            break

        path.append(best_word)
        current = best_word

    return path


if __name__ == "__main__":
    path = greedy_walk("thought", "work")
    print(" → ".join(path))