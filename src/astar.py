import heapq
import numpy as np
from .neighbours import get_neighbours, EMBEDDINGS, WORD_TO_IDX


def cosine_sim(word1: str, word2: str) -> float:
    v1 = EMBEDDINGS[WORD_TO_IDX[word1]]
    v2 = EMBEDDINGS[WORD_TO_IDX[word2]]
    return float(np.dot(v1, v2))


def astar_search(
    start: str,
    target: str,
    max_steps: int = 20,
    k: int = 10,
    alpha: float = 1.0,
):
    """
    A* search in semantic space.
    Returns the path from start to target if found.
    """

    # Priority queue: (f_cost, current_word)
    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from = {}          # child -> parent
    g_score = {start: 0}   # cost so far
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            return reconstruct_path(came_from, current)

        if current in visited:
            continue

        visited.add(current)

        if g_score[current] >= max_steps:
            continue

        neighbors = get_neighbours(current, k=k)

        for neighbor, _ in neighbors:
            tentative_g = g_score[current] + 1

            if neighbor in visited:
                continue

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                h = 1 - cosine_sim(neighbor, target)
                f = tentative_g + alpha * h

                heapq.heappush(open_set, (f, neighbor))

    return None  # no path found


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


if __name__ == "__main__":
    path = astar_search("thought", "work")
    print(" → ".join(path))
