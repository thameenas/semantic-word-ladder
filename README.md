# Semantic Word Ladder 🪜

A small hobby project to explore how words are connected in embedding space by treating semantic similarity as a search problem.

Given a start word and a target word, the system finds a **semantic ladder** - a sequence of intermediate words that gradually move from the start to the target based on **cosine similarity**.

---

## Core Ideas

The project is built around a few key concepts.

---

### 1. Word Embeddings

Words are converted into dense vectors using a pretrained **sentence-transformer** model.

Semantic similarity between words is measured using **cosine similarity**, which captures how closely two words align in meaning.

---

### 2. Semantic Space as a Graph

- Each word is treated as a **node**
- Edges are implicit: the **top-k most similar words** form the local neighborhood
- The full graph is **never built explicitly**

This allows semantic navigation to be treated as a **graph search problem** rather than a static similarity lookup.

---

### 3. A* Search for Semantic Navigation

The ladder is found using **A\*** search, where:

- Each step moves to a nearby semantic neighbor
- The algorithm balances:
  - **local smoothness** (don’t jump too far)
  - **progress toward the target**

---

## Project Structure

### Setup & Usage

This project is designed to be run locally using **Python + uv**.


### Prerequisites

- Python **3.10** or **3.11**
- `uv` installed

```bash
pip install uv
```

### Install dependencies
```
uv pip install -r requirements.txt
```
### Prepare dataset
Create or edit the vocabulary file: 
```data/words.txt```

### Generate embeddings (one-time step)

Convert words into vectors and save them locally:
```
uv run python src/embed.py
```

This generates:
```
data/embeddings.npy
data/word_to_idx.json
```

### Build / load FAISS index

The FAISS index is loaded automatically when the app runs.

To test it manually:
```
uv run python -m src.index
```

### Run streamlit app
```
uv run streamlit run app.py
```