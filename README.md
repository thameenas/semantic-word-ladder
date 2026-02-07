---
title: Semantic Word Ladder
emoji: 🪜
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
tags:
    - streamlit
app_file: app.py
pinned: false
---
# Semantic Word Ladder 🪜

Access the app at: https://huggingface.co/spaces/thameena/semantic-word-ladder

A small hobby project to explore how words are connected in embedding space by treating semantic similarity as a search problem.

Given a start word and a target word, the system finds a sequence of intermediate words that gradually move from the start to the target based on **cosine similarity**.

A semantic ladder between 'sleep' and 'weather' would look something like:
```
sleep → wake → sunrise → sunny → cloudy → rain → weather
```
---

## Core Ideas

---

### 1. Word Embeddings

Words are converted into embeddings using all-MiniLM-L6-v2 from the sentence-transformers library.

Semantic similarity between words is measured using **cosine similarity**, which captures how closely two words align in meaning.

---

### 2. Semantic Space as a Graph

- Each word is treated as a **node**
- Top K most similar neighbours to a node is computed using FAISS library
- The full graph is **never built explicitly**

This allows semantic navigation to be treated as a **graph search problem** rather than a static similarity lookup.

---

### 3. A* Search for Semantic Navigation

The ladder is found using **A\*** search, where each step moves to a nearby semantic neighbor using a heuristic function.

---

## Project Structure

### Setup & Usage

This project is designed to be run locally using **Python + uv**.


### Prerequisites

- Python **3.10** 
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
```
data/words.txt
```
The dataset that is used for this project is from: https://github.com/first20hours/google-10000-english/blob/master/google-10000-english-no-swears.txt

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