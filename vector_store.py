# vector_store.py

import os
import json
import faiss
import numpy as np
from config import INDEX_FOLDER

INDEX_PATH = os.path.join(INDEX_FOLDER, "vectors.index")
META_PATH = os.path.join(INDEX_FOLDER, "metadata.json")


def normalize(vectors):

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    norms[norms == 0] = 1

    return vectors / norms


def create_or_load_index(dim):

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        return index

    index = faiss.IndexFlatIP(dim)

    return index


def add_vectors(index, vectors):

    vectors = normalize(vectors.astype(np.float32))

    index.add(vectors)


def save_index(index, metadata):

    os.makedirs(INDEX_FOLDER, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_index():

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def search(index, query_vector, top_k=5):

    query = np.array([query_vector]).astype(np.float32)

    scores, indices = index.search(query, top_k)

    return scores[0], indices[0]