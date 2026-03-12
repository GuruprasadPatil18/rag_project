# embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

model = None

progress = {
    "step": "initializing",
    "percent": 0,
    "done": False
}


def load_model():
    global model

    if model is None:
        print("[EMBEDDER] loading embedding model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[EMBEDDER] embedding model ready")


def embed_texts(texts):

    load_model()

    if not texts:
        return np.array([])

    vectors = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return vectors


def embed_query(text):

    load_model()

    vec = model.encode([text], convert_to_numpy=True)

    return vec[0]