# config.py

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_FOLDER = os.path.join(BASE_DIR, "documents")
INDEX_FOLDER = os.path.join(BASE_DIR, "faiss_index")

# chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# embedding model (~80MB, produces 384-dim vectors)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# generation model (~1GB, runs locally)
GENERATION_MODEL = "google/flan-t5-base"
MAX_ANSWER_TOKENS = 512

# retrieval
TOP_K = 5

# flask
ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024
