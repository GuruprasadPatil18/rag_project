# RAG Document Search

Local RAG system with sentence-transformers, FAISS, and flan-t5.
No API keys. Everything runs on your machine.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

First run: models load from HuggingFace cache (~30-60 sec).
UI shows loading progress, then unlocks when ready.

## Usage

Open http://localhost:5000
- Click + to upload PDFs
- Type a question, hit send
- Get answers with source references

## Tech Stack

- **sentence-transformers** (all-MiniLM-L6-v2) — converts text to 384-dim vectors
- **FAISS** — fast vector similarity search
- **Cosine Similarity** — measures how similar two vectors are
- **flan-t5-base** — generates answers from retrieved context
- **pdfplumber** — PDF text extraction
- **Flask** — web server

## How It Works

1. PDFs → pdfplumber extracts text (loader.py)
2. Text → split into 1000-char overlapping chunks (chunker.py)
3. Chunks → sentence-transformer converts to vectors (embedder.py)
4. Vectors → stored in FAISS index (vector_store.py)
5. Question → embedded, FAISS finds closest chunks
6. Chunks + question → flan-t5 generates answer (generator.py)

## Files

```
config.py           settings
loader.py           PDF reading
chunker.py          text splitting
embedder.py         sentence-transformers embeddings
vector_store.py     FAISS + cosine similarity search
generator.py        flan-t5 answer generation
rag_pipeline.py     connects all modules
app.py              flask web server
main.py             entry point
templates/index.html    web UI
```
