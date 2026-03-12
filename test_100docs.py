# test_100docs.py
# tests the full pipeline with all your PDFs, measures time and memory

import os
import time
import psutil

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

folder = "documents"
pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
print(f"found {len(pdfs)} PDFs in documents/\n")

if len(pdfs) == 0:
    print("put PDFs in documents/ folder first")
    exit()

# --- step 1: load all PDFs ---
print("=" * 50)
print("STEP 1: LOADING PDFs")
print("=" * 50)

mem_before = get_memory_mb()
start = time.time()

from loader import load_all_pdfs
docs = list(load_all_pdfs(folder))

load_time = time.time() - start
mem_after = get_memory_mb()

total_chars = sum(len(d["text"]) for d in docs)
total_pages = len(docs)

print(f"\n  pages extracted: {total_pages}")
print(f"  total text: {total_chars:,} chars")
print(f"  time: {load_time:.1f}s")
print(f"  memory: {mem_before:.0f}MB -> {mem_after:.0f}MB (+{mem_after - mem_before:.0f}MB)\n")

# --- step 2: chunk ---
print("=" * 50)
print("STEP 2: CHUNKING")
print("=" * 50)

start = time.time()
from chunker import split_text

all_chunks = []
for doc in docs:
    chunks = split_text(doc["text"])
    for i, c in enumerate(chunks):
        all_chunks.append({
            "text": c,
            "filename": doc["filename"],
            "page": doc["page"],
            "chunk_id": i
        })

chunk_time = time.time() - start
mem_after_chunk = get_memory_mb()

print(f"\n  chunks created: {len(all_chunks)}")
print(f"  avg chunk size: {sum(len(c['text']) for c in all_chunks) // len(all_chunks)} chars")
print(f"  time: {chunk_time:.1f}s")
print(f"  memory: {mem_after_chunk:.0f}MB (+{mem_after_chunk - mem_after:.0f}MB)\n")

# --- step 3: embed ---
print("=" * 50)
print("STEP 3: EMBEDDING (sentence-transformers)")
print("=" * 50)

start = time.time()
from embedder import embed_texts

texts = [c["text"] for c in all_chunks]

# embed in batches to track progress
batch_size = 32
import numpy as np
all_vectors = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    vecs = embed_texts(batch)
    all_vectors.append(vecs)
    done = min(i + batch_size, len(texts))
    elapsed = time.time() - start
    rate = done / elapsed if elapsed > 0 else 0
    print(f"  embedded {done}/{len(texts)} chunks ({rate:.1f} chunks/sec)")

vectors = np.vstack(all_vectors)
embed_time = time.time() - start
mem_after_embed = get_memory_mb()

print(f"\n  vectors shape: {vectors.shape}")
print(f"  time: {embed_time:.1f}s")
print(f"  memory: {mem_after_embed:.0f}MB (+{mem_after_embed - mem_after_chunk:.0f}MB)\n")

# --- step 4: FAISS index ---
print("=" * 50)
print("STEP 4: FAISS INDEX")
print("=" * 50)

start = time.time()
from vector_store import create_or_load_index, add_vectors, search
import faiss

dim = vectors.shape[1]
index = faiss.IndexFlatIP(dim)

# normalize before adding
from vector_store import normalize
norm_vecs = normalize(vectors.astype(np.float32))
index.add(norm_vecs)

index_time = time.time() - start

print(f"\n  vectors in index: {index.ntotal}")
print(f"  time: {index_time:.2f}s\n")

# --- step 5: search speed ---
print("=" * 50)
print("STEP 5: SEARCH SPEED TEST")
print("=" * 50)

from embedder import embed_query

queries = [
    "what is attention mechanism",
    "how does transformer work",
    "what is self attention",
    "training details and hardware",
    "what are the results"
]

for q in queries:
    start = time.time()
    q_vec = embed_query(q)
    q_norm = normalize(np.array([q_vec]).astype(np.float32))
    scores, indices = index.search(q_norm, 5)
    search_time = time.time() - start

    print(f"\n  Q: {q}")
    print(f"  search time: {search_time*1000:.0f}ms")
    for j, idx in enumerate(indices[0]):
        if idx != -1:
            c = all_chunks[idx]
            print(f"    [{j+1}] score={scores[0][j]:.3f} | {c['filename']} p{c['page']} | {c['text'][:80]}...")

# --- summary ---
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  PDFs:       {len(pdfs)}")
print(f"  pages:      {total_pages}")
print(f"  chunks:     {len(all_chunks)}")
print(f"  vectors:    {vectors.shape}")
print(f"  load time:  {load_time:.1f}s")
print(f"  chunk time: {chunk_time:.1f}s")
print(f"  embed time: {embed_time:.1f}s")
print(f"  index time: {index_time:.2f}s")
print(f"  total:      {load_time + chunk_time + embed_time + index_time:.1f}s")
print(f"  memory:     {get_memory_mb():.0f}MB")
print(f"\n  estimated 100 PDFs: ~{(load_time + chunk_time + embed_time) * (100 / max(len(pdfs), 1)):.0f}s")
print("=" * 50)