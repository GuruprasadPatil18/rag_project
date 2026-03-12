# rag_pipeline.py

import faiss
from config import DOCS_FOLDER, TOP_K
from loader import load_all_pdfs
from chunker import split_text
from embedder import embed_texts, embed_query, progress
from vector_store import normalize, save_index, load_index, search
from generator import generate_answer
import numpy as np


class RAGPipeline:

    def __init__(self):
        self.index = None
        self.metadata = []
        self.ready = False

    def build_index(self, folder=DOCS_FOLDER):

        print("\n==============================")
        print("[PIPELINE] INDEX BUILD STARTED")
        print("==============================\n")

        self.metadata = []

        docs = load_all_pdfs(folder)

        all_vectors = []

        # track chunk_id per file, not globally
        file_chunk_counts = {}

        for doc in docs:

            fname = doc["filename"]
            page = doc["page"]

            print(f"[PIPELINE] processing page {page+1} from {fname}")

            chunks = split_text(doc["text"])

            if not chunks:
                continue

            print(f"[PIPELINE] created {len(chunks)} chunks")

            vectors = embed_texts(chunks)

            if vectors.size == 0:
                continue

            all_vectors.append(vectors)

            # get current chunk count for this file
            if fname not in file_chunk_counts:
                file_chunk_counts[fname] = 0

            for text in chunks:
                self.metadata.append({
                    "filename": fname,
                    "page": page,
                    "chunk_id": file_chunk_counts[fname],
                    "text": text
                })
                file_chunk_counts[fname] += 1

            print(f"[PIPELINE] {fname}: {file_chunk_counts[fname]} chunks so far")

        if not all_vectors:
            return {"ok": False, "msg": "no text extracted"}

        # build fresh index
        combined = np.vstack(all_vectors).astype(np.float32)
        combined = normalize(combined)

        dim = combined.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(combined)

        save_index(self.index, self.metadata)

        self.ready = True

        progress["step"] = "done"
        progress["percent"] = 100
        progress["done"] = True

        total_chunks = len(self.metadata)
        total_docs = len(self.get_doc_list())

        print("\n==============================")
        print("[PIPELINE] INDEX COMPLETE")
        print(f"  documents: {total_docs}")
        print(f"  chunks: {total_chunks}")
        print(f"  vectors: {self.index.ntotal}")
        print("==============================")

        return {
            "ok": True,
            "msg": f"indexed {total_chunks} chunks from {total_docs} documents",
            "doc_count": total_docs,
            "chunk_count": total_chunks
        }

    def load_existing(self):

        print("[PIPELINE] loading existing index")

        self.index, self.metadata = load_index()

        if self.index is not None and self.metadata:
            if self.index.ntotal == len(self.metadata):
                self.ready = True
                print(f"[PIPELINE] loaded: {self.index.ntotal} vectors, {len(self.metadata)} metadata")
                return True
            else:
                print(f"[PIPELINE] mismatch! {self.index.ntotal} vectors vs {len(self.metadata)} metadata")
                self.index = None
                self.metadata = []
                return False

        self.metadata = []
        return False

    def ask(self, question, top_k=TOP_K):

        print(f"\n[QUERY] {question}")

        if not self.ready:
            return {"answer": "upload PDFs first", "sources": []}

        q_vec = embed_query(question)

        scores, indices = search(self.index, q_vec, top_k)

        chunks_text = []
        sources = []

        for i, idx in enumerate(indices):

            if idx == -1 or idx >= len(self.metadata):
                continue

            chunk = self.metadata[idx]

            chunks_text.append(chunk["text"])

            sources.append({
                "filename": chunk["filename"],
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"] + 1,
                "score": round(float(scores[i]), 4)
            })

        if not chunks_text:
            return {"answer": "no relevant chunks found", "sources": []}

        answer = generate_answer(question, chunks_text[:5])

        return {"answer": answer, "sources": sources}

    def get_doc_list(self):

        if not self.metadata:
            return []

        seen = set()
        names = []

        for m in self.metadata:
            if m["filename"] not in seen:
                seen.add(m["filename"])
                names.append(m["filename"])

        return names