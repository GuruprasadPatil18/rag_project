# chunker.py

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """break text into overlapping pieces"""
    chunks = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        end = min(pos + size, text_len)

        # try to cut at a sentence boundary
        if end < text_len:
            cut = text.rfind(".", pos + (size // 2), end)
            if cut == -1:
                cut = text.rfind("\n", pos + (size // 2), end)
            if cut > pos:
                end = cut + 1

        piece = text[pos:end].strip()
        if piece:
            chunks.append(piece)

        # make sure we always move forward
        new_pos = end - overlap
        if new_pos <= pos:
            new_pos = pos + size
        pos = new_pos

    return chunks


def chunk_all_docs(documents):
    """takes loader output, returns flat list with metadata"""
    result = []

    for doc in documents:
        parts = split_text(doc["text"])
        for i, part in enumerate(parts):
            result.append({
                "text": part,
                "filename": doc["filename"],
                "chunk_id": i
            })

    return result