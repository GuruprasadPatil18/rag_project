# loader.py

import os
from PyPDF2 import PdfReader


def load_all_pdfs(folder):
    """
    Generator that yields one page at a time.
    This avoids loading full PDFs into memory.
    """

    if not os.path.exists(folder):
        print(f"folder not found: {folder}")
        return

    for filename in os.listdir(folder):

        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(folder, filename)

        try:
            reader = PdfReader(path)

            print(f"[LOADER] reading {filename} ({len(reader.pages)} pages)")

            for page_id, page in enumerate(reader.pages):

                text = page.extract_text()

                if not text:
                    continue

                yield {
                    "filename": filename,
                    "page": page_id,
                    "text": text
                }

        except Exception as e:
            print(f"[LOADER ERROR] {filename}: {e}")