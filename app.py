# app.py

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from config import DOCS_FOLDER, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from rag_pipeline import RAGPipeline
import embedder

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

rag = RAGPipeline()

if rag.load_existing():
    print(f"loaded existing index ({rag.index.ntotal} vectors)")
else:
    print("no existing index — upload PDFs to get started")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/progress")
def get_progress():

    from embedder import progress

    return {
        "percent": progress["percent"],
        "step": progress["step"],
        "done": progress["done"]
    }
@app.route("/upload", methods=["POST"])
def upload():

    print("UPLOAD ROUTE HIT")

    if "files" not in request.files:
        return jsonify({"ok": False, "msg": "no files in request"}), 400

    files = request.files.getlist("files")
    saved = []

    # reset progress
    embedder.progress = {"step": "saving files...", "percent": 0, "done": False}

    for f in files:
        if f and f.filename and allowed_file(f.filename):
            name = secure_filename(f.filename)
            f.save(os.path.join(DOCS_FOLDER, name))
            saved.append(name)

    if not saved:
        embedder.progress = {"step": "failed", "percent": 0, "done": True}
        return jsonify({"ok": False, "msg": "no valid PDF files"}), 400

    embedder.progress["step"] = "reading PDFs..."
    embedder.progress["percent"] = 5

    result = rag.build_index()

    embedder.progress = {"step": "done", "percent": 100, "done": True}

    result["uploaded"] = saved
    return jsonify(result)


@app.route("/ask", methods=["POST"])
def ask():
    if not rag.ready:
        return jsonify({"answer": "upload some PDFs first", "sources": []}), 400

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "type a question", "sources": []}), 400

    result = rag.ask(question)
    return jsonify(result)


@app.route("/documents")
def documents():
    docs = rag.get_doc_list()
    return jsonify({"documents": docs, "count": len(docs)})


if __name__ == "__main__":
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    app.run(debug=False, port=5000, threaded=True)