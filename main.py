# main.py

import sys
import os

os.makedirs("documents", exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

if "--cli" in sys.argv:
    from rag_pipeline import RAGPipeline
    rag = RAGPipeline()

    if not rag.load_existing():
        print("no index found, building...")
        result = rag.build_index()
        print(result["msg"])

    if not rag.ready:
        print("put PDFs in documents/ and try again")
    else:
        print("\nready — type question (or 'quit' to exit)\n")
        while True:
            try:
                q = input("you: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                break
            result = rag.ask(q)
            print(f"\nanswer: {result['answer']}")
            for s in result["sources"]:
                print(f"  <- {s['filename']} (chunk {s['chunk_id']}, score: {s['score']})")
            print()
else:
    from app import app
    print("\nstarting at http://localhost:5000")
    print("models load in background (30-60 sec), UI shows progress\n")
    app.run(debug=False, port=5000)
