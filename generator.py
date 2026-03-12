# generator.py

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:latest"


def generate_answer(question, context_chunks):

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful AI assistant.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Give a clear answer.
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        data = response.json()

        if "response" in data:
            return data["response"]

        print("OLLAMA ERROR:", data)
        return "LLM returned unexpected output."

    except Exception as e:

        print("GENERATION ERROR:", e)
        return "Error generating answer."