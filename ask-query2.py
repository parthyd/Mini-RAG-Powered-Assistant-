import os
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# -----------------------------
# CONFIG
# -----------------------------
FAISS_INDEX_FILE = "case_index.faiss"
METADATA_FILE = "case_metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral"   # the model you pulled


# -----------------------------
# Load embedding model
# -----------------------------
print("Loading MiniLM embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

def embed(text):
    return embedder.encode(text).astype("float32")


# -----------------------------
# Load FAISS + metadata
# -----------------------------
def load_index_and_metadata():
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


# -----------------------------
# Retrieve top-k pages
# -----------------------------
def retrieve(query, k=3):
    q_emb = embed(query)
    index, meta = load_index_and_metadata()
    D, I = index.search(np.array([q_emb]), k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({
            "page": meta[idx]["page"],
            "snippet": meta[idx]["text"],
            "distance": float(dist)
        })

    return results


# -----------------------------
# Build context for LLM
# -----------------------------
def build_context(retrieved):
    text = ""
    for item in retrieved:
        text += item["snippet"] + "\n"
    return text.strip()


# -----------------------------
# Call Mistral locally via Ollama
# -----------------------------
def call_mistral(prompt):
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_predict": 350
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, stream=True)

    final_text = ""

    for line in r.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            chunk = data.get("response", "")
            final_text += chunk
        except Exception:
            continue

    return final_text.strip()



# -----------------------------
# End-to-end RAG QA
# -----------------------------
def ask_question(query, k=3):
    retrieved = retrieve(query, k)
    context = build_context(retrieved)

    prompt = f"""
You are a helpful legal assistant.
Read the following document text and give a simple, clear summary.
Use short human-friendly sentences.
Ignore broken OCR text or noise.
Explain the core meaning of the document in 5–6 sentences.

DOCUMENT CONTENT:
{context}

QUESTION:
{query}

Write the answer in plain, easy English:
"""

    answer = call_mistral(prompt)
    cited_pages = ", ".join([f"[Page {r['page']}]" for r in retrieved])
    return answer + f"\n\nCited pages: {cited_pages}"


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    print("\n=== Answer ===\n")
    print(ask_question("give me the summary of the doc"))
