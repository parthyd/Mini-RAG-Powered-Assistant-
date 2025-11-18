import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------
# CONFIG
# -----------------------

FAISS_INDEX_FILE = "case_index.faiss"
METADATA_FILE = "case_metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "google/flan-t5-small"   # You can upgrade later

# -----------------------
# Load embedding model
# -----------------------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)


def embed(text: str):
    return embedder.encode(text).astype("float32")


# -----------------------
# Load FAISS + metadata
# -----------------------
def load_index_and_meta():
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError("FAISS index missing!")

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("Metadata missing!")

    index = faiss.read_index(FAISS_INDEX_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return index, meta


# -----------------------
# Retrieve Relevant Pages
# -----------------------
def retrieve(query: str, k=3):
    q_emb = embed(query)
    index, meta = load_index_and_meta()

    D, I = index.search(np.array([q_emb]), k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta):
            continue

        results.append({
            "page": meta[idx]["page"],
            "snippet": meta[idx]["text"],
            "distance": float(dist)
        })

    return results


# -----------------------
# Build the context
# -----------------------
def build_context(retrieved_pages):
    parts = []
    for item in retrieved_pages:
        parts.append(f"[Page {item['page']}]\n{item['snippet']}")
    return "\n\n".join(parts)


# -----------------------
# Local HuggingFace LLM
# -----------------------
print("Loading local LLM...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL)
generator = pipeline("text2text-generation", model=hf_model, tokenizer=tokenizer)


def answer_locally(query: str, context: str):
    prompt = (
        "Use the context to answer the question in VERY simple language. "
        "Use short sentences. Always cite page numbers like [Page 2]. "
        "If the answer is not in the context, say you don't have enough information.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    )

    output = generator(prompt, max_length=300, do_sample=False)[0]["generated_text"]
    return output.strip()


# -----------------------
# Main Query Function
# -----------------------
def ask_question(query: str, k=3):
    retrieved = retrieve(query, k)
    if not retrieved:
        return "No relevant pages found."

    context = build_context(retrieved)

    print("Retrieved pages:", [r["page"] for r in retrieved])

    answer = answer_locally(query, context)

    citations = ", ".join([f"[Page {r['page']}]" for r in retrieved])

    return answer + f"\n\nCited pages: {citations}"


# -----------------------
# Example Run
# -----------------------
if __name__ == "__main__":
    user_query = "who is plaintiff and who was defendant"
    response = ask_question(user_query)

    print("\n===== ANSWER =====\n")
    print(response)
