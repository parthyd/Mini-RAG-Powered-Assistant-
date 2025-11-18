import os
import json
import uuid
import time
import faiss
import fitz
import tiktoken
import numpy as np
import requests
import traceback
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from sentence_transformers import SentenceTransformer

# --------------------
# CONFIG
# --------------------
UPLOAD_DIR = "uploads"
INDEX_DIR = "indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Flask
app = Flask(__name__)
app.secret_key = "supersecret"


# --------------------
# Logging helper (terminal only)
# --------------------
def log(msg):
    print(f"[SERVER] {msg}", flush=True)


# --------------------
# PDF text extraction
# --------------------
def extract_pages(pdf_path):
    log("Reading PDF pages...")
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        text = text.replace("\u0000", "").replace("\x00", "")
        pages.append({"page": i+1, "text": text})

    log(f"Total pages extracted: {len(pages)}")
    return pages


# --------------------
# Chunking
# --------------------
def chunk_text(text, max_tokens=500, overlap=50):
    tokens = enc.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i: i + max_tokens]
        chunk_text = enc.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        i += max_tokens - overlap

    return chunks


# --------------------
# Build FAISS index
# --------------------
def build_faiss_for_pdf(pdf_path, base_name):
    log("Starting indexing pipeline...")
    pages = extract_pages(pdf_path)

    all_chunks = []
    for p in pages:
        if not p["text"]:
            continue
        chunks = chunk_text(p["text"])
        for c in chunks:
            all_chunks.append({"page": p["page"], "text": c})

    log(f"Total chunks created: {len(all_chunks)}")
    if not all_chunks:
        raise ValueError("No text found in PDF")

    # Generate embeddings
    log("Generating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    emb_matrix = np.array(embeddings).astype("float32")

    # FAISS index
    log("Building FAISS index...")
    d = emb_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(emb_matrix)

    index_file = f"{INDEX_DIR}/{base_name}_index.faiss"
    meta_file = f"{INDEX_DIR}/{base_name}_meta.json"

    faiss.write_index(index, index_file)

    metadata = []
    for c in all_chunks:
        metadata.append({
            "page": c["page"],
            "text": c["text"][:300]
        })

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log("Index building completed.")
    return index_file, meta_file


# --------------------
# Retrieve
# --------------------
def retrieve_top_k(index_path, meta_path, query, k=3):
    log("Retrieving relevant chunks...")
    idx = faiss.read_index(index_path)

    q_emb = embedder.encode(query).astype("float32")
    D, I = idx.search(np.array([q_emb]), k)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    results = []
    for dist, i in zip(D[0], I[0]):
        results.append(metadata[i])

    return results


# --------------------
# Extract Legal Sections
# --------------------
import re

SECTION_PATTERNS = [
    r"\bSection\s+\d+[A-Za-z]?\b",
    r"\bSec\.?\s*\d+[A-Za-z]?\b",
    r"\bU/s\s*\d+[A-Za-z]?\b",
    r"\b\d+\s*IPC\b",
    r"\b\d+\s*NIA\b",
    r"\b\d+\s*CrPC\b",
]

def extract_sections(text):
    found = set()
    for pat in SECTION_PATTERNS:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            found.add(m.strip())
    return list(found)


# --------------------
# Call Ollama LLM
# --------------------
def call_ollama(prompt):
    log("Calling Mistral via Ollama...")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }

    r = requests.post(OLLAMA_URL, json=payload, stream=True)
    final_text = ""

    for line in r.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode())
            if "response" in data:
                final_text += data["response"]
        except:
            continue

    log("Mistral generated response.")
    return final_text.strip()


# --------------------
# Routes
# --------------------
@app.route("/")
def index():
    docs = [f.replace("_index.faiss", "") for f in os.listdir(INDEX_DIR) if f.endswith("_index.faiss")]
    return render_template("index.html", docs=docs)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    filename = file.filename
    base_name = os.path.splitext(filename)[0]
    unique = uuid.uuid4().hex[:8]
    safe_base = f"{base_name}_{unique}"

    save_path = f"{UPLOAD_DIR}/{safe_base}.pdf"
    file.save(save_path)

    try:
        index_file, meta_file = build_faiss_for_pdf(save_path, safe_base)
        flash("Document indexed successfully!", "success")
    except Exception as e:
        flash(str(e), "danger")

    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    doc = request.form.get("doc")
    query = request.form.get("query")

    index_path = f"{INDEX_DIR}/{doc}_index.faiss"
    meta_path = f"{INDEX_DIR}/{doc}_meta.json"

    retrieved = retrieve_top_k(index_path, meta_path, query, k=4)

    # Build context
    pages = []
    context = ""
    for r in retrieved:
        pages.append(r["page"])
        context += r["text"] + "\n"

    sections = extract_sections(context)

    prompt = f"""
You are a legal assistant. Read the context and answer in very simple English.

CONTEXT:
{context}

QUESTION:
{query}

Write a clear answer:
"""

    answer = call_ollama(prompt)

    return render_template(
        "index.html",
        docs=[f.replace("_index.faiss", "") for f in os.listdir(INDEX_DIR) if f.endswith("_index.faiss")],
        answer=answer,
        pages=pages,
        sections=sections,
        selected_doc=doc,
        query=query
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
