import fitz
import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_all_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append((i+1, text))
    return pages


if __name__ == "__main__":
    pdf_path = "case.pdf"

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    metadata_file = f"{base_name}_metadata.json"
    faiss_file = f"{base_name}_index.faiss"

    print("Extracting all pages...")
    pages = extract_all_pages(pdf_path)

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = []
    metadata = []

    print("\nGenerating embeddings & preparing FAISS index...\n")

    for page_num, text in pages:
        if not text.strip():
            print(f"⚠ Page {page_num} empty. Skipping.")
            continue

        print(f"📄 Embedding Page {page_num}...")

        emb = model.encode(text)
        embeddings.append(emb)

        metadata.append({
            "page": page_num,
            "text": text[:300] + "...",
        })

    # Convert embeddings to numpy array
    emb_matrix = np.array(embeddings).astype("float32")

    print("\nBuilding FAISS index...")
    dimension = emb_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(emb_matrix)

    print(f"Indexed {index.ntotal} vectors.")

    # Save FAISS index
    faiss.write_index(index, faiss_file)

    # Save metadata
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n====================================")
    print(f"🔥 FAISS index saved as: {faiss_file}")
    print(f"🔥 Metadata saved as:   {metadata_file}")
    print("You're officially a vector database daddy now. 😏💥")
