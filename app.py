import os
import json
import streamlit as st
import numpy as np
import faiss
import urllib.parse
from sentence_transformers import SentenceTransformer

# =========================================
# CONFIG
# =========================================
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ✅ Use jsDelivr CDN for serving PDFs inline
JSDELIVR_BASE = "https://cdn.jsdelivr.net/gh/rydrbot/go-search-app-final@main/pdfs"

# =========================================
# LOAD FAISS INDEX + METADATA
# =========================================
@st.cache_resource
def load_index():
    # Load FAISS index
    index = faiss.read_index("go_index.faiss")

    # Load metadata
    with open("metadata.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Load embedding model (for queries only)
    model = SentenceTransformer(MODEL_NAME)

    return documents, index, model

documents, index, model = load_index()

# =========================================
# SEARCH FUNCTION
# =========================================
def search(query, top_k=5):
    # Create query embedding
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

    # Search FAISS
    similarities, indices = index.search(query_emb, top_k)

    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        doc = documents[idx]
        pdf_file = doc["file_name"].replace("_raw.txt", ".pdf")

        # ✅ Encode filename for URL safety
        pdf_file_encoded = urllib.parse.quote(pdf_file)

        # Build jsDelivr link
        pdf_link = f"{JSDELIVR_BASE}/{pdf_file_encoded}"

        results.append({
            "chunk_id": doc["chunk_id"],
            "doc_id": doc["doc_id"],
            "page": doc["page"],
            "similarity": round(float(sim), 4),
            "translated_text": doc["translated_text"][:500],
            "original_text": doc["original_text"][:500],
            "pdf_link": pdf_link
        })
    return results

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="GO Search", layout="wide")

st.title("📑 Government Order Semantic Search")
st.write("Search across Government Orders (English queries supported).")

query = st.text_input("Enter your search query (English):", "")

top_k = st.slider("Number of results:", 1, 10, 3)

if query:
    results = search(query, top_k=top_k)
    st.write(f"### 🔎 Results for: `{query}`")

    for r in results:
        with st.container():
            st.markdown(f"**📄 Document:** {r['doc_id']} | **Page:** {r['page']}")
            st.markdown(f"**Chunk ID:** {r['chunk_id']} | **Similarity:** {r['similarity']}")
            st.markdown(f"**➡️ English (Translated):** {r['translated_text']}")
            st.markdown(f"**➡️ Malayalam (Original):** {r['original_text']}")
            st.markdown(f"[📎 Open PDF]({r['pdf_link']})")
            st.markdown("---")
