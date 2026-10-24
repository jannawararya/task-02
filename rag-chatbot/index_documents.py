"""Build the searchable index from every PDF in data/.

Run:  python index_documents.py
Creates index/ (FAISS index, embeddings, chunk text + page metadata, manifest).
Re-run whenever you add or change PDFs.
"""
from rag_core import build_index

if __name__ == "__main__":
    build_index()
