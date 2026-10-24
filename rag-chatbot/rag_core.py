"""Core RAG pipeline: PDF ingestion, chunking, retrieval, grounded generation.

Heavy libraries (torch, faiss, transformers) are imported lazily inside the
functions that need them, so chunking and PDF parsing can be tested on their own.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

# Models can be swapped with environment variables, e.g.  RAG_GEN_MODEL=google/flan-t5-large
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("RAG_GEN_MODEL", "google/flan-t5-base")

CHUNK_WORDS = 110      # words per chunk (keeps 3 chunks inside the generator's input window)
OVERLAP_WORDS = 25     # words shared between neighbouring chunks
TOP_K = 3              # passages passed to the generator
MIN_SCORE = 0.30       # below this top similarity, the bot abstains (tune on your eval set)

NO_ANSWER = "I couldn't find enough information in the documents to answer that."


# ---------------------------------------------------------------- ingestion
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def extract_pages(pdf_path: Path):
    """Yield (page_number, text) for every page that contains extractable text."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            yield number, text
        else:
            print(f"  warning: {pdf_path.name} page {number} has no extractable text "
                  "(scanned image? this page will be skipped)")


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(pdf_dir: Path = DATA_DIR) -> list[dict]:
    pdfs = sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}. Add at least one PDF.")
    chunks: list[dict] = []
    for pdf in pdfs:
        before = len(chunks)
        pages = 0
        for page, text in extract_pages(pdf):
            pages += 1
            for piece in chunk_words(text):
                chunks.append({"id": len(chunks), "source": pdf.name, "page": page, "text": piece})
        print(f"  {pdf.name}: {pages} pages -> {len(chunks) - before} chunks")
    if not chunks:
        raise RuntimeError("PDFs were found but no text could be extracted (scanned PDFs need OCR).")
    return chunks


def build_index() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print("Reading PDFs...")
    chunks = build_chunks()

    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} (first run downloads the model)...")
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode([c["text"] for c in chunks], batch_size=32,
                       normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_PATH, emb)  # also used as a fallback if FAISS is unavailable
    try:
        import faiss

        index = faiss.IndexFlatIP(emb.shape[1])  # inner product == cosine on normalised vectors
        index.add(emb)
        faiss.write_index(index, str(INDEX_PATH))
        store = "faiss"
    except ImportError:
        print("  faiss not installed, falling back to NumPy search")
        store = "numpy"

    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    MANIFEST_PATH.write_text(json.dumps({
        "embed_model": EMBED_MODEL,
        "vector_store": store,
        "num_chunks": len(chunks),
        "sources": sorted({c["source"] for c in chunks}),
        "chunk_words": CHUNK_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print(f"Done. Index saved to {INDEX_DIR}/ ({store}, {len(chunks)} chunks)")


# ---------------------------------------------------------------- retrieval
class Retriever:
    def __init__(self) -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if not CHUNKS_PATH.exists():
            raise FileNotFoundError("Index not found. Run `python index_documents.py` first.")
        self.manifest = json.loads(MANIFEST_PATH.read_text())
        self.chunks = json.loads(CHUNKS_PATH.read_text())
        self.model = SentenceTransformer(self.manifest["embed_model"])
        self._index = None
        self._emb = None
        if self.manifest["vector_store"] == "faiss" and INDEX_PATH.exists():
            try:
                import faiss

                self._index = faiss.read_index(str(INDEX_PATH))
            except ImportError:
                pass
        if self._index is None:
            self._emb = np.load(EMBEDDINGS_PATH)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        import numpy as np

        k = min(k, len(self.chunks))
        q = np.asarray(self.model.encode([query], normalize_embeddings=True), dtype="float32")
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            pairs = zip(scores[0], ids[0])
        else:
            sims = (self._emb @ q[0])
            top = np.argsort(-sims)[:k]
            pairs = ((sims[i], i) for i in top)
        return [{**self.chunks[int(i)], "score": float(s)} for s, i in pairs if int(i) != -1]


# --------------------------------------------------------------- generation
class Generator:
    """Local seq2seq model that answers strictly from the supplied passages."""

    def __init__(self, model_name: str = GEN_MODEL) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()

    def generate(self, question: str, passages: list[dict]) -> str:
        import torch

        context = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, 1))
        prompt = (
            "Answer the question using only the context below. Give a complete, factual answer. "
            "If the context does not contain the answer, reply exactly: I don't know.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=160, num_beams=4)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return (not t) or any(s in t for s in ("don't know", "do not know", "unanswerable", "not mentioned"))


def answer_question(question: str, retriever: Retriever, generator: Generator | None,
                    k: int = TOP_K, min_score: float = MIN_SCORE) -> dict:
    """Retrieve, decide whether to abstain, then generate a grounded answer."""
    t0 = time.perf_counter()
    passages = retriever.search(question, k)
    top = passages[0]["score"] if passages else 0.0

    if top < min_score:
        answer, abstained, reason = NO_ANSWER, True, "low_retrieval_score"
    elif generator is None:
        answer, abstained, reason = "(retrieval only, no generation)", False, "retrieval_only"
    else:
        raw = generator.generate(question, passages)
        if _is_refusal(raw):
            answer, abstained, reason = NO_ANSWER, True, "model_refused"
        else:
            answer, abstained, reason = raw, False, "answered"

    return {"answer": answer, "abstained": abstained, "reason": reason,
            "top_score": top, "sources": passages, "seconds": time.perf_counter() - t0}
