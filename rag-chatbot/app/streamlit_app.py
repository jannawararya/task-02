"""Chat with your PDFs. Every answer shows the passages (file + page) it was based on.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag_core import (  # noqa: E402
    CHUNKS_PATH, GEN_MODEL, MIN_SCORE, TOP_K, Generator, Retriever, answer_question,
)

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading search index...")
def get_retriever():
    return Retriever()


@st.cache_resource(show_spinner="Loading language model (first run downloads it)...")
def get_generator():
    return Generator()


def render_sources(sources: list[dict], abstained: bool) -> None:
    if not sources:
        return
    label = "Closest passages found (not used for an answer)" if abstained else f"Sources ({len(sources)})"
    with st.expander(label):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['source']}, page {s['page']}**  ·  similarity {s['score']:.2f}")
            st.caption(s["text"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    k = st.slider("Passages retrieved", 1, 5, TOP_K)
    min_score = st.slider("Minimum similarity to answer", 0.0, 0.8, MIN_SCORE, 0.05,
                          help="If the best passage scores below this, the bot says it doesn't know.")
    if st.button("Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()
    st.divider()
    st.caption(
        "Answers are generated only from retrieved passages, but a small local model can still "
        "make mistakes. Check the cited page. Similarity scores measure retrieval, "
        "not whether a passage truly answers the question."
    )
    st.caption(f"Generator: `{GEN_MODEL}`")

# ------------------------------------------------------------------- main
st.title("📄 Document Q&A")
st.write("Ask a question about the indexed documents. Answers come with the source page.")

if not CHUNKS_PATH.exists():
    st.error("No index found. Put PDFs in `data/` and run `python index_documents.py` first.")
    st.stop()

retriever = get_retriever()
st.caption("Indexed: " + ", ".join(retriever.manifest["sources"])
           + f"  ({retriever.manifest['num_chunks']} chunks)")

st.session_state.setdefault("messages", [])
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []), m.get("abstained", False))

if question := st.chat_input("Ask a question about your documents"):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and answering..."):
            result = answer_question(question, retriever, get_generator(), k=k, min_score=min_score)
        st.write(result["answer"])
        render_sources(result["sources"], result["abstained"])
        st.caption(f"{result['seconds']:.1f}s · top similarity {result['top_score']:.2f}")

    st.session_state["messages"].append({
        "role": "assistant", "content": result["answer"],
        "sources": result["sources"], "abstained": result["abstained"],
    })
