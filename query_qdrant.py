#!/usr/bin/env python3
import argparse
from functools import lru_cache

from sentence_transformers import SentenceTransformer
import numpy as np

from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    from langchain.embeddings.base import Embeddings

from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Default configuration for Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "pdf_rag")
QDRANT_DEVICE = os.environ.get("QDRANT_DEVICE", "cpu")
QDRANT_DIMS = int(os.environ.get("QDRANT_DIMS", "768"))
QDRANT_MODEL_ID = os.environ.get("QDRANT_MODEL_ID", "Snowflake/snowflake-arctic-embed-m-v1.5")





class ArcticEmbedder(Embeddings):
    def __init__(self, model_id: str, device: str = "cpu", dims: int = 768, batch_size: int = 32):
        self.model = SentenceTransformer(model_id, device=device)
        self.dims = dims
        self.batch_size = batch_size

    def _post(self, emb: np.ndarray) -> np.ndarray:
        if self.dims is not None and emb.shape[-1] != self.dims:
            emb = emb[..., : self.dims]
            norm = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12
            emb = emb / norm
        return emb

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) >= 256,
        )
        return self._post(np.asarray(emb)).tolist()

    def embed_query(self, text: str) -> list[float]:
        emb = self.model.encode(
            [text],
            prompt_name="query",          # ✅ important for arctic-embed
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._post(np.asarray(emb))[0].tolist()

@lru_cache(maxsize=1)
def get_vectorstore(qdrant_url: str, collection: str, device: str, dims: int, model_id: str):
    embedder = ArcticEmbedder(model_id=model_id, device=device, dims=dims)
    # LangChain docs: from_existing_collection(...) :contentReference[oaicite:7]{index=7}
    return QdrantVectorStore.from_existing_collection(
        embedding=embedder,
        collection_name=collection,
        url=qdrant_url,
    )


def format_hits(hits):
    lines = []
    for rank, item in enumerate(hits, start=1):
        # item can be Document or (Document, score)
        if isinstance(item, tuple):
            doc, score = item
            score_str = f"{score:.4f}"
        else:
            doc, score_str = item, "n/a"

        meta = doc.metadata or {}
        src = meta.get("source_file") or meta.get("source") or meta.get("source_path") or "unknown"
        page = meta.get("page", meta.get("page_number", ""))
        chunk_id = meta.get("chunk_id", "")

        snippet = doc.page_content.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."

        lines.append(
            f"{rank}) score={score_str} | source={src} | page={page} | chunk={chunk_id}\n   {snippet}"
        )
    return "\n".join(lines) if lines else "No matches found."

@tool("qdrant_pdf_search")
def qdrant_pdf_search(query: str, k: int = 5) -> str:
    """Search the past succesfull itinerary data and return top matching chunks with sources."""
    # Use global defaults if not in main
    url = globals().get('args', type('obj', (object,), {'qdrant_url': QDRANT_URL})).qdrant_url
    collection = globals().get('args', type('obj', (object,), {'collection': QDRANT_COLLECTION})).collection
    device = globals().get('args', type('obj', (object,), {'device': QDRANT_DEVICE})).device
    dims = globals().get('args', type('obj', (object,), {'dims': QDRANT_DIMS})).dims
    model_id = globals().get('args', type('obj', (object,), {'model_id': QDRANT_MODEL_ID})).model_id

    _vs = get_vectorstore(url, collection, device, dims, model_id)
    try:
        _hits = _vs.similarity_search_with_score(query, k=k)
    except Exception:
        _hits = _vs.similarity_search(query, k=k)
    return format_hits(_hits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--collection", default="pdf_rag")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dims", type=int, default=768)
    ap.add_argument("--model_id", default="Snowflake/snowflake-arctic-embed-m-v1.5")
    ap.add_argument("--as_agent", action="store_true", help="Run a LangGraph-backed agent that uses the search tool")
    ap.add_argument("--gemini_model", default="gemini-2.5-flash", help="Gemini model name")
    args = ap.parse_args()

    vs = get_vectorstore(args.qdrant_url, args.collection, args.device, args.dims, args.model_id)

    # Plain retrieval
    try:
        hits = vs.similarity_search_with_score(args.query, k=args.k)
    except Exception:
        # fallback if with_score isn't available in your environment
        hits = vs.similarity_search(args.query, k=args.k)

    print("\n=== Top Matches ===")
    print(format_hits(hits))

    # Optional: expose as Tool + run agent
    if args.as_agent:

        @tool("qdrant_pdf_search")
        def qdrant_pdf_search(query: str, k: int = 5) -> str:
            """Search the PDF knowledge base (Qdrant) and return top matching chunks with sources."""
            _vs = get_vectorstore(args.qdrant_url, args.collection, args.device, args.dims, args.model_id)
            try:
                _hits = _vs.similarity_search_with_score(query, k=k)
            except Exception:
                _hits = _vs.similarity_search(query, k=k)
            return format_hits(_hits)

        # LangChain Agents docs: create_agent + invoke({"messages":[...]}) :contentReference[oaicite:8]{index=8}
        llm = ChatGoogleGenerativeAI(
        model=args.gemini_model,
        temperature=0,
        # Helps avoid "system message not supported" issues in some Gemini setups
        # (If you don't get system-message errors, you can remove this.)
        convert_system_message_to_human=True,
    )

        agent = create_agent(
            model=llm,
            tools=[qdrant_pdf_search],
            system_prompt="You answer using the qdrant_pdf_search tool . Cite sources from the tool output."
        )


        result = agent.invoke({"messages": [{"role": "user", "content": args.query}]})
        print("\n=== Agent Answer ===")
        # the final answer is usually the last message
        last = result["messages"][-1]
        print(getattr(last, "content", last))


if __name__ == "__main__":
    main()
