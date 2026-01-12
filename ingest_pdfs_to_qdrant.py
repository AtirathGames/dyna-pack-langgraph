#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings


class ArcticEmbedder(Embeddings):
    """
    LangChain-compatible embeddings wrapper.

    Uses prompt_name="query" for queries (recommended for arctic-embed).
    """
    def __init__(self, model_id: str, device: str = "cpu", dims: int = 768, batch_size: int = 32):
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)
        self.dims = dims
        self.batch_size = batch_size

    def _post(self, emb: np.ndarray) -> np.ndarray:
        # optional truncation (e.g., 256) + re-normalize
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
        emb = self._post(np.asarray(emb))
        return emb.tolist()

    def embed_query(self, text: str) -> list[float]:
        emb = self.model.encode(
            [text],
            prompt_name="query",          # ✅ important for this model
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = self._post(np.asarray(emb))[0]
        return emb.tolist()


def load_all_pdfs(pdf_dir: Path):
    pdf_paths = sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])
    all_docs = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()  # one Document per page
        # add a cleaner source name
        for d in docs:
            d.metadata["source_file"] = pdf_path.name
            d.metadata["source_path"] = str(pdf_path)
        all_docs.extend(docs)

    return all_docs, pdf_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Folder containing PDFs (recursively scanned)")
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--collection", default="pdf_rag")
    ap.add_argument("--recreate", action="store_true", help="Delete and recreate the collection")
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=64, help="Upsert batch size (documents)")
    ap.add_argument("--embed_batch", type=int, default=32, help="Embedding batch size")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--dims", type=int, default=768, help="768 (default) or 256 (truncate)")
    ap.add_argument("--model_id", default="Snowflake/snowflake-arctic-embed-m-v1.5")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    if not pdf_dir.exists():
        raise SystemExit(f"PDF dir not found: {pdf_dir}")

    print(f"[1/5] Loading PDFs from: {pdf_dir}")
    docs, pdfs = load_all_pdfs(pdf_dir)
    print(f"  Found PDFs: {len(pdfs)} | Loaded pages/docs: {len(docs)}")

    if not docs:
        raise SystemExit("No PDFs/pages loaded. Are they scanned images with no text?")

    print("[2/5] Splitting into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    print(f"  Chunks: {len(chunks)}")

    print("[3/5] Setting up embedder")
    embedder = ArcticEmbedder(
        model_id=args.model_id,
        device=args.device,
        dims=args.dims,
        batch_size=args.embed_batch,
    )

    print("[4/5] Connecting to Qdrant + creating collection if needed")
    client = QdrantClient(url=args.qdrant_url)

    # determine vector size (dims)
    vector_size = args.dims

    # (re)create collection
    if args.recreate:
        try:
            client.delete_collection(collection_name=args.collection)
            print(f"  Deleted existing collection: {args.collection}")
        except Exception:
            pass

    exists = True
    try:
        client.get_collection(args.collection)
    except Exception:
        exists = False

    if not exists:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )
        print(f"  Created collection: {args.collection} (size={vector_size}, cosine)")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=args.collection,
        embedding=embedder,
    )

    print("[5/5] Uploading chunks to Qdrant")
    total = len(chunks)
    for start in range(0, total, args.batch_size):
        batch = chunks[start : start + args.batch_size]
        ids = [str(uuid4()) for _ in batch]
        vector_store.add_documents(documents=batch, ids=ids)
        print(f"  Upserted {min(start + args.batch_size, total)}/{total}")

    print("\n✅ Done.")
    print(f"Collection: {args.collection}")
    print(f"Qdrant URL:  {args.qdrant_url}")


if __name__ == "__main__":
    main()
