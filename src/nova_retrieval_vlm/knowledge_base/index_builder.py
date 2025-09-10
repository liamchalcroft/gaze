from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

import faiss
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever as HS_BM25
from haystack.document_stores.in_memory import InMemoryDocumentStore
from sentence_transformers import SentenceTransformer

from .ingest_nice import ingest_nice


def build_bm25(
    docs: list[dict[str, Any]],
    index_dir: Path | str,
) -> None:
    """
    Build and persist a BM25 index for the given docs.

    Args:
        docs: List of docs as returned by ingest_nice.
        index_dir: Directory to store bm25_docs.jsonl.
    """
    if not docs:
        return

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    # Write docs to JSONL
    with open(index_path / "bm25_docs.jsonl", "w") as fw:
        for doc in docs:
            fw.write(json.dumps(doc) + "\n")
    # Load into Haystack InMemoryDocumentStore
    from haystack.dataclasses import Document

    store = InMemoryDocumentStore()
    hay_docs = [
        Document(content=doc["text"], meta={"doc_id": doc["doc_id"], "chunk_id": doc["chunk_id"]})
        for doc in docs
    ]
    store.write_documents(hay_docs)
    # Initialize retriever to ensure indexing
    _ = HS_BM25(document_store=store)


def build_faiss(
    docs: list[dict[str, Any]],
    index_dir: Path | str,
    batch_size: int = 64,
) -> None:
    """
    Build and persist a FAISS dense index for the given docs.

    Args:
        docs: List of docs as returned by ingest_nice.
        index_dir: Directory to store faiss.index and faiss_docs.jsonl.
        batch_size: Batch size for embedding computation.
    """
    if not docs:
        return

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    texts = [doc["text"] for doc in docs]

    # --------------------------------------------------------------
    # Choose a state-of-the-art embedding model tuned for retrieval.
    # "intfloat/e5-base-v2" consistently outperforms MiniLM on RAG
    # benchmarks and understands biomedical language well.
    # We follow the recommended "passage: " prefix for document
    # embeddings (queries will be prefixed with "query: " at search
    # time by the DenseRetriever).
    # --------------------------------------------------------------
    model_name = "intfloat/e5-base-v2"
    model = SentenceTransformer(model_name)
    passages = [f"passage: {t}" for t in texts]

    embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )

    # L2-normalise so that inner product ~ cosine similarity
    faiss.normalize_L2(embeddings)

    if embeddings.size == 0:
        return

    dim = embeddings.shape[1]

    # Use HNSW for sub-linear ANN search while maintaining high recall.
    # Optimized parameters based on retrieval research:
    # - M=64: Higher connections for better recall (vs default 32)
    # - efConstruction=400: More thorough construction (vs default 200)
    # - efSearch=128: More candidates during search (vs default 64)
    try:
        index = faiss.IndexHNSWFlat(dim, 64)  # M=64 for better accuracy
        index.hnsw.efConstruction = 400  # Higher for better index quality
        index.hnsw.efSearch = 128  # Higher for better search recall
    except AttributeError:
        # Fallback to exact search if HNSW not available in the FAISS build
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    # Persist index
    faiss.write_index(index, str(index_path / "faiss.index"))
    # Persist docs metadata
    with open(index_path / "faiss_docs.jsonl", "w") as fw:
        for doc in docs:
            fw.write(json.dumps(doc) + "\n")


def build_indexes(
    config_path: str,
    raw_dir: str,
    index_dir: str | Path,
    *,
    num_workers: int = 4,
    parallel_index_build: bool = True,
    robots_mode: str = "strict",
) -> None:
    """
    Ingest guideline documents (see `ingest_nice`) and build both BM25 and
    FAISS indexes.  The resulting artefacts are saved under `index_dir`.

    Args:
        config_path: Path to guidelines YAML configuration.
        raw_dir: Directory where raw downloaded files will be cached.
        index_dir: Directory to save index artifacts.
        num_workers: Number of workers for parallel ingestion.
        parallel_index_build: Whether to build BM25 and FAISS indexes in parallel.
        robots_mode: Mode for robots (default "strict").
    """
    docs = ingest_nice(
        config_path,
        raw_dir,
        verbose=True,
        num_workers=num_workers,
        robots_mode=robots_mode,
    )
    if not docs:
        return

    if parallel_index_build:
        # Build BM25 and FAISS indexes in parallel
        with ProcessPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(build_bm25, docs, Path(index_dir) / "bm25")
            faiss_future = executor.submit(build_faiss, docs, Path(index_dir) / "faiss")

            # Wait for both to complete
            for future in as_completed([bm25_future, faiss_future]):
                try:
                    future.result()  # This will raise any exception that occurred
                except Exception:
                    raise
    else:
        # Sequential build
        build_bm25(docs, Path(index_dir) / "bm25")
        build_faiss(docs, Path(index_dir) / "faiss")
