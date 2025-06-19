from __future__ import annotations
import json
from pathlib import Path
from typing import Any, List

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever as HS_BM25
import faiss
from sentence_transformers import SentenceTransformer

from .ingest_nice import ingest_nice


def build_bm25(
    docs: List[dict[str, Any]],
    index_dir: Path | str,
) -> None:
    """
    Build and persist a BM25 index for the given docs.

    Args:
        docs: List of docs as returned by ingest_nice.
        index_dir: Directory to store bm25_docs.jsonl.
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    # Write docs to JSONL
    with open(index_path / 'bm25_docs.jsonl', 'w') as fw:
        for doc in docs:
            fw.write(json.dumps(doc) + '\n')
    # Load into Haystack InMemoryDocumentStore
    from haystack.dataclasses import Document

    store = InMemoryDocumentStore()
    hay_docs = [Document(content=doc['text'], meta={'doc_id': doc['doc_id'], 'chunk_id': doc['chunk_id']}) for doc in docs]
    store.write_documents(hay_docs)
    # Initialize retriever to ensure indexing
    _ = HS_BM25(document_store=store)


def build_faiss(
    docs: List[dict[str, Any]],
    index_dir: Path | str,
) -> None:
    """
    Build and persist a FAISS dense index for the given docs.

    Args:
        docs: List of docs as returned by ingest_nice.
        index_dir: Directory to store faiss.index and faiss_docs.jsonl.
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    texts = [doc['text'] for doc in docs]
    # Compute embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    # Normalize if using inner-product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    # Persist index
    faiss.write_index(index, str(index_path / 'faiss.index'))
    # Persist docs metadata
    with open(index_path / 'faiss_docs.jsonl', 'w') as fw:
        for doc in docs:
            fw.write(json.dumps(doc) + '\n')


def build_indexes(
    config_yaml: str,
    raw_dir: str,
    output_dir: Path | str,
) -> None:
    """
    Ingest NICE docs and build both BM25 and FAISS indexes.
    """
    docs = ingest_nice(config_yaml, raw_dir)
    build_bm25(docs, Path(output_dir) / 'bm25')
    build_faiss(docs, Path(output_dir) / 'faiss')
