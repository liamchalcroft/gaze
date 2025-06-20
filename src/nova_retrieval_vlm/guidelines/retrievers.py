from __future__ import annotations

import json
from typing import Protocol, List
from pathlib import Path

try:
    import numpy as np
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
    faiss = None  # type: ignore

# sentence-transformers (requires torch)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# haystack (used for BM25)
try:
    from haystack.document_stores.in_memory import InMemoryDocumentStore  # type: ignore
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever as HS_BM25  # type: ignore
    from haystack.dataclasses import Document  # type: ignore
except Exception:  # pragma: no cover
    InMemoryDocumentStore = HS_BM25 = Document = None  # type: ignore

class BaseRetriever(Protocol):  # typing.Protocol
    def __call__(self, query: str, k: int = 6) -> List[str]: ...

class BM25Retriever:
    """BM25-based retriever using Haystack InMemoryDocumentStore."""
    def __init__(self, index_dir: str | Path):
        index_path = Path(index_dir)
        docs = [json.loads(line) for line in open(index_path / 'bm25_docs.jsonl')]

        if InMemoryDocumentStore is None or HS_BM25 is None:
            raise ImportError("haystack not available - BM25Retriever cannot be instantiated.")

        store = InMemoryDocumentStore()
        hay_docs = [Document(content=d['text'], meta=d) for d in docs]
        store.write_documents(hay_docs)
        self.retriever = HS_BM25(document_store=store)
        
        # Verify the retriever has the expected method (try 'run' first, then 'retrieve')
        if not hasattr(self.retriever, 'run') and not hasattr(self.retriever, 'retrieve'):
            available_methods = [attr for attr in dir(self.retriever) if not attr.startswith('_')]
            raise AttributeError(
                f"BM25Retriever.retriever object has no 'run' or 'retrieve' method. "
                f"Available methods: {available_methods}. "
                f"This may indicate a Haystack version compatibility issue."
            )

    def __call__(self, query: str, k: int = 6) -> List[str]:
        try:
            # Try 'run' method first (newer Haystack versions)
            if hasattr(self.retriever, 'run'):
                results = self.retriever.run(query=query, top_k=k)
                return [doc.content for doc in results['documents']]
            # Fallback to 'retrieve' method (older versions)
            elif hasattr(self.retriever, 'retrieve'):
                results = self.retriever.retrieve(query=query, top_k=k)
                return [doc.content for doc in results]
            else:
                available_methods = [attr for attr in dir(self.retriever) if not attr.startswith('_')]
                raise AttributeError(
                    f"BM25Retriever failed. Available methods: {available_methods}. "
                    f"Haystack version compatibility issue detected."
                )
        except Exception as e:
            available_methods = [attr for attr in dir(self.retriever) if not attr.startswith('_')]
            raise AttributeError(
                f"BM25Retriever failed during execution. Available methods: {available_methods}. "
                f"Error: {e}"
            ) from e

class DenseRetriever:
    """Dense retriever using FAISS and SentenceTransformer embeddings."""
    def __init__(self, index_dir: str | Path):
        index_path = Path(index_dir)

        if faiss is None or SentenceTransformer is None or np is None:
            raise ImportError("FAISS, NumPy, and sentence-transformers are required for DenseRetriever.")

        self.index = faiss.read_index(str(index_path / 'faiss.index'))
        self.docs = [json.loads(line) for line in open(index_path / 'faiss_docs.jsonl')]
        self.texts = [d['text'] for d in self.docs]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def __call__(self, query: str, k: int = 6) -> List[str]:
        if faiss is None or np is None:
            raise RuntimeError("DenseRetriever cannot run without FAISS and NumPy.")
        q_emb = self.model.encode(query, convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, indices = self.index.search(np.array([q_emb]), k)
        return [self.texts[i] for i in indices[0]]

class HybridRetriever:
    """Hybrid retriever using Reciprocal Rank Fusion of BM25 and dense."""
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        alpha: float = 0.5,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha

    def __call__(self, query: str, k: int = 6) -> List[str]:
        bm25_res = self.bm25(query, k)
        dense_res = self.dense(query, k)
        scores: dict[str, float] = {}
        for rank, doc in enumerate(bm25_res):
            scores[doc] = scores.get(doc, 0.0) + self.alpha * (1.0 / (rank + 1))
        for rank, doc in enumerate(dense_res):
            scores[doc] = scores.get(doc, 0.0) + (1 - self.alpha) * (1.0 / (rank + 1))
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in fused][:k]
