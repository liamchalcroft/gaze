"""Retrieval implementations for BM25, dense, and hybrid search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever as HS_BM25
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer


class BaseRetriever(Protocol):
    def __call__(self, query: str, k: int = 6) -> list[str]: ...


class BM25Retriever:
    """BM25-based retriever using Haystack InMemoryDocumentStore."""

    def __init__(self, index_dir: str | Path):
        index_path = Path(index_dir)
        docs_file = index_path / "bm25_docs.jsonl"
        if not docs_file.exists():
            raise FileNotFoundError(f"BM25 index file not found: {docs_file}")

        with docs_file.open() as f:
            docs = [json.loads(line) for line in f]

        store = InMemoryDocumentStore()
        hay_docs = [Document(content=d["text"], meta=d) for d in docs]
        store.write_documents(hay_docs)
        self.retriever = HS_BM25(document_store=store)

    def __call__(self, query: str, k: int = 6) -> list[str]:
        results = self.retriever.run(query=query, top_k=k)
        return [doc.content for doc in results["documents"]]


class DenseRetriever:
    """Dense retriever using FAISS and SentenceTransformer embeddings."""

    def __init__(self, index_dir: str | Path, *, model_name: str = "all-MiniLM-L6-v2"):
        index_path = Path(index_dir)
        self.index = faiss.read_index(str(index_path / "faiss.index"))
        with (index_path / "faiss_docs.jsonl").open() as f:
            self.docs = [json.loads(line) for line in f]
        self.texts = [d["text"] for d in self.docs]
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._use_e5_prefix = "e5" in model_name.lower()

    def __call__(self, query: str, k: int = 6) -> list[str]:
        if self._use_e5_prefix:
            query = f"query: {query}"

        q_emb = self.model.encode(query, convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        q_emb = np.array([q_emb])
        _, indices = self.index.search(q_emb, k)
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]


class CrossEncoderReranker:
    """Cross-encoder re-ranker for improving retrieval precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_k: int = None) -> list[str]:
        """Re-rank documents using cross-encoder."""
        if not documents:
            return documents

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score (descending)
        ranked_docs = [doc for _, doc in sorted(zip(scores, documents, strict=False), reverse=True)]

        return ranked_docs[:top_k] if top_k else ranked_docs


class MedicalQueryExpander:
    """Expand medical queries with synonyms and abbreviations."""

    def __init__(self):
        # Common medical synonyms and abbreviations
        self.medical_synonyms = {
            "mri": ["magnetic resonance imaging", "mr imaging"],
            "ct": ["computed tomography", "cat scan"],
            "brain": ["cerebral", "cranial", "intracranial"],
            "tumor": ["tumour", "neoplasm", "mass", "lesion"],
            "stroke": ["cerebrovascular accident", "cva", "brain attack"],
            "hemorrhage": ["haemorrhage", "bleeding", "bleed"],
            "aneurysm": ["aneurysmal", "vascular malformation"],
            "ms": ["multiple sclerosis", "demyelinating disease"],
            "tia": ["transient ischemic attack", "mini stroke"],
            "ich": ["intracerebral hemorrhage", "intracerebral haemorrhage"],
            "sah": ["subarachnoid hemorrhage", "subarachnoid haemorrhage"],
        }

    def expand_query(self, query: str, max_expansions: int = 3) -> list[str]:
        """Expand query with medical synonyms."""
        query_lower = query.lower()
        expanded_queries = [query]  # Always include original

        for abbrev, synonyms in self.medical_synonyms.items():
            if abbrev in query_lower:
                for synonym in synonyms[: max_expansions - 1]:  # Limit expansions
                    expanded_query = query_lower.replace(abbrev, synonym)
                    if expanded_query != query_lower:
                        expanded_queries.append(expanded_query.title())

        return list(set(expanded_queries))  # Remove duplicates


class HybridRetriever:
    """Hybrid retriever using Reciprocal Rank Fusion of BM25 and dense."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        alpha: float = 0.5,
        reranker: CrossEncoderReranker = None,
        query_expander: MedicalQueryExpander = None,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.reranker = reranker
        self.query_expander = query_expander

    def __call__(self, query: str, k: int = 6) -> list[str]:
        # Expand query if expander is available
        queries = [query]
        if self.query_expander:
            queries = self.query_expander.expand_query(query)

        # Retrieve more candidates for re-ranking
        retrieval_k = k * 3 if self.reranker else k

        all_results = {}
        for q in queries:
            bm25_res = self.bm25(q, retrieval_k)
            dense_res = self.dense(q, retrieval_k)

            scores: dict[str, float] = {}
            for rank, doc in enumerate(bm25_res):
                scores[doc] = scores.get(doc, 0.0) + self.alpha * (1.0 / (rank + 1))
            for rank, doc in enumerate(dense_res):
                scores[doc] = scores.get(doc, 0.0) + (1 - self.alpha) * (1.0 / (rank + 1))

            # Aggregate scores across query variants
            for doc, score in scores.items():
                all_results[doc] = all_results.get(doc, 0.0) + score

        fused = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        initial_results = [doc for doc, _ in fused]

        # Apply re-ranking if available
        if self.reranker:
            return self.reranker.rerank(query, initial_results, k)
        else:
            return initial_results[:k]
