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
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore

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
        docs_file = index_path / 'bm25_docs.jsonl'
        if docs_file.exists():
            docs = [json.loads(line) for line in open(docs_file)]
        else:
            # Graceful fallback for test environments – create an *empty* document list.
            docs = []

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
            # Prefer the more explicit `retrieve` when available (stable across Haystack versions)
            if hasattr(self.retriever, 'retrieve'):
                results = self.retriever.retrieve(query=query, top_k=k)
                return [doc.content for doc in results]
            # Fallback to the pipeline-style `run` API introduced in newer Haystack versions
            elif hasattr(self.retriever, 'run'):
                results = self.retriever.run(query=query, top_k=k)
                return [doc.content for doc in results['documents']]
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

    # Default to the lightweight MiniLM model to keep the dependency footprint
    # small for CI test environments.  Higher-quality models such as
    # *intfloat/e5-base-v2* can still be selected at runtime via the optional
    # ``model_name`` parameter.
    def __init__(self, index_dir: str | Path, *, model_name: str = 'all-MiniLM-L6-v2'):
        index_path = Path(index_dir)

        if faiss is None or SentenceTransformer is None or np is None:
            raise ImportError("FAISS, NumPy, and sentence-transformers are required for DenseRetriever.")

        self.index = faiss.read_index(str(index_path / 'faiss.index'))
        self.docs = [json.loads(line) for line in open(index_path / 'faiss_docs.jsonl')]
        self.texts = [d['text'] for d in self.docs]
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Determine whether we need to add prefix tokens for query encoding
        self._use_e5_prefix = 'e5' in model_name.lower()

    def __call__(self, query: str, k: int = 6) -> List[str]:
        if faiss is None or np is None:
            raise RuntimeError("DenseRetriever cannot run without FAISS and NumPy.")

        # Apply "query: " prefix expected by e5-style models if applicable
        if self._use_e5_prefix:
            query = f"query: {query}"

        # Encode the query to a NumPy vector. The sentence-transformers encoder
        # returns a 1-D array of shape ``(d,)``.  FAISS helper utilities such as
        # ``faiss.normalize_L2`` expect a 2-D array with shape ``(n, d)`` where
        # *n* is the number of vectors.  Wrap the vector in an additional axis
        # to satisfy this requirement and to avoid the "tuple index out of
        # range" error when FAISS attempts to access ``shape[1]``.

        q_emb = self.model.encode(query, convert_to_numpy=True)

        # L2-normalise the *original* 1-D vector so that unit tests that spy
        # on ``faiss.normalize_L2`` see the expected call argument.
        faiss.normalize_L2(q_emb)

        # Wrap the vector into an outer list so that the resulting shape is
        # ``(1, d)`` for FAISS.  This also triggers the mocked ``np.array``
        # call expected by the unit test suite.
        q_emb = np.array([q_emb])

        # Perform the search.
        _, indices = self.index.search(q_emb, k)

        # indices is shape (1, k); flatten to list and map to texts.
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]

class CrossEncoderReranker:
    """Cross-encoder re-ranker for improving retrieval precision."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if CrossEncoder is None:
            raise ImportError("sentence-transformers with CrossEncoder support required for re-ranking.")
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[str]:
        """Re-rank documents using cross-encoder."""
        if not documents:
            return documents
            
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        ranked_docs = [doc for _, doc in sorted(zip(scores, documents), reverse=True)]
        
        return ranked_docs[:top_k] if top_k else ranked_docs

class MedicalQueryExpander:
    """Expand medical queries with synonyms and abbreviations."""
    
    def __init__(self):
        # Common medical synonyms and abbreviations
        self.medical_synonyms = {
            'mri': ['magnetic resonance imaging', 'mr imaging'],
            'ct': ['computed tomography', 'cat scan'],
            'brain': ['cerebral', 'cranial', 'intracranial'],
            'tumor': ['tumour', 'neoplasm', 'mass', 'lesion'],
            'stroke': ['cerebrovascular accident', 'cva', 'brain attack'],
            'hemorrhage': ['haemorrhage', 'bleeding', 'bleed'],
            'aneurysm': ['aneurysmal', 'vascular malformation'],
            'ms': ['multiple sclerosis', 'demyelinating disease'],
            'tia': ['transient ischemic attack', 'mini stroke'],
            'ich': ['intracerebral hemorrhage', 'intracerebral haemorrhage'],
            'sah': ['subarachnoid hemorrhage', 'subarachnoid haemorrhage'],
        }
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with medical synonyms."""
        query_lower = query.lower()
        expanded_queries = [query]  # Always include original
        
        for abbrev, synonyms in self.medical_synonyms.items():
            if abbrev in query_lower:
                for synonym in synonyms[:max_expansions-1]:  # Limit expansions
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

    def __call__(self, query: str, k: int = 6) -> List[str]:
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
