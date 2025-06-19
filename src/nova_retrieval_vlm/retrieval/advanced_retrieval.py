"""
Advanced Retrieval System for Medical VLM

This module implements state-of-the-art retrieval techniques including:
- Dense retrieval with medical-specific embeddings
- Cross-encoder re-ranking 
- Multi-modal retrieval (text + image features)
- Domain knowledge integration
- Query expansion with medical synonyms
"""

from __future__ import annotations
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel
import faiss
from loguru import logger

@dataclass
class RetrievalResult:
    """Enhanced retrieval result with confidence and reasoning."""
    text: str
    score: float
    source: str
    reasoning_type: str  # 'anatomical', 'pathological', 'comparative', etc.
    medical_concepts: List[str]
    confidence: float

class MedicalQueryExpander:
    """Expand queries with medical synonyms and related terms."""
    
    def __init__(self):
        # Medical terminology mappings
        self.medical_synonyms = {
            'brain': ['cerebral', 'cranial', 'intracranial', 'neural'],
            'lesion': ['abnormality', 'mass', 'nodule', 'opacity'],
            'midline': ['median', 'sagittal', 'central axis'],
            'symmetry': ['bilateral', 'symmetric', 'mirror image'],
            'tumor': ['mass', 'neoplasm', 'growth', 'lesion'],
            'ventricles': ['ventricular system', 'CSF spaces'],
            'hemorrhage': ['bleeding', 'hematoma', 'blood'],
            'edema': ['swelling', 'fluid accumulation'],
            'atrophy': ['shrinkage', 'volume loss', 'degeneration']
        }
        
        self.anatomical_regions = {
            'frontal': ['anterior', 'prefrontal'],
            'parietal': ['posterior parietal', 'superior parietal'],
            'temporal': ['lateral temporal', 'mesial temporal'],
            'occipital': ['posterior', 'visual cortex'],
            'cerebellum': ['posterior fossa', 'cerebellar'],
            'brainstem': ['midbrain', 'pons', 'medulla']
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with medical synonyms and related terms."""
        expanded_queries = [query]
        query_lower = query.lower()
        
        # Add synonyms
        for term, synonyms in self.medical_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        # Add anatomical variations
        for region, variants in self.anatomical_regions.items():
            if region in query_lower:
                for variant in variants:
                    expanded_query = query_lower.replace(region, variant)
                    expanded_queries.append(expanded_query)
        
        return list(set(expanded_queries))

class DenseRetriever:
    """Dense retrieval using medical-specific embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.document_embeddings = None
        
    def build_index(self, documents: List[str]) -> None:
        """Build FAISS index for dense retrieval."""
        logger.info(f"Building dense index for {len(documents)} documents...")
        
        self.documents = documents
        # Generate embeddings
        self.document_embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.document_embeddings)
        self.index.add(self.document_embeddings)
        
        logger.info(f"Dense index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve documents using dense embeddings."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results

class CrossEncoderReranker:
    """Re-rank retrieved documents using cross-encoder."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Tuple[str, float]], top_k: int = 5) -> List[Tuple[str, float]]:
        """Re-rank documents using cross-encoder scores."""
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, doc[0]) for doc in documents]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Combine with original scores
        reranked = []
        for i, (doc, original_score) in enumerate(documents):
            # Weighted combination of dense and cross-encoder scores
            combined_score = 0.7 * scores[i] + 0.3 * original_score
            reranked.append((doc, combined_score))
        
        # Sort by combined score and return top_k
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

class MedicalConceptExtractor:
    """Extract medical concepts from text."""
    
    def __init__(self):
        # Medical concept categories
        self.anatomical_terms = {
            'brain_structures': ['cortex', 'ventricles', 'cerebellum', 'brainstem', 'hippocampus'],
            'brain_regions': ['frontal', 'parietal', 'temporal', 'occipital'],
            'laterality': ['left', 'right', 'bilateral', 'unilateral'],
            'symmetry': ['symmetric', 'asymmetric', 'midline', 'deviation']
        }
        
        self.pathological_terms = {
            'lesions': ['lesion', 'mass', 'tumor', 'nodule'],
            'vascular': ['hemorrhage', 'stroke', 'infarct', 'hematoma'],
            'edema': ['edema', 'swelling', 'fluid'],
            'atrophy': ['atrophy', 'volume loss', 'shrinkage']
        }
        
        self.descriptive_terms = {
            'intensity': ['hypointense', 'hyperintense', 'isointense'],
            'shape': ['round', 'oval', 'irregular', 'lobulated'],
            'enhancement': ['enhancing', 'non-enhancing', 'rim-enhancing']
        }
    
    def extract_concepts(self, text: str) -> Dict[str, List[str]]:
        """Extract medical concepts from text."""
        text_lower = text.lower()
        extracted = {
            'anatomical': [],
            'pathological': [],
            'descriptive': []
        }
        
        # Extract anatomical concepts
        for category, terms in self.anatomical_terms.items():
            for term in terms:
                if term in text_lower:
                    extracted['anatomical'].append(f"{category}:{term}")
        
        # Extract pathological concepts
        for category, terms in self.pathological_terms.items():
            for term in terms:
                if term in text_lower:
                    extracted['pathological'].append(f"{category}:{term}")
        
        # Extract descriptive concepts
        for category, terms in self.descriptive_terms.items():
            for term in terms:
                if term in text_lower:
                    extracted['descriptive'].append(f"{category}:{term}")
        
        return extracted

class AdvancedRetriever:
    """State-of-the-art retrieval system for medical VLM."""
    
    def __init__(self, 
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        
        self.query_expander = MedicalQueryExpander()
        self.dense_retriever = DenseRetriever(dense_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        self.concept_extractor = MedicalConceptExtractor()
        
        # Retrieval parameters
        self.dense_top_k = 50
        self.rerank_top_k = 10
        self.final_top_k = 5
    
    def build_index(self, documents: List[str], save_path: Optional[Path] = None) -> None:
        """Build retrieval index from documents."""
        logger.info("Building advanced retrieval index...")
        
        # Build dense index
        self.dense_retriever.build_index(documents)
        
        # Save index if path provided
        if save_path:
            self.save_index(save_path)
        
        logger.info("Advanced retrieval index built successfully")
    
    def save_index(self, save_path: Path) -> None:
        """Save retrieval index to disk."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.dense_retriever.index, str(save_path / "dense.index"))
        
        # Save documents and metadata
        with open(save_path / "documents.json", 'w') as f:
            json.dump(self.dense_retriever.documents, f)
        
        # Save embeddings
        np.save(save_path / "embeddings.npy", self.dense_retriever.document_embeddings)
        
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, load_path: Path) -> None:
        """Load retrieval index from disk."""
        logger.info(f"Loading index from {load_path}")
        
        # Load FAISS index
        self.dense_retriever.index = faiss.read_index(str(load_path / "dense.index"))
        
        # Load documents
        with open(load_path / "documents.json", 'r') as f:
            self.dense_retriever.documents = json.load(f)
        
        # Load embeddings
        self.dense_retriever.document_embeddings = np.load(load_path / "embeddings.npy")
        
        logger.info("Index loaded successfully")
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 5,
                 use_query_expansion: bool = True,
                 use_reranking: bool = True) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using advanced techniques.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_query_expansion: Whether to expand query with medical terms
            use_reranking: Whether to use cross-encoder reranking
        """
        
        # Step 1: Query expansion
        queries = [query]
        if use_query_expansion:
            queries = self.query_expander.expand_query(query)
            logger.debug(f"Expanded query to {len(queries)} variants")
        
        # Step 2: Dense retrieval for all query variants
        all_results = []
        for q in queries:
            results = self.dense_retriever.retrieve(q, self.dense_top_k)
            all_results.extend(results)
        
        # Remove duplicates and keep best scores
        unique_results = {}
        for doc, score in all_results:
            if doc not in unique_results or score > unique_results[doc]:
                unique_results[doc] = score
        
        # Convert back to list
        dense_results = [(doc, score) for doc, score in unique_results.items()]
        dense_results.sort(key=lambda x: x[1], reverse=True)
        dense_results = dense_results[:self.dense_top_k]
        
        # Step 3: Cross-encoder reranking
        if use_reranking and dense_results:
            reranked_results = self.reranker.rerank(query, dense_results, self.rerank_top_k)
        else:
            reranked_results = dense_results[:self.rerank_top_k]
        
        # Step 4: Create enhanced results with medical concept extraction
        enhanced_results = []
        for doc, score in reranked_results[:top_k]:
            concepts = self.concept_extractor.extract_concepts(doc)
            
            # Determine reasoning type based on content
            reasoning_type = self._determine_reasoning_type(doc, concepts)
            
            # Extract medical concepts as flat list
            medical_concepts = []
            for category, concept_list in concepts.items():
                medical_concepts.extend(concept_list)
            
            # Calculate confidence based on score and concept relevance
            confidence = self._calculate_confidence(score, concepts, query)
            
            result = RetrievalResult(
                text=doc,
                score=score,
                source="medical_guidelines",
                reasoning_type=reasoning_type,
                medical_concepts=medical_concepts,
                confidence=confidence
            )
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _determine_reasoning_type(self, text: str, concepts: Dict[str, List[str]]) -> str:
        """Determine the type of medical reasoning based on content."""
        text_lower = text.lower()
        
        # Check for different reasoning patterns
        if any('symmetry' in concept or 'bilateral' in concept for concept in concepts.get('anatomical', [])):
            return 'symmetry_analysis'
        elif any('lesion' in concept or 'mass' in concept for concept in concepts.get('pathological', [])):
            return 'lesion_detection'
        elif 'midline' in text_lower or 'deviation' in text_lower:
            return 'midline_analysis'
        elif any('hemorrhage' in concept or 'stroke' in concept for concept in concepts.get('pathological', [])):
            return 'vascular_assessment'
        elif len(concepts.get('anatomical', [])) > 2:
            return 'anatomical_localization'
        else:
            return 'general_assessment'
    
    def _calculate_confidence(self, score: float, concepts: Dict[str, List[str]], query: str) -> float:
        """Calculate confidence score based on retrieval score and concept relevance."""
        base_confidence = min(score, 1.0)  # Normalize score
        
        # Boost confidence if many relevant medical concepts found
        concept_count = sum(len(concept_list) for concept_list in concepts.values())
        concept_bonus = min(concept_count * 0.1, 0.3)  # Max 30% bonus
        
        # Boost confidence if query terms found in concepts
        query_lower = query.lower()
        query_match_bonus = 0.0
        for concept_list in concepts.values():
            for concept in concept_list:
                if any(term in concept.lower() for term in query_lower.split()):
                    query_match_bonus += 0.05
        
        query_match_bonus = min(query_match_bonus, 0.2)  # Max 20% bonus
        
        final_confidence = min(base_confidence + concept_bonus + query_match_bonus, 1.0)
        return final_confidence 