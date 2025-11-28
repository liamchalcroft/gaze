"""Retrieval manager for agentic processing.

Provides a unified interface for retrieving relevant medical context
based on image analysis results and metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype
from loguru import logger

from nova_retrieval_vlm.visual_reasoning.radiology_analyzer import RadiologyAnalysis


class RetrievalManager:
    """Manages retrieval for agentic medical image analysis.

    Builds queries from visual analysis and metadata, then retrieves
    relevant passages from the knowledge base.
    """

    def __init__(
        self,
        index_dir: Path | str,
        retrieval_type: str = "hybrid",
        top_k: int = 5,
    ):
        """Initialize retrieval manager.

        Args:
            index_dir: Directory containing retrieval indexes
            retrieval_type: Type of retrieval ('bm25', 'dense', 'hybrid')
            top_k: Number of passages to retrieve
        """
        self.index_dir = Path(index_dir)
        self.retrieval_type = retrieval_type
        self.top_k = top_k
        self._retriever = None
        self._initialized = False

    def _initialize_retriever(self) -> None:
        """Lazily initialize the retriever."""
        if self._initialized:
            return

        try:
            if self.retrieval_type == "bm25":
                from nova_retrieval_vlm.retrieval import BM25Retriever

                self._retriever = BM25Retriever(self.index_dir)
            elif self.retrieval_type == "dense":
                from nova_retrieval_vlm.retrieval import DenseRetriever

                self._retriever = DenseRetriever(self.index_dir)
            elif self.retrieval_type == "hybrid":
                from nova_retrieval_vlm.retrieval import HybridRetriever

                self._retriever = HybridRetriever(self.index_dir)
            else:
                logger.warning(f"Unknown retrieval type: {self.retrieval_type}, using BM25")
                from nova_retrieval_vlm.retrieval import BM25Retriever

                self._retriever = BM25Retriever(self.index_dir)

            self._initialized = True
            logger.info(f"Initialized {self.retrieval_type} retriever")

        except FileNotFoundError as e:
            logger.warning(f"Retrieval index not found: {e}")
            self._retriever = None
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            self._retriever = None

    @beartype
    def retrieve(
        self,
        metadata: dict[str, Any],
        visual_analysis: RadiologyAnalysis | None = None,
        query_override: str | None = None,
    ) -> list[str]:
        """Retrieve relevant passages based on context.

        Args:
            metadata: Image/patient metadata
            visual_analysis: Optional visual analysis results
            query_override: Optional explicit query to use

        Returns:
            List of relevant passages
        """
        self._initialize_retriever()

        if self._retriever is None:
            logger.debug("Retriever not available, returning empty passages")
            return []

        # Build query
        query = query_override or self._build_query(metadata, visual_analysis)

        if not query:
            return []

        try:
            passages = self._retriever(query, k=self.top_k)
            logger.debug(f"Retrieved {len(passages)} passages for query: {query[:100]}...")
            return passages
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _build_query(
        self,
        metadata: dict[str, Any],
        visual_analysis: RadiologyAnalysis | None,
    ) -> str:
        """Build retrieval query from metadata and visual analysis."""
        query_parts = []

        # Add modality info
        if modality := metadata.get("modality"):
            query_parts.append(f"{modality} imaging")

        # Add clinical history
        if history := metadata.get("clinical_history"):
            query_parts.append(history)

        # Add findings from visual analysis
        if visual_analysis:
            # Add overall assessment
            if visual_analysis.overall_assessment:
                query_parts.append(visual_analysis.overall_assessment)

            # Add detected abnormalities
            abnormal_features = [
                f.name for f in visual_analysis.visual_features if f.feature_type == "abnormal"
            ]
            if abnormal_features:
                query_parts.append(f"findings: {', '.join(abnormal_features)}")

            # Add symmetry concerns
            if visual_analysis.symmetry_analysis.symmetry_score < 0.85:
                query_parts.append("asymmetric findings brain MRI")

        # Add any detected structures of interest
        if diagnosis := metadata.get("suspected_diagnosis"):
            query_parts.append(diagnosis)

        return " ".join(query_parts) if query_parts else "brain MRI analysis guidelines"

    @beartype
    def retrieve_for_task(
        self,
        task: str,
        metadata: dict[str, Any],
        visual_analysis: RadiologyAnalysis | None = None,
    ) -> list[str]:
        """Retrieve passages optimized for a specific task.

        Args:
            task: Task type ('localization', 'diagnosis', 'caption')
            metadata: Image/patient metadata
            visual_analysis: Optional visual analysis results

        Returns:
            Task-relevant passages
        """
        task_prefixes = {
            "localization": "lesion localization brain MRI",
            "diagnosis": "differential diagnosis brain MRI",
            "caption": "brain MRI findings description",
        }

        prefix = task_prefixes.get(task, "")
        base_query = self._build_query(metadata, visual_analysis)
        query = f"{prefix} {base_query}".strip()

        return self.retrieve(metadata, visual_analysis, query_override=query)
