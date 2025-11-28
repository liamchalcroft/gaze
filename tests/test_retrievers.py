import json
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Import retrievers inside test functions to avoid collection-time import issues
pytest_plugins = []


class TestBM25Retriever:
    @pytest.fixture
    def mock_index(self) -> Generator[str, None, None]:
        # Create a temporary directory for the index
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_index_data(self, mock_index: str) -> Generator[str, None, None]:
        # Create a mock directory structure for the BM25 index
        index_dir = Path(mock_index)
        docs_file = index_dir / "bm25_docs.jsonl"

        # Create sample docs
        docs = [
            {"id": 1, "text": "Result 1", "metadata": {"source": "doc1"}},
            {"id": 2, "text": "Result 2", "metadata": {"source": "doc2"}},
            {"id": 3, "text": "Result 3", "metadata": {"source": "doc3"}},
        ]

        with open(docs_file, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

        yield mock_index

    @patch("nova_retrieval_vlm.retrieval.retrievers.InMemoryDocumentStore")
    @patch("nova_retrieval_vlm.retrieval.retrievers.HS_BM25")
    def test_init(self, mock_bm25: MagicMock, mock_store: MagicMock, mock_index_data: str):
        """Test initialization of BM25Retriever."""
        from nova_retrieval_vlm.retrieval.retrievers import BM25Retriever

        # Configure the mocks
        mock_store_instance = MagicMock()
        mock_store.return_value = mock_store_instance

        mock_retriever = MagicMock()
        mock_bm25.return_value = mock_retriever

        # Create the retriever
        retriever = BM25Retriever(mock_index_data)

        # Verify the mocks were called correctly
        mock_store.assert_called_once()
        mock_store_instance.write_documents.assert_called_once()
        mock_bm25.assert_called_once_with(document_store=mock_store_instance)
        assert retriever.retriever == mock_retriever

    @patch("nova_retrieval_vlm.retrieval.retrievers.InMemoryDocumentStore")
    @patch("nova_retrieval_vlm.retrieval.retrievers.HS_BM25")
    def test_call(self, mock_bm25: MagicMock, mock_store: MagicMock, mock_index_data: str):
        """Test the call method of BM25Retriever."""
        from nova_retrieval_vlm.retrieval.retrievers import BM25Retriever

        # Configure the mocks
        mock_doc1 = MagicMock()
        mock_doc1.content = "Result 1"

        mock_doc2 = MagicMock()
        mock_doc2.content = "Result 2"

        mock_doc3 = MagicMock()
        mock_doc3.content = "Result 3"

        mock_results = [mock_doc1, mock_doc2, mock_doc3]

        mock_retriever = MagicMock()
        # Haystack BM25 uses .run() method returning {"documents": [...]}
        mock_retriever.run.return_value = {"documents": mock_results}
        mock_bm25.return_value = mock_retriever

        # Create the retriever
        retriever = BM25Retriever(mock_index_data)

        # Call the retriever
        query = "test query"
        k = 3
        results = retriever(query, k)

        # Verify the results
        assert len(results) == 3
        assert results == ["Result 1", "Result 2", "Result 3"]

        # Verify the mock was called with the right arguments
        mock_retriever.run.assert_called_once_with(query=query, top_k=k)


class TestDenseRetriever:
    @pytest.fixture
    def mock_index(self) -> Generator[str, None, None]:
        # Create a temporary directory for the index
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_index_data(self, mock_index: str) -> Generator[str, None, None]:
        # Create a mock directory structure for the FAISS index
        index_dir = Path(mock_index)
        index_file = index_dir / "faiss.index"
        docs_file = index_dir / "faiss_docs.jsonl"

        # Write dummy data to the files
        with open(index_file, "wb") as f:
            f.write(b"dummy data")

        # Create sample docs
        docs = [
            {"id": 1, "text": "Passage 1", "metadata": {"source": "doc1"}},
            {"id": 2, "text": "Passage 2", "metadata": {"source": "doc2"}},
            {"id": 3, "text": "Passage 3", "metadata": {"source": "doc3"}},
        ]

        with open(docs_file, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

        yield mock_index

    @patch("nova_retrieval_vlm.retrieval.retrievers.faiss")
    @patch("nova_retrieval_vlm.retrieval.retrievers.SentenceTransformer")
    def test_init(self, mock_transformer: MagicMock, mock_faiss: MagicMock, mock_index_data: str):
        """Test initialization of DenseRetriever."""
        from nova_retrieval_vlm.retrieval.retrievers import DenseRetriever

        # Configure the mocks
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        mock_index_obj = MagicMock()
        mock_faiss.read_index.return_value = mock_index_obj

        # Create the retriever
        retriever = DenseRetriever(mock_index_data)

        # Verify the mocks were called
        mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        mock_faiss.read_index.assert_called_once()

        # Verify the attributes
        assert retriever.index == mock_index_obj  # type: ignore[comparison-overlap]
        assert retriever.model == mock_model
        assert len(retriever.texts) == 3
        assert retriever.texts == ["Passage 1", "Passage 2", "Passage 3"]

    @patch("nova_retrieval_vlm.retrieval.retrievers.faiss")
    @patch("nova_retrieval_vlm.retrieval.retrievers.SentenceTransformer")
    @patch("nova_retrieval_vlm.retrieval.retrievers.np")
    def test_call(
        self,
        mock_np: MagicMock,
        mock_transformer: MagicMock,
        mock_faiss: MagicMock,
        mock_index_data: str,
    ):
        """Test the call method of DenseRetriever."""
        from nova_retrieval_vlm.retrieval.retrievers import DenseRetriever

        # Configure the mocks
        mock_model = MagicMock()
        mock_emb = MagicMock()
        mock_model.encode.return_value = mock_emb
        mock_transformer.return_value = mock_model

        mock_index_obj = MagicMock()
        mock_index_obj.search.return_value = (
            MagicMock(),  # Distances (not used in the function)
            [[1, 0, 2]],  # Indices of the top passages
        )
        mock_faiss.read_index.return_value = mock_index_obj

        mock_np_array = MagicMock()
        mock_np.array.return_value = mock_np_array

        # Create the retriever
        retriever = DenseRetriever(mock_index_data)

        # Call the retriever
        query = "test query"
        k = 3
        results = retriever(query, k)

        # Verify the results
        assert len(results) == 3
        # The order should be based on the mock search results indices: [1, 0, 2]
        assert results == ["Passage 2", "Passage 1", "Passage 3"]

        # Verify the mocks were called
        mock_model.encode.assert_called_once_with(query, convert_to_numpy=True)
        mock_faiss.normalize_L2.assert_called_once_with(mock_emb)
        mock_np.array.assert_called_once_with([mock_emb])
        mock_index_obj.search.assert_called_once_with(mock_np_array, k)


class TestHybridRetriever:
    @pytest.fixture
    def mock_bm25_retriever(self) -> MagicMock:
        """Return a mock BM25Retriever."""
        retriever = MagicMock()
        retriever.return_value = ["BM25 Result 1", "BM25 Result 2", "BM25 Result 3"]
        return retriever

    @pytest.fixture
    def mock_dense_retriever(self) -> MagicMock:
        """Return a mock DenseRetriever."""
        retriever = MagicMock()
        retriever.return_value = ["Dense Result 1", "Dense Result 2", "Dense Result 3"]
        return retriever

    def test_init(self, mock_bm25_retriever: MagicMock, mock_dense_retriever: MagicMock) -> None:
        """Test initialization of HybridRetriever."""
        from nova_retrieval_vlm.retrieval.retrievers import HybridRetriever

        # Create the retriever
        retriever = HybridRetriever(mock_bm25_retriever, mock_dense_retriever, alpha=0.7)

        # Verify the attributes
        assert retriever.bm25 == mock_bm25_retriever
        assert retriever.dense == mock_dense_retriever
        assert retriever.alpha == 0.7

    def test_call(self, mock_bm25_retriever: MagicMock, mock_dense_retriever: MagicMock) -> None:
        """Test the call method of HybridRetriever."""
        from nova_retrieval_vlm.retrieval.retrievers import HybridRetriever

        # Configure the retrievers to return some overlapping results
        mock_bm25_retriever.return_value = ["Shared Result 1", "BM25 Result 2", "Shared Result 3"]

        mock_dense_retriever.return_value = ["Dense Result 1", "Shared Result 1", "Shared Result 3"]

        # Create the retriever with high weight for BM25
        retriever = HybridRetriever(mock_bm25_retriever, mock_dense_retriever, alpha=0.7)

        # Call the retriever
        query = "test query"
        k = 4
        results = retriever(query, k)

        # Verify the results
        assert len(results) == 4
        # The order should prioritize shared results, then BM25 (high alpha), then dense
        assert "Shared Result 1" in results
        assert "Shared Result 3" in results

        # Verify the retrievers were called with the right arguments
        mock_bm25_retriever.assert_called_once_with(query, k)
        mock_dense_retriever.assert_called_once_with(query, k)

        # Test with low alpha (prioritize dense)
        retriever.alpha = 0.3
        results = retriever(query, k)

        # Verify the results still include the shared ones
        assert "Shared Result 1" in results
        assert "Shared Result 3" in results
