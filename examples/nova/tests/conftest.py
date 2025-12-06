import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from nova_retrieval_vlm.config import Config
from nova_retrieval_vlm.config import ModelConfig
from nova_retrieval_vlm.config import PathsConfig
from nova_retrieval_vlm.types import BatchData


@pytest.fixture
def test_config():
    """Return a test configuration."""
    return Config(
        model=ModelConfig(
            name="test-model",
            max_retries=1,
            timeout=10,
            temperature=0.5,
            max_tokens=100,
        ),
        paths=PathsConfig(data_dir="./tests/data", output_dir="./tests/outputs"),
        task="localization",
        batch_size=1,
        max_iterations=1,
        request_delay=0.1,
        strict_mode=True,
    )


@pytest.fixture
def mock_image():
    """Return a path to a mock image."""
    image_path = Path(__file__).parent / "data" / "test_image.png"
    image_path.parent.mkdir(exist_ok=True, parents=True)

    # Create a small test image if it doesn't exist
    if not image_path.exists():
        import numpy as np
        from PIL import Image

        # Create a simple black image
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        image.save(str(image_path))

    return image_path


@pytest.fixture
def mock_passages():
    """Return mock passages for testing."""
    return [
        "This is the first test passage.",
        "This is the second test passage with more content.",
        "This is the third test passage discussing neurological findings.",
    ]


@pytest.fixture
def mock_adapter():
    """Return a mock adapter for testing."""
    adapter = MagicMock()

    async def mock_generate(*args, **kwargs):
        return (
            json.dumps(
                {
                    "boxes": [[10, 20, 30, 40]],
                    "labels": ["anomaly"],
                    "scores": [0.95],
                    "caption": "Test caption",
                    "diagnosis": "Test diagnosis",
                }
            ),
            MagicMock(tokens=100, cost=0.0),
        )

    async def mock_generate_text(*args, **kwargs):
        return "This is a test response.", MagicMock(tokens=20, cost=0.0)

    adapter.generate = mock_generate
    adapter.generate_text = mock_generate_text
    return adapter


@pytest.fixture
def mock_retriever():
    """Return a mock retriever for testing."""
    retriever = MagicMock()
    retriever.return_value = [
        "Relevant passage 1",
        "Relevant passage 2",
        "Relevant passage 3",
    ]
    return retriever


@pytest.fixture
def mock_batch_data(mock_image: Path) -> BatchData:
    """Create mock batch data."""
    return BatchData(
        images=[mock_image, mock_image],
        metadata=[
            {"modality": "CT", "patient_info": "Test patient 1"},
            {"modality": "MRI", "patient_info": "Test patient 2"},
        ],
    )
