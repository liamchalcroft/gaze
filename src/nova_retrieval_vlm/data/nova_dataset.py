from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PIL import Image

from nova_retrieval_vlm.types import ConfigurationError, ValidationError

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Compose

logger = logging.getLogger(__name__)

class NovaDataset:
    """PyTorch Dataset for the NOVA brain-MRI HuggingFace dataset (test split only)."""
    
    def __init__(
        self,
        data_dir: str,
        transform: Compose | None = None,
    ):
        """
        Args:
            data_dir: Path to cache or download dataset.
            transform: torchvision transforms to apply to images.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        try:
            from datasets import load_dataset
            self.dataset = load_dataset(
                "Ano-2090/Nova", split="test", cache_dir=self.data_dir
            )
            logger.info("Loaded NOVA test split with %d samples.", len(self.dataset))
        except ImportError as e:
            raise ConfigurationError(
                "datasets library not available. Install with: pip install datasets"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to load dataset: {e}") from e

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self.dataset):
            raise ValidationError(f"Index {idx} out of range [0, {len(self.dataset)})")
            
        try:
            item = self.dataset[idx]
        except Exception as e:
            raise ValidationError(f"Failed to get item at index {idx}: {e}") from e
            
        # Load image: either in-memory HF image or on-disk
        if "image" in item and item["image"] is not None:
            img = item["image"]
            if not isinstance(img, Image.Image):
                try:
                    import numpy as np
                    img = Image.fromarray(np.array(img))
                except ImportError as e:
                    raise ConfigurationError("numpy required for array conversion") from e
            image = img.convert("RGB")
        else:
            image_path = item.get("image_path")
            if not image_path:
                raise ValidationError(f"No image or image_path for index {idx}")
            try:
                image = Image.open(image_path).convert("RGB")
            except (OSError, IOError) as e:
                raise ValidationError(f"Failed to load image {image_path}: {e}") from e
                
        # Apply transforms if provided
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                raise ValidationError(f"Transform failed on image {idx}: {e}") from e
                
        return {
            "image": image,
            "metadata": item.get("metadata", {}),
            "image_id": item.get("image_id", idx),
        }

    @property
    def hf_dataset(self) -> HFDataset:
        """Access the underlying HuggingFace dataset for reference data."""
        return self.dataset


def get_dataloader(
    batch_size: int,
    data_dir: str,
) -> Any:  # DataLoader[dict[str, Any]] - simplified for type checking
    """
    Create a DataLoader for the NOVA dataset.

    Args:
        batch_size: Number of samples per batch.
        data_dir: Path to cache or download dataset (test split only).

    Returns:
        DataLoader yielding batches of dicts.
        
    Raises:
        ConfigurationError: If required dependencies are missing.
    """
    try:
        from torch.utils.data import DataLoader
        from torchvision.transforms import Compose, ToTensor, Normalize
    except ImportError as e:
        raise ConfigurationError(
            "PyTorch and torchvision required. Install with: pip install torch torchvision"
        ) from e
    
    if batch_size <= 0:
        raise ValidationError(f"Batch size must be positive, got {batch_size}")
    
    default_transforms = Compose([
        ToTensor(), 
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = NovaDataset(data_dir=data_dir, transform=default_transforms)
    # Deterministic order – no shuffling required for zero-shot evaluation
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
