"""Visual reasoning module for medical image analysis.

Provides modality-agnostic image manipulation primitives that can be
used by VLMs during interactive analysis.
"""

from .image_ops import adjust_contrast
from .image_ops import apply_intensity_threshold
from .image_ops import crop_image
from .image_ops import flip_horizontal
from .image_ops import flip_vertical
from .image_ops import rotate_90
from .image_ops import zoom_image

__all__ = [
    "adjust_contrast",
    "apply_intensity_threshold",
    "crop_image",
    "flip_horizontal",
    "flip_vertical",
    "rotate_90",
    "zoom_image",
]
