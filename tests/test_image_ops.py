import numpy as np
from PIL import Image

from nova_retrieval_vlm.visual_reasoning.image_ops import (
    adjust_contrast,
    apply_intensity_threshold,
    crop_image,
    zoom_image,
)


def test_zoom_and_crop(mock_image):
    img = Image.open(mock_image)
    zoomed = zoom_image(img, 2.0)
    assert zoomed.size[0] == img.size[0] * 2
    cropped = crop_image(img, (10, 10, 50, 50))
    assert cropped.size == (40, 40)


def test_contrast_and_threshold(mock_image):
    img = Image.open(mock_image).convert("L")
    contrast_img = adjust_contrast(img, 1.5)
    assert contrast_img.size == img.size
    thresh_img = apply_intensity_threshold(img, 0, 128)
    arr = np.array(thresh_img)
    assert arr.max() <= 255 and arr.min() >= 0
