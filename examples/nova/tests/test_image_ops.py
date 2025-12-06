import numpy as np
from PIL import Image

from radiant_harness.tools.image_ops import adjust_contrast
from radiant_harness.tools.image_ops import apply_intensity_threshold
from radiant_harness.tools.image_ops import crop_image
from radiant_harness.tools.image_ops import zoom_image


def test_zoom_and_crop(mock_image):
    img = Image.open(mock_image)
    zoomed = zoom_image(img, 2.0)
    assert zoomed.size[0] == img.size[0] * 2

    # crop_image expects normalized coordinates (0-1)
    # For a 100x100 image, (0.1, 0.1, 0.5, 0.5) = pixel coords (10, 10, 50, 50)
    cropped = crop_image(img, (0.1, 0.1, 0.5, 0.5))
    assert cropped.size == (40, 40)


def test_contrast_and_threshold(mock_image):
    img = Image.open(mock_image).convert("L")
    contrast_img = adjust_contrast(img, 1.5)
    assert contrast_img.size == img.size
    thresh_img = apply_intensity_threshold(img, 0, 128)
    arr = np.array(thresh_img)
    assert arr.max() <= 255 and arr.min() >= 0
