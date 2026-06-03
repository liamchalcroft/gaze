import json
import random
from pathlib import Path

from beartype import beartype
from datasets import Dataset
from datasets import load_dataset
from loguru import logger
from PIL import Image
from PIL import ImageDraw


@beartype
def visualize_samples(
    num_samples: int = 5,
    out_dir: str | None = None,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
    overlay: bool = False,
) -> None:
    """
    Save a few NOVA samples (images + metadata) via Hugging Face.

    Args:
        num_samples: Number of random samples.
        out_dir: Directory to save outputs.
        cache_dir: HF datasets cache_dir.
        trust_remote_code: Whether to set trust_remote_code in load_dataset.
        overlay: Whether to overlay bbox_gold and bbox_raters.
    """
    logger.info("Loading NOVA test split from Hugging Face...")
    ds = load_dataset(
        "c-i-ber/Nova",
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    # load_dataset with split returns a Dataset (not IterableDataset)
    assert isinstance(ds, Dataset), f"Expected Dataset, got {type(ds).__name__}"
    total = len(ds)
    count = min(num_samples, total)
    logger.info(f"Selecting {count}/{total} samples.")

    base = Path(out_dir) if out_dir else Path.cwd() / "viz_samples"
    images_dir = base / "images"
    meta_dir = base / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(total), count)
    for idx, i in enumerate(indices):
        rec = ds[i]
        img = rec.get("image")
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if overlay:
            overlay_img = img.copy()
            draw = ImageDraw.Draw(overlay_img)
            # Gold standard boxes
            bg = rec.get("bbox_gold", {})
            for x, y, w, h in zip(
                bg.get("x", []),
                bg.get("y", []),
                bg.get("width", []),
                bg.get("height", []),
                strict=False,
            ):
                draw.rectangle([x, y, x + w, y + h], outline="gold", width=2)
            # Rater boxes
            br = rec.get("bbox_raters", {})
            colors = ["#40E0D0", "#FA8072"]
            xs, ys, ws, hs, raters = (
                br.get("x", []),
                br.get("y", []),
                br.get("width", []),
                br.get("height", []),
                br.get("rater", []),
            )
            for j, (x, y, w, h) in enumerate(zip(xs, ys, ws, hs, strict=False)):
                color = colors[j % len(colors)]
                draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
                label = raters[j] if j < len(raters) else ""
                draw.text((x, y - 10), label, fill=color)
            out_img = images_dir / f"sample_{idx}_overlay.png"
            overlay_img.save(out_img)
        else:
            out_img = images_dir / f"sample_{idx}.png"
            img.save(out_img)

        # Save metadata
        meta = {
            "filename": rec.get("filename"),
            "caption": rec.get("caption"),
            "clinical_history": rec.get("clinical_history"),
            "differential_diagnosis": rec.get("differential_diagnosis"),
            "final_diagnosis": rec.get("final_diagnosis"),
            "bbox_gold": rec.get("bbox_gold", {}),
            "bbox_raters": rec.get("bbox_raters", {}),
        }
        with open(meta_dir / f"sample_{idx}.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved sample {idx} to {out_img}")

    logger.info("Visualization complete.")
