#!/usr/bin/env python3
"""
Visualize raw NOVA dataset samples directly from Hugging Face.

Usage:
    python scripts/visualize_samples.py \
        [--num-samples 5] [--out-dir viz_samples] [--data-dir CACHE_DIR] [--trust-remote-code] [--overlay]
"""
import argparse
import random
import json
from pathlib import Path
import PIL.Image
import PIL.ExifTags
# Monkey-patch missing EXIF Base.Orientation tag for decoding
PIL.ExifTags.Base = type('Base', (), {'Orientation': 274})
PIL.Image.ExifTags = PIL.ExifTags
from datasets import load_dataset
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(
        description="Save a few NOVA samples (images + metadata) via Hugging Face"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of random samples to save"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to save outputs (default: ./viz_samples)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Cache directory for Hugging Face dataset (optional)"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Pass trust_remote_code=True to load_dataset"
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="Overlay gold and rater bounding boxes on saved images"
    )
    args = parser.parse_args()

    print("Loading NOVA test split from Hugging Face...")
    ds = load_dataset(
        "Ano-2090/Nova",
        split="test",
        cache_dir=args.data_dir,
        trust_remote_code=args.trust_remote_code
    )

    total = len(ds)
    count = min(args.num_samples, total)
    print(f"Selecting {count}/{total} samples.")

    base = Path(args.out_dir) if args.out_dir else Path.cwd() / "viz_samples"
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
        if args.overlay:
            overlay = img.copy()
            draw = ImageDraw.Draw(overlay)
            # Gold standard boxes
            bg = rec.get("bbox_gold", {})
            for x, y, w, h in zip(
                bg.get("x", []), bg.get("y", []), bg.get("width", []), bg.get("height", [])
            ):
                draw.rectangle([x, y, x + w, y + h], outline="gold", width=2)
            # Rater boxes
            br = rec.get("bbox_raters", {})
            colors = ["#40E0D0", "#FA8072"]
            xs, ys, ws, hs, raters = (
                br.get("x", []), br.get("y", []), br.get("width", []), br.get("height", []), br.get("rater", [])
            )
            for j, (x, y, w, h) in enumerate(zip(xs, ys, ws, hs)):
                color = colors[j % len(colors)]
                draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
                label = raters[j] if j < len(raters) else ""
                draw.text((x, y - 10), label, fill=color)
            out_img = images_dir / f"sample_{idx}_overlay.png"
            overlay.save(out_img)
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
        print(f"Saved sample {idx} to {out_img}")

    print("Done.")

if __name__ == "__main__":
    main() 