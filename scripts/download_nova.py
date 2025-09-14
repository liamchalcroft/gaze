#!/usr/bin/env python3
"""
Download the NOVA dataset from Hugging Face and save it to disk.
Usage:
    python scripts/download_nova.py --data-dir /path/to/data_dir
"""

import argparse

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download and cache the NOVA HuggingFace dataset.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="${DATA_DIR}",
        help="Directory where the NOVA dataset will be cached/saved",
    )
    args = parser.parse_args()

    # Load all splits
    print(f"Loading NOVA dataset into cache at {args.data_dir}...")
    ds = load_dataset("Ano-2090/Nova", cache_dir=args.data_dir)

    # Save to disk for reproducibility
    print(f"Saving dataset to disk at {args.data_dir}/nova_")
    if hasattr(ds, "items"):
        for split, dataset in ds.items():
            if isinstance(split, str):
                path = f"{args.data_dir}/nova_{split}"
                if hasattr(dataset, "save_to_disk"):
                    dataset.save_to_disk(path)
                    print(f"Saved split '{split}' to {path}")


if __name__ == "__main__":
    main()
