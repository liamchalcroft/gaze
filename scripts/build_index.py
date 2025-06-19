#!/usr/bin/env python3
"""
Build BM25 and FAISS indexes for guideline sources specified in a YAML.
Usage:
    python scripts/build_index.py \
        --config docs/guidelines.yaml \
        --raw-dir data/raw/guidelines \
        --output-dir indexes
"""
import argparse
from pathlib import Path
from nova_retrieval_vlm.guidelines.index_builder import build_indexes

def main():
    parser = argparse.ArgumentParser(description="Ingest guidelines and build retrieval indexes.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to guidelines YAML listing source URLs"
    )
    parser.add_argument(
        "--raw-dir", type=str, required=True,
        help="Directory to cache raw guideline documents"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write BM25 and FAISS indexes"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose crawl progress"
    )
    args = parser.parse_args()

    build_indexes(
        config_yaml=args.config,
        raw_dir=args.raw_dir,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
    )
    print(f"Built indexes under {args.output_dir}")

if __name__ == "__main__":
    main()
