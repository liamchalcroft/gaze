#!/usr/bin/env python3
"""
Build BM25 and FAISS indexes for guideline sources specified in a YAML.
Usage:
    python scripts/build_index.py \
        --config docs/guidelines.yaml \
        --raw-dir data/raw/guidelines \
        --output-dir indexes \
        --num-workers 8 \
        --parallel-index-build
"""

import argparse

try:
    from nova_retrieval_vlm.retrieval.web_search import build_indexes
except ImportError:
    # Fallback if the module doesn't exist
    def build_indexes(**kwargs):
        print("Warning: build_indexes function not available. Index building disabled.")


def main():
    parser = argparse.ArgumentParser(description="Ingest guidelines and build retrieval indexes.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to guidelines YAML listing source URLs"
    )
    parser.add_argument(
        "--raw-dir", type=str, required=True, help="Directory to cache raw guideline documents"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to write BM25 and FAISS indexes"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose crawl progress")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for crawling / indexing",
    )
    parser.add_argument(
        "--robots-mode",
        type=str,
        default="strict",
        choices=["strict", "lax", "off"],
        help="Robots.txt handling: strict (default), lax (fall back to Googlebot rules), off (ignore)",
    )
    parser.add_argument(
        "--no-parallel-index-build",
        action="store_true",
        help="Disable parallel building of BM25 and FAISS indexes (build sequentially)",
    )
    args = parser.parse_args()

    build_indexes(
        config_path=args.config,
        raw_dir=args.raw_dir,
        index_dir=args.output_dir,
        num_workers=args.num_workers,
        parallel_index_build=not args.no_parallel_index_build,
        robots_mode=args.robots_mode,
    )
    print(f"Built indexes under {args.output_dir}")


if __name__ == "__main__":
    main()
