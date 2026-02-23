#!/usr/bin/env python
"""Download AgentClinic NEJM Extended dataset.

Downloads the agentclinic_nejm_extended.jsonl file from the
AgentClinic GitHub repository with atomic writes, JSONL validation,
and retry with exponential backoff.

Usage:
    python download.py
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.error import URLError

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/SamuelSchmidgall/AgentClinic/"
    "main/agentclinic_nejm_extended.jsonl"
)

OUTPUT_PATH = Path(__file__).parent / "agentclinic_nejm_extended.jsonl"

MAX_RETRIES = 3
BACKOFF_BASE = 2.0
TIMEOUT = 30


def _validate_jsonl(path: Path) -> int:
    """Validate every non-empty line is valid JSON. Returns line count."""
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON on line {line_no}: {exc}"
                raise ValueError(msg) from exc
            count += 1
    return count


def download_dataset() -> None:
    """Download the dataset file with retries and validation."""
    if OUTPUT_PATH.exists():
        logger.info("Dataset already exists at %s", OUTPUT_PATH)
        return

    last_exc: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        tmp_path: Path | None = None
        try:
            logger.info(
                "Downloading dataset (attempt %d/%d) from %s ...",
                attempt,
                MAX_RETRIES,
                DATASET_URL,
            )
            with urllib.request.urlopen(DATASET_URL, timeout=TIMEOUT) as resp:  # noqa: S310
                # Write to a temp file in the same directory for atomic rename
                fd = tempfile.NamedTemporaryFile(
                    dir=OUTPUT_PATH.parent,
                    suffix=".tmp",
                    delete=False,
                )
                tmp_path = Path(fd.name)
                try:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        fd.write(chunk)
                finally:
                    fd.close()

            # Validate before committing
            count = _validate_jsonl(tmp_path)
            if count == 0:
                msg = "Downloaded file contains no valid JSON lines"
                raise ValueError(msg)

            # Atomic rename
            tmp_path.rename(OUTPUT_PATH)
            logger.info("Downloaded to %s (%d cases)", OUTPUT_PATH, count)
            return

        except (URLError, OSError, ValueError) as exc:
            last_exc = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)
            # Clean up temp file on failure
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
            if attempt < MAX_RETRIES:
                delay = BACKOFF_BASE**attempt
                logger.info("Retrying in %.1f seconds ...", delay)
                time.sleep(delay)

    msg = f"Failed to download after {MAX_RETRIES} attempts"
    raise RuntimeError(msg) from last_exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_dataset()
