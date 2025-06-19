from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

def ingest_nice(
    config_yaml: str,
    raw_dir: str,
) -> List[Dict[str, Any]]:
    """
    Ingest NICE guideline documents from URLs specified in a YAML file.

    Args:
        config_yaml: Path to YAML file with key 'nice_urls' listing URLs.
        raw_dir: Directory to save raw downloaded files.

    Returns:
        List of dicts with keys: doc_id, chunk_id, text.
    """
    with open(config_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    # ------------------------------------------------------------------
    # 1. Collect seed URLs from known top-level keys and the optional
    #    structured  `sources:` list.
    # ------------------------------------------------------------------
    source_keys = [
        'nice_urls',
        'radiopaedia_urls',
        'mriquestions_urls',
        'aan_urls',
        'acr_urls',
    ]

    urls: list[str] = []
    # Top-level key lists
    for key in source_keys:
        urls.extend(cfg.get(key, []))

    # Structured list with explicit metadata
    for item in cfg.get('sources', []):
        if not isinstance(item, dict):
            continue
        url_val = item.get('url')
        if url_val:
            urls.append(url_val)
    # Expand URLs by crawling each site's sitemap to collect all relevant pages
    expanded_urls: list[str] = []
    for base_url in urls:
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        sitemap_url = f"{domain}/sitemap.xml"
        try:
            sitemap_resp = requests.get(sitemap_url, timeout=10)
            sitemap_resp.raise_for_status()
            tree = ET.fromstring(sitemap_resp.content)
            locs = [loc.text for loc in tree.findall(".//{*}loc") if base_url.rstrip('/') in loc.text]
            if locs:
                expanded_urls.extend(locs)
            else:
                expanded_urls.append(base_url)
        except Exception:
            expanded_urls.append(base_url)
    # Deduplicate while preserving order
    seen = set()
    urls = [u for u in expanded_urls if not (u in seen or seen.add(u))]
    if not urls:
        raise ValueError(f"No URLs found in config_keys {source_keys}")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Allow user to restrict ingestion to documents that match certain
    #    keywords (e.g. brain-related MRI).  This prevents unrelated
    #    modalities such as lung CT from entering the index.
    # ------------------------------------------------------------------
    settings = cfg.get('settings', {}) if isinstance(cfg, dict) else {}

    include_keywords = [
        kw.lower() for kw in settings.get(
            'include_keywords',
            [
                'mri',
                'magnetic resonance',
                'brain',
                'cerebral',
                'head',
            ],
        )
    ]
    exclude_keywords = [kw.lower() for kw in settings.get('exclude_keywords', [])]

    docs: List[Dict[str, Any]] = []

    for url in urls:
        doc_id = url.rstrip('/').split('/')[-1]
        ext = '.pdf' if url.lower().endswith('.pdf') else '.html'
        out_file = Path(raw_dir) / f"{doc_id}{ext}"

        # --------------------------------------------------------------
        # Download if needed
        # --------------------------------------------------------------
        if not out_file.exists():
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 406 and "radiopaedia.org" in url:
                alt_url = url.replace("radiopaedia.org", "www.radiopaedia.org")
                resp = requests.get(alt_url, headers=headers, timeout=10)

            if not resp.ok:
                print(f"Warning: skipping URL {url}, status code {resp.status_code}")
                continue
            out_file.write_bytes(resp.content)

        # --------------------------------------------------------------
        # Extract raw text
        # --------------------------------------------------------------
        if ext == '.pdf':
            reader = PdfReader(str(out_file))
            text = ''.join(page.extract_text() or '' for page in reader.pages)
        else:
            soup = BeautifulSoup(out_file.read_text(), 'html.parser')
            text = soup.get_text(separator=' ')

        # --------------------------------------------------------------
        # Relevance filter — skip document if it does **not** mention at
        # least one include_keyword or if it contains any exclude_keyword.
        # --------------------------------------------------------------
        lowered = text.lower()
        if not any(kw in lowered for kw in include_keywords):
            # No required keywords present → irrelevant (e.g. lung CT)
            continue
        if exclude_keywords and any(ex_kw in lowered for ex_kw in exclude_keywords):
            # Explicitly excluded topic → skip
            continue

        # ------------------------------------------------------------------
        # 3. Text chunking – allow YAML to override default parameters.
        # ------------------------------------------------------------------
        chunk_size = settings.get('chunk_size', 512)
        chunk_overlap = settings.get('chunk_overlap', 64)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        )
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append({'doc_id': doc_id, 'chunk_id': idx, 'text': chunk})
    return docs
