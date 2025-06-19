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
    # Aggregate URLs from various guideline sources
    source_keys = ['nice_urls', 'radiopaedia_urls', 'mriquestions_urls', 'aan_urls', 'acr_urls']
    urls: List[str] = []
    for key in source_keys:
        urls.extend(cfg.get(key, []))
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
    docs: List[Dict[str, Any]] = []
    for url in urls:
        doc_id = url.rstrip('/').split('/')[-1]
        ext = '.pdf' if url.lower().endswith('.pdf') else '.html'
        out_file = Path(raw_dir) / f"{doc_id}{ext}"
        if not out_file.exists():
            # Request with enhanced headers to avoid 406 from Radiopaedia
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
            }
            # Initial request
            resp = requests.get(url, headers=headers, timeout=10)
            # Retry with www prefix if Radiopaedia returns 406
            if resp.status_code == 406 and "radiopaedia.org" in url:
                alt_url = url.replace("radiopaedia.org", "www.radiopaedia.org")
                resp = requests.get(alt_url, headers=headers, timeout=10)
            # Skip URLs that don't return HTTP 200
            if not resp.ok:
                print(f"Warning: skipping URL {url}, status code {resp.status_code}")
                continue
            out_file.write_bytes(resp.content)
        # extract text
        if ext == '.pdf':
            reader = PdfReader(str(out_file))
            text = ''.join(page.extract_text() or '' for page in reader.pages)
        else:
            soup = BeautifulSoup(out_file.read_text(), 'html.parser')
            text = soup.get_text(separator=' ')
        # chunk text
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append({'doc_id': doc_id, 'chunk_id': idx, 'text': chunk})
    return docs
