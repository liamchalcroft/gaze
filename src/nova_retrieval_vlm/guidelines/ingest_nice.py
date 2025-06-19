from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
import yaml
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET
import time
import re

def ingest_nice(
    config_yaml: str,
    raw_dir: str,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Ingest radiology guideline documents whose seed URLs are defined in
    `docs/guidelines.yaml` (or another compatible YAML file).

    The YAML can hold multiple top-level lists e.g. `nice_urls`, `acr_urls`,
    `rcr_urls`, etc.  This function will gather all those seed URLs, crawl
    each site (via its sitemap where available), download any HTML/PDF found,
    filter for brain-related content, chunk the text and return it ready for
    indexing.

    Args:
        config_yaml: Path to the guidelines YAML configuration.
        raw_dir: Directory where raw downloaded files will be cached.
        verbose: Whether to print verbose output during the crawling process.

    Returns:
        A list of dicts with keys: `doc_id`, `chunk_id`, and `text`.
    """
    with open(config_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    # ------------------------------------------------------------------
    # 1. Collect seed URLs from known top-level keys and the optional
    #    structured  `sources:` list.
    # ------------------------------------------------------------------
    # These top-level YAML keys each hold a list of seed URLs we want to ingest.
    # If you add another guidelines provider to `docs/guidelines.yaml`, remember
    # to include the new key here so it gets picked up automatically.
    source_keys = [
        'nice_urls',            # National Institute for Health and Care Excellence
        'radiopaedia_urls',     # Radiopaedia reference articles
        'mriquestions_urls',    # MRIquestions educational pages
        'aan_urls',             # American Academy of Neurology guidelines
        'acr_urls',             # American College of Radiology (ACR) resources
        'rcr_urls',             # Royal College of Radiologists (RCR) resources
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

    crawl_depth = int(settings.get('crawl_depth', 2))
    request_delay = float(settings.get('request_delay', 1.0))  # seconds between requests per domain

    # Pre-compiled patterns for URLs we intentionally skip (heavy search pages etc.)
    skip_regex = re.compile(r"(\?|=|/search|/page/)", re.IGNORECASE)

    docs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 3. Breadth-first crawl within each domain up to `crawl_depth`.
    # ------------------------------------------------------------------
    queue: List[Tuple[str, int]] = [(u, 0) for u in urls]
    visited: Set[str] = set()

    while queue:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if verbose:
            print(f"[crawl] depth={depth} queue={len(queue)} → {url}")

        parsed = urlparse(url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"

        # Unique file name based on URL hash (avoids clashes)
        doc_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        ext = '.pdf' if url.lower().endswith('.pdf') else '.html'
        out_file = Path(raw_dir) / f"{doc_hash}{ext}"

        # --------------------------------------------------------------
        # Download if needed
        # --------------------------------------------------------------
        if not out_file.exists():
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
            }

            # Respect per-domain delay
            time.sleep(request_delay)
            try:
                resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            except Exception as e:
                print(f"Warning: request failed for {url}: {e}")
                continue

            if not resp.ok:
                if resp.status_code == 429:
                    # Too many requests – back off and retry once after delay*5
                    print(f"Rate-limited (429) – backing off and retrying {url}")
                    time.sleep(request_delay * 5)
                    try:
                        resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
                    except Exception:
                        continue
                    if not resp.ok:
                        print(f"Warning: still failing {url}, status code {resp.status_code}")
                        continue
                else:
                    print(f"Warning: skipping URL {url}, status code {resp.status_code}")
                    continue

            # If final URL after redirects contains search/query params, skip to avoid crawler traps
            final_url = str(resp.url)
            if skip_regex.search(final_url):
                continue

            # Update extension based on actual Content-Type header
            content_type = resp.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type and ext != '.pdf':
                ext = '.pdf'
                out_file = Path(raw_dir) / f"{doc_hash}{ext}"
            elif 'html' in content_type and ext != '.html':
                ext = '.html'
                out_file = Path(raw_dir) / f"{doc_hash}{ext}"

            if verbose:
                print(f"  ↳ downloaded → {out_file.name} ({len(resp.content)//1024} KB)")
            out_file.write_bytes(resp.content)

        # --------------------------------------------------------------
        # Extract raw text
        # --------------------------------------------------------------
        text = ""
        soup: BeautifulSoup | None = None
        if ext == '.pdf':
            try:
                reader = PdfReader(str(out_file))
                text = ''.join(page.extract_text() or '' for page in reader.pages)
            except Exception as e:
                # Fallback: treat as HTML/text if PDF parsing fails
                try:
                    raw_html = out_file.read_text(errors='ignore')
                    soup = BeautifulSoup(raw_html, 'html.parser')
                    text = soup.get_text(separator=' ')
                    ext = '.html'
                except Exception:
                    print(f"Warning: unable to extract text from {url}: {e}")
                    continue
        else:
            try:
                raw_html = out_file.read_text(errors='ignore')
                soup = BeautifulSoup(raw_html, 'html.parser')
                text = soup.get_text(separator=' ')
            except Exception as e:
                print(f"Warning: HTML parsing failed for {url}: {e}")
                continue

        # --------------------------------------------------------------
        # If HTML and we still have depth budget, enqueue internal links.
        # --------------------------------------------------------------
        if soup is not None and depth < crawl_depth:
            for a in soup.find_all('a', href=True):
                href = a['href']
                new_url = urljoin(url, href)
                if new_url.startswith(base_domain) and new_url not in visited:
                    # Skip mailto:, javascript:, etc.
                    if (new_url.startswith('http')
                        and 'mailto:' not in new_url
                        and 'javascript:' not in new_url
                        and not skip_regex.search(new_url)):
                        queue.append((new_url, depth + 1))

        # --------------------------------------------------------------
        # Relevance filter — skip document if it does **not** mention at
        # least one include_keyword or if it contains any exclude_keyword.
        # --------------------------------------------------------------
        lowered = text.lower()
        if include_keywords and not any(kw in lowered for kw in include_keywords):
            if verbose:
                print("  ✗ skipped (no include keywords)")
            continue
        if exclude_keywords and any(ex_kw in lowered for ex_kw in exclude_keywords):
            if verbose:
                print("  ✗ skipped (exclude keyword matched)")
            continue

        # ------------------------------------------------------------------
        # 4. Text chunking – allow YAML to override default parameters.
        # ------------------------------------------------------------------
        chunk_size = settings.get('chunk_size', 512)
        chunk_overlap = settings.get('chunk_overlap', 64)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        )
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append({'doc_id': doc_hash, 'chunk_id': idx, 'text': chunk})

    return docs
