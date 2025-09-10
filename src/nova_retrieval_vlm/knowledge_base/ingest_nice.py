from __future__ import annotations

import datetime
import hashlib
import html
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
import yaml
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect  # language detection
from pypdf import PdfReader
from readability import Document as ReadabilityDocument  # content extraction

# -----------------------------------------------------------------------------
# Global session pool (one persistent Session per domain to reuse TCP/TLS)
# -----------------------------------------------------------------------------
_session_pool: dict[str, requests.Session] = defaultdict(requests.Session)


def _get_session(domain: str) -> requests.Session:
    """Return (and lazily create) a persistent Session for a domain."""
    return _session_pool[domain]


# -----------------------------------------------------------------------------
# Robots.txt cache
# -----------------------------------------------------------------------------
_robot_cache: dict[str, RobotFileParser] = {}


def _can_fetch_cached(domain: str, url: str, agent: str) -> bool:
    if domain not in _robot_cache:
        rp = RobotFileParser()
        try:
            rp.set_url(urljoin(domain, "/robots.txt"))
            rp.read()
        except (OSError, requests.RequestException, ValueError):
            # Failed to fetch robots.txt, create empty parser
            rp = RobotFileParser()
            rp.parse("")
        _robot_cache[domain] = rp
    return _robot_cache[domain].can_fetch(agent, url)


def _is_allowed(url: str, *, robots_mode: str = "strict") -> bool:
    """Return True if URL is crawlable under given robots_mode.

    robots_mode:
        "strict" – honour robots.txt for our UA (default)
        "lax"    – if blocked for '*', allow if Googlebot is allowed
        "off"    – ignore robots.txt entirely
    """
    if robots_mode == "off":
        return True

    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    # First try with wildcard
    if _can_fetch_cached(domain, url, "*"):
        return True

    if robots_mode == "lax":
        # Allow if Googlebot may crawl (many sites explicitly whitelist it)
        return _can_fetch_cached(domain, url, "Googlebot")
    return False


# -----------------------------------------------------------------------------
# ETag cache for conditional GETs (incremental crawling)
# -----------------------------------------------------------------------------
_etag_path = None
_etag_cache: dict[str, str] = {}


def _load_etag_cache(path: str):
    global _etag_path, _etag_cache
    _etag_path = path
    if Path(path).exists():
        try:
            _etag_cache = yaml.safe_load(Path(path).read_text()) or {}
        except (OSError, yaml.YAMLError, FileNotFoundError):
            _etag_cache = {}


def _save_etag_cache():
    if _etag_path:
        try:
            Path(_etag_path).write_text(yaml.safe_dump(_etag_cache))
        except (OSError, yaml.YAMLError, FileNotFoundError, PermissionError):
            # Log the error but don't crash - ETag cache is not critical
            pass


# -----------------------------------------------------------------------------
# Helper: perform GET with retries & backoff, using conditional headers
# -----------------------------------------------------------------------------


def _http_get(
    url: str,
    *,
    timeout: int = 15,
    allow_redirects: bool = True,
    head_only: bool = False,
    user_agent: str | None = None,
):
    parsed = urlparse(url)
    session = _get_session(parsed.netloc)
    headers = {
        "User-Agent": user_agent or "Mozilla/5.0 (X11; Linux x86_64)",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.8",
    }
    if url in _etag_cache and not head_only:
        headers["If-None-Match"] = _etag_cache[url]

    method = session.head if head_only else session.get
    backoff = 2
    for _attempt in range(4):
        try:
            resp = method(url, headers=headers, timeout=timeout, allow_redirects=allow_redirects)
            # 304 Not Modified shortcut
            if resp.status_code == 304:
                return resp
            if resp.ok:
                if not head_only and resp.headers.get("ETag"):
                    _etag_cache[url] = resp.headers["ETag"]
                return resp
            # Retry on 429/503 or unpredictable server errors (>500)
            if resp.status_code in {429, 500, 502, 503, 504}:
                time.sleep(backoff)
                backoff *= 2
                continue
            return resp  # non-retryable error
        except requests.exceptions.ConnectionError as e:
            if "Name or service not known" in str(e) or "nodename nor servname provided" in str(e):
                # DNS resolution failed
                return None
            time.sleep(backoff)
            backoff *= 2
        except requests.exceptions.Timeout:
            time.sleep(backoff)
            backoff *= 2
        except requests.exceptions.TooManyRedirects:
            # Too many redirects, return the last response
            return None
        except Exception:
            time.sleep(backoff)
            backoff *= 2
    return None


# -----------------------------------------------------------------------------
# PDF sanity check (fast-fail on >25 MB or encrypted PDFs)
# -----------------------------------------------------------------------------


def _should_skip_pdf(url: str, max_mb: int = 25) -> bool:
    head = _http_get(url, head_only=True)
    if not head or not head.ok:
        return False
    size = int(head.headers.get("Content-Length", 0)) / (1024 * 1024)
    return size > max_mb


# -----------------------------------------------------------------------------
# SimHash deduplication at URL level (raw HTML)
# -----------------------------------------------------------------------------
from simhash import Simhash

_url_simhashes: set[int] = set()


def _is_duplicate_html(raw_html: str) -> bool:
    if not raw_html.strip():
        return False
    h = Simhash(re.sub(r"\s+", " ", raw_html)).value
    if h in _url_simhashes:
        return True
    _url_simhashes.add(h)
    return False


def clean_text(text: str) -> str:
    """
    Clean and normalize text content by removing junk, empty lines, and improving readability.

    Args:
        text: Raw text content to clean

    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple newlines to a single newline
    text = re.sub(r"\n{2,}", "\n", text)
    # Remove leading/trailing whitespace on each line and normalize internal whitespace
    lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in text.split("\n")]
    # Remove empty lines
    lines = [line for line in lines if line]
    text = "\n".join(lines)

    # Remove common web artifacts only if they appear as standalone lines or at boundaries
    web_artifacts = [
        r"cookie( policy)?",
        r"privacy( policy)?",
        r"terms( and conditions)?",
        r"conditions",
        r"subscribe( to (our|the) newsletter)?",
        r"newsletter",
        r"sign up",
        r"log in",
        r"login",
        r"register",
        r"all rights reserved",
    ]
    for artifact in web_artifacts:
        text = re.sub(rf"^\s*{artifact}\s*\.?$", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(rf"(^|[\.!?]\s*){artifact}(\.|!|\?|$)", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"©\s*\d{4}.*?\.", "", text)

    # Remove navigation and UI elements, including sequences
    nav_artifacts = [
        r"home",
        r"about",
        r"contact",
        r"search",
        r"menu",
        r"navigation",
        r"sidebar",
        r"back to top",
        r"scroll to top",
        r"next page",
        r"previous page",
    ]
    nav_seq = r"(?:" + r"|".join(nav_artifacts) + r")(?:\s+(?:" + r"|".join(nav_artifacts) + r"))*"
    # Remove whole lines that are just a sequence of navigation words
    text = re.sub(rf"^\s*{nav_seq}\s*\.?$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    # Remove trailing/leading navigation sequences at sentence boundaries
    text = re.sub(rf"(^|[\.!?]\s*){nav_seq}(\.|!|\?|$)", r"\1", text, flags=re.IGNORECASE)

    # Remove common medical website boilerplate
    med_boilerplate = [
        r"this information is provided for educational purposes only",
        r"consult your healthcare provider",
        r"talk to your doctor",
        r"medical disclaimer",
        r"disclaimer",
    ]
    for artifact in med_boilerplate:
        text = re.sub(rf"^\s*{artifact}\s*\.?$", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(rf"(^|[\.!?]\s*){artifact}(\.|!|\?|$)", r"\1", text, flags=re.IGNORECASE)

    # Remove excessive punctuation and formatting artifacts
    text = re.sub(r"[•·▪▫◦‣⁃]\s*", "\n", text)  # Bullet points to newlines
    text = re.sub(r"[|¦]\s*", "\n", text)  # Vertical bars to newlines
    text = re.sub(r"[─━═─]\s*", "\n", text)  # Horizontal lines to newlines
    text = re.sub(r"[■□▪▫▭▯▮▯]\s*", "\n", text)  # Box characters to newlines

    # Remove page numbers and headers/footers
    text = re.sub(r"page\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+\s+of\s+\d+", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)  # Standalone numbers

    # Remove common PDF artifacts
    text = re.sub(r"www\.[^\s]+", "", text)  # URLs
    text = re.sub(r"http[s]?://[^\s]+", "", text)  # URLs
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}", "", text)  # Email addresses

    # Remove very short lines that are likely navigation artifacts
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 10 or (len(line) > 3 and any(c.isalpha() for c in line)):
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Final cleanup
    text = text.strip()
    return text


def _should_skip_url(url: str) -> bool:
    """Check if URL should be skipped based on patterns."""
    # Only skip obviously problematic patterns, not entire domains
    skip_patterns = [
        r"/(search|filter|page|tag|category)/",
        r"\?(page|search|filter|p)=",
        r"/calendar",
        r"/events",
        r"/login",
        r"/register",
        r"/subscribe",
        r"/newsletter",
        r"/media/",
        r"/images/",
        r"/css/",
        r"/js/",
        r"/assets/",
        r"/static/",
        r"/api/",
        r"/admin/",
        r"/dashboard/",
        r"\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot)$",  # Static assets
    ]

    return any(re.search(pattern, url, re.IGNORECASE) for pattern in skip_patterns)


def _is_promising_url(url: str) -> bool:
    """Check if URL looks promising for containing guidelines."""
    promising_patterns = [
        r"/(guidance|guidelines|practice|clinical|protocols|standards)/",
        r"\.(pdf|html?)$",
        r"/(mri|magnetic|resonance|brain|cerebral|neuro|neuroradiology)/",
        r"/(imaging|radiology|diagnostic)/",
        r"/(publications|resources|documents|papers)/",
        r"/(acr|nice|aan|rcr)/",  # Organization-specific paths
    ]

    return any(re.search(pattern, url, re.IGNORECASE) for pattern in promising_patterns)


def _download_url(
    url: str, raw_dir: str, request_delay: float, verbose: bool = False, robots_mode: str = "strict"
) -> tuple[str, str, bool]:
    """
    Download a single URL and return (url, filepath, success).
    Returns empty filepath and False if download failed or disallowed.
    """
    # Skip problematic URLs early
    if _should_skip_url(url):
        if verbose:
            pass
        return url, "", False

    # Respect robots.txt
    if not _is_allowed(url, robots_mode=robots_mode):
        if verbose:
            pass
        return url, "", False

    # Skip huge PDFs early
    if url.lower().endswith(".pdf") and _should_skip_pdf(url):
        if verbose:
            pass
        return url, "", False

    urlparse(url)
    doc_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    ext = ".pdf" if url.lower().endswith(".pdf") else ".html"
    out_file = Path(raw_dir) / f"{doc_hash}{ext}"

    # Skip if ETag indicates unchanged and file already exists
    if out_file.exists() and url in _etag_cache:
        return url, str(out_file), True

    # Perform HTTP GET with retries/backoff
    ua = "Googlebot" if robots_mode == "lax" else "Mozilla/5.0 (X11; Linux x86_64)"
    resp = _http_get(url, user_agent=ua)
    if resp is None:
        if verbose:
            pass
        return url, "", False
    if not resp.ok:
        if verbose:
            pass
        return url, "", False

    # If conditional GET returns 304, reuse existing file
    if resp.status_code == 304 and out_file.exists():
        if verbose:
            pass
        return url, str(out_file), True

    # Update extension based on Content-Type header
    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type and ext != ".pdf":
        ext = ".pdf"
        out_file = Path(raw_dir) / f"{doc_hash}{ext}"
    elif "html" in content_type and ext != ".html":
        ext = ".html"
        out_file = Path(raw_dir) / f"{doc_hash}{ext}"

    # Duplicate HTML detection via SimHash (skip writing)
    if ext == ".html" and _is_duplicate_html(resp.text):
        if verbose:
            pass
        return url, "", False

    if verbose:
        len(resp.content) // 1024
    out_file.write_bytes(resp.content)

    # Throttle between requests
    time.sleep(request_delay)

    return url, str(out_file), True


def _process_file(
    url: str,
    filepath: str,
    include_keywords: list[str],
    exclude_patterns: list[re.Pattern],
    settings: dict[str, Any],
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Process a downloaded file and return (docs, new_urls).
    """
    if not filepath or not Path(filepath).exists():
        return [], []

    docs = []
    new_urls = []

    # Extract metadata from URL and file
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    parsed_url.path.split("/")
    filename = Path(filepath).stem

    # Extract potential publication date from URL or filename
    current_year = datetime.datetime.now().year
    date_patterns = [
        r"(\d{4})",  # 4-digit year
        r"(\d{2}-\d{2}-\d{4})",  # DD-MM-YYYY
        r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
    ]

    extracted_date = None
    for pattern in date_patterns:
        for text in [url, filename]:
            match = re.search(pattern, text)
            if match:
                year_candidate = match.group(1)
                if len(year_candidate) == 4 and 1990 <= int(year_candidate) <= current_year:
                    extracted_date = year_candidate
                    break
        if extracted_date:
            break

    # Extract text from file
    text = ""
    soup = None
    document_title = ""
    ext = Path(filepath).suffix.lower()

    try:
        if ext == ".pdf":
            try:
                reader = PdfReader(filepath)
                text = "".join(page.extract_text() or "" for page in reader.pages)
                # Try to extract title from first page
                if reader.pages:
                    first_page_text = reader.pages[0].extract_text() or ""
                    lines = first_page_text.strip().split("\n")[:5]  # First 5 lines
                    document_title = " ".join(lines).strip()[:100]  # Limit length
                text = clean_text(text)
            except Exception:
                if verbose:
                    pass
                # Fallback: treat as HTML if PDF parsing fails
                try:
                    raw_html = Path(filepath).read_text(errors="ignore")
                    soup = BeautifulSoup(raw_html, "html.parser")

                    # Extract title from HTML
                    title_tag = soup.find("title")
                    if title_tag:
                        document_title = title_tag.get_text().strip()
                    if not document_title:
                        h1_tag = soup.find("h1")
                        if h1_tag:
                            document_title = h1_tag.get_text().strip()

                    # Link harvesting - be more aggressive
                    if soup is not None:
                        parsed_base = urlparse(url)
                        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

                        # Harvest links from both the original HTML and processed HTML
                        all_links = []

                        # Get links from original HTML (before readability processing)
                        original_soup = BeautifulSoup(raw_html, "html.parser")
                        for a in original_soup.find_all("a", href=True):
                            href = a["href"]
                            nxt = urljoin(url, href)
                            if (
                                nxt.startswith(base_domain)
                                and nxt.startswith("http")
                                and "mailto:" not in nxt
                                and "javascript:" not in nxt
                                and not _should_skip_url(nxt)
                            ):
                                all_links.append(nxt)

                        # Also get links from processed HTML
                        for a in soup.find_all("a", href=True):
                            href = a["href"]
                            nxt = urljoin(url, href)
                            if (
                                nxt.startswith(base_domain)
                                and nxt.startswith("http")
                                and "mailto:" not in nxt
                                and "javascript:" not in nxt
                                and not _should_skip_url(nxt)
                            ):
                                all_links.append(nxt)

                        # Deduplicate and add to new_urls
                        seen_links = set()
                        for link in all_links:
                            if link not in seen_links:
                                seen_links.add(link)
                                new_urls.append(link)

                    _clean_soup(soup)
                    text = soup.get_text(separator=" ")
                    text = clean_text(text)
                except Exception:
                    if verbose:
                        pass
                    return [], []
        else:
            # Handle HTML files with better error handling
            try:
                raw_html = Path(filepath).read_text(errors="ignore")

                # Use the new robust text extraction function
                text, soup = _extract_text_from_html(raw_html, url)

                if not text or len(text.strip()) < 50:
                    if verbose:
                        pass
                    return [], []

                # Extract title from HTML
                if soup:
                    title_tag = soup.find("title")
                    if title_tag:
                        document_title = title_tag.get_text().strip()
                    if not document_title:
                        h1_tag = soup.find("h1")
                        if h1_tag:
                            document_title = h1_tag.get_text().strip()

                # Link harvesting - be more aggressive
                if soup is not None:
                    parsed_base = urlparse(url)
                    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

                    # Harvest links from both the original HTML and processed HTML
                    all_links = []

                    # Get links from original HTML (before readability processing)
                    original_soup = BeautifulSoup(raw_html, "html.parser")
                    for a in original_soup.find_all("a", href=True):
                        href = a["href"]
                        nxt = urljoin(url, href)
                        if (
                            nxt.startswith(base_domain)
                            and nxt.startswith("http")
                            and "mailto:" not in nxt
                            and "javascript:" not in nxt
                            and not _should_skip_url(nxt)
                        ):
                            all_links.append(nxt)

                    # Also get links from processed HTML
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        nxt = urljoin(url, href)
                        if (
                            nxt.startswith(base_domain)
                            and nxt.startswith("http")
                            and "mailto:" not in nxt
                            and "javascript:" not in nxt
                            and not _should_skip_url(nxt)
                        ):
                            all_links.append(nxt)

                    # Deduplicate and add to new_urls
                    seen_links = set()
                    for link in all_links:
                        if link not in seen_links:
                            seen_links.add(link)
                            new_urls.append(link)

                text = clean_text(text)

            except Exception:
                if verbose:
                    pass
                return [], []

    except Exception:
        if verbose:
            pass
        return [], []

    # Apply lightweight heuristics to avoid obvious junk links.
    # Keep query strings unless they look like pagination / filter noise.
    new_urls = [
        u
        for u in new_urls
        if not re.search(r"(\?page=|\?p=|/search/|/filter/|/tag/)", u, flags=re.I)
    ]

    if verbose and new_urls:
        pass

    # Skip if insufficient content - relaxed threshold
    if len(text.strip()) < 50:  # Reduced from 100 to 50
        if verbose:
            pass
        return [], new_urls

    # Language detection – skip non-English content
    try:
        if len(text) > 100:
            lang = detect(text[:500])
            if lang != "en":
                if verbose:
                    pass
                return [], new_urls
    except Exception:
        pass

    # Relevance filtering - make it optional for discovery
    lowered = text.lower()
    if include_keywords and not any(kw in lowered for kw in include_keywords):
        # Don't skip immediately, just mark as low priority
        if verbose:
            pass
        # Continue processing but mark as low priority

    if exclude_patterns and any(p.search(lowered) for p in exclude_patterns):
        if verbose:
            pass
        return [], new_urls

    # Chunk the text
    chunk_size = settings.get("chunk_size", 512)
    chunk_overlap = settings.get("chunk_overlap", 64)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
    )
    chunks = splitter.split_text(text)

    # Prepare metadata
    metadata = {
        "source_url": url,
        "domain": domain,
        "filename": filename,
        "document_title": document_title,
        "extracted_date": extracted_date,
    }

    # Process chunks with enhanced metadata
    doc_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    for idx, chunk in enumerate(chunks):
        cleaned_chunk = clean_text(chunk)

        # Skip short or low-quality chunks
        if len(cleaned_chunk.strip()) < 50:
            continue

        alpha_ratio = sum(1 for c in cleaned_chunk if c.isalpha()) / max(1, len(cleaned_chunk))
        if alpha_ratio < 0.3:
            continue

        # Skip junk chunks
        junk_words = ["cookie", "privacy", "terms", "subscribe", "newsletter", "log in", "register"]
        chunk_lower = cleaned_chunk.lower()
        junk_count = sum(1 for word in junk_words if word in chunk_lower)
        if junk_count > 0 and len(cleaned_chunk.split()) < 20:
            continue

        # Create enhanced text with metadata context
        enhanced_text = cleaned_chunk
        if document_title:
            enhanced_text = f"Document: {document_title}\n\n{enhanced_text}"
        if extracted_date:
            enhanced_text = f"Date: {extracted_date}\n{enhanced_text}"

        docs.append(
            {
                "doc_id": doc_hash,
                "chunk_id": idx,
                "text": enhanced_text,
                "original_text": cleaned_chunk,  # Keep original for reference
                "metadata": metadata,
            }
        )

    return docs, new_urls


def _clean_soup(soup: BeautifulSoup) -> None:
    """Clean common navigation and junk elements from soup in-place."""
    # Remove common navigation / peripheral sections
    for tag in soup(
        [
            "nav",
            "header",
            "footer",
            "aside",
            "script",
            "style",
            "noscript",
            "iframe",
            "embed",
            "object",
        ]
    ):
        tag.decompose()

    # Remove elements with common junk classes/IDs
    for tag in soup.find_all(
        class_=re.compile(
            r"(cookie|privacy|newsletter|subscribe|social|share|ad|banner|popup)", re.I
        )
    ):
        tag.decompose()
    for tag in soup.find_all(
        id=re.compile(r"(cookie|privacy|newsletter|subscribe|social|share|ad|banner|popup)", re.I)
    ):
        tag.decompose()


def ingest_nice(
    config_yaml: str,
    raw_dir: str,
    *,
    verbose: bool = False,
    num_workers: int = 4,
    robots_mode: str = "strict",  # strict | lax | off
) -> list[dict[str, Any]]:
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
        num_workers: Number of concurrent workers for downloading files.
        robots_mode: Crawling mode for robots.txt (strict | lax | off).

    Returns:
        A list of dicts with keys: `doc_id`, `chunk_id`, and `text`.
    """
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)

    # Collect seed URLs
    source_keys = [
        "nice_urls",
        "radiopaedia_urls",
        "mriquestions_urls",
        "aan_urls",
        "acr_urls",
        "rcr_urls",
    ]

    urls: list[str] = []
    for key in source_keys:
        urls.extend(cfg.get(key, []))

    for item in cfg.get("sources", []):
        if not isinstance(item, dict):
            continue
        url_val = item.get("url")
        if url_val:
            urls.append(url_val)

    # Expand URLs via sitemaps but be more aggressive
    expanded_urls: list[str] = []
    for base_url in urls:
        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        sitemap_url = f"{domain}/sitemap.xml"

        try:
            sitemap_resp = requests.get(sitemap_url, timeout=10)
            sitemap_resp.raise_for_status()
            tree = ET.fromstring(sitemap_resp.content)
            locs: list[str] = []

            for loc in tree.findall(".//{*}loc"):
                if loc.text:
                    txt = html.unescape(loc.text).strip()
                    # More permissive filtering for sitemap URLs
                    if (
                        base_url.rstrip("/") in txt
                        and not _should_skip_url(txt)
                        and not re.search(r"\?(page|search|filter)=", txt)
                    ):
                        locs.append(txt)

            if locs:
                # Much higher limit for sitemap URLs
                expanded_urls.extend(locs[:1000])  # Increased from 200 to 1000
            else:
                expanded_urls.append(base_url)
        except Exception:
            if verbose:
                pass
            expanded_urls.append(base_url)

    # Deduplicate
    seen = set()
    urls = [u for u in expanded_urls if not (u in seen or seen.add(u))]

    if not urls:
        raise ValueError("No URLs found in config")

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    _load_etag_cache(str(Path(raw_dir) / "etag_cache.yaml"))

    # Get settings
    settings = cfg.get("settings", {})
    include_keywords = [
        kw.lower()
        for kw in settings.get(
            "include_keywords", ["mri", "magnetic resonance", "brain", "cerebral", "head"]
        )
    ]
    exclude_keywords = [kw.lower() for kw in settings.get("exclude_keywords", [])]
    exclude_patterns = [
        re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in exclude_keywords
    ]

    crawl_depth = int(settings.get("crawl_depth", 2))
    request_delay = float(settings.get("request_delay", 1.0))

    # Parallel processing
    all_docs = []
    _seen_chunk_hashes = set()
    visited = set()
    queue = [(url, 0) for url in urls]

    # Process in batches by depth to maintain some BFS behavior
    for current_depth in range(crawl_depth + 1):
        current_batch = [(url, depth) for url, depth in queue if depth == current_depth]
        if not current_batch:
            continue

        if verbose:
            pass

        # Download files in parallel
        download_results = []
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                download_futures = {
                    executor.submit(
                        _download_url, url, raw_dir, request_delay, verbose, robots_mode
                    ): url
                    for url, _ in current_batch
                    if url not in visited
                }

                for future in as_completed(download_futures):
                    url, filepath, success = future.result()
                    visited.add(url)
                    if success:
                        download_results.append((url, filepath))
        else:
            # Sequential fallback
            for url, _ in current_batch:
                if url in visited:
                    continue
                visited.add(url)
                url, filepath, success = _download_url(
                    url, raw_dir, request_delay, verbose, robots_mode
                )
                if success:
                    download_results.append((url, filepath))

        # Process files and extract new URLs
        new_urls_for_next_depth = []
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                process_futures = {
                    executor.submit(
                        _process_file,
                        url,
                        filepath,
                        include_keywords,
                        exclude_patterns,
                        settings,
                        verbose,
                    ): url
                    for url, filepath in download_results
                }

                for future in as_completed(process_futures):
                    docs, new_urls = future.result()

                    # Deduplicate chunks
                    for doc in docs:
                        chunk_hash = hashlib.md5(doc["text"].encode("utf-8")).hexdigest()
                        if chunk_hash not in _seen_chunk_hashes:
                            _seen_chunk_hashes.add(chunk_hash)
                            all_docs.append(doc)

                    # Collect new URLs for next depth
                    if current_depth < crawl_depth:
                        for new_url in new_urls:
                            if new_url not in visited:
                                new_urls_for_next_depth.append((new_url, current_depth + 1))
        else:
            # Sequential fallback
            for url, filepath in download_results:
                docs, new_urls = _process_file(
                    url, filepath, include_keywords, exclude_patterns, settings, verbose
                )

                # Deduplicate chunks
                for doc in docs:
                    chunk_hash = hashlib.md5(doc["text"].encode("utf-8")).hexdigest()
                    if chunk_hash not in _seen_chunk_hashes:
                        _seen_chunk_hashes.add(chunk_hash)
                        all_docs.append(doc)

                # Collect new URLs for next depth
                if current_depth < crawl_depth:
                    for new_url in new_urls:
                        if new_url not in visited:
                            new_urls_for_next_depth.append((new_url, current_depth + 1))

        # Add new URLs to queue for next depth
        queue.extend(new_urls_for_next_depth)

        if verbose:
            pass

    _save_etag_cache()

    return all_docs


def _extract_text_from_html(raw_html: str, url: str) -> tuple[str, BeautifulSoup]:
    """
    Extract clean text from HTML using readability with fallback.
    Returns (text, soup) tuple.
    """
    if not raw_html or not raw_html.strip():
        return "", None

    # Try readability first
    try:
        doc = ReadabilityDocument(raw_html)
        summary = doc.summary()
        if summary and summary.strip():
            soup = BeautifulSoup(summary, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            if text and len(text.strip()) > 50:  # Ensure we have meaningful content
                return text, soup
    except Exception:
        # Fallback to raw HTML processing if readability fails
        pass

    # Fallback: process raw HTML directly
    try:
        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Get text content
        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text, soup
    except Exception:
        # Last resort: return empty text
        return "", None
