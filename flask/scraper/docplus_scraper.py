"""Docplus scraper for Region Uppsala search pages.

This script runs as one operation: fetch paginated search result pages, download
linked documents, extract text, and write parsed JSON + summary metadata.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urldefrag, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from scraper.parsers import extract_page_count, extract_text
from scraper.storage import DocumentStore, DownloadResult


DEFAULT_USER_AGENT = "ThesisSearchDocplusScraper/1.0"
DEFAULT_BASE_URL = "https://publikdocplus.regionuppsala.se/"
DEFAULT_START_PATH = "/Home/Search?searchValue=&oldFilter=&facet=&facetVal=&page=1"
DEFAULT_PAGE_START = 1
DEFAULT_PAGE_END = 620
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_DELAY_SECONDS = 1.0
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}


@dataclass
class ScraperConfig:
    base_url: str
    start_path: str
    page_start: int
    page_end: int
    output_dir: str = DEFAULT_OUTPUT_DIR
    delay_seconds: float = DEFAULT_DELAY_SECONDS
    user_agent: str = DEFAULT_USER_AGENT
    request_timeout: int = 30


@dataclass
class PageResult:
    url: str
    status_code: int
    document_links: tuple[str, ...]


class DocplusScraper:
    def __init__(self, config: ScraperConfig, session: Optional[requests.Session] = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})
        self.store = DocumentStore(config.output_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def scrape(self) -> list[DownloadResult]:
        downloads: list[DownloadResult] = []
        for page in range(self.config.page_start, self.config.page_end + 1):
            current = build_page_url(self.config.base_url, self.config.start_path, page)
            self.logger.info("Fetching search page %s", current)
            page_result = self._fetch_page(current)

            for link in page_result.document_links:
                downloads.append(self._download_document(link))

            time.sleep(self.config.delay_seconds)

        return downloads

    def _fetch_page(self, url: str) -> PageResult:
        try:
            response = self.session.get(url, timeout=self.config.request_timeout)
            if response.status_code in {404, 410}:
                self.logger.warning("Skipping missing page %s (status %s)", url, response.status_code)
                return PageResult(url=url, status_code=response.status_code, document_links=tuple())
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else 0
            if status_code in {404, 410}:
                self.logger.warning("Skipping missing page %s (status %s)", url, status_code)
                return PageResult(url=url, status_code=status_code, document_links=tuple())
            raise

        soup = BeautifulSoup(response.text, "html.parser")
        document_links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = normalize_url(urljoin(url, anchor["href"]))
            if self._is_document_link(href):
                document_links.append(href)

        return PageResult(
            url=url,
            status_code=response.status_code,
            document_links=tuple(sorted(set(document_links))),
        )

    def _download_document(self, url: str) -> DownloadResult:
        self.logger.info("Downloading %s", url)
        try:
            response = self.session.get(url, timeout=self.config.request_timeout)
            if response.status_code in {404, 410}:
                self.logger.warning("Skipping missing document %s (status %s)", url, response.status_code)
                return DownloadResult(url=url, filename="", extracted_text_len=0, metadata={})
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else 0
            if status_code in {404, 410}:
                self.logger.warning("Skipping missing document %s (status %s)", url, status_code)
                return DownloadResult(url=url, filename="", extracted_text_len=0, metadata={})
            raise

        filename = self.store.write_binary(url, response.content)
        extracted_text = extract_text(filename)
        page_count = extract_page_count(filename)
        document_name = extract_document_name_from_url(url)
        metadata = {
            "source_url": url,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "content_type": response.headers.get("Content-Type", ""),
            "document_name": document_name,
            "title": extract_title_from_document_name(document_name),
            "title_source": "url_filename",
            "page_count": page_count,
        }
        self.store.write_text(filename, extracted_text, metadata)
        return DownloadResult(
            url=url,
            filename=filename,
            extracted_text_len=len(extracted_text),
            metadata=metadata,
        )

    def _is_document_link(self, url: str) -> bool:
        parsed = urlparse(url)
        _, ext = os.path.splitext(parsed.path.lower())
        if ext in DOCUMENT_EXTENSIONS:
            return True

        if "getdocument" in parsed.path.lower():
            return True

        query = parse_qs(parsed.query)
        for values in query.values():
            for value in values:
                _, value_ext = os.path.splitext(value.lower())
                if value_ext in DOCUMENT_EXTENSIONS:
                    return True
        return False


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Region Uppsala Docplus pages into parsed JSON + summary.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Docplus base URL.")
    parser.add_argument("--start-path", default=DEFAULT_START_PATH, help="Search path template containing page query.")
    parser.add_argument("--page-start", type=int, default=DEFAULT_PAGE_START, help="Starting page number.")
    parser.add_argument("--page-end", type=int, default=DEFAULT_PAGE_END, help="Ending page number (inclusive).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to store downloads and parsed text.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Delay between requests.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def normalize_url(url: str) -> str:
    normalized, _ = urldefrag(url)
    return normalized


def set_page_param(url: str, page: int) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["page"] = [str(page)]
    updated_query = urlencode(query, doseq=True)
    updated = parsed._replace(query=updated_query)
    return normalize_url(urlunparse(updated))


def build_page_url(base_url: str, start_path: str, page: int) -> str:
    if re.match(r"^https?://", start_path):
        template_url = normalize_url(start_path)
    else:
        base = base_url if base_url.endswith("/") else f"{base_url}/"
        template_url = normalize_url(urljoin(base, start_path.lstrip("/")))
    return set_page_param(template_url, page)


def extract_document_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key in ("filename", "file", "name"):
        values = query.get(key)
        if values:
            candidate = unquote(values[0]).strip()
            if candidate:
                return candidate

    last = parsed.path.split("/")[-1].strip()
    if last and last.lower() != "getdocument":
        return unquote(last)
    return ""


def extract_title_from_document_name(document_name: str) -> str:
    stem, ext = os.path.splitext(document_name.strip())
    if stem and ext:
        return stem
    return document_name.strip()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.page_start > args.page_end:
        raise ValueError("page-start must be less than or equal to page-end.")

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    config = ScraperConfig(
        base_url=args.base_url,
        start_path=args.start_path,
        page_start=args.page_start,
        page_end=args.page_end,
        output_dir=args.output_dir,
        delay_seconds=args.delay,
        user_agent=args.user_agent,
        request_timeout=args.timeout,
    )

    scraper = DocplusScraper(config)
    results = scraper.scrape()
    successful = [result for result in results if result.filename]
    summary = {
        "downloaded": len(successful),
        "attempted": len(results),
        "documents": [dataclasses.asdict(result) for result in results],
    }

    output_path = os.path.join(config.output_dir, "summary.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logging.getLogger("DocplusScraper").info("Wrote summary to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
