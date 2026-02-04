"""Docplus scraper to collect documents and extract text for downstream use."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import queue
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import parse_qs, urlencode, urljoin, urldefrag, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from scraper.parsers import extract_text
from scraper.storage import DocumentStore, DownloadResult


DEFAULT_USER_AGENT = "ThesisSearchDocplusScraper/1.0"
DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}


@dataclass
class ScraperConfig:
    base_url: str
    start_paths: tuple[str, ...]
    output_dir: str
    delay_seconds: float = 0.5
    max_pages: Optional[int] = None
    user_agent: str = DEFAULT_USER_AGENT
    request_timeout: int = 30


@dataclass
class PageResult:
    url: str
    status_code: int
    document_links: tuple[str, ...]
    next_links: tuple[str, ...]


class DocplusScraper:
    def __init__(self, config: ScraperConfig, session: Optional[requests.Session] = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})
        self.store = DocumentStore(config.output_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def scrape(self) -> list[DownloadResult]:
        visited: set[str] = set()
        to_visit: queue.Queue[str] = queue.Queue()
        for path in self.config.start_paths:
            to_visit.put(self._build_start_url(path))

        downloads: list[DownloadResult] = []
        pages_processed = 0

        while not to_visit.empty():
            if self.config.max_pages and pages_processed >= self.config.max_pages:
                break

            current = to_visit.get()
            if current in visited:
                continue

            visited.add(current)
            self.logger.info("Fetching %s", current)
            page_result = self._fetch_page(current)
            pages_processed += 1

            for link in page_result.document_links:
                download = self._download_document(link)
                downloads.append(download)

            for link in page_result.next_links:
                if link not in visited:
                    to_visit.put(link)

            time.sleep(self.config.delay_seconds)

        return downloads

    def _fetch_page(self, url: str) -> PageResult:
        try:
            response = self.session.get(url, timeout=self.config.request_timeout)
            if response.status_code in {404, 410}:
                self.logger.warning("Skipping missing page %s (status %s)", url, response.status_code)
                return PageResult(
                    url=url,
                    status_code=response.status_code,
                    document_links=tuple(),
                    next_links=tuple(),
                )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else 0
            if status_code in {404, 410}:
                self.logger.warning("Skipping missing page %s (status %s)", url, status_code)
                return PageResult(
                    url=url,
                    status_code=status_code,
                    document_links=tuple(),
                    next_links=tuple(),
                )
            raise
        soup = BeautifulSoup(response.text, "html.parser")
        document_links: list[str] = []
        next_links: list[str] = []

        for anchor in soup.find_all("a", href=True):
            href = self._normalize_url(urljoin(url, anchor["href"]))
            if self._is_document_link(href):
                document_links.append(href)
            elif self._is_same_domain(href):
                next_links.append(href)

        return PageResult(
            url=url,
            status_code=response.status_code,
            document_links=tuple(sorted(set(document_links))),
            next_links=tuple(sorted(set(next_links))),
        )

    def _download_document(self, url: str) -> DownloadResult:
        self.logger.info("Downloading %s", url)
        try:
            response = self.session.get(url, timeout=self.config.request_timeout)
            if response.status_code in {404, 410}:
                self.logger.warning("Skipping missing document %s (status %s)", url, response.status_code)
                return DownloadResult(url=url, filename="", extracted_text_len=0)
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else 0
            if status_code in {404, 410}:
                self.logger.warning("Skipping missing document %s (status %s)", url, status_code)
                return DownloadResult(url=url, filename="", extracted_text_len=0)
            raise
        filename = self.store.write_binary(url, response.content)
        extracted_text = extract_text(filename)
        metadata = {
            "source_url": url,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "content_type": response.headers.get("Content-Type", ""),
        }
        self.store.write_text(filename, extracted_text, metadata)
        return DownloadResult(url=url, filename=filename, extracted_text_len=len(extracted_text))

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

    def _is_same_domain(self, url: str) -> bool:
        return urlparse(url).netloc == urlparse(self.config.base_url).netloc

    def _normalize_url(self, url: str) -> str:
        normalized, _ = urldefrag(url)
        return normalized

    def _build_start_url(self, path: str) -> str:
        if re.match(r"^https?://", path):
            return self._normalize_url(path)
        base_url = self.config.base_url
        if not base_url.endswith("/"):
            base_url += "/"
        return self._normalize_url(urljoin(base_url, path.lstrip("/")))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Docplus documents for later parsing.")
    parser.add_argument("--base-url", required=True, help="Base URL for Docplus, e.g. https://docplus.example.com/")
    parser.add_argument(
        "--start-paths",
        default="/",
        help="Comma-separated list of paths to seed the crawl.",
    )
    parser.add_argument("--page-start", type=int, help="Starting page number for search pagination.")
    parser.add_argument("--page-end", type=int, help="Ending page number for search pagination (inclusive).")
    parser.add_argument("--output-dir", default="output", help="Directory to store downloads and parsed text.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests.")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to visit.")
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


def build_start_urls(
    base_url: str,
    start_paths: tuple[str, ...],
    page_start: Optional[int],
    page_end: Optional[int],
) -> tuple[str, ...]:
    def build_start_url(path: str) -> str:
        if re.match(r"^https?://", path):
            return normalize_url(path)
        base = base_url
        if not base.endswith("/"):
            base += "/"
        return normalize_url(urljoin(base, path.lstrip("/")))

    if page_start is None and page_end is None:
        return tuple(build_start_url(path) for path in start_paths)

    if page_start is None or page_end is None:
        raise ValueError("Both page_start and page_end must be provided when using page range.")
    if len(start_paths) != 1:
        raise ValueError("Provide a single start path when using page range options.")
    if page_start > page_end:
        raise ValueError("page_start must be less than or equal to page_end.")

    template_url = build_start_url(start_paths[0])
    return tuple(set_page_param(template_url, page) for page in range(page_start, page_end + 1))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    config = ScraperConfig(
        base_url=args.base_url,
        start_paths=build_start_urls(
            base_url=args.base_url,
            start_paths=tuple(path.strip() for path in args.start_paths.split(",") if path.strip()),
            page_start=args.page_start,
            page_end=args.page_end,
        ),
        output_dir=args.output_dir,
        delay_seconds=args.delay,
        max_pages=args.max_pages,
        user_agent=args.user_agent,
        request_timeout=args.timeout,
    )
    scraper = DocplusScraper(config)
    results = scraper.scrape()
    summary = {
        "downloaded": len(results),
        "documents": [dataclasses.asdict(result) for result in results],
    }
    output_path = os.path.join(config.output_dir, "summary.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logging.getLogger("DocplusScraper").info("Wrote summary to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
