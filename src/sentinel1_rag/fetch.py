"""Polite, cached download of the source pages.

The client identifies the project in its User-Agent, honours robots.txt and sends
at most one request per second. Pages already in data/raw/ are not downloaded again;
delete the file (or run `make clean`) to refresh one.
"""

import json
import re
import time
import urllib.robotparser
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from sentinel1_rag.config import RAW_DIR, USER_AGENT

MIN_INTERVAL_S = 1.0


@dataclass
class RawPage:
    url: str
    html: str
    fetched_at: str  # ISO 8601, UTC
    from_cache: bool


def slug(url: str) -> str:
    """File name for a page URL: .../web/s1-mission -> s1-mission."""
    last = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1] or "index"
    return re.sub(r"[^a-z0-9-]+", "-", last.lower())


class PoliteClient:
    """HTTP client that honours robots.txt and spaces requests MIN_INTERVAL_S apart."""

    def __init__(self) -> None:
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True
        )
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request = 0.0
        self.requests = 0

    def get(self, url: str) -> httpx.Response:
        if not self._robots_for(url).can_fetch(USER_AGENT, url):
            raise PermissionError(f"robots.txt disallows {url}")
        response = self._request(url)
        response.raise_for_status()
        return response

    def _request(self, url: str) -> httpx.Response:
        wait = self._last_request + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return self._http.get(url)
        finally:
            self._last_request = time.monotonic()
            self.requests += 1

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            response = self._request(f"{origin}/robots.txt")
            parser = urllib.robotparser.RobotFileParser()
            if response.status_code in (401, 403):
                parser.parse(["User-agent: *", "Disallow: /"])
            elif response.status_code >= 500:
                response.raise_for_status()
            elif response.status_code >= 400:  # no robots.txt: everything is allowed
                parser.parse([])
            else:
                parser.parse(response.text.splitlines())
            self._robots[origin] = parser
        return self._robots[origin]


def fetch(url: str, client: PoliteClient, raw_dir: Path = RAW_DIR) -> RawPage:
    """Return the page HTML, from data/raw/ if cached, otherwise downloading it."""
    html_path = raw_dir / f"{slug(url)}.html"
    meta_path = html_path.with_suffix(".json")
    if html_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return RawPage(url, html_path.read_text(encoding="utf-8"), meta["fetched_at"], True)

    response = client.get(url)
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(response.text, encoding="utf-8")
    # Written last: a page only counts as cached once its metadata exists.
    meta = {"url": url, "final_url": str(response.url), "fetched_at": fetched_at}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return RawPage(url, response.text, fetched_at, False)
