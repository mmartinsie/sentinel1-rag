"""Turn a place name into a bounding box with Nominatim, OpenStreetMap's geocoder (D-028).

The public Nominatim server has a usage policy (https://operations.osmfoundation.org/policies/nominatim/),
and this module follows it:
- at most one request per second, from a User-Agent that identifies the project;
- results cached on our side: data/geocode.json keeps every answer, including "not found", so
  asking again for a place sends no request;
- the OpenStreetMap attribution is shown wherever a result is used (ATTRIBUTION);
- the server can be switched without changing the code (NOMINATIM_URL).
The policy forbids heavy, systematic or autocomplete use; here each lookup is one question asked
by a person.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from sentinel1_rag.config import GEOCODE_CACHE, NOMINATIM_URL

ATTRIBUTION = "Places geocoded with Nominatim. Data © OpenStreetMap contributors, ODbL."
MIN_INTERVAL_S = 1.0


@dataclass
class Place:
    name: str  # Nominatim's display name, e.g. "Madrid, Comunidad de Madrid, España"
    kind: str  # what OpenStreetMap says it is: city, national_park, peak, road...
    bbox: tuple[float, float, float, float]  # west, south, east, north, in degrees (WGS84)

    def size_km(self) -> tuple[float, float]:
        """Width and height of the box, roughly: a degree of latitude is about 111 km."""
        west, south, east, north = self.bbox
        width = (east - west) * 111.32 * math.cos(math.radians((south + north) / 2))
        return width, (north - south) * 110.57


class Geocoder:
    def __init__(self, http: httpx.Client, cache_path: Path = GEOCODE_CACHE, base_url: str = NOMINATIM_URL) -> None:
        self._http = http
        self._cache_path = cache_path
        self._base_url = base_url
        self._cache: dict[str, dict | None] = (
            json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        )
        self._last_request = 0.0

    def lookup(self, query: str) -> Place | None:
        """The best match for a place name, or None if Nominatim knows no such place.

        Only the first result is used. A short name can be ambiguous: "Doñana, Spain" is a peak in
        Almería and "Delta del Ebro, Spain" a street, and the right place is not among the next
        four results either. Only a more precise name finds it, so the caller should show what
        was found (D-028).
        """
        key = " ".join(query.split()).casefold()
        if key not in self._cache:
            self._cache[key] = self._search(query)
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        found = self._cache[key]
        return Place(found["name"], found["kind"], tuple(found["bbox"])) if found else None

    def _search(self, query: str) -> dict | None:
        wait = self._last_request + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            response = self._http.get(f"{self._base_url}/search", params={"q": query, "format": "jsonv2", "limit": 1})
        finally:
            self._last_request = time.monotonic()
        response.raise_for_status()
        results = response.json()
        if not results:
            return None
        # Nominatim gives the box as [south, north, west, east], in strings.
        south, north, west, east = map(float, results[0]["boundingbox"])
        kind = results[0].get("addresstype") or results[0].get("type", "place")
        return {"name": results[0]["display_name"], "kind": kind, "bbox": [west, south, east, north]}
