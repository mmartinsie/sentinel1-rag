"""Search the Copernicus Data Space Ecosystem STAC catalogue for Sentinel-1 scenes (D-027).

STAC (SpatioTemporal Asset Catalog) is a standard API for satellite imagery catalogues: a POST to
/search with a collection, a bounding box, a time range and optional filters returns the matching
scenes ("items") as GeoJSON. Searching needs no account; downloading a product does, and is out
of scope here.

A full item is about 17 KB of JSON, so the request asks only for the properties used here (the
"fields" extension), and the filters on mode and orbit direction use CQL2, STAC's filter language.
"""

from dataclasses import dataclass
from datetime import date

import httpx

from sentinel1_rag.config import STAC_URL

COLLECTIONS = {"GRD": "sentinel-1-grd", "SLC": "sentinel-1-slc"}
MODES = ("IW", "EW", "SM")
ORBITS = ("ascending", "descending")
# Scenes returned per search, newest first. A month over one place already gives 10 or more.
SCENE_LIMIT = 10

FIELDS = [
    "id",
    "properties.datetime",
    "properties.platform",
    "properties.sar:instrument_mode",
    "properties.sar:polarizations",
    "properties.sat:orbit_state",
    "properties.sat:relative_orbit",
]


@dataclass
class Scene:
    id: str  # the product name, e.g. S1C_IW_GRDH_1SDV_20261009T061713_..._COG
    acquired: str  # start of the acquisition, ISO 8601 in UTC
    platform: str  # sentinel-1a, -1c or -1d
    mode: str  # IW, EW or SM
    polarisations: list[str]
    orbit: str  # ascending or descending
    relative_orbit: int  # the track: scenes on the same track see the place from the same geometry


def item_url(collection: str, scene_id: str) -> str:
    """Where a scene's full metadata is, including the download link (which needs an account)."""
    return f"{STAC_URL}/collections/{collection}/items/{scene_id}"


def search(
    http: httpx.Client,
    bbox: tuple[float, float, float, float],
    collection: str,
    start: date,
    end: date,
    mode: str | None = None,
    orbit: str | None = None,
    limit: int = SCENE_LIMIT,
) -> tuple[list[Scene], bool]:
    """Up to `limit` scenes that intersect the box between the two days (inclusive), newest first,
    and whether the catalogue has more. One more scene than needed is asked for: the API does not
    return the number of matches, and its `next` link does not say whether a next page has any."""
    filters = []
    if mode:
        filters.append({"op": "=", "args": [{"property": "sar:instrument_mode"}, mode]})
    if orbit:
        filters.append({"op": "=", "args": [{"property": "sat:orbit_state"}, orbit]})
    body: dict = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "fields": {"include": FIELDS, "exclude": ["links", "assets", "geometry", "bbox"]},
        "limit": limit + 1,
    }
    if filters:
        body["filter-lang"] = "cql2-json"
        body["filter"] = filters[0] if len(filters) == 1 else {"op": "and", "args": filters}
    response = http.post(f"{STAC_URL}/search", json=body)
    response.raise_for_status()
    features = response.json()["features"]
    scenes = [
        Scene(
            id=f["id"],
            acquired=f["properties"]["datetime"],
            platform=f["properties"]["platform"],
            mode=f["properties"]["sar:instrument_mode"],
            polarisations=f["properties"]["sar:polarizations"],
            orbit=f["properties"]["sat:orbit_state"],
            relative_orbit=f["properties"]["sat:relative_orbit"],
        )
        for f in features
    ]
    return scenes[:limit], len(scenes) > limit
