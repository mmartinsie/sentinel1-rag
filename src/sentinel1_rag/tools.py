"""The agent's two tools, each a Python method with a JSON Schema that tells the model how to call it (D-027).

- search_docs: the retrieval of `make ask`, returning numbered passages to cite.
- search_scenes: Sentinel-1 acquisitions over a place and dates, from the Copernicus STAC catalogue.

A tool's result is text for the model. What the user is shown afterwards (the cited sections and
the scenes found) comes from the records kept here, never from the model's words.

The MCP server (mcp_server.py) exposes the same tools to other programs: it builds its schemas
from SCHEMAS and runs scene searches through SceneFinder, but writes its own results (D-032).
"""

import inspect
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx
import psycopg

from sentinel1_rag.geocode import Geocoder, Place
from sentinel1_rag.ollama import Ollama
from sentinel1_rag.retrieval import DEFAULT_METHOD, DEFAULT_RANKING, TOP_K, Hit, retrieve
from sentinel1_rag.scenes import COLLECTIONS, MODES, ORBITS, SCENE_LIMIT, Scene, search
from sentinel1_rag.translate import to_english, use_translation

# A filter's value for "no filter". Without it in the enums, the model chose a mode for searches
# that did not ask for one: IW, and once SM, which found nothing (D-027).
ANY = "any"

# What the model reads about each tool: the descriptions are part of the prompt (D-027).
SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the Sentinel-1 documentation (SentiWiki pages and technical documents). "
                "Returns numbered passages; cite them by number, like [2]."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for, as a question."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_scenes",
            "description": (
                "Find Sentinel-1 acquisitions (scenes) over a place in a date range, in the Copernicus "
                f"Data Space catalogue. Returns the place found and a summary of the scenes (up to {SCENE_LIMIT}, newest first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place": {
                        "type": "string",
                        "description": (
                            "Name of the place with its region and country, "
                            "e.g. 'Lake Constance, Germany'. The tool finds its coordinates."
                        ),
                    },
                    "start": {"type": "string", "description": "First day, YYYY-MM-DD."},
                    "end": {"type": "string", "description": "Last day, YYYY-MM-DD."},
                    "days": {
                        "type": "integer",
                        "description": (
                            "Only for 'the last N days' or 'the past N weeks': the number of days up to "
                            "today, e.g. 10 or 14. When given, it replaces start and end."
                        ),
                    },
                    "product_type": {
                        "type": "string",
                        "enum": list(COLLECTIONS),
                        "description": "GRD (detected, in ground range; the default) or SLC (complex, in slant range).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": [ANY, *MODES],
                        "description": f"Acquisition mode: {ANY} (the default) unless the question names one.",
                    },
                    "orbit": {
                        "type": "string",
                        "enum": [ANY, *ORBITS],
                        "description": f"Orbit direction: {ANY} (the default) unless the question names one.",
                    },
                },
                # start and end stay required even though days can replace them: when they were
                # optional, the model left out end for "September 2026" and repeated that call (D-027).
                "required": ["place", "start", "end"],
            },
        },
    },
]


class ToolError(Exception):
    """A problem the model can fix or report: a bad argument, an unknown place, a service down."""


class SmallPlace(ToolError):
    """The place found is less than 1 km across: usually the wrong match for a short name.
    "Doñana, Spain" is a peak and "Delta del Ebro, Spain" a street (D-028). The caller adds how
    to confirm the place, because the agent and the MCP server confirm it differently."""

    def __init__(self, asked: str, found: Place) -> None:
        super().__init__(
            f"{asked!r} was found as {found.name}, a {found.kind} less than 1 km across, which is "
            "probably not the place the question means."
        )


@dataclass
class SceneSearch:
    """One search_scenes call, as it was run."""

    place: Place
    product_type: str
    start: date
    end: date
    mode: str | None
    orbit: str | None
    scenes: list[Scene]
    more: bool  # the catalogue has more scenes than were returned

    @property
    def collection(self) -> str:
        return COLLECTIONS[self.product_type]

    def describe(self) -> str:
        filters = ", ".join(
            [f"{self.mode} mode" if self.mode else "any mode", f"{self.orbit} orbits" if self.orbit else "both orbit directions"]
        )
        return f"{self.product_type} scenes over {self.place.name}, {self.start} to {self.end} ({filters})"


class SceneFinder:
    """Runs scene searches: checks the arguments, finds the place and searches the catalogue.
    Shared by the agent's Toolbox and the MCP server, which makes a new one for every call (D-032)."""

    def __init__(self, http: httpx.Client, geocoder: Geocoder, today: date) -> None:
        self._http = http
        self._geocoder = geocoder
        self._today = today  # what `days` counts back from
        self._questioned: set[str] = set()  # small places already sent back once
        self.searches: list[SceneSearch] = []

    def find(
        self,
        place: str,
        start: str | None = None,
        end: str | None = None,
        days: int | None = None,
        product_type: str = "GRD",
        mode: str | None = None,
        orbit: str | None = None,
        confirm_place: bool = False,
    ) -> SceneSearch:
        """Raises ToolError, or SmallPlace for a place under 1 km across unless it is confirmed:
        by `confirm_place`, or by asking for the same place a second time."""
        product_type = product_type.strip().upper()
        mode = mode.strip().upper() if mode and mode.strip().lower() != ANY else None
        orbit = orbit.strip().lower() if orbit and orbit.strip().lower() != ANY else None
        if product_type not in COLLECTIONS:
            raise ToolError(f"product_type must be one of {', '.join(COLLECTIONS)}, not {product_type!r}.")
        if mode and mode not in MODES:
            raise ToolError(f"mode must be one of {', '.join(MODES)}, not {mode!r}.")
        if orbit and orbit not in ORBITS:
            raise ToolError(f"orbit must be one of {', '.join(ORBITS)}, not {orbit!r}.")
        first, last = self._dates(start, end, days)

        try:
            found = self._geocoder.lookup(place)
        except httpx.HTTPError as exc:
            raise ToolError(f"the geocoding service did not answer ({exc}).") from None
        if found is None:
            raise ToolError(f"no place called {place!r} was found. Try another name, with its region and country.")
        width, height = found.size_km()
        if max(width, height) < 1 and not confirm_place and found.name not in self._questioned:
            # A second call that finds the same place means it is the one meant, such as a real
            # peak, and goes through (D-028). The MCP server makes a new SceneFinder for every
            # call, so there only confirm_place lets a small place through (D-032).
            self._questioned.add(found.name)
            raise SmallPlace(place, found)
        try:
            scenes, more = search(self._http, found.bbox, COLLECTIONS[product_type], first, last, mode, orbit)
        except httpx.HTTPError as exc:
            raise ToolError(f"the Copernicus catalogue did not answer ({exc}).") from None
        record = SceneSearch(found, product_type, first, last, mode, orbit, scenes, more)
        self.searches.append(record)
        return record

    def _dates(self, start: str | None, end: str | None, days: int | None) -> tuple[date, date]:
        """The first and last day to search. `days` wins over start and end: counting back from
        today is the code's job, because the model got "the last 10 days" wrong (D-027)."""
        if isinstance(days, str) and days.strip().isdigit():
            days = int(days)
        if days is not None:
            if not isinstance(days, int) or days < 1:
                raise ToolError(f"days must be a whole number of at least 1, not {days!r}.")
            return self._today - timedelta(days=days), self._today
        # Errors name what is missing: told only "give start and end, or days", the model sent the
        # same call without `end` three times.
        if start and not end:
            raise ToolError(f"end is missing. Give the last day as end (YYYY-MM-DD) along with start {start}.")
        if end and not start:
            raise ToolError(f"start is missing. Give the first day as start (YYYY-MM-DD) along with end {end}.")
        if not start:
            raise ToolError("the dates are missing. Give start and end (YYYY-MM-DD), or days.")
        try:
            first, last = date.fromisoformat(start), date.fromisoformat(end)
        except ValueError:
            raise ToolError(f"start and end must be dates written YYYY-MM-DD, not {start!r} and {end!r}.") from None
        if first > last:
            raise ToolError(f"start ({start}) is after end ({end}).")
        return first, last


class Toolbox:
    """Runs the agent's tools for one question and keeps what they returned."""

    def __init__(
        self, conn: psycopg.Connection, ollama: Ollama, http: httpx.Client, geocoder: Geocoder, today: date
    ) -> None:
        self._conn = conn
        self._ollama = ollama
        self._scenes = SceneFinder(http, geocoder, today)
        self.passages: list[Hit] = []  # numbered from 1 across all search_docs calls

    @property
    def searches(self) -> list[SceneSearch]:
        return self._scenes.searches

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool call. Errors become the result, so the model can correct the call or report it."""
        tool = {"search_docs": self.search_docs, "search_scenes": self.search_scenes}.get(name)
        if tool is None:
            return f"Error: there is no tool called {name!r}."
        try:
            # Checked first, so that a TypeError raised inside a tool is never mistaken for a bad call.
            inspect.signature(tool).bind(**arguments)
        except TypeError as exc:
            return f"Error: wrong arguments for {name}: {exc}."
        try:
            return tool(**arguments)
        except ToolError as exc:
            return f"Error: {exc}"

    def search_docs(self, query: str) -> str:
        """The default retrieval of `make ask` (hybrid, the full-text half in English, top 5), run
        on the model's query. A chunk found again keeps its number."""
        english = to_english(self._ollama, query) if use_translation(DEFAULT_METHOD, None) else None
        hits = retrieve(self._conn, self._ollama, query, k=TOP_K, method=DEFAULT_METHOD, english=english, ranking=DEFAULT_RANKING)
        if not hits:
            raise ToolError("the documentation index is empty.")
        known = {hit.chunk_id: n for n, hit in enumerate(self.passages, start=1)}
        passages = []
        for hit in hits:
            if hit.chunk_id not in known:
                self.passages.append(hit)
                known[hit.chunk_id] = len(self.passages)
            passages.append(f"[{known[hit.chunk_id]}] {hit.section}\n{hit.content}")
        return "\n\n".join(passages)

    def search_scenes(
        self,
        place: str,
        start: str | None = None,
        end: str | None = None,
        days: int | None = None,
        product_type: str = "GRD",
        mode: str | None = None,
        orbit: str | None = None,
    ) -> str:
        try:
            record = self._scenes.find(place, start, end, days, product_type, mode, orbit)
        except SmallPlace as exc:
            # As a warning next to the scenes, the model ignored a tiny place; as an error, it
            # searches again. A second call that finds the same place goes through (SceneFinder).
            raise ToolError(
                f"{exc} Call search_scenes again with a more precise name, such as its full official "
                "name, or with the same name if this is the right place."
            ) from None
        width, height = record.place.size_km()
        lines = [
            f"Place found: {record.place.name} (OpenStreetMap type: {record.place.kind}, about {width:.0f} x {height:.0f} km).",
            f"Searched: {record.describe()}.",
            *summarise(record.scenes, record.more),
        ]
        if record.scenes:
            lines.append("The user sees the scenes in a table under your answer: summarise them, do not list them.")
        # Without this line, the model cited "[1]" after scene facts in 4 of 8 answers about
        # scenes only, although no passage had been read (D-027).
        lines.append("These results are not documentation passages and have no number: do not cite them with [n].")
        return "\n".join(lines)


def summarise(scenes: list[Scene], more: bool) -> list[str]:
    """A summary of the scenes found, with counts computed here. The agent's model reads only
    this, never the scene names: listed one by one, the scenes got copied into the answer (about
    20 s for ten), although the user sees them in a table printed by the code."""
    if not scenes:
        return ["No scenes found."]
    # Consecutive scenes ("slices") of one pass share the day, the satellite and the track.
    passes = Counter((s.acquired[:10], s.platform, s.orbit, s.relative_orbit) for s in scenes)
    tracks: dict[str, set[int]] = {}
    for s in scenes:
        tracks.setdefault(s.orbit, set()).add(s.relative_orbit)
    satellites = Counter(satellite(s.platform) for s in scenes)
    modes = Counter(f"{s.mode} {'+'.join(s.polarisations)}" for s in scenes)
    days = sorted({s.acquired[:10] for s in scenes})
    found = f"{len(scenes)} newest scenes; the catalogue has more" if more else f"{len(scenes)} scenes"
    return [
        f"Found: {found}, from {len(passes)} passes of a satellite over the place.",
        f"Days: {', '.join(days)}.",
        f"Satellites: {', '.join(f'{name} ({n} scenes)' for name, n in sorted(satellites.items()))}.",
        "Orbits: "
        + "; ".join(f"{orbit}, track {', '.join(map(str, sorted(t)))}" for orbit, t in sorted(tracks.items()))
        + ".",
        f"Modes and polarisations: {', '.join(f'{m} ({n} scenes)' for m, n in sorted(modes.items()))}.",
    ]


def scene_line(s: Scene) -> str:
    """One scene as a table row: when, satellite, mode, polarisations, orbit, track and name."""
    return (
        f"{s.acquired[:10]} {s.acquired[11:16]}  {s.id[:3]} {s.mode:<3} {'+'.join(s.polarisations):<6}"
        f" {s.orbit:<10} track {s.relative_orbit:<4} {s.id}"
    )


def satellite(platform: str) -> str:
    """sentinel-1c -> Sentinel-1C"""
    return "Sentinel-" + platform.removeprefix("sentinel-").upper()
