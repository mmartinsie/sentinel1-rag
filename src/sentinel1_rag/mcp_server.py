"""`make mcp`: an MCP server that offers search_docs and search_scenes to other programs (D-031, D-032).

MCP (Model Context Protocol) lets an application with a language model, the host (Claude Code,
Claude Desktop, an IDE), discover and call tools that another program offers. The host starts this
server as a subprocess and speaks JSON-RPC 2.0 over its stdin and stdout, one message per line:
`tools/list` returns the tools with their JSON Schemas, and `tools/call` runs one of them.

The protocol is handled by the official SDK's low-level Server, which serves clients of both eras:
the `initialize` handshake of revisions up to 2025-11-25, and the per-request metadata of 2026-07-28.
The tools are the agent's (tools.py), adapted to a host and a model this project does not control:

- Every call stands alone, because revision 2026-07-28 has no sessions: passages are numbered
  within each result and carry their section and URL, and a place under 1 km across needs
  `confirm_place` instead of being asked for twice (D-032).
- search_docs runs the default retrieval of `make ask` on a query the host's model writes in
  English, so the server needs no language model of its own (D-032).
- Each result is text for the host's model plus structured data (`structuredContent`), which a
  client checks against the tool's `outputSchema` and can show as sources or a table of scenes.

stdout carries the protocol, and anything else written there would corrupt it: logs go to stderr.
"""

import copy
import inspect
import json
import sys
import time
from datetime import date
from importlib.metadata import version
from typing import Any

import anyio
import httpx
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)

from sentinel1_rag.config import USER_AGENT
from sentinel1_rag.db import connect
from sentinel1_rag.geocode import ATTRIBUTION, Geocoder
from sentinel1_rag.ollama import Ollama, OllamaError
from sentinel1_rag.retrieval import DEFAULT_METHOD, DEFAULT_RANKING, TOP_K, retrieve
from sentinel1_rag.scenes import SCENE_LIMIT, item_url
from sentinel1_rag.tools import SCHEMAS, SceneFinder, SmallPlace, ToolError, scene_line, summarise

NAME = "sentinel1-rag"
# JSON-RPC's code for invalid parameters, which MCP uses for an unknown tool.
INVALID_PARAMS = -32602

# Sent to the host when it connects. The host decides whether its model reads them.
INSTRUCTIONS = f"""\
Tools for the Sentinel-1 radar satellite mission.
- search_docs searches the official documentation: the Sentinel-1 SentiWiki and six technical \
documents. Write the query in English, the language of the documents. It returns the {TOP_K} best \
passages; cite the ones you use with their section and URL, and say so when they do not answer.
- search_scenes finds Sentinel-1 acquisitions over a named place in the Copernicus Data Space \
catalogue. It reports the place it found: check that it is the one meant, because short names \
can match another place. Show the OpenStreetMap attribution with its results."""


def agent_schema(name: str) -> dict[str, Any]:
    """A tool's JSON Schema as the agent's model gets it (tools.py), copied so it can be adapted."""
    return copy.deepcopy(next(s["function"] for s in SCHEMAS if s["function"]["name"] == name))


def build_tools() -> list[Tool]:
    """The agent's tools, with descriptions for a host's model and the shape of their results."""
    docs = agent_schema("search_docs")
    docs["parameters"]["properties"]["query"]["description"] = (
        "What to look for, as a question in English, the language of the documents."
    )
    scenes = agent_schema("search_scenes")
    scenes["parameters"]["properties"]["confirm_place"] = {
        "type": "boolean",
        "description": (
            "Only after an error saying that the place found is less than 1 km across: "
            "true if that small place (a peak, a building) is the one meant."
        ),
    }
    passage = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "section": {"type": "string", "description": "Heading path, starting with the page or document title."},
            "url": {"type": "string", "description": "Link to the section: a page anchor, or #page=N in a PDF."},
            "content": {"type": "string"},
        },
        "required": ["n", "section", "url", "content"],
    }
    scene = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The product name."},
            "acquired": {"type": "string", "description": "Start of the acquisition, ISO 8601 in UTC."},
            "platform": {"type": "string"},
            "mode": {"type": "string"},
            "polarisations": {"type": "array", "items": {"type": "string"}},
            "orbit": {"type": "string"},
            "relative_orbit": {"type": "integer"},
            "metadata_url": {"type": "string"},
        },
        "required": ["id", "acquired", "platform", "mode", "polarisations", "orbit", "relative_orbit", "metadata_url"],
    }
    nullable_string = {"type": ["string", "null"]}
    return [
        Tool(
            name="search_docs",
            title="Search the Sentinel-1 documentation",
            description=(
                "Search the Sentinel-1 documentation: the SentiWiki pages and six technical documents "
                "(radiometric calibration, thermal denoising, GRD border noise, TOPS deramping, geocoding, "
                f"product definition). Returns the {TOP_K} best passages, each with its section and the URL to cite."
            ),
            input_schema=docs["parameters"],
            output_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "passages": {"type": "array", "items": passage}},
                "required": ["query", "passages"],
            },
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        ),
        Tool(
            name="search_scenes",
            title="Find Sentinel-1 scenes",
            description=(
                "Find Sentinel-1 acquisitions (scenes) over a place in a date range, in the Copernicus Data "
                "Space catalogue. Returns the place found, a summary and the scenes "
                f"(up to {SCENE_LIMIT}, newest first), with the product names."
            ),
            input_schema=scenes["parameters"],
            output_schema={
                "type": "object",
                "properties": {
                    "place": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "kind": {"type": "string", "description": "What OpenStreetMap says the place is."},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "West, south, east, north, in degrees.",
                            },
                        },
                        "required": ["name", "kind", "bbox"],
                    },
                    "product_type": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "mode": nullable_string,
                    "orbit": nullable_string,
                    "scenes": {"type": "array", "items": scene},
                    "more": {"type": "boolean", "description": "The catalogue has more scenes than were returned."},
                    "attribution": {"type": "string"},
                },
                "required": ["place", "product_type", "start", "end", "mode", "orbit", "scenes", "more", "attribution"],
            },
            # Open world: it asks two outside services, Nominatim and the Copernicus catalogue.
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
        ),
    ]


TOOLS = build_tools()


def failure(message: str) -> CallToolResult:
    """A tool execution error: the host's model reads it and can correct the call (isError)."""
    return CallToolResult(content=[TextContent(type="text", text=f"Error: {message}")], is_error=True)


class Tools:
    """Runs the tools. Only what must hold across calls is kept here: the clients, and the
    geocoder with its cache and its limit of one request per second. Everything else is made
    for each call, because the host may send calls from different conversations (D-032)."""

    def __init__(self, http: httpx.Client) -> None:
        self._ollama = Ollama()
        self._http = http
        self._geocoder = Geocoder(http)

    def call(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        tool = {"search_docs": self.search_docs, "search_scenes": self.search_scenes}[name]
        try:
            # Checked first, so that a TypeError raised inside a tool is never mistaken for a bad call.
            inspect.signature(tool).bind(**arguments)
        except TypeError as exc:
            return failure(f"wrong arguments for {name}: {exc}.")
        try:
            return tool(**arguments)
        except ToolError as exc:
            return failure(str(exc))

    def search_docs(self, query: str) -> CallToolResult:
        """The default retrieval (hybrid, BM25, top 5), with the host's English query for both
        halves: measured on the eval questions, it finds the same 22 of 25 (D-032)."""
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a question, in English.")
        try:
            conn = connect()
        except RuntimeError:
            raise ToolError("the documentation database is not running; start it with `make up`.") from None
        with conn:
            try:
                hits = retrieve(conn, self._ollama, query, k=TOP_K, method=DEFAULT_METHOD, ranking=DEFAULT_RANKING)
            except OllamaError as exc:
                raise ToolError(f"the embedding model is not available: {exc}") from None
        if not hits:
            raise ToolError("the documentation index is empty; build it with `make ingest` and `make index`.")
        passages = [
            {"n": n, "section": hit.section, "url": hit.url, "content": hit.content}
            for n, hit in enumerate(hits, start=1)
        ]
        text = "\n\n".join(f"[{p['n']}] {p['section']}\nURL: {p['url']}\n{p['content']}" for p in passages)
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structured_content={"query": query, "passages": passages},
        )

    def search_scenes(
        self,
        place: str,
        start: str | None = None,
        end: str | None = None,
        days: int | None = None,
        product_type: str = "GRD",
        mode: str | None = None,
        orbit: str | None = None,
        confirm_place: bool = False,
    ) -> CallToolResult:
        if not isinstance(confirm_place, bool):
            raise ToolError(f"confirm_place must be true or false, not {confirm_place!r}.")
        # A new finder for every call: a small place sent back once is not remembered (D-032).
        finder = SceneFinder(self._http, self._geocoder, date.today())
        try:
            record = finder.find(place, start, end, days, product_type, mode, orbit, confirm_place)
        except SmallPlace as exc:
            raise ToolError(
                f"{exc} Call search_scenes again with a more precise name, such as its full official name, "
                "or with confirm_place set to true if this is the place meant."
            ) from None
        width, height = record.place.size_km()
        metadata = item_url(record.collection, "<name>")
        lines = [
            f"Place found: {record.place.name} (OpenStreetMap type: {record.place.kind}, about {width:.0f} x {height:.0f} km).",
            f"Searched: {record.describe()}.",
            *summarise(record.scenes, record.more),
        ]
        if record.scenes:
            lines += ["Scenes, newest first:", *(f"  {scene_line(s)}" for s in record.scenes)]
            if record.more:
                lines.append(f"  (the {len(record.scenes)} newest; the catalogue has more: narrow the dates to see them)")
            lines.append(f"Metadata of a scene: {metadata}")
        lines.append(ATTRIBUTION)
        structured = {
            "place": {"name": record.place.name, "kind": record.place.kind, "bbox": list(record.place.bbox)},
            "product_type": record.product_type,
            "start": record.start.isoformat(),
            "end": record.end.isoformat(),
            "mode": record.mode,
            "orbit": record.orbit,
            "scenes": [
                {
                    "id": s.id,
                    "acquired": s.acquired,
                    "platform": s.platform,
                    "mode": s.mode,
                    "polarisations": s.polarisations,
                    "orbit": s.orbit,
                    "relative_orbit": s.relative_orbit,
                    "metadata_url": item_url(record.collection, s.id),
                }
                for s in record.scenes
            ],
            "more": record.more,
            "attribution": ATTRIBUTION,
        }
        return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))], structured_content=structured)


def log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def result_summary(result: CallToolResult) -> str:
    """One line about a result, for the log."""
    if result.is_error:
        return result.content[0].text
    data = result.structured_content
    if "passages" in data:
        return f"{len(data['passages'])} passages"
    return f"{len(data['scenes'])} scenes over {data['place']['name']}"


def build_server(tools: Tools) -> Server:
    # One call at a time: the geocoder's cache and rate limit, and the Ollama client, are not
    # made for threads, and a person asking questions does not need calls in parallel.
    lock = anyio.Lock()
    names = {tool.name for tool in TOOLS}

    async def list_tools(ctx: ServerRequestContext, params: PaginatedRequestParams | None) -> ListToolsResult:
        log(f"tools/list, protocol revision {ctx.protocol_version}")
        return ListToolsResult(tools=TOOLS)

    async def call_tool(ctx: ServerRequestContext, params: CallToolRequestParams) -> CallToolResult:
        if params.name not in names:
            # An unknown tool is a protocol error, not a result the model could act on.
            raise MCPError(INVALID_PARAMS, f"Unknown tool: {params.name}")
        arguments = params.arguments or {}
        async with lock:
            started = time.perf_counter()
            # The tools block (database, Ollama, HTTP), so they run in a worker thread while the
            # event loop keeps reading the protocol.
            result = await anyio.to_thread.run_sync(tools.call, params.name, arguments)
        seconds = time.perf_counter() - started
        log(f"{params.name}({json.dumps(arguments, ensure_ascii=False)})  {seconds:.1f}s -> {result_summary(result)}")
        return result

    return Server(
        NAME,
        version=version("sentinel1-rag"),
        title="Sentinel-1 documentation and scenes",
        instructions=INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def serve() -> None:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as http:
        server = build_server(Tools(http))
        async with stdio_server() as (read, write):
            # stdio_server has moved fd 1 to stderr and keeps the protocol on a copy of it. A
            # print() would still wait in Python's buffer and reach the protocol when the server
            # exits, after fd 1 is put back; pointing sys.stdout at stderr sends it there at once (D-031).
            sys.stdout = sys.stderr
            log(f"{NAME} {version('sentinel1-rag')}: MCP server on stdio, tools {', '.join(t.name for t in TOOLS)}")
            await server.run(read, write, server.create_initialization_options())


def main() -> None:
    anyio.run(serve)


if __name__ == "__main__":
    main()
