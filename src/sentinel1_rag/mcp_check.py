"""`make mcp-check`: start the MCP server as a host would and check it through the protocol (D-031).

The SDK's client launches `python -m sentinel1_rag.mcp_server` as a subprocess and talks to it
over stdio twice: once with the `initialize` handshake of the older revisions (the client offers
2025-11-25) and once with the per-request metadata of 2026-07-28. Each time it lists the tools and
calls them: the client checks every structured result against the tool's outputSchema. The checks:

- the tools and their arguments are the agent's (tools.SCHEMAS), plus confirm_place;
- search_docs returns 5 passages, each with its section and URL;
- search_scenes finds a place and its scenes, and returns a table with their names;
- a wrong argument, a bad date and a place under 1 km across are errors the model can read
  (isError), and confirm_place lets the small place through;
- an unknown tool is a protocol error (-32602).

It needs the database, Ollama (bge-m3 only) and the network (Nominatim and the STAC catalogue).
"""

import sys
import time
from typing import Any

import anyio
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError

from sentinel1_rag.config import ROOT
from sentinel1_rag.mcp_server import INVALID_PARAMS
from sentinel1_rag.retrieval import TOP_K
from sentinel1_rag.tools import SCHEMAS

# Past dates, so the scenes found do not change from one run to the next.
SCENES = {"place": "Rotterdam, Netherlands", "start": "2026-09-01", "end": "2026-09-30"}
SMALL_PLACE = {"place": "Doñana, Spain", "start": "2026-09-01", "end": "2026-09-30"}
DOCS_QUERY = "What does the Interferometric Wide swath mode acquire?"


class Checks:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, ok: bool, what: str, detail: str = "") -> None:
        self.failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f": {detail}" if detail else ""), flush=True)


def text(result: Any) -> str:
    return result.content[0].text if result.content else ""


async def check_server(mode: str, checks: Checks) -> None:
    server = StdioServerParameters(command=sys.executable, args=["-m", "sentinel1_rag.mcp_server"], cwd=ROOT)
    started = time.perf_counter()
    async with Client(server, mode=mode) as client:
        print(f"\n{'Handshake (initialize)' if mode == 'legacy' else 'Per-request metadata'}: protocol {client.protocol_version}", flush=True)
        checks.check(client.server_info.name == "sentinel1-rag", "server", f"{client.server_info.name} {client.server_info.version}")
        checks.check(bool(client.instructions), "instructions sent")

        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        checks.check(list(tools) == ["search_docs", "search_scenes"], "tools", ", ".join(tools))
        for schema in SCHEMAS:
            name, agent = schema["function"]["name"], schema["function"]["parameters"]
            served = tools[name].input_schema
            same = set(served["properties"]) - {"confirm_place"} == set(agent["properties"]) and served["required"] == agent["required"]
            checks.check(same, f"{name}: the agent's arguments", ", ".join(served["properties"]))

        result = await client.call_tool("search_docs", {"query": DOCS_QUERY})
        passages = (result.structured_content or {}).get("passages", [])
        checks.check(
            not result.is_error and len(passages) == TOP_K and all(p["url"] in text(result) for p in passages),
            "search_docs: passages with section and URL",
            f"[1] {passages[0]['section']}" if passages else text(result),
        )

        if mode == "legacy":
            # The rest goes through the same handler in both eras; once is enough for the network.
            print(f"  ({time.perf_counter() - started:.1f}s)")
            return

        result = await client.call_tool("search_scenes", SCENES)
        data = result.structured_content or {}
        names = [s["id"] for s in data.get("scenes", [])]
        checks.check(
            not result.is_error and bool(names) and all(n in text(result) for n in names),
            "search_scenes: place, summary and table",
            f"{len(names)} scenes over {data['place']['name']}" if names else text(result),
        )

        result = await client.call_tool("search_scenes", SCENES | {"radius": 10})
        checks.check(result.is_error and "wrong arguments" in text(result), "unknown argument -> isError", text(result))
        result = await client.call_tool("search_scenes", SCENES | {"start": "1 September 2026"})
        checks.check(result.is_error and "YYYY-MM-DD" in text(result), "bad date -> isError", text(result))

        result = await client.call_tool("search_scenes", SMALL_PLACE)
        checks.check(result.is_error and "confirm_place" in text(result), "small place -> isError", text(result)[:120])
        result = await client.call_tool("search_scenes", SMALL_PLACE)
        checks.check(result.is_error, "the same call again -> still an error (no state across calls)")
        result = await client.call_tool("search_scenes", SMALL_PLACE | {"confirm_place": True})
        checks.check(not result.is_error, "confirm_place -> searched", (result.structured_content or {}).get("place", {}).get("name", text(result)))

        try:
            await client.call_tool("search_everything", {})
            checks.check(False, "unknown tool -> protocol error", "it answered")
        except MCPError as exc:
            checks.check(exc.code == INVALID_PARAMS, "unknown tool -> protocol error", f"{exc.code} {exc.message}")
        print(f"  ({time.perf_counter() - started:.1f}s)")


async def run() -> int:
    checks = Checks()
    for mode in ("legacy", "auto"):
        await check_server(mode, checks)
    print(f"\n{'All checks passed.' if not checks.failed else f'{checks.failed} checks failed.'}")
    return 1 if checks.failed else 0


def main() -> None:
    raise SystemExit(anyio.run(run))


if __name__ == "__main__":
    main()
