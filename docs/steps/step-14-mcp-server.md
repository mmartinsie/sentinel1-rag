# Step 14: An MCP server for the documentation and scene searches

*Built and verified on 2026-10-11 and 2026-10-12, approved by the user on 2026-10-12, and tested with Claude Code as the host on 2026-10-14. The next extension in the README, chosen by the user.*

## What this step was about

In step 13 the agent's tools lived inside our own program: Gemma asked for a call, and our loop ran it. MCP, the Model Context Protocol, takes those tools out of the program, so that any application with a language model can use them. That application is called the host: Claude Code, Claude Desktop or an IDE. The host starts our server as a separate program and talks to it in JSON-RPC, a simple format for requests and answers. It asks `tools/list` to learn what tools exist, with the same kind of JSON Schema that Gemma received, and `tools/call` to run one. The host's own model decides when to call them. As before, the model never runs anything: the host calls, and our code runs.

The transport here is stdio. The host launches the server as a subprocess and writes one JSON message per line to its standard input; the server answers on its standard output. That makes standard output special: anything else written there, even a stray debug line, breaks the host's reading of the protocol.

One fact changed the design. The MCP specification had a new revision on 28 July 2026, and it removed sessions. Before, a client opened a connection with an `initialize` handshake and then talked within that session. Now every request carries its own protocol version and capabilities, and the specification says that a connection is not a conversation: a host may send calls from different chats through the same server process. So the server must not remember anything from one call to the next.

## What we built

`make mcp` runs the server, built on the official Python SDK for MCP. We used its low-level `Server`, which accepts our JSON Schemas as they are. The SDK's higher-level option derives the schemas from Python type hints, which would have meant describing each tool a second time. The SDK serves both kinds of client on the same process: older ones that open with the handshake, and newer ones that send their metadata with every request. We checked this on the wire before writing any code, with a tiny throwaway server.

The tools are the agent's, with the same arguments, checks and error messages, but three things had to change for a host and a model we do not control:

- **Nothing is remembered between calls.** In the agent, passage numbers ran across the whole conversation, and the code printed the sources under the answer. A server cannot print anything in someone else's application, so each result numbers its passages from 1 and gives each one its section and URL, for the host's model to cite. In the agent, a place under 1 km across was rejected once and accepted if asked for again; the server cannot know that a call is the second one, so the call has to say so with a new argument, `confirm_place`.
- **The host's model writes the query in English.** Our documents are in English, and the keyword half of the search only matches English words. In `make ask`, Gemma translates the question first. A host already has a capable model, so the server asks for an English query and needs no language model of its own: only the embedding model and the database. We measured this before choosing it. With Gemma's translations standing in for the host's, the search finds the same 22 of 25 eval questions in its top 5, directly and through the protocol.
- **Results come in two forms.** Each result has text for the host's model and structured data, checked against a declared schema, so that a client can show the sources or the scenes from data rather than from the model's words. The scene names are now in the text as well: they were hidden from Gemma only because it copied them slowly.

A tool error that the model can fix, like a bad date, comes back as a result marked as an error, so the model can read it and try again. An unknown tool is a protocol error instead. The scene search moved into a small `SceneFinder` class that the agent and the server share. Before and after that change, 18 calls through the agent's tools returned exactly the same text.

`make mcp-check` starts the server the way a host would and checks it through the SDK's client, with both kinds of client: 19 checks, all passing.

## What we found

**A stray print can still reach the protocol.** The SDK protects standard output: while it serves, it moves the process's output channel to the error stream and keeps the protocol on a private copy. In the probe, a `print()` inside a tool still reached the client. Python had kept the text in a buffer, and wrote it out when the server exited, after the SDK had moved the channel back. One line fixes it: once the transport is open, Python's `sys.stdout` points at the error stream, so a print goes there at once.

**v0.2 has the same retrieval as v0.1.** The user tagged the end of step 12 as v0.1, because a public post links to it. v0.2 is the agent plus this server. Both use the same search function, and `make eval` on v0.2 returned the same 20 chunks, in the same order, for all 25 questions. The README says so in one sentence instead of repeating the table.

**The agent eval passed an empty answer.** Rerunning the agent's eval for v0.2, the checks counted 14 of 18 instead of 13. The extra pass was an answer that no longer cited "[1]" after facts about scenes, which looked like an improvement. Reading it showed that it was empty: Gemma now ends its turn without writing anything for "¿Hay imágenes Sentinel-1 GRD de Madrid de la última semana?". The citation check passes an answer that cites nothing, and an empty answer cites nothing. It happens every time now, also with the code from before this step, so this step did not cause it; none of the 72 answers in earlier agent runs was empty. At the review, the user chose to fix the measurement: the eval now also checks that there is an answer, which gives 13 of 18 again, and `make agent` says when the answer is empty instead of printing a blank line. Asking the model again after an empty answer would change the agent, so it is left for a later step. As in step 13, the lesson is to read the answers, not only the scores.

**It works in a real host.** In a new Claude Code session, after the user approved the server, Claude used the tools for four questions, in 7 calls. Two things stood out. First, a host's model can search more than once: for the difference between GRD and SLC products it searched twice, and the second, more specific query found the section about the phase information in SLC products, which the single search of `make ask` misses. Second, the error for a tiny place did its job: for "Doñana" the model searched again with the national park's full name. The connection itself took a detour. The server was slow to start the first time after a reboot, Claude Code's first probe timed out, and it fell back to the old handshake, which the server refused because the probe had already set it to the new protocol. A second probe connected. Claude knew how the server was designed, so this test shows that the server works with a real application, not how an unknown model would use it.

## What it means for the next steps

The tools can now be used by any MCP host on this machine, and the agent keeps working as before. Still pending: the v0.2 tag, created only with the user's OK. The agent's loop could ask the model again after an empty answer. OpenSearch is the last planned extension, and the reranker is still among the smaller experiments.

## Check your understanding

**What is the difference between MCP and the tool calling of step 13?**
Tool calling is the mechanism inside one program: the model receives JSON Schemas, asks for a call, and the program runs it and returns the result. MCP is a protocol between programs: a server offers tools, with the same kind of schemas, and any host can discover them with `tools/list` and run them with `tools/call` over JSON-RPC, here through the server's standard input and output. The host's model still only asks; the host sends the call, and the server's code runs it. Writing a tool once as an MCP server makes it usable from every host that speaks the protocol, instead of being written again for each application.

**Why can the MCP server not keep state between calls, and what did that change?**
Revision 2026-07-28 of the specification removed sessions: every request carries its own version and capabilities, and a server must not treat a connection as a conversation, because a host can send calls from different chats through the same process. State that the agent kept in memory had to move into the call or into the result. Passage numbers now start at 1 in every result, and each passage carries its section and URL. Accepting a small place on the second request became an explicit argument, `confirm_place`.
