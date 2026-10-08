"""Split page sections into chunks for embedding.

- A section that fits in `max_tokens` becomes a single chunk.
- A longer section is cut into chunks of similar size, no larger than `target_tokens`.
  Cuts fall between sentences; a table or list stays whole when it fits in one chunk,
  and is otherwise cut between rows or items. Each of those chunks starts by repeating
  up to `overlap_tokens` from the end of the previous one, so that an idea on a
  boundary appears whole in some chunk.

Token counts are estimated from the number of words (see TOKENS_PER_WORD).
"""

import math
import re
from dataclasses import dataclass

from sentinel1_rag.extract import Page

# bge-m3 tokens per whitespace-separated word, measured on this corpus with Ollama's
# prompt_eval_count (prose 1.61, tables 1.86). Counting words predicts the real size
# better than counting characters here (error spread 12% vs 16%).
TOKENS_PER_WORD = 1.68

# A sentence ends with . ! or ? followed by a space and an upper-case letter, digit or bracket.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class ChunkingSettings:
    max_tokens: int = 500
    target_tokens: int = 400
    overlap_tokens: int = 60
    version: int = 1  # bump when the algorithm changes, so that every page is re-indexed


SETTINGS = ChunkingSettings()


@dataclass
class Chunk:
    chunk_index: int  # position within the page
    section: str  # heading path: "Page title > Section > Subsection"
    anchor: str | None
    content: str
    tokens: int  # estimated


@dataclass
class _Unit:
    """Smallest piece of text a chunk boundary may fall between."""

    text: str
    sep: str  # what goes before it when it follows another unit
    tokens: int


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * TOKENS_PER_WORD)


def chunk_page(page: Page, settings: ChunkingSettings = SETTINGS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in page.sections:
        for content in split_section(section.blocks, settings):
            chunks.append(
                Chunk(len(chunks), " > ".join(section.path), section.anchor, content,
                      estimate_tokens(content))
            )
    return chunks


def split_section(blocks: list[str], settings: ChunkingSettings = SETTINGS) -> list[str]:
    units = _units(blocks, settings.target_tokens)
    total = _size(units)
    if total <= settings.max_tokens:
        return [_render(units)]

    # Aim for chunks of equal size instead of full chunks plus a small remainder.
    goal = math.ceil(total / math.ceil(total / settings.target_tokens))
    chunks: list[str] = []
    current: list[_Unit] = []
    has_new = False  # does `current` hold anything beyond the overlap?
    for unit in units:
        if has_new and _size(current) + unit.tokens > goal:
            chunks.append(_render(current))
            current, has_new = _tail(current, settings.overlap_tokens), False
            if _size(current) + unit.tokens > goal:
                current = []  # no room for the overlap before this unit
        current.append(unit)
        has_new = True
    chunks.append(_render(current))
    return chunks


def _units(blocks: list[str], limit: int) -> list[_Unit]:
    units = []
    for block in blocks:
        if "\n" not in block:  # prose: cut between sentences
            pieces, sep = SENTENCE_BREAK.split(block), " "
        elif estimate_tokens(block) <= limit:  # table or list that fits: keep it whole
            pieces, sep = [block], "\n"
        else:  # table or list too long for one chunk: cut between rows or items
            pieces, sep = block.split("\n"), "\n"
        for i, piece in enumerate(pieces):
            for j, part in enumerate(_split_words(piece, limit)):
                if i == 0 and j == 0:
                    part_sep = "\n\n"  # a new block starts a new paragraph
                else:
                    part_sep = sep if j == 0 else " "
                units.append(_Unit(part, part_sep, estimate_tokens(part)))
    return units


def _split_words(text: str, limit: int) -> list[str]:
    """Cut a piece longer than `limit` tokens into word groups (rare: very long sentences)."""
    words = text.split()
    per_part = max(1, int(limit / TOKENS_PER_WORD))
    if len(words) <= per_part:
        return [text]
    return [" ".join(words[i : i + per_part]) for i in range(0, len(words), per_part)]


def _tail(units: list[_Unit], budget: int) -> list[_Unit]:
    """The last units whose tokens add up to at most `budget`."""
    tail: list[_Unit] = []
    for unit in reversed(units):
        if _size(tail) + unit.tokens > budget:
            break
        tail.insert(0, unit)
    return tail


def _size(units: list[_Unit]) -> int:
    return sum(u.tokens for u in units)


def _render(units: list[_Unit]) -> str:
    return units[0].text + "".join(u.sep + u.text for u in units[1:])
