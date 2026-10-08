"""Turn a SentiWiki page into its title and a list of sections.

A section is the text under one heading (h2-h6) together with its heading path and
the heading id, which becomes the citation anchor (page URL + "#" + id). Text before
the first heading forms a section of its own, without an anchor.

Text is kept in blocks: one per paragraph, list or table. Tables become one line per
row ("cell | cell | ..."), lists one line per item, and figures keep only their caption.
"""

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

HEADING_LEVELS = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# Elements without useful text: the headings' "copy link" buttons, icons and images.
NOISE = ["script", "style", "noscript", "svg", "button", "img", "copy-clipboard", "i18n-message"]
BLOCK_TAGS = [
    *HEADING_LEVELS, "p", "div", "section", "table", "ul", "ol", "li",
    "figure", "figcaption", "pre", "blockquote", "dl",
]
TEXT_BLOCKS = {"p", "pre", "blockquote", "figcaption", "dl"}
# "[1]", "[2, 3]", and "[ 2 ]" when the number is a link: [<a href="#Ref_2">2</a>].
REFERENCE_MARKER = re.compile(r"\[\s*\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\s*\]")


@dataclass
class Section:
    path: list[str]  # heading path, starting with the page title
    anchor: str | None  # id of the heading; None for the text before the first heading
    blocks: list[str] = field(default_factory=list)


@dataclass
class Page:
    url: str
    title: str
    sections: list[Section]

    @property
    def text(self) -> str:
        """All the extracted text, headings included; used to detect page changes."""
        return "\n".join(line for s in self.sections for line in [" > ".join(s.path), *s.blocks])


def clean(text: str) -> str:
    """Normalise a piece of text.

    Collapses whitespace, drops the spaces that inline tags leave around punctuation,
    and removes the wiki's numeric reference markers ("[1]", "[2, 3]"), which would
    clash with the [n] citations of our answers.
    """
    text = REFERENCE_MARKER.sub("", text)
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    return re.sub(r"([(\[])\s+", r"\1", text)


def extract(html: str, url: str) -> Page:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article#content")
    body = article.select_one("section.article-body") if article else None
    if body is None or article.h1 is None:
        raise ValueError(f"unexpected page layout: {url}")
    for tag in body.find_all(NOISE):
        tag.decompose()
    for br in body.find_all("br"):
        br.replace_with(" ")

    walker = _Walker(clean(article.h1.get_text(" ")))
    walker.visit(body)
    return Page(url, walker.title, [s for s in walker.sections if s.blocks])


class _Walker:
    """Walks the page body in document order, filling one Section per heading."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.headings: list[tuple[int, str]] = []  # currently open headings: (level, text)
        self.sections = [Section([title], None)]

    def visit(self, node: Tag) -> None:
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                self.add(clean(child))
            elif child.name in HEADING_LEVELS:
                self.open_section(child)
            elif child.name == "table":
                self.add_table(child)
            elif child.name in ("ul", "ol"):
                self.add("\n".join(self.list_lines(child)))
            elif child.name in TEXT_BLOCKS or "expand-control" in child.get("class", []):
                self.add(clean(child.get_text(" ")))
            elif child.find(BLOCK_TAGS) is None:  # container holding inline content only
                self.add(clean(child.get_text(" ")))
            else:
                self.visit(child)

    def open_section(self, heading: Tag) -> None:
        text = clean(heading.get_text(" "))
        if not text:
            return
        level = HEADING_LEVELS[heading.name]
        while self.headings and self.headings[-1][0] >= level:
            self.headings.pop()
        self.headings.append((level, text))
        path = [self.title, *(t for _, t in self.headings)]
        self.sections.append(Section(path, heading.get("id")))

    def add(self, text: str) -> None:
        if text:
            self.sections[-1].blocks.append(text)

    def add_table(self, table: Tag) -> None:
        rows = []
        for tr in table.find_all("tr"):
            cells = [clean(cell.get_text(" ")) for cell in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(" | ".join(c for c in cells if c))
        self.add("\n".join(rows))

    def list_lines(self, lst: Tag, depth: int = 0) -> list[str]:
        lines = []
        for item in lst.find_all("li", recursive=False):
            nested = item.find_all(["ul", "ol"], recursive=False)
            for sub in nested:
                sub.extract()
            text = clean(item.get_text(" "))
            if text:
                lines.append("  " * depth + "- " + text)
            for sub in nested:
                lines.extend(self.list_lines(sub, depth + 1))
        return lines
