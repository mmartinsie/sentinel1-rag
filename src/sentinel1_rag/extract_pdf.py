"""Turn a technical PDF from the Sentinel-1 document library into a list of sections.

A PDF only places characters on pages, so the structure has to be rebuilt (D-025):

- Running headers, footers and page numbers are removed: lines in the top or bottom margin
  that repeat on at least half of the pages, once their digits are ignored.
- A section starts at a numbered heading, such as "3.1 Calibration LUT Description", or
  "A1.4 Data Size and Volume" in an appendix. A heading is a line that starts with a section
  number, is set in a larger font than the body text, and continues the numbering (3.1 can
  follow 3 or 2.4, not 3.4). Everything before section 1 is left out: the title page, the
  change log and the table of contents.
- A section's anchor is "page=N", the physical page of its heading, so that the citation URL
  (url#page=N, RFC 8118) opens the PDF there.
- Lines are joined into paragraphs, which end where the gap between two lines is larger than
  inside a paragraph. Bulleted lines become list items ("- item"), and the tables that
  pdfplumber finds become one line per row ("cell | cell"), as in extract.py. A cell merged
  across several rows is repeated in each of them, and an empty cell is kept, as an empty
  field, in the columns that the first row names, so that every row lines up with the header.
- A word split by a hyphen at the end of a line ("Correc-" + "tions") is joined without the
  hyphen when the whole word appears elsewhere in the document, and keeps it otherwise
  ("cross-section"): most of these documents only break lines at real hyphens.
- Lines without a single word are dropped: the labels and axis numbers of charts, subscripts
  set on a line of their own, and what is left of equations. Equations lose their symbols,
  because their fonts do not say which character each glyph is ("(cid:1830)").

The result has the same shape as a SentiWiki page, so chunking and indexing do not change.
The document's title comes from sources.yaml.
"""

import io
import re
from collections import Counter
from dataclasses import dataclass

import pdfplumber

from sentinel1_rag.extract import Page, Section, clean

# "3", "3.1", "3.1.2." or, in an appendix, "B", "B1", "A1.4", then the heading text.
HEADING = re.compile(
    r"^(?P<number>\d{1,2}(?:\.\d{1,2}){0,4}|[A-H](?:\d{1,2}(?:\.\d{1,2}){0,3})?)\.?\s+(?P<text>\S.*)$"
)
DOT_LEADER = re.compile(r"\.{4,}|…{2,}")  # table of contents: "3.1 Purpose ........ 1-1"
# Bullet glyphs, including those of symbol fonts (Unicode private use area, such as U+F0B7);
# "o" and "-" only before a capital letter, so that prose is not taken for a list.
BULLET = re.compile(r"^(?:[•▪◦●■\ue000-\uf8ff]|[o\-–](?=\s+[A-Z]))\s*")
# A glyph whose font has no mapping to a character, as in most equations here.
UNMAPPED_GLYPH = re.compile(r"\(cid:\d+\)")
WORD = re.compile(r"[A-Za-z]{3,}")
APPENDIX = 100  # appendix A is numbered after any chapter: A -> 100, B -> 101, ...

# Share of the page height, at the top and at the bottom, where running headers and footers are.
MARGIN = 0.15
# A margin line repeated on at least this share of the pages is a header or a footer.
REPEATED = 0.5
# A gap between two lines larger than this share of the line height starts a new paragraph.
# Inside a paragraph the gap is about 0.2 of the line height in these documents, between
# paragraphs about 0.8.
PARAGRAPH_GAP = 0.5
# Headings are set at least this many points larger than the body text.
HEADING_SIZE = 0.5


@dataclass
class _Line:
    page: int  # physical page, from 1
    text: str
    top: float
    bottom: float
    x0: float
    size: float  # largest font size on the line, in points
    cells: list[list[str | None]] | None = None  # set for a whole table: its rows of cells

    @property
    def table(self) -> bool:
        return self.cells is not None


def extract_pdf(data: bytes, url: str, title: str) -> Page:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [_read_page(number, page) for number, page in enumerate(pdf.pages, start=1)]
        heights = [page.height for page in pdf.pages]

    repeated = _repeated_margin_lines(pages, heights)
    # The body font is the size that most characters are set in, to the nearest half point.
    sizes: Counter[float] = Counter()
    for line in (line for lines in pages for line in lines if not line.table):
        sizes[round(line.size * 2) / 2] += len(line.text)
    texts = [line.text for lines in pages for line in lines] + [
        cell for lines in pages for line in lines if line.cells for row in line.cells for cell in row if cell
    ]
    vocabulary = {word.lower() for text in texts for word in re.findall(r"[A-Za-z]+", text)}
    walker = _Walker(title, sizes.most_common(1)[0][0], vocabulary)
    for lines, height in zip(pages, heights, strict=True):
        for line in lines:
            if not line.table and (_in_margin(line, height) and _key(line.text) in repeated
                                   or _is_debris(line.text)):
                continue
            walker.add(line)
    walker.flush()
    sections = [s for s in walker.sections if s.blocks]
    if not sections:
        raise ValueError(f"no numbered sections found: {url}")
    return Page(url, title, sections)


def _read_page(number: int, page: pdfplumber.page.Page) -> list[_Line]:
    """The page's text lines and tables, top to bottom. Table text is not repeated in the lines.
    Rotated text is left out: chart axes, margin notes and the odd vertical table header."""
    page = page.filter(lambda obj: obj.get("object_type") != "char" or obj["upright"])
    tables = page.find_tables()
    boxes = [table.bbox for table in tables]

    def outside_tables(obj: dict) -> bool:
        if obj.get("object_type") != "char":
            return True
        x, y = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= x <= x1 and top <= y <= bottom for x0, top, x1, bottom in boxes)

    lines = [
        _Line(number, text, line["top"], line["bottom"], line["x0"], max(char["size"] for char in line["chars"]))
        for line in page.filter(outside_tables).extract_text_lines()
        if (text := UNMAPPED_GLYPH.sub("", line["text"]).strip())
    ]
    for table, (x0, top, _, bottom) in zip(tables, boxes, strict=True):
        lines.append(_Line(number, "", top, bottom, x0, 0.0, _table_cells(table)))
    return sorted(lines, key=lambda line: line.top)


def _table_cells(table: pdfplumber.table.Table) -> list[list[str | None]]:
    """The table's cells row by row; None where a cell has nothing of its own to show.

    pdfplumber gives None for the positions that a merged cell covers. A cell merged down
    several rows is repeated in each ("SM" in front of every SM row), so that each row reads
    on its own; one merged across columns appears once. An empty cell stays as "" in a column
    that the first row names, and is dropped (None) elsewhere.
    """
    text = [[UNMAPPED_GLYPH.sub("", cell) if cell is not None else None for cell in row] for row in table.extract()]
    header = text[0]
    rows: list[list[str | None]] = []
    for r, row in enumerate(table.rows):
        y = (row.bbox[1] + row.bbox[3]) / 2
        cells: list[str | None] = []
        for c, bbox in enumerate(row.cells):
            if bbox is not None:
                cells.append(text[r][c] if text[r][c] or (r and header[c]) else None)
                continue
            # The nearest cell above in this column: repeated if it reaches down into this row.
            above = next(((up, b) for up in range(r - 1, -1, -1) if (b := table.rows[up].cells[c])), None)
            cells.append(text[above[0]][c] if above and above[1][1] < y < above[1][3] else None)
        rows.append(cells)
    return rows


def _is_debris(text: str) -> bool:
    """A line with no word of three letters or more, or mostly single characters, as chart
    labels are ("N e a r R a n g e", "5.4]m[5.2 noituloseR S 1 5.0 S 2")."""
    tokens = text.split()
    single = sum(len(token) == 1 for token in tokens)
    return not WORD.search(text) or (len(tokens) >= 6 and single >= len(tokens) / 2)


def _join(lines: list[str], vocabulary: set[str]) -> str:
    """Join the lines of a paragraph or a table cell. After a line ending in "word-", the next
    line follows without a space, and the hyphen goes only if the joined word is in `vocabulary`."""
    text = ""
    for line in (line.strip() for line in lines):
        head = re.search(r"([A-Za-z]+)-$", text)
        if head and line[:1].isalnum():  # "S-" + "1 IPF" -> "S-1 IPF"
            tail = re.match(r"[A-Za-z]+", line)
            text = (text[:-1] if tail and (head[1] + tail[0]).lower() in vocabulary else text) + line
        else:
            text += (" " if text and line else "") + line
    return text


def _table(cells: list[list[str | None]], vocabulary: set[str]) -> str:
    """One line per row, its cells joined with " | "; rows without any text are dropped."""
    rows = [[clean(_join(cell.splitlines(), vocabulary)) for cell in row if cell is not None] for row in cells]
    return "\n".join(" | ".join(row) for row in rows if any(row))


def _key(text: str) -> str:
    """A margin line with its numbers ignored: "Page 4/13" and "Page 5/13" are the same footer."""
    return re.sub(r"\d+", "#", " ".join(text.split()))


def _in_margin(line: _Line, height: float) -> bool:
    return line.top < MARGIN * height or line.bottom > (1 - MARGIN) * height


def _repeated_margin_lines(pages: list[list[_Line]], heights: list[float]) -> set[str]:
    counts = Counter(
        key
        for lines, height in zip(pages, heights, strict=True)
        for key in {_key(line.text) for line in lines if not line.table and _in_margin(line, height)}
    )
    return {key for key, n in counts.items() if n >= max(3, REPEATED * len(pages))}


def _number(text: str) -> tuple[int, ...]:
    """ "3.1" -> (3, 1); "A1.4" -> (100, 1, 4): appendices come after the chapters."""
    if text[0].isalpha():
        letter, rest = text[0], text[1:]
        return (APPENDIX + ord(letter) - ord("A"), *(int(n) for n in rest.split(".") if n))
    return tuple(int(n) for n in text.split("."))


def _follows(previous: tuple[int, ...] | None, number: tuple[int, ...]) -> bool:
    """Whether `number` can come right after `previous`: its first subsection, the next section
    at the same or a higher level, or the first appendix."""
    if previous is None:
        return number == (1,)
    if number == (*previous, 1):
        return True
    if number == (APPENDIX,) and previous[0] < APPENDIX:
        return True
    return any(number == (*previous[:level], previous[level] + 1) for level in range(len(previous)))


class _Walker:
    """Reads the lines in order, opening a Section at each heading and building its blocks."""

    def __init__(self, title: str, body_size: float, vocabulary: set[str]) -> None:
        self.title = title
        self.body_size = body_size
        self.vocabulary = vocabulary  # every word in the document, lower case
        self.headings: list[tuple[tuple[int, ...], str]] = []  # open headings: (number, text)
        self.sections: list[Section] = []  # nothing is kept before the first heading
        self.previous: _Line | None = None  # previous text line of the current block
        self.heading_line: _Line | None = None  # set right after a heading, which may wrap
        self.paragraph: list[str] = []
        self.items: list[tuple[int, list[str]]] = []  # list in progress: (depth, lines)
        self.bullet_x: list[float] = []  # x positions of the list's bullets, one per depth

    def add(self, line: _Line) -> None:
        if line.cells is not None:
            self.flush()
            self.heading_line = None
            self.block(_table(line.cells, self.vocabulary))
            return
        text = line.text.strip()
        heading = self.heading(line, text)
        if heading:
            self.open_section(line, *heading)
        elif self.continues_heading(line):
            number, words = self.headings[-1]
            self.headings[-1] = (number, f"{words} {text}")
            self.sections[-1].path[-1] = self.headings[-1][1]
            self.previous = line
        else:
            self.heading_line = None
            self.add_text(line, text)

    def heading(self, line: _Line, text: str) -> tuple[tuple[int, ...], str] | None:
        match = HEADING.match(text)
        if not match or not self.is_heading_font(line) or DOT_LEADER.search(text):
            return None
        number = _number(match["number"])
        last = self.headings[-1][0] if self.headings else None
        if not _follows(last, number):
            return None
        return number, f"{match['number']} {match['text']}"

    def is_heading_font(self, line: _Line) -> bool:
        return line.size >= self.body_size + HEADING_SIZE

    def continues_heading(self, line: _Line) -> bool:
        """A heading that wraps onto a second line, in the same font and close below it."""
        heading = self.heading_line
        return (
            heading is not None
            and abs(line.size - heading.size) <= 0.3
            and not self.new_paragraph(line, PARAGRAPH_GAP)
        )

    def open_section(self, line: _Line, number: tuple[int, ...], text: str) -> None:
        self.flush()
        while self.headings and len(self.headings[-1][0]) >= len(number):
            self.headings.pop()
        self.headings.append((number, text))
        path = [self.title, *(words for _, words in self.headings)]
        self.sections.append(Section(path, f"page={line.page}"))
        self.previous = self.heading_line = line

    def add_text(self, line: _Line, text: str) -> None:
        if not self.sections:
            return  # front matter, before section 1
        bullet = BULLET.match(text)
        new_paragraph = self.new_paragraph(line, PARAGRAPH_GAP)
        if bullet:
            if not self.items or self.new_paragraph(line, 1.5):
                self.flush()
            self.items.append((self.depth(line.x0), [text[bullet.end():]]))
        elif self.items and not new_paragraph and line.x0 > self.bullet_x[self.items[-1][0]] + 2:
            self.items[-1][1].append(text)  # an item that wraps onto the next line
        elif self.paragraph and not new_paragraph:
            self.paragraph.append(text)
        else:
            self.flush()
            self.paragraph = [text]
        self.previous = line

    def new_paragraph(self, line: _Line, gap: float) -> bool:
        """Whether `line` starts a new paragraph after the previous line. Across a page break,
        a paragraph continues only if it had not ended a sentence and the new line is lower case."""
        previous = self.previous
        if previous is None:
            return True
        if line.page != previous.page:
            return bool(re.search(r"[.:;!?]$", previous.text.strip())) or not line.text.strip()[:1].islower()
        return line.top - previous.bottom > gap * (previous.bottom - previous.top)

    def depth(self, x0: float) -> int:
        """List depth from the bullet's position: each new indentation is one level deeper."""
        if not self.items:
            self.bullet_x = [x0]
        for depth, x in enumerate(self.bullet_x):
            if abs(x0 - x) <= 2:
                return depth
        self.bullet_x.append(x0)
        self.bullet_x.sort()
        return self.bullet_x.index(x0)

    def flush(self) -> None:
        if self.paragraph:
            self.block(clean(_join(self.paragraph, self.vocabulary)))
        if self.items:
            self.block("\n".join(
                "  " * depth + "- " + clean(_join(lines, self.vocabulary)) for depth, lines in self.items
            ))
        self.paragraph, self.items = [], []

    def block(self, text: str) -> None:
        if text and self.sections:
            self.sections[-1].blocks.append(text)
            self.previous = None
