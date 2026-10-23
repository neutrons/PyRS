"""Citation extraction for the audit toolkit (A3/A7).

A plan series carries hundreds of ``path:LINE`` pointers over actively moving
code, and referent drift -- a citation correct when written, now pointing at
unrelated code -- is the dominant staleness sub-class. This module finds those
pointers. It deliberately does **not** judge them: see ``check_citations.py``.

The style inventory is empirical, taken from ``plans/NXstress-prod/``. Five of
these are not described in ``process.md`` section 7.4 and were found only by
counting the real corpus, which is the tuning round that document demands:

======================================  =========================================
Style                                   Example
======================================  =========================================
backticked, bare basename               ``_instrument.py:70``
backticked, range                       ``_peaks.py:235-239``
backticked, comma list                  ``_definitions.py:106,109,116,122,127-135``
backticked, space after comma           ``nexus_conversion.py:118, 374``
backticked, path-qualified              ``pyrs/core/workspaces.py:981``
backticked, prefix-inherited            ``:494`` after an earlier path
**slash-joined line list**              ``model.py:316/344``
**``path::Symbol``** anchor             ``_peaks.py::PeakIndex``
heading parenthetical                   ``### `_peaks.py` -- restore (L235-239)``
**two L-tokens in one paren**           ``(L425-429, L430-438)``
**L-prefix on first element only**      ``(L239, 611, 816, ... 1184)``
**en-dash range**                       ``(L212-229)`` with U+2013
**bare ``Lnn``, no parentheses**        ``at L180-181``
======================================  =========================================

Prefix inheritance is the subtle one. It is **not line-local**: the antecedent
may sit on the previous line (``README.md:444`` to ``:445``), or two lines back
through an intervening inherited citation (``open-questions/09`` lines 63-65).
So inheritance is scoped to the enclosing *paragraph* -- except inside a table,
where it is scoped to the *row*, because each row is an independent record and
paragraph scope would let one row silently borrow another's path.

And sometimes there is no antecedent at all: ``open-questions/04c:138`` emits
four bare ``:nnn`` with no filename anywhere in the paragraph. That case is
**reported as unresolved, never guessed** -- guessing is precisely how a tool
buries a real defect under a plausible-looking wrong answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import markdown

# A path token must carry a real extension, or every "ratio:2" becomes a citation.
_EXT = r"(?:py|toml|yml|yaml|lock|rst|md|txt|h5|nxs|xml|html|cfg|ini)"
_PATH = rf"[A-Za-z0-9_][A-Za-z0-9_./-]*\.{_EXT}"
# Line specs: digits, ranges (ASCII hyphen or en dash), comma lists, slash lists.
_LINESPEC = r"\d+(?:\s*[-–]\s*\d+)?(?:\s*[,/]\s*\d+(?:\s*[-–]\s*\d+)?)*"

_SPAN_CITE_RE = re.compile(rf"^(?P<path>{_PATH})(?:::(?P<symbol>[A-Za-z_][\w.]*))?:(?P<lines>{_LINESPEC})$")
_SPAN_BARE_RE = re.compile(rf"^:(?P<lines>{_LINESPEC})$")
_SPAN_ANCHOR_RE = re.compile(rf"^(?P<path>{_PATH})(?::{{1,2}}(?P<symbol>[A-Za-z_][\w.]*))?$")

# Parenthetical line lists: "(L463)", "(L425-429, L430-438)", "(L239, 611, 816)".
_PAREN_LIST = r"L?\d+(?:\s*[-\u2013]\s*L?\d+)?"
_PAREN_RE = re.compile(rf"\((?P<body>L\d+(?:\s*[-\u2013]\s*L?\d+)?(?:\s*,\s*{_PAREN_LIST})*)\)")
# Bare Lnn outside parentheses. Two digits minimum: this corpus says "Detector L2"
# as a physics term, and a one-digit rule turns that into a phantom citation.
_BARE_L_RE = re.compile(r"(?<![\w(#])L(?P<lines>\d{2,}(?:\s*[-–]\s*L?\d+)?)")
_PAREN_BODY_TOKEN_RE = re.compile(r"L?(\d+)(?:\s*[-–]\s*(\d+))?")


@dataclass
class Citation:
    """One extracted pointer, before any judgement is made about it.

    Attributes:
        doc: Document the citation appears in.
        line: 1-indexed line of the document.
        raw: The citation exactly as written.
        path: Path as written, or as inherited from an antecedent.
        lines: Every line number named, ranges expanded to (start, end) pairs.
        style: Which of the styles above produced it.
        inherited: True when ``path`` was not stated at this citation.
        symbol: The ``::Symbol`` qualifier, when one was written.
    """

    doc: Path
    line: int
    raw: str
    path: str | None
    lines: list[tuple[int, int]]
    style: str
    inherited: bool = False
    symbol: str | None = None


def _parse_linespec(spec: str) -> list[tuple[int, int]]:
    """Expand a line spec into (start, end) pairs. ``"1,5-7"`` -> ``[(1,1),(5,7)]``."""
    out: list[tuple[int, int]] = []
    for chunk in re.split(r"[,/]", spec):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*[-–]\s*(\d+))?", chunk)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        out.append((start, end))
    return out


def _parse_paren_body(body: str) -> list[tuple[int, int]]:
    """Expand a parenthetical body, where only the first element may carry ``L``."""
    out: list[tuple[int, int]] = []
    for match in _PAREN_BODY_TOKEN_RE.finditer(body):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        out.append((start, end))
    return out


def _paragraph_bounds(text: str) -> list[tuple[int, int]]:
    """Character offsets of each blank-line-delimited paragraph."""
    bounds, start = [], 0
    for match in re.finditer(r"\n[ \t]*\n", text):
        bounds.append((start, match.start()))
        start = match.end()
    bounds.append((start, len(text)))
    return bounds


def _scope_key(text: str, offset: int, paragraphs: list[tuple[int, int]]) -> tuple[int, int]:
    """The inheritance scope containing ``offset``.

    A table row is its own scope; anything else inherits within its paragraph.
    """
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    line_end = len(text) if line_end == -1 else line_end
    if text[line_start:line_end].lstrip().startswith("|"):
        return (line_start, line_end)
    for para in paragraphs:
        if para[0] <= offset <= para[1]:
            return para
    return (0, len(text))


def extract(doc: Path) -> tuple[list[Citation], list[Citation]]:
    """Extract every citation from ``doc``.

    The scan is a **single positional pass** over spans, parentheticals and bare
    ``Lnn`` tokens interleaved in document order. That ordering is load-bearing:
    an earlier implementation collected every code span first and then resolved
    parentheticals against the resulting table, which silently handed each
    parenthetical the *last* path in its paragraph rather than the last path
    *before* it. ``README.md:192`` inherited ``nexus_conversion.py`` from a
    later bullet instead of the ``fields.py`` sitting on its own line.

    Returns:
        A ``(citations, unresolved)`` pair. ``unresolved`` holds citations with
        no antecedent in scope -- reported, never guessed.
    """
    text = doc.read_text(encoding="utf-8")
    fenced = markdown.mask_fences(text)
    paragraphs = _paragraph_bounds(text)

    span_ranges = [(m.start(), m.end()) for m in markdown._SPAN_RE.finditer(fenced)]

    def _in_span(pos: int) -> bool:
        return any(a <= pos < b for a, b in span_ranges)

    paren_ranges = [(m.start(), m.end()) for m in _PAREN_RE.finditer(fenced) if not _in_span(m.start())]

    events: list[tuple[int, str, object]] = []
    for match in markdown._SPAN_RE.finditer(fenced):
        events.append((match.start(), "span", match))
    for match in _PAREN_RE.finditer(fenced):
        if not _in_span(match.start()):
            events.append((match.start(), "paren", match))
    for match in _BARE_L_RE.finditer(fenced):
        pos = match.start()
        if _in_span(pos) or any(a <= pos < b for a, b in paren_ranges):
            continue
        events.append((pos, "bare", match))
    events.sort(key=lambda e: e[0])

    found: list[Citation] = []
    unresolved: list[Citation] = []
    anchors: dict[tuple[int, int], str] = {}

    for offset, kind, match in events:
        line = markdown.line_of(text, offset)
        scope = _scope_key(text, offset, paragraphs)

        if kind == "span":
            content = re.sub(r"\s*\n\s*", " ", match.group(2).strip())

            cite = _SPAN_CITE_RE.match(content)
            if cite:
                anchors[scope] = cite.group("path")
                found.append(
                    Citation(
                        doc=doc,
                        line=line,
                        raw=content,
                        path=cite.group("path"),
                        lines=_parse_linespec(cite.group("lines")),
                        style="backticked",
                        symbol=cite.group("symbol"),
                    )
                )
                continue

            bare = _SPAN_BARE_RE.match(content)
            if bare:
                inherited = anchors.get(scope)
                citation = Citation(
                    doc=doc,
                    line=line,
                    raw=content,
                    path=inherited,
                    lines=_parse_linespec(bare.group("lines")),
                    style="prefix-inherited",
                    inherited=True,
                )
                (found if inherited else unresolved).append(citation)
                continue

            anchor = _SPAN_ANCHOR_RE.match(content)
            if anchor:
                anchors[scope] = anchor.group("path")
            continue

        if kind == "paren":
            lines = _parse_paren_body(match.group("body"))
            style, raw = "parenthetical", match.group(0)
        else:
            lines = _parse_linespec(match.group("lines"))
            style, raw = "bare-L", match.group(0)

        if not lines:
            continue
        inherited = anchors.get(scope)
        citation = Citation(
            doc=doc,
            line=line,
            raw=raw,
            path=inherited,
            lines=lines,
            style=style,
            inherited=True,
        )
        (found if inherited else unresolved).append(citation)

    return found, unresolved
