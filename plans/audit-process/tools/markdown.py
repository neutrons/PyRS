"""Shared Markdown tokenizing for the audit toolkit.

Every checker in this toolkit has to answer the same two questions before it can
answer its own: *is this text actually prose, or is it inside a code block?* and
*where does this construct really end?* Getting either wrong is how a tool
"whose first output is wrong in a plausible-looking way" gets built, so the
logic lives here once rather than three times.

Two properties earned their place the hard way, both observed in
``plans/NXstress-prod/``:

* **Masking preserves offsets and line count.** Code spans and fences are
  replaced character-for-character with a filler, never deleted. Deleting them
  would shift every column and line number the tools report, and -- worse --
  would destroy the two live links whose *link text* is itself a code span
  (``README.md:6`` and ``README.md:84``): ``[`pyrs/...`](../../pyrs/...)``.
  Filler keeps the surrounding brackets exactly where they were.

* **Code spans wrap across lines.** Eleven do in this series, and one real
  citation (``09-fit-spectrum-nxstress.md:152``) is only visible once the lines
  are joined, while two others put ``arr[cur:] = ...`` in a span that a
  line-at-a-time tokenizer would mistake for a citation. So spans are matched
  over the whole document, not per line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FILLER = "\x00"

_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})", re.MULTILINE)
# Longest-run-first so ``double`` spans are consumed before single ones.
_SPAN_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
_EXPLICIT_ANCHOR_RE = re.compile(r"\{#([A-Za-z0-9_-]+)\}\s*$")
# GitHub DOES honour an inline HTML anchor, which is the portable way to give a
# heading a stable id -- unlike `{#id}`, which it renders as literal text.
_HTML_ANCHOR_RE = re.compile(r"""<a\s+(?:id|name)\s*=\s*["']([A-Za-z0-9_-]+)["']""", re.IGNORECASE)


@dataclass(frozen=True)
class Heading:
    """One ATX heading, with the GitHub anchor slug it generates.

    Attributes:
        explicit_anchor: The id declared with a trailing ``{#id}``, if any.
            GitHub-Flavored Markdown does **not** implement that syntax -- it
            renders the braces as literal heading text and folds them into the
            generated slug -- so an explicit anchor is recorded separately and
            never treated as a working target. Reporting it as a plain dead
            anchor would hide the actual cause.
    """

    level: int
    text: str
    line: int
    slug: str
    explicit_anchor: str | None = None


def mask_fences(text: str) -> str:
    """Replace fenced-code content with filler, preserving offsets and newlines.

    The fence markers themselves are masked too, so a fence line can never be
    mistaken for a heading or a citation.
    """
    out = list(text)
    in_fence = False
    fence_marker = ""
    for match in _FENCE_RE.finditer(text):
        marker = match.group(2)
        if not in_fence:
            in_fence, fence_marker, start = True, marker[0], match.start()
            fence_start = start
        elif marker[0] == fence_marker:
            in_fence = False
            line_end = text.find("\n", match.start())
            line_end = len(text) if line_end == -1 else line_end
            for i in range(fence_start, line_end):
                if out[i] != "\n":
                    out[i] = FILLER
    if in_fence:
        for i in range(fence_start, len(text)):
            if out[i] != "\n":
                out[i] = FILLER
    return "".join(out)


def mask_code_spans(text: str) -> str:
    """Replace inline-code content with filler, preserving offsets and newlines.

    Applied *after* :func:`mask_fences`. The brackets of a surrounding Markdown
    link are outside the span and therefore survive untouched -- which is the
    only reason a link whose text is a code span still resolves.
    """
    out = list(text)
    for match in _SPAN_RE.finditer(text):
        for i in range(match.start(), match.end()):
            if out[i] != "\n":
                out[i] = FILLER
    return "".join(out)


def mask_all(text: str) -> str:
    """Mask fences then inline spans. The order matters; see module docstring."""
    return mask_code_spans(mask_fences(text))


def line_of(text: str, offset: int) -> int:
    """1-indexed line number containing ``offset``."""
    return text.count("\n", 0, offset) + 1


def slugify(heading_text: str) -> str:
    """Render a heading's GitHub anchor slug.

    Markdown emphasis and code markers are stripped first, then everything that
    is not alphanumeric, a space, a hyphen or an underscore is removed, the
    result lowercased, and spaces folded to hyphens.
    """
    text = re.sub(r"[`*_~]", "", heading_text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return text.strip().lower().replace(" ", "-")


def html_anchors(text: str) -> set[str]:
    """Every id declared by an inline ``<a id="...">`` outside fenced code.

    These are real, linkable targets on GitHub, so a checker that only collects
    heading slugs reports them as dead.
    """
    return {m.group(1).lower() for m in _HTML_ANCHOR_RE.finditer(mask_fences(text))}


def headings(text: str) -> list[Heading]:
    """Every ATX heading outside fenced code, in document order."""
    masked = mask_fences(text)
    found = []
    for match in _HEADING_RE.finditer(masked):
        raw = text[match.start(2) : match.end(2)]
        explicit = _EXPLICIT_ANCHOR_RE.search(raw)
        found.append(
            Heading(
                level=len(match.group(1)),
                text=raw,
                line=line_of(text, match.start()),
                slug=slugify(raw),
                explicit_anchor=explicit.group(1) if explicit else None,
            )
        )
    return found
