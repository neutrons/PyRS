"""A6: pair the grounded README against the superseded draft it replaced.

A pre-codebase draft that is never retired is a hazard with a long half-life:
it shares the grounded document's title and its ``## 1``-``## 6`` numbering, so a
reader who opens the wrong "section 2.3" gets a different architecture. In
``plans/NXstress-prod/`` that pair specified different config libraries,
different config-file locations, a different validation story and a different
CLI story -- discovered by comparing **one** heading pair out of roughly twenty.

This tool generates the other nineteen.

Blind spots
-----------
* **It is a shortlist generator, not an oracle.** Its whole value is producing a
  list short enough to read; the reading is still the audit.
* It cannot see a divergence that is not heading-aligned.
* It says nothing about whether a divergence was *recorded* -- the reading half
  of A6, and the half that matters.
* **A6 has weaker tooling here than the other axes.** Stated plainly rather than
  implying parity: there is no mechanical check that a divergence was justified.

Note the direction of the asymmetry. Agreement with the draft is *not* evidence
of health -- the draft is older, so agreeing with it is evidence of staleness.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import manifest as manifest_mod
import markdown

_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")


def sections(path: Path, max_level: int = 3) -> dict[str, tuple[str, int, str]]:
    """Map each numbered heading's number to ``(title, line, body)``."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    heads = [h for h in markdown.headings(text) if h.level <= max_level]
    out: dict[str, tuple[str, int, str]] = {}
    for i, head in enumerate(heads):
        match = _NUMBERED_RE.match(head.text.strip())
        if not match:
            continue
        end = heads[i + 1].line - 1 if i + 1 < len(heads) else len(lines)
        body = "\n".join(lines[head.line : end]).strip()
        out[match.group(1)] = (match.group(2).strip(), head.line, body)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series_dir")
    parser.add_argument("--format", choices=("tsv", "md"), default="tsv")
    args = parser.parse_args(argv)

    m = manifest_mod.load_or_exit(args.series_dir)
    if m.earlier_draft is None:
        print("not applicable: [series.earlier_draft] is empty for this series")
        return 0

    current = sections(m.readme)
    draft = sections(m.earlier_draft)

    rows: list[tuple[str, str, str, str]] = []
    for number in sorted(set(current) | set(draft), key=lambda n: [int(p) for p in n.split(".")]):
        cur = current.get(number)
        old = draft.get(number)
        if cur and not old:
            rows.append((number, "only-in-readme", cur[0], f"{m.rel(m.readme)}:{cur[1]}"))
        elif old and not cur:
            rows.append((number, "only-in-draft", old[0], f"{m.rel(m.earlier_draft)}:{old[1]}"))
        else:
            ratio = difflib.SequenceMatcher(None, cur[2], old[2]).ratio()
            verdict = "identical" if cur[2] == old[2] else f"diverged ({ratio:.0%} similar)"
            title = cur[0] if cur[0] == old[0] else f"{cur[0]!r} vs {old[0]!r}"
            rows.append((number, verdict, title, f"{m.rel(m.readme)}:{cur[1]} / {m.rel(m.earlier_draft)}:{old[1]}"))

    diverged = [r for r in rows if r[1] != "identical"]

    if args.format == "md":
        print(f"**{len(diverged)} of {len(rows)} numbered section(s) diverge** between the README and its draft.\n")
        print("| Section | Verdict | Title | Where |")
        print("|---|---|---|---|")
        for number, verdict, title, where in rows:
            print(f"| `{number}` | {verdict} | {title} | `{where}` |")
    else:
        for number, verdict, title, where in rows:
            print(f"{number}\t{verdict}\t{title}\t{where}")
        print(f"# {len(diverged)} of {len(rows)} sections diverge", file=sys.stderr)

    return 1 if diverged else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
