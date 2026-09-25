"""A3/A7: resolve every ``path:LINE`` citation and show what is there now.

**This is a location report, not a verdict**, and the distinction is the whole
point. In ``plans/NXstress-prod/`` there is a worked pair: four citations in one
README table cell had drifted by two to twenty lines while *every prose claim
around them remained exactly true*, and one heading named a file that does not
exist at all. Conflating those buries real defects or manufactures false ones.
So this tool prints where a citation lands and what that line now says, and a
human decides whether the claim survived.

Output columns: ``doc:line | cited path:LINE | current text of that line``.

Resolution
----------
A path written with directories resolves from the repo root, or by longest
matching suffix under ``code.roots`` for a partial path like
``manual_reduction/pyrs_api.py``. A bare basename is searched under
``code.roots``.

**An ambiguous bare basename is a finding, not noise.** It is reported and never
guessed. Note that uniqueness here is accidental rather than structural:
``model.py`` resolves to exactly one file today *only because*
``pyrs/interface/texture_fitting/model.py`` does not exist -- which is itself one
of the defects this series carries, at ``README.md:738``. Add that file and two
different subspecs start citing the same basename for different code.

Blind spots
-----------
* **Reports locations, not claims.** A drifted line number does not imply a
  stale claim, and an accurate line number does not imply a true one.
* Cannot see a *missing* citation -- an assertion that cites nothing.
* Cannot see a citation "corrected" only in a Follow-up while the body still
  cites the old number. That is exactly why a ``path:LINE`` pointer is fixed in
  place rather than only logged.
* A citation naming a sibling plan document (``README.md:729``) is resolved
  against the series directory and its line number checked there. That is how
  an audit's own Follow-up citations get checked -- see section 7.7.
* Markdown ``#Lnn`` links are deliberately **not** handled here;
  ``check_links.py`` owns those, and the two tools must not double-report.
* Inheritance is scoped to a paragraph (a row, inside a table). A long bullet
  list with no blank lines is one paragraph, so an inherited path is always
  marked ``inherited`` for the reader to check rather than trusted silently.

Section 7.7, the self-check property: this tool reads ``## Follow-up N``
sections too, so an audit that cites line numbers has its own pointers checked
on the next run -- an audit generates exactly the artifact it exists to catch.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import citations as citations_mod
import manifest as manifest_mod

OK = "ok"
NOT_FOUND = "path-not-found"
AMBIGUOUS = "ambiguous-basename"
OUT_OF_RANGE = "line-out-of-range"
UNRESOLVED = "unresolved-prefix"
OUT_OF_SCOPE = "out-of-scope"

PROBLEM_STATUSES = frozenset({NOT_FOUND, AMBIGUOUS, OUT_OF_RANGE, UNRESOLVED, OUT_OF_SCOPE})


@dataclass
class Row:
    doc: Path
    doc_line: int
    raw: str
    status: str
    resolved: Path | None
    cited_line: int
    text: str
    inherited: bool


@lru_cache(maxsize=1)
def _index(repo_root: Path, roots: tuple[str, ...]) -> dict[str, list[Path]]:
    """Map every basename under ``code.roots`` to the files carrying it."""
    index: dict[str, list[Path]] = {}
    for root in roots:
        base = repo_root / root
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts:
                index.setdefault(path.name, []).append(path)
    return index


def resolve(
    path_text: str,
    repo_root: Path,
    roots: tuple[str, ...],
    series_dir: Path | None = None,
) -> tuple[str, list[Path]]:
    """Resolve a cited path to real files. Returns ``(status, matches)``.

    A series cites its own sibling documents as well as code -- ``README.md:729``
    means the plan README, not anything under ``code.roots``. Those are resolved
    against the series directory first, so a cross-document citation is checked
    rather than reported as a missing source file.
    """
    if series_dir is not None and path_text.endswith(".md"):
        for candidate in (series_dir / path_text, series_dir / "open-questions" / path_text):
            if candidate.is_file():
                return (OK, [candidate])

    direct = repo_root / path_text
    if direct.is_file():
        in_scope = any(str(direct).startswith(str(repo_root / r)) for r in roots)
        return (OK if in_scope else OUT_OF_SCOPE, [direct])

    candidates = _index(repo_root, roots).get(Path(path_text).name, [])
    if "/" in path_text:
        candidates = [c for c in candidates if str(c).endswith(path_text)]

    if not candidates:
        return (NOT_FOUND, [])
    if len(candidates) > 1:
        return (AMBIGUOUS, sorted(candidates))
    return (OK, candidates)


def check_document(doc: Path, repo_root: Path, roots: tuple[str, ...], series_dir: Path | None = None) -> list[Row]:
    found, unresolved = citations_mod.extract(doc)
    rows: list[Row] = []

    for cite in unresolved:
        rows.append(Row(doc, cite.line, cite.raw, UNRESOLVED, None, 0, "no antecedent path in scope", True))

    for cite in found:
        status, matches = resolve(cite.path, repo_root, roots, series_dir)
        if status != OK:
            detail = ", ".join(str(m.relative_to(repo_root)) for m in matches) or "no file of that name"
            rows.append(Row(doc, cite.line, cite.raw, status, None, 0, detail, cite.inherited))
            continue

        target = matches[0]
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        for start, end in cite.lines:
            if start > len(lines):
                rows.append(
                    Row(
                        doc,
                        cite.line,
                        cite.raw,
                        OUT_OF_RANGE,
                        target,
                        start,
                        f"file has {len(lines)} lines",
                        cite.inherited,
                    )
                )
                continue
            text = lines[start - 1].strip()
            rows.append(Row(doc, cite.line, cite.raw, OK, target, start, text, cite.inherited))
            _ = end
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series_dir", help="directory containing audit.toml")
    parser.add_argument("--format", choices=("tsv", "md"), default="tsv")
    parser.add_argument("--all", action="store_true", help="report every citation, not only problems")
    parser.add_argument("--doc", help="restrict to one document by filename")
    args = parser.parse_args(argv)

    m = manifest_mod.load_or_exit(args.series_dir)
    roots = tuple(m.data["code"]["roots"])

    rows: list[Row] = []
    for doc in m.scanned:
        if args.doc and doc.name != args.doc:
            continue
        rows.extend(check_document(doc, m.repo_root, roots, m.series_dir))

    problems = [r for r in rows if r.status in PROBLEM_STATUSES]
    shown = rows if args.all else problems

    if args.format == "md":
        print(f"**{len(problems)} problem(s)** out of {len(rows)} resolved citation(s).\n")
        print("| Document | Cited | Status | Resolves to | Current text of that line |")
        print("|---|---|---|---|---|")
        for r in shown:
            where = f"{m.rel(r.resolved)}:{r.cited_line}" if r.resolved else "—"
            mark = " _(inherited)_" if r.inherited else ""
            print(f"| `{m.rel(r.doc)}:{r.doc_line}` | `{r.raw}`{mark} | {r.status} | `{where}` | `{r.text[:90]}` |")
    else:
        for r in shown:
            where = f"{m.rel(r.resolved)}:{r.cited_line}" if r.resolved else "-"
            print(f"{m.rel(r.doc)}:{r.doc_line}\t{r.raw}\t{r.status}\t{where}\t{r.text[:110]}")
        print(f"# {len(problems)} problem(s) out of {len(rows)} resolved citation(s)", file=sys.stderr)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
