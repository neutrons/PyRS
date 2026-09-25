"""A2: reconcile the README ownership table against each subspec's own headings.

``README.md``'s ``## 5. Files to be Modified`` says which document owns which
file; each subspec's change sections say which files that subspec actually
touches. When those two drift apart, nobody notices, because checking requires
holding thirteen documents in mind at once.

Three parsing hazards, all verified present in ``plans/NXstress-prod/``:

* ``**Depends on:**`` takes **three shapes** -- an inline link (02, 03, 04, 04c,
  10), a bullet list on the following lines (04b, 05, 09), and an em dash with an
  optional parenthetical (01, 06, 07, 08). Spec 01's is a bare em dash with
  *nothing after it*, so the parser must tolerate the empty case rather than
  treating it as malformed.
* ``:LINE`` suffixes must be stripped before any path comparison.
* **Brace notation must be expanded**, in three distinct suffix shapes:
  ``{a,b}.py``, ``{a.py,b.yml}`` with extensions inside the braces, and
  ``{a,b}/...`` on a directory. The bad path this series carries --
  ``texture_fitting/model.py``, a file that does not exist -- hides inside one of
  them at ``README.md:738``, which is exactly why expansion is not optional.

Blind spots
-----------
* **Covers paths, not names.** Types, config keys and vocabularies are traced by
  reading; nothing here checks them.
* The table's first column is prose (``1``, ``2/3 (04b)``, ``3 (05)``), so the
  phase-to-spec mapping is approximate. **Ambiguous rows are reported, not
  dropped** -- a row this tool cannot attribute is a finding about the table.
* It surfaces, rather than papers over, that the change heading itself is
  non-uniform across the series. That is an A2 finding in its own right.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import manifest as manifest_mod

_BRACE_RE = re.compile(r"\{([^{}]*)\}")
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_FILE_EXT = ("py", "toml", "yml", "yaml", "lock", "rst", "md", "txt", "h5", "nxs", "xml", "html", "cfg", "ini")
_PATHISH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./{},*-]*(?:\.(?:" + "|".join(_FILE_EXT) + r")|/(?:…|\.\.\.)?)$")


@dataclass
class SubspecInfo:
    path: Path
    spec_id: str
    depends_on: list[str] = field(default_factory=list)
    change_headings: list[str] = field(default_factory=list)
    files: set[str] = field(default_factory=set)


def expand_braces(token: str) -> list[str]:
    """Expand ``a/{b,c}.py`` into ``['a/b.py', 'a/c.py']``, recursively."""
    match = _BRACE_RE.search(token)
    if not match:
        return [token]
    prefix, suffix = token[: match.start()], token[match.end() :]
    out: list[str] = []
    for alt in match.group(1).split(","):
        out.extend(expand_braces(f"{prefix}{alt.strip()}{suffix}"))
    return out


def canonical(token: str, repo_root: Path, roots: list[str]) -> str | None:
    """Repo-relative path a token names, or ``None`` when nothing matches.

    Resolves a fully qualified path directly, and searches ``code.roots`` for a
    bare basename or a partial path. A basename matching more than one file
    returns the token unchanged, so the caller reports it rather than guessing.
    """
    bare = token.rstrip("/")
    if (repo_root / bare).exists():
        return str(Path(bare))
    name = Path(bare).name
    hits = []
    for root in roots:
        base = repo_root / root
        if base.is_dir():
            hits.extend(h for h in base.rglob(name) if "__pycache__" not in h.parts and str(h).endswith(bare))
    if len(hits) == 1:
        return str(hits[0].relative_to(repo_root))
    return None if not hits else token


def exists_in_repo(token: str, repo_root: Path, roots: list[str]) -> bool:
    """True when ``token`` names something real, whether qualified or bare."""
    return canonical(token, repo_root, roots) is not None


def _normalize(token: str) -> str:
    """Strip a ``:LINE`` suffix, trailing ellipsis and surrounding punctuation."""
    token = token.strip().strip("`* ,;()")
    token = re.sub(r":[\d,\s–-]+$", "", token)
    token = re.sub(r"/(?:…|\.\.\.)$", "/", token)
    return token


def paths_in(text: str) -> set[str]:
    """Every path-shaped code span in ``text``, brace-expanded and normalized."""
    out: set[str] = set()
    for span in _CODE_SPAN_RE.findall(text):
        token = _normalize(span)
        if not token:
            continue
        # Expand BEFORE testing the shape: `pyrs/resources/{__init__.py,app.yml}`
        # carries its extensions *inside* the braces, so the unexpanded token
        # ends in `}` and fails every path shape there is.
        for expanded in expand_braces(token):
            expanded = _normalize(expanded)
            if expanded and _PATHISH_RE.match(expanded):
                out.add(expanded)
    return out


def parse_ownership_table(
    readme: Path, heading: str, spec_col: int, file_cols: list[int]
) -> list[tuple[str, set[str], int]]:
    """Return ``(phase_cell, files, line)`` for each row of the ownership table."""
    lines = readme.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == heading)
    except StopIteration:  # pragma: no cover - manifest.py already checked this
        return []

    rows: list[tuple[str, set[str], int]] = []
    seen_header = False
    for offset, raw in enumerate(lines[start + 1 :], start=start + 2):
        stripped = raw.strip()
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not seen_header:
            seen_header = True
            continue
        if set(stripped) <= set("|- :"):
            continue
        files: set[str] = set()
        for col in file_cols:
            if col - 1 < len(cells):
                files |= paths_in(cells[col - 1])
        rows.append((cells[spec_col - 1] if spec_col - 1 < len(cells) else "", files, offset))
    return rows


def parse_subspec(path: Path, change_headings: list[str], marker: str) -> SubspecInfo:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    spec_id = re.match(r"(\d+[a-z]?)", path.name).group(1)
    info = SubspecInfo(path=path, spec_id=spec_id)

    # `Depends on:` -- inline links, or a bullet list on following lines, or an
    # em dash that may be followed by nothing at all (spec 01).
    for i, line in enumerate(lines):
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1].strip()
        info.depends_on.extend(re.findall(r"\]\((\d+[a-z]?)[^)]*\.md\)", tail))
        if not tail or tail.startswith("—"):
            for follow in lines[i + 1 :]:
                if not follow.strip().startswith("-"):
                    break
                info.depends_on.extend(re.findall(r"\]\((\d+[a-z]?)[^)]*\.md\)", follow))
        break

    current = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            current = stripped if stripped in change_headings else None
            if stripped in change_headings:
                info.change_headings.append(stripped)
        elif current and stripped.startswith("### "):
            info.files |= paths_in(stripped)
        elif current and line.startswith("- "):
            # A bullet's *first* code span names the file; later spans in the
            # same bullet are prose references, not claims of ownership.
            first = _CODE_SPAN_RE.search(line)
            if first:
                info.files |= paths_in(f"`{first.group(1)}`")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series_dir")
    parser.add_argument("--format", choices=("tsv", "md"), default="tsv")
    args = parser.parse_args(argv)

    m = manifest_mod.load_or_exit(args.series_dir)
    own = m.data["series"]["ownership"]
    sections = m.data["series"]["subspec_sections"]

    rows = parse_ownership_table(m.readme, own["heading"], own["spec_column"], own["file_columns"])
    specs = [
        parse_subspec(p, sections["change_headings"], m.data["series"]["dependencies"]["marker"]) for p in m.subspecs
    ]

    roots = m.data["code"]["roots"]

    def canon(token: str) -> str:
        return canonical(token, m.repo_root, roots) or token

    table_files: set[str] = set()
    for _, files, _line in rows:
        table_files |= {canon(f) for f in files}
    spec_files: set[str] = set()
    for spec in specs:
        spec.files = {canon(f) for f in spec.files}
        spec_files |= spec.files

    findings: list[tuple[str, str, str]] = []

    for phase, files, line in rows:
        if not re.search(own["spec_pattern"], phase or ""):
            findings.append((f"{m.rel(m.readme)}:{line}", "unattributable-row", f"phase cell {phase!r}"))
        for f in sorted(files):
            if not exists_in_repo(f, m.repo_root, m.data["code"]["roots"]):
                findings.append((f"{m.rel(m.readme)}:{line}", "table-path-missing", f))

    for spec in specs:
        for f in sorted(spec.files):
            if f not in table_files:
                findings.append((m.rel(spec.path), "claimed-but-not-in-table", f))
            if not exists_in_repo(f, m.repo_root, m.data["code"]["roots"]):
                findings.append((m.rel(spec.path), "spec-path-missing", f))

    for f in sorted(table_files - spec_files):
        findings.append((m.rel(m.readme), "in-table-but-unclaimed", f))

    variants = sorted({h for s in specs for h in s.change_headings if h != "## PyRS Changes"})
    if len(variants) > 1:
        findings.append((m.rel(m.readme), "non-uniform-change-heading", " | ".join(variants)))

    if args.format == "md":
        print(f"**{len(findings)} ownership finding(s)** over {len(rows)} table rows and {len(specs)} subspecs.\n")
        print("| Where | Finding | Detail |")
        print("|---|---|---|")
        for where, kind, detail in findings:
            print(f"| `{where}` | {kind} | `{detail}` |")
    else:
        for where, kind, detail in findings:
            print(f"{where}\t{kind}\t{detail}")
        print(f"# {len(findings)} finding(s) over {len(rows)} rows, {len(specs)} subspecs", file=sys.stderr)

    return 1 if findings else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
