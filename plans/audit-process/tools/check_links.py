"""A1/A2: resolve every Markdown link in a plan series.

Markdown renders a dead link exactly like a live one, which is why seven of them
survived every reading pass of ``plans/NXstress-prod/``. This tool is the answer
to that: it resolves each inline link's target against the filesystem and, where
the link carries a fragment, against the target's own contents.

What it resolves
----------------
* relative ``*.md`` targets, and relative paths into the repo (``../../pyrs/...``),
  including directory targets;
* a bare ``#anchor`` against the containing file's own headings;
* ``path#Lnn`` against the target's actual line count;
* ``path#heading-anchor`` against a real heading's GitHub slug, or against an
  inline ``<a id="...">`` anchor, which GitHub honours and ``{#id}`` is not.

What it cannot do
-----------------
* It cannot tell whether a resolved target **says** what the citing sentence
  claims. That is the reading half of A1, and no checker covers it.
* It is blind to a claim that cites nothing at all.
* It does not handle reference-style ``[x][y]`` links. This series has none --
  stated explicitly so that a future one is not missed silently. The only
  ``[x][y]``-shaped text in the corpus is a Python subscript inside a code span,
  which the masking removes before matching.
* Line numbers inside a link fragment are checked for *existence* only; whether
  ``#L42`` still points at the right code is ``check_citations.py``'s half of the
  job, and the two must not double-report.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import manifest as manifest_mod
import markdown
import re

# Inline links only. The target runs to the matching paren; no nested parens
# occur in this corpus, and a target containing one would be reported as broken
# rather than silently mis-parsed.
_LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")


@dataclass
class Finding:
    doc: Path
    line: int
    target: str
    reason: str
    detail: str


def _split_fragment(target: str) -> tuple[str, str]:
    path, sep, fragment = target.partition("#")
    return path, fragment if sep else ""


def _check_fragment(target_path: Path, fragment: str) -> tuple[str, str] | None:
    """Validate a fragment against the target file. Returns (reason, detail) or None."""
    if not fragment:
        return None

    if re.fullmatch(r"L\d+(-L?\d+)?", fragment):
        first = int(re.match(r"L(\d+)", fragment).group(1))
        try:
            total = len(target_path.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError) as exc:
            return ("unreadable-target", str(exc))
        if first > total:
            return ("line-out-of-range", f"#{fragment} but target has {total} lines")
        return None

    try:
        text = target_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ("unreadable-target", str(exc))
    heads = markdown.headings(text)
    if fragment.lower() in {h.slug for h in heads} | markdown.html_anchors(text):
        return None

    declared = [h for h in heads if h.explicit_anchor == fragment]
    if declared:
        h = declared[0]
        return (
            "explicit-anchor-unsupported",
            f"line {h.line} declares {{#{fragment}}}, which GitHub-Flavored Markdown "
            f"does not implement; its real slug is '{h.slug}'",
        )
    return ("dead-anchor", f"no heading slugs to '{fragment}'")


def check_document(doc: Path, repo_root: Path) -> list[Finding]:
    text = doc.read_text(encoding="utf-8")
    masked = markdown.mask_all(text)
    findings: list[Finding] = []

    for match in _LINK_RE.finditer(masked):
        target = match.group("target")
        line = markdown.line_of(text, match.start())
        path_part, fragment = _split_fragment(target)

        if target.startswith(("http://", "https://", "mailto:")):
            continue

        if not path_part:
            # Bare "#anchor" -- resolve against the containing document.
            problem = _check_fragment(doc, fragment)
            if problem:
                findings.append(Finding(doc, line, target, *problem))
            continue

        resolved = (doc.parent / path_part).resolve()

        if not resolved.exists():
            findings.append(Finding(doc, line, target, "dead-link", f"resolves to {resolved}, which does not exist"))
            continue

        try:
            resolved.relative_to(repo_root)
        except ValueError:
            findings.append(Finding(doc, line, target, "out-of-repo", f"resolves outside {repo_root}"))
            continue

        if resolved.is_dir():
            if fragment:
                findings.append(Finding(doc, line, target, "fragment-on-directory", f"{resolved} is a directory"))
            continue

        problem = _check_fragment(resolved, fragment)
        if problem:
            findings.append(Finding(doc, line, target, *problem))

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series_dir", help="directory containing audit.toml")
    parser.add_argument("--format", choices=("tsv", "md"), default="tsv")
    args = parser.parse_args(argv)

    m = manifest_mod.load_or_exit(args.series_dir)

    findings: list[Finding] = []
    scanned = 0
    for doc in m.scanned:
        scanned += 1
        findings.extend(check_document(doc, m.repo_root))

    if args.format == "md":
        print(f"**{len(findings)} link finding(s)** across {scanned} scanned document(s).\n")
        if findings:
            print("| Document | Line | Target | Reason | Detail |")
            print("|---|---|---|---|---|")
            for f in findings:
                print(f"| `{m.rel(f.doc)}` | {f.line} | `{f.target}` | {f.reason} | {f.detail} |")
    else:
        for f in findings:
            print(f"{m.rel(f.doc)}:{f.line}\t{f.target}\t{f.reason}\t{f.detail}")
        print(f"# {len(findings)} finding(s) across {scanned} document(s)", file=sys.stderr)

    return 1 if findings else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
