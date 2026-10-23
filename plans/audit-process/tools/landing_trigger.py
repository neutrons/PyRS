"""A7: which documents cite code that has changed over a commit range.

When a subspec lands, the files it touched are exactly the files whose citations
elsewhere just went stale. Intersecting ``git diff --name-only`` with the repo
paths each document cites gives that set mechanically, in seconds, precisely
targeted at claims that just moved.

The PyRS wrinkle, and the reason ``[git]`` exists in the manifest: **``git log``
on a plan document does not date its claims.** These documents were authored on
an unsquashed parent branch and re-added wholesale later, so their commit dates
describe the move, not the prose. The commit their claims were written against
is *declared* in the manifest, never derived.

Blind spots
-----------
* **Flags citations, not claims.** A touched file does not mean a false sentence.
* **Flags boilerplate.** ``pyrs/resources/application.yml`` appears in five of
  six ownership rows and ``pyproject.toml`` in most documents, asserting nothing
  specific about either. Expect those rows and discount them.
* A rename by a PR that never touched the citing document is invisible here.
  That is why **A3 and A7 are complements, not duplicates**: A7 finds a stale
  citation because a PR touched the file, A3 by asking whether the symbol is
  there at all.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import check_citations
import citations as citations_mod
import manifest as manifest_mod


def changed_files(repo_root: Path, rev_range: str) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", rev_range],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"git diff {rev_range} failed: {result.stderr.strip()}", file=sys.stderr)
        raise SystemExit(2)
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series_dir")
    parser.add_argument("--format", choices=("tsv", "md"), default="tsv")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--base", help="override the manifest's base commit")
    group.add_argument("--since", help="range start; equivalent to --base")
    group.add_argument("--pr", action="store_true", help="use the manifest's main branch as the base")
    args = parser.parse_args(argv)

    m = manifest_mod.load_or_exit(args.series_dir)
    git_cfg = m.data["git"]
    base = args.base or args.since or (git_cfg["main_branch"] if args.pr else git_cfg["base_commit"])
    rev_range = f"{base}..HEAD"

    touched = changed_files(m.repo_root, rev_range)
    roots = tuple(m.data["code"]["roots"])

    hits: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for doc in m.scanned:
        found, _ = citations_mod.extract(doc)
        for cite in found:
            status, matches = check_citations.resolve(cite.path, m.repo_root, roots, m.series_dir)
            if status != check_citations.OK:
                continue
            rel = str(matches[0].relative_to(m.repo_root))
            if rel in touched:
                hits[rel].append((m.rel(doc), cite.line, cite.raw))

    total = sum(len(v) for v in hits.values())
    if args.format == "md":
        print(
            f"**{total} citation(s)** in {len({d for v in hits.values() for d, _, _ in v})} document(s) "
            f"point at {len(hits)} file(s) changed over `{rev_range}`.\n"
        )
        if hits:
            print("| Changed file | Document | Line | Citation |")
            print("|---|---|---|---|")
            for rel in sorted(hits):
                for doc, line, raw in sorted(hits[rel]):
                    print(f"| `{rel}` | `{doc}` | {line} | `{raw}` |")
    else:
        for rel in sorted(hits):
            for doc, line, raw in sorted(hits[rel]):
                print(f"{rel}\t{doc}:{line}\t{raw}")
        print(f"# {total} citation(s) into {len(hits)} changed file(s) over {rev_range}", file=sys.stderr)

    return 1 if hits else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
