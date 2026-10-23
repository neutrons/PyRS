"""Validating loader for a plan series' audit manifest (``audit.toml``).

This is not a tool; it is the single gate every tool passes through. No tool
parses ``audit.toml`` itself, so validation cannot be skipped by a tool that
forgot -- see ``plans/audit-process/process.md`` section 7.2 for the reasoning
and the full failure table.

Two properties are worth stating because they are easy to get backwards:

* **Every key is required, and a wrong value fails at load, not at use.** An
  unknown key is an error, not a no-op; typos are the thing this catches.
* **Discovery is by directory walk, and classification is mandatory.** Tools do
  not scan only what is declared. Every ``*.md`` under the series directory must
  match exactly one classification, or the load fails -- a document nobody
  classified is the silent gap the whole audit process exists to prevent.

``series.earlier_draft`` is a *pointer*, not a classification. The worked
example proves it: ``archive/overview.md`` is named by ``earlier_draft`` *and*
matched by ``exclude.paths``, and that pair is deliberate rather than a
double-classification error -- the draft is excluded from every scanner because
it is stale by definition, yet ``check_drift.py`` still reads it through the
pointer. ``series.open_questions`` classifies; ``series.earlier_draft`` does not.
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAME = "audit.toml"

# Keys whose value is a path that is CREATED by the audit rather than read by it.
# Every other path key fails loudly when absent; these two are absent on a
# series' first pass by definition, so applying the same rule would make the
# manifest unloadable exactly when it is first needed.
OUTPUT_PATH_KEYS = frozenset({"probes", "findings"})

SCHEMA: dict[str, object] = {
    "series": {
        "name": str,
        "readme": str,
        "earlier_draft": str,
        "subspecs": str,
        "open_questions": str,
        "probes": str,
        "findings": str,
        "ownership": {
            "heading": str,
            "spec_column": int,
            "file_columns": list,
            "spec_pattern": str,
            "brace_expand": bool,
        },
        "dependencies": {
            "marker": str,
            "wraps": bool,
        },
        "subspec_sections": {
            "change_headings": list,
            "verification": str,
            "followup_prefix": str,
        },
    },
    "git": {
        "main_branch": str,
        "base_commit": str,
        "base_subject": str,
        "claims_written_against": str,
    },
    "scope": {
        "full": list,
        "freshness_only": list,
    },
    "exclude": {
        "paths": list,
    },
    "code": {
        "roots": list,
    },
}


class ManifestError(Exception):
    """Raised when the manifest is structurally wrong or points at nothing."""


@dataclass
class Manifest:
    """A loaded, fully validated audit manifest.

    Attributes:
        series_dir: Directory holding ``audit.toml``; every relative path key is
            resolved against it.
        repo_root: The git top level containing ``series_dir``.
        data: The raw parsed TOML, after validation.
        readme: Absolute path to the series README.
        earlier_draft: Absolute path to the superseded draft, or ``None`` when
            ``series.earlier_draft`` is ``""`` (legal, and means "not applicable").
        subspecs: Absolute paths of the numbered subspecs, sorted.
        open_questions: Absolute paths of the open-questions documents, sorted.
        excluded: Absolute paths excluded from every scanner, sorted.
        audit_output: Probe index and findings document -- this audit's own
            output. Classified so the exhaustiveness rule holds, but never
            scanned, because they are not plan documents.
        full: Subset of ``subspecs`` + readme getting the full seven-axis pass.
        freshness_only: Subset getting an A7 check and nothing else.
    """

    series_dir: Path
    repo_root: Path
    data: dict
    readme: Path
    earlier_draft: Path | None
    subspecs: list[Path] = field(default_factory=list)
    open_questions: list[Path] = field(default_factory=list)
    excluded: list[Path] = field(default_factory=list)
    audit_output: list[Path] = field(default_factory=list)
    full: list[Path] = field(default_factory=list)
    freshness_only: list[Path] = field(default_factory=list)

    @property
    def probes_dir(self) -> Path:
        """Output path; created on demand, never existence-checked."""
        return self.series_dir / self.data["series"]["probes"]

    @property
    def findings_path(self) -> Path:
        """Output path; created on demand, never existence-checked."""
        return self.series_dir / self.data["series"]["findings"]

    @property
    def code_roots(self) -> list[Path]:
        return [self.repo_root / r for r in self.data["code"]["roots"]]

    @property
    def scanned(self) -> list[Path]:
        """Every document a scanner should read: readme, subspecs, open questions.

        Excludes ``excluded`` by construction -- excluding is not hiding, but it
        is exclusion from every scanner.
        """
        return [self.readme, *self.subspecs, *self.open_questions]

    def is_freshness_only(self, doc: Path) -> bool:
        return doc in self.freshness_only

    def rel(self, path: Path) -> str:
        """Render ``path`` relative to the repo root, for stable reporting."""
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)


def _fail(message: str) -> None:
    raise ManifestError(message)


def _validate_types(node: dict, schema: dict, trail: str) -> None:
    """Check one TOML table against its schema slice, both directions.

    Missing keys and unknown keys are equally errors: the first catches an
    incomplete manifest, the second catches a typo that would otherwise be a
    silent no-op.
    """
    for key, expected in schema.items():
        where = f"{trail}.{key}" if trail else key
        if key not in node:
            _fail(f"[{where}] is required and is missing")
        value = node[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                _fail(f"[{where}] must be a table, got {type(value).__name__}")
            _validate_types(value, expected, where)
        elif not isinstance(value, expected):
            _fail(f"[{where}] must be {expected.__name__}, got {type(value).__name__}: {value!r}")

    for key in node:
        if key not in schema:
            where = f"{trail}.{key}" if trail else key
            _fail(f"[{where}] is not a known key -- typo, or a key this toolkit does not implement")


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _validate_git(repo_root: Path, git_cfg: dict) -> None:
    """Triple-check ``base_commit``; this is the point of the ``[git]`` section.

    Every other manifest error is loud on its own -- a wrong path finds no file.
    A stale base commit fails *silently*: ``git diff`` against an unrelated
    commit succeeds and reports a plausible count over the wrong range. A hash
    from another branch will exist and may even be an ancestor; what it will not
    do is match the subject line a human wrote down beside it.
    """
    base = git_cfg["base_commit"]

    kind = _git(repo_root, "cat-file", "-t", base)
    if kind.returncode != 0 or kind.stdout.strip() != "commit":
        _fail(f"[git.base_commit] {base!r} is not a commit object in {repo_root}")

    ancestor = _git(repo_root, "merge-base", "--is-ancestor", base, "HEAD")
    if ancestor.returncode != 0:
        _fail(f"[git.base_commit] {base!r} is not an ancestor of HEAD")

    subject = _git(repo_root, "log", "-1", "--format=%s", base)
    actual = subject.stdout.strip()
    expected = git_cfg["base_subject"]
    if actual != expected:
        _fail(
            f"[git.base_subject] does not match [git.base_commit] {base!r}\n"
            f"  manifest says: {expected!r}\n"
            f"  commit says:   {actual!r}"
        )

    claims = git_cfg["claims_written_against"]
    kind = _git(repo_root, "cat-file", "-t", claims)
    if kind.returncode != 0 or kind.stdout.strip() != "commit":
        _fail(f"[git.claims_written_against] {claims!r} is not a commit object in {repo_root}")

    branch = git_cfg["main_branch"]
    if _git(repo_root, "rev-parse", "--verify", "--quiet", branch).returncode != 0:
        _fail(f"[git.main_branch] {branch!r} does not resolve to a ref in {repo_root}")


def _classify(series_dir: Path, cfg: dict) -> dict[str, list[Path]]:
    """Walk the series directory and classify every ``*.md`` found.

    Zero classifications and more than one are both errors. The first is the
    silent gap; the second means two rules claim the same document and the
    manifest does not say which wins.
    """
    series = cfg["series"]
    readme_rel = series["readme"]
    subspec_glob = series["subspecs"]
    oq_dir = series["open_questions"].rstrip("/")
    exclude_patterns = cfg["exclude"]["paths"]

    # `series.probes` and `series.findings` are OUTPUT paths (section 7.2): they
    # are created by the audit, never existence-checked. But once created they
    # are `*.md` under the series directory, so the exhaustiveness rule has to
    # account for them or the loader fails the moment the first probe index is
    # written -- which is exactly what happened on this series' first pass.
    # They are classified, and deliberately NOT scanned: they are the audit's
    # own output, not plan documents. Section 7.7's self-check property is
    # unaffected, because `## Follow-up` sections live inside the plan documents
    # themselves, which are scanned.
    probes_dir = series["probes"].rstrip("/")
    findings_path = series["findings"]

    buckets: dict[str, list[Path]] = {
        "readme": [],
        "subspec": [],
        "open_questions": [],
        "excluded": [],
        "audit_output": [],
    }
    exclude_hits = {pattern: 0 for pattern in exclude_patterns}

    for path in sorted(series_dir.rglob("*.md")):
        rel = path.relative_to(series_dir).as_posix()
        hits: list[str] = []

        if rel == readme_rel:
            hits.append("readme")
        # `subspecs` is a root-level glob: a nested document is never a subspec,
        # or `open-questions/02-*.md` would match `[0-9]*.md` too.
        if "/" not in rel and fnmatch.fnmatch(rel, subspec_glob):
            hits.append("subspec")
        if rel.startswith(f"{oq_dir}/"):
            hits.append("open_questions")
        if rel == findings_path or rel.startswith(f"{probes_dir}/"):
            hits.append("audit_output")
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(rel, pattern):
                if "excluded" not in hits:
                    hits.append("excluded")
                exclude_hits[pattern] += 1

        if not hits:
            _fail(
                f"{rel} matches no classification. Every *.md under the series "
                f"directory must match exactly one of [series.readme], "
                f"[series.subspecs], [series.open_questions], [series.probes], "
                f"[series.findings] or [exclude.paths]."
            )
        if len(hits) > 1:
            _fail(f"{rel} matches more than one classification: {', '.join(hits)}")

        buckets[hits[0]].append(path)

    for pattern, count in exclude_hits.items():
        if count == 0:
            _fail(
                f"[exclude.paths] pattern {pattern!r} matches zero files -- a stale exclude is as bad as a missing one"
            )

    if not buckets["subspec"]:
        _fail(f"[series.subspecs] glob {subspec_glob!r} matched zero files")

    return buckets


def _apply_scope(series_dir: Path, cfg: dict, readme: Path, subspecs: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split the readme and subspecs into the full and freshness-only tiers.

    Exhaustiveness is the mitigation for ``[scope]`` being a second copy of where
    each document sits in its lifecycle: it converts silent under-scope into a
    load error.
    """
    scope = cfg["scope"]
    candidates = [readme, *subspecs]
    resolved: dict[str, list[Path]] = {"full": [], "freshness_only": []}

    for tier in ("full", "freshness_only"):
        for pattern in scope[tier]:
            matched = [p for p in candidates if fnmatch.fnmatch(p.name, pattern)]
            if not matched:
                _fail(f"[scope.{tier}] pattern {pattern!r} matches zero files")
            resolved[tier].extend(matched)

    both = set(resolved["full"]) & set(resolved["freshness_only"])
    if both:
        names = ", ".join(sorted(p.name for p in both))
        _fail(f"[scope] classifies the same file in both tiers: {names}")

    classified = set(resolved["full"]) | set(resolved["freshness_only"])
    missing = [p for p in candidates if p not in classified]
    if missing:
        names = ", ".join(sorted(p.name for p in missing))
        _fail(f"[scope] leaves these documents in neither tier: {names}")

    _ = series_dir
    return sorted(set(resolved["full"])), sorted(set(resolved["freshness_only"]))


def load(series_dir: str | Path) -> Manifest:
    """Load and fully validate the audit manifest for one plan series.

    Args:
        series_dir: Directory containing ``audit.toml``.

    Returns:
        A validated :class:`Manifest`.

    Raises:
        ManifestError: On any structural problem, missing input path, failed git
            check, unclassified document, or zero-match pattern.
    """
    series_dir = Path(series_dir).resolve()
    manifest_path = series_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        _fail(f"no {MANIFEST_NAME} in {series_dir}")

    with manifest_path.open("rb") as handle:
        cfg = tomllib.load(handle)

    _validate_types(cfg, SCHEMA, "")

    series = cfg["series"]

    readme = series_dir / series["readme"]
    if not readme.is_file():
        _fail(f"[series.readme] = {series['readme']!r} resolves to {readme}, which does not exist")

    draft_value = series["earlier_draft"]
    earlier_draft: Path | None = None
    if draft_value:
        earlier_draft = series_dir / draft_value
        if not earlier_draft.is_file():
            _fail(f"[series.earlier_draft] = {draft_value!r} resolves to {earlier_draft}, which does not exist")

    oq_dir = series_dir / series["open_questions"]
    if not oq_dir.is_dir():
        _fail(f"[series.open_questions] = {series['open_questions']!r} resolves to {oq_dir}, which is not a directory")

    for key in OUTPUT_PATH_KEYS:
        if not series[key]:
            _fail(f"[series.{key}] must name an output path, even though it is not existence-checked")

    top = _git(series_dir, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        _fail(f"{series_dir} is not inside a git repository")
    repo_root = Path(top.stdout.strip())

    _validate_git(repo_root, cfg["git"])

    for root in cfg["code"]["roots"]:
        if not (repo_root / root).exists():
            _fail(f"[code.roots] entry {root!r} does not exist under {repo_root}")

    heading = series["ownership"]["heading"]
    readme_text = readme.read_text(encoding="utf-8")
    if heading not in readme_text:
        _fail(f"[series.ownership.heading] = {heading!r} is not found verbatim in {readme}")

    buckets = _classify(series_dir, cfg)

    marker = series["dependencies"]["marker"]
    missing_marker = [p.name for p in buckets["subspec"] if marker not in p.read_text(encoding="utf-8")]
    if missing_marker:
        _fail(f"[series.dependencies.marker] = {marker!r} is missing from: {', '.join(sorted(missing_marker))}")

    full, freshness_only = _apply_scope(series_dir, cfg, readme, buckets["subspec"])

    return Manifest(
        series_dir=series_dir,
        repo_root=repo_root,
        data=cfg,
        readme=readme,
        earlier_draft=earlier_draft,
        subspecs=buckets["subspec"],
        open_questions=buckets["open_questions"],
        excluded=buckets["excluded"],
        audit_output=buckets["audit_output"],
        full=full,
        freshness_only=freshness_only,
    )


def load_or_exit(series_dir: str | Path) -> Manifest:
    """Load the manifest, or exit 2 with a message naming key, value and path.

    Exit 2 is reserved for manifest and usage errors precisely so a broken
    manifest is never mistaken for "no findings".
    """
    try:
        return load(series_dir)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <series-dir>", file=sys.stderr)
        raise SystemExit(2)
    m = load_or_exit(sys.argv[1])
    print(f"series           {m.data['series']['name']}")
    print(f"repo root        {m.repo_root}")
    print(f"readme           {m.rel(m.readme)}")
    print(f"earlier draft    {m.rel(m.earlier_draft) if m.earlier_draft else '(not applicable)'}")
    print(f"subspecs         {len(m.subspecs)}")
    print(f"open questions   {len(m.open_questions)}")
    print(f"excluded         {len(m.excluded)}")
    print(f"audit output     {len(m.audit_output)} (classified, not scanned)")
    print(f"full pass        {', '.join(p.name for p in m.full)}")
    print(f"freshness only   {', '.join(p.name for p in m.freshness_only)}")
    print(f"probes dir       {m.rel(m.probes_dir)} (output; not checked)")
    print(f"findings         {m.rel(m.findings_path)} (output; not checked)")
