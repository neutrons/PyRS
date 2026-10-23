# Audit toolkit

Checkers for a numbered plan series, built to the specification in
[`../process.md`](../process.md) section 7. They exist because **every artifact
the planning process produces is prose, and every check it ran was a human
reading prose** — and a reader cannot see that a relative link resolves to
nothing, because Markdown renders a dead link exactly like a live one.

**Stdlib only.** A tool that adds a dependency violates the rule it is meant to
enforce. `tomllib` is stdlib at this repo's Python 3.13, so the manifest costs
nothing.

**Series-agnostic, never copied per series.** Everything series-specific lives
in one manifest, `plans/<series>/audit.toml`. Tools are never edited to retarget
them. Probes and findings are the opposite — they belong to the series that
motivated them and live under `plans/<series>/`.

## Usage

```console
$ pixi run python plans/audit-process/tools/check_links.py plans/NXstress-prod
$ pixi run python plans/audit-process/tools/check_citations.py plans/NXstress-prod --all
$ pixi run python plans/audit-process/tools/landing_trigger.py plans/NXstress-prod --base=<sha>
```

One positional argument, the series directory. `--format=tsv|md`.
**Exit 0 = clean, 1 = findings, 2 = manifest or usage error**, so a broken
manifest is never mistaken for "no findings".

## The tools

| Tool | Axis | What it covers | What it cannot |
|---|---|---|---|
| `manifest.py` | — | The validating loader every tool passes through. Required keys, unknown-key rejection, the `base_commit` triple-check, directory-walk discovery with mandatory classification. Not a tool. | Validates structure, not truth: a `readme` key pointing at the wrong *existing* file passes. |
| `markdown.py` | — | Shared tokenizer: fence and code-span masking that preserves offsets, heading extraction, GitHub slugs. Not a tool. | Not a CommonMark parser. Nested or exotic constructs are out of scope. |
| `citations.py` | — | Citation extraction across all fifteen styles the corpus actually uses. Not a tool. | See `check_citations.py`. |
| `check_links.py` | A1/A2 | Resolves all 107 inline links: relative `.md`, repo paths, `#anchor`, `path#Lnn`, `path#heading-anchor`. Detects `{#id}` anchors that GFM does not implement. | Cannot tell whether a resolved target **says** what the citing sentence claims. Blind to a claim citing nothing. No reference-style `[x][y]` links — this series has none. |
| `check_citations.py` | A3/A7 | Resolves every `path:LINE` pointer and prints **the current text of that line**. A location report, not a verdict. | Reports locations, not claims — a drifted number does not imply a stale claim, and an accurate one does not imply a true one. Cannot see a missing citation. Owns the line-number half of `#Lnn`; `check_links.py` owns existence, and they must not double-report. |
| `check_ownership.py` | A2 | README §5 against each subspec's change sections: brace expansion, three `Depends on:` shapes, `:LINE` stripping, canonical path comparison. | **Covers paths, not names** — types, config keys and vocabularies are traced by reading. The phase column is prose, so attribution is approximate and ambiguous rows are reported, not dropped. |
| `landing_trigger.py` | A7 | `git diff --name-only` over a range, intersected with cited repo paths. | Flags citations, not claims, and **flags boilerplate**. A rename by a PR that never touched the citing document is invisible — which is why A3 and A7 are complements, not duplicates. |
| `check_drift.py` | A6 | Pairs README and the superseded draft by shared section numbering; reports identical / diverged / only-in-one. | **A shortlist generator, not an oracle.** Cannot see a non-heading-aligned divergence, and says nothing about whether a divergence was *recorded*. **A6 has weaker tooling than the other axes**; this says so rather than implying parity. |

**Axes that stay reading tasks**, stated so gaps are not mistaken for coverage:
the semantic half of A1 and A2, and all of A4 and A5 — those are probes, not
checkers.

**Recommended against building**, and why: a *symbol-existence checker* (a
series cites the symbols its subspecs will **create** far more often than
symbols that exist, so a containment check flags every planned API as missing);
a *claims-file runner* (a second, hand-maintained copy of the document's
assertions, drifting from the document exactly as the document drifts from the
code — the original problem with one more place to keep in sync).

## What the tuning round changed

`process.md` section 7.6 warned that **a tool whose first output is wrong in a
plausible-looking way is worse than no tool.** Every defect below was in a first
draft of these tools and was caught only by checking output against the
hand-verified calibration set. Recorded because a toolkit that documents only
its successes teaches nothing about its failure modes.

| Defect | Symptom | Fix |
|---|---|---|
| Code spans masked before links were found | The two links whose *link text* is itself a code span (`README.md:6`, `:84`) vanished | Mask with a same-length filler so surrounding brackets survive |
| Citation anchors resolved non-positionally | `README.md:192` inherited `nexus_conversion.py` from a *later* bullet instead of the `fields.py` on its own line | Single positional pass over spans, parentheticals and bare `Lnn` interleaved in document order |
| `L\d+` matched anywhere | "Detector **L2**" (a physics term) became a citation to line 2 | Require two digits, and require a parenthetical body to be a *pure* line list — `(L2, arm shift)` is prose |
| `#Lnn` matched inside link targets | `04c:17-18` double-reported what `check_links.py` already owns | Exclude `#` in the lookbehind |
| `.strip("_")` on path tokens | `_sample.py` became `sample.py`, then "missing" | Never strip a leading underscore |
| Path shape tested before brace expansion | `pyrs/resources/{__init__.py,application.yml}` ends in `}`, matched no shape, and was dropped whole | Expand first, test each result |
| Any `.ext` counted as a file | Config keys `nxstress.enable` / `legacy_io.enable` reported as missing files | Match against a known extension set |
| Change sections read as `###` only | Specs that list files as **bullets** (05's `## PyRS Changes`) looked like they claimed nothing | Parse both shapes |
| Literal string comparison of paths | 04b writes `### ...`_input_data.py`, `_sample.py`, ...` — a comma list where only the first is qualified — so bare siblings looked unclaimed | Compare canonical repo-relative paths |

Two of those were not tool bugs at all but **findings about the series**, and
they are carried into the audit rather than silently absorbed: the change-section
structure is non-uniform (`###` children in some specs, bullets in others), and
so is the change heading itself.
