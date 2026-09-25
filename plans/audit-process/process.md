# What a complete audit of a plan series consists of

**Status:** the working definition for PyRS plan audits. Section 4 has migrated to
[`CLAUDE.md`](../../CLAUDE.md); section 7 specifies a toolkit that does not exist yet.
**Scope:** any numbered plan series under [`plans/`](../). Nothing here is specific
to one feature; the worked examples come from
[`plans/NXstress-prod/`](../NXstress-prod/) because that is the series this was
first exercised against.

---

## 1. Why this document exists

A PyRS feature is planned in four steps:

> **I.** Generate an overview implementation plan (no codebase awareness).
> **II.** Match the plan to the codebase; extend where necessary.
> **III.** Break the plan into PR-sized subspecs against a shared `README.md`.
> **IV.** Audit the plan repeatedly, against the codebase, until consistent.

That is what `plans/NXstress-prod/` literally is: `archive/overview.md` is step I,
`README.md` step II, the twelve subspecs step III, and the eighteen-row
[Decisions Log](../NXstress-prod/README.md#4-decisions-log) is step IV's ledger --
nine entries recording a correction found by re-reading documents against code.

Step IV has run, repeatedly and conscientiously, and the series still carries
defects a twenty-line script finds in seconds (section 2). The problem is not
diligence. **Every artifact this process produces is prose, and every check it runs
is a human reading prose.** A reader cannot see that a relative link resolves to
nothing -- Markdown renders a dead link exactly like a live one -- and re-reading a
sentence that was true when written produces agreement, not detection.

**Keep the claim narrow.** These defects have not cost anyone a week. The argument
is not "our audits are failing" but something more specific: the rounds that ran
covered two of the seven axes, and every defect that remains lives on an axis never
run. The gap is coverage, not effort.

---

## 2. The evidence

Five defects in `plans/NXstress-prod/`, each confirmed by running the command
shown. They are also the **calibration set** for section 7's tools: a checker that
cannot rediscover them is not ready to be trusted.

### 2.1 Seven dead links, five documents, one dependency table

Every reference to subspec 01 targets `01-config-and-test-infra.md`. The file is
[`01-config-and-test-infra-PR.md`](../NXstress-prod/01-config-and-test-infra-PR.md).

```console
$ grep -rn '01-config-and-test-infra\.md' plans/ | wc -l
7
```

They sit in `README.md:21` -- the `## Sub-specifications` table on which the
series' ordering rests -- plus `README.md:723`, `02:5`, `03:5`, `04b:6`, `04b:25`,
`05:6`. Four are a subspec's `**Depends on:**` header, the one line stating what
must land first.

**Nothing about `[01](01-config-and-test-infra.md)` looks wrong.** Be precise about
what this is: a broken pointer, not a false claim -- nobody would build the wrong
thing because of it. What makes it evidence is that it is mechanically detectable,
sits in the most load-bearing table in the series, and survived every pass.

### 2.2 Four line citations drifted in a single table cell

`README.md:43` places `HidraWorkspace`'s methods at L463, L497, L1025 and L1096.

```console
$ grep -n 'def load_hidra_project\|def append_hidra_project\|def save_experimental_data\|def save_reduced_diffraction_data' pyrs/core/workspaces.py
465:    def load_hidra_project(self, hidra_file, load_raw_counts, load_reduced_diffraction):
517:    def append_hidra_project(self, hidra_file):
1045:    def save_experimental_data(self, hidra_project, sub_runs=None, ignore_raw_counts=False):
1116:    def save_reduced_diffraction_data(self, hidra_project, sub_runs=None):
```

All four drifted. **All four prose claims remain exactly true** -- only the
pointers rotted, and `:463` now lands on plausible, unrelated code. This is why a
location report is not a claim report (section 7.7). The same wrong sentence also
sits in the archived draft, copied forward and never maintained (section 2.5).

The harder parsing case is in the same table: `README.md:58` cites `model.py:37`
and `model.py:141-165` for methods at `:42` and `:175` of
`texture_fitting_model.py`. That bare basename resolves to no path without the
surrounding context, and the series uses it for two different files.

### 2.3 A cited file that does not exist, in a subspec that has shipped

`02-peak-and-texture-nxstress.md:83` is a section heading naming
`pyrs/interface/texture_fitting/model.py`. There is no such file; it is
`texture_fitting_model.py`. The same wrong path hides in `README.md`'s ownership
table inside a brace expansion (`{texture_fitting_viewer,model}.py`).

**Subspec 02 was implemented and landed with that heading intact** (`a3eb8427`).
The implementer read it, went to the right file anyway, and nobody noticed -- the
shape of a defect that costs nothing until someone automates against it.

### 2.4 The class no reading can reach -- exposure, not defect

Unlike the others, **these are not known to be wrong.** The point is that none is
*knowable* by reading, each is minutes of probing, and the series has no probes.

| Where | The claim |
|---|---|
| `README.md:267-282` | `neutrons_standard.init("pyrs")` must precede importing `neutrons_standard.config`; a stray import "would race `init()` and silently corrupt `package_name`"; `env=<file>` deep-merges over the default |
| `04c-nxstress-append.md` | Tail-append works via `resize(cur+N); arr[cur:] = ...` on an h5py resizable dataset -- the whole spec rests on it |
| Decisions Log item 16 | Installed `nexusformat` 1.0.8 "has no validator capability at all" |
| `10-flip-defaults.md` | `self.statusBar()` creates lazily; `setEnabled` leaves a `QAction` visible. `docs/ground_truths.md` records that this repo moved PyQt5 to PyQt6 -- a plan asserting a PyQt5 API reads perfectly and fails at import |

Every one requires **executing** something.

### 2.5 The superseded draft that specifies a different architecture

`overview.md` and `README.md` shared the same `## 1`-`## 6` numbering, much text,
and the same title. `README.md` superseded it when the subspecs were split out; it
was never retired, updated, or marked. Their `### 2.3` sections are both "New
capability" and are mutually exclusive:

| | draft section 2.3 | `README.md` section 2.3 |
|---|---|---|
| Library | `pyyaml` | `neutrons_standard.Config`; "no `pyyaml` dependency needed" |
| Config file | `config/pyrs.default.yml` + `~/.config/pyrs/config.yml` | `pyrs/resources/application.yml`, "a hard requirement" |
| Validation | a validated pydantic `Config` | "provides no schema validation of its own" |
| Extension key | `nxstress.default_extension` | paired `nxstress` / `legacy_io` sections |
| CLI | "`argparse` wiring on each installed script" | "no `argparse` + `--config` plumbing ... **and this work does not add any**" |

**Subspec 01 landed implementing the README design.** Decisions Log item 11 records
the schema's evolution and never mentions that the draft now describes something
unbuilt. A reader opening the wrong "section 2.3" gets a different architecture.

Found by comparing **one** heading pair out of roughly twenty; the rest have never
been compared. The draft is now at
[`archive/overview.md`](../NXstress-prod/archive/overview.md) behind a banner,
which removes the hazard but not the open question.

---

## 3. The three failure classes

**Class A -- staleness.** A document asserts a fact about the codebase; later work
falsifies it; nothing links the two.

> Repeated auditing makes this **worse**: each pass re-reads the same sentence,
> finds it plausible, re-confirms it, and confidence rises while correctness falls.
> The dominant PyRS sub-class is **referent drift** -- a citation correct when
> written, now pointing at unrelated code -- because this series carries 172
> `path:LINE` citations over actively moving code. Decisions Log item 15 records a
> round of "cited line numbers ... corrected against the current code"; section
> 2.2's four survived it.

**Class B -- unexecutable claims.** Assertions about third-party behaviour, or
about how another module treats our output at runtime.

> Unfixable by review at any level of care (section 2.4). These surface at first
> execution because that is the only place they *can* -- unless someone probes
> first (section 5.2).

**Class C -- contradiction, within a document or across siblings.**

> Cheap to catch, consistently missed, because audit instructions point *outward*
> at the codebase (sections 2.1, 2.3). Its two habitats are the `README.md`/draft
> pair (section 2.5) and `README.md`'s `## 5. Files to be Modified` table against
> each subspec's own change headings.

---

## 4. Definition of a complete audit

An audit is complete when **all seven axes** are covered with the stated evidence.
Anything less is a partial audit, recorded as partial rather than reported as
"consistent".

| # | Axis | Question | Referent in a PyRS series |
|---|---|---|---|
| **A1** | Doc vs self | Does every prose claim agree with every code block, table and link in the same document? | Section-by-section pass; **every Markdown link resolved to an existing file, every anchor to an existing heading** |
| **A2** | Doc vs siblings | Do shared paths, config keys, vocabularies and ordering agree across subspecs? | `README.md`'s `## Sub-specifications` table and `## 5. Files to be Modified`; each shared name traced to one owning document |
| **A3** | Doc vs codebase | Does every cited file, symbol and line range exist and say what is claimed? | The citation opened and read, not recalled |
| **A4** | Doc vs library | Is every claim about third-party behaviour true *of the installed version*? | A committed probe against `h5py`, `nexusformat`, `mantid`, `qtpy`, `neutrons_standard`, with real output pasted in |
| **A5** | Doc vs runtime contract | Will another module in this repo treat our output as we assume? | A probe through the real code path of `HidraProjectFile`, `HidraWorkspace`, a viewer model, or a Qt action |
| **A6** | Doc vs earlier draft | Is each divergence from the pre-codebase draft acknowledged and justified? | `README.md` against `plans/<series>/archive/overview.md` |
| **A7** | Freshness | Are the claims true *now*, given what has landed since? | Re-verification dated later than the most recent landed subspec the document depends on |

For `plans/NXstress-prod/`, the recorded rounds covered **A2 and A3**. Every defect
in section 2 falls in A1, A4, A5, A6 or A7.

**Exemption is recorded, never assumed.** A subspec asserting nothing third-party
and nothing cross-module is genuinely A4/A5-exempt -- but write it down, because
afterwards "no probes were needed" and "no probes were written" look identical.

### 4.1 Anti-patterns -- things that look like auditing and are not

- **Re-reading a claim and finding it plausible.** Plausibility is what a stale
  claim has in abundance. A claim is verified when checked *against its referent*.
- **Checking the grounded document against the earlier draft.** That confirms
  transcription, not correctness -- and since the draft is *older*, agreement with
  it is evidence of staleness. Section 2.5 is the worked example.
- **Treating an exemption label as exemption from everything.** "Blocked",
  "tracked follow-up", "not scheduled in this plan's phases" and `open-questions/`
  are *scheduling* statements. `open-questions/08` Q3 is externally blocked and
  genuinely A4/A5-exempt; `open-questions/04b` Q1 is a "tracked follow-up" **while
  04b, 04c and all of Phase 3 proceed on the assumption it makes**. Exempt from
  scheduling, and from nothing else.
- **Assessing "self-consistent" as "does not conflict with the codebase."** A
  document can match the code perfectly and contradict itself on the next page.
- **Counting audit passes.** Eighteen Decisions Log entries over two axes is two
  axes covered, not thirty-six.
- **Recording the correction in the log and not in the document.** A log preserves
  what was believed, which is right for a *claim*. A line number is a pointer, not
  a belief: log the finding **and** apply the number.

---

## 5. Mechanisms

Ordered by leverage per unit of effort; each attacks a named class.

### 5.1 Invariants belong in tests, not prose
*(Class A, permanently)*

A finding that can be expressed as a test must be, and the prose becomes a comment
on the test. A paragraph is re-read once per audit; a test is re-checked on every
commit. The idiom to copy is
[`test_definitions.py`](../../tests/unit/pyrs/utilities/NXstress/test_definitions.py),
which iterates the schema enumeration rather than restating it:

```python
for group in GROUP_NAME:
    ...  # every member carries allowMultiple / nxClass, of the right type
```

Add a member and the guarantee extends automatically. Related precedents:
[`tests/test_conftest.py`](../../tests/test_conftest.py) and
[`tests/util/test_peak_collection_helpers.py`](../../tests/util/test_peak_collection_helpers.py)
pin the fixture layer;
[`tests/ui/test_peak_fitting.py:243`](../../tests/ui/test_peak_fitting.py#L243)
asserts a widget surface a plan document promised.

**Be accurate about the starting point:** nothing in PyRS pins a third-party API
surface, and nothing scans source for a convention. Both shapes are net-new. Do not
write a sentence claiming otherwise.

Placement follows `CLAUDE.md`'s tier rules -- consult them, do not restate them.
**An audit flags; the implementing PR writes.** A test written during the audit has
no subject.

### 5.2 Probe before asserting
*(Class B, moved from weeks of implementation to minutes of probing)*

**Rule:** a document may not be marked audited if it asserts (a) third-party
library behaviour or (b) how another module treats its output, without a committed
probe under `plans/<series>/probes/` whose **real output is pasted into the
document at the claim**.

A probe is **not** the subspec's `## Verification` section. The two share exactly
one thing -- an evidentiary standard: run it, paste the *real* output.

| | `## Verification` | Probe |
|---|---|---|
| Audience | the PR reviewer | the spec author / auditor |
| When | after implementation | before it, during the audit |
| Purpose | show the shipped thing works end to end | test whether a design assumption is true |
| Lives in | the subspec's own section | a probe file; output pasted at the claim |
| If it fails | the PR is not finished | the **document** is wrong and the design must change |

That last row is the separation that matters. (`## Delivered Feature`, the
blockquoted release note, is neither.)

The A4/A5 boundary is **which counterparty, not which interpreter**: a claim about
`h5py` is A4; a claim about `pyrs/utilities/NXstress/` is A5, because that library
is in this repo and already landed.

### 5.3 Cross-subspec dependencies are imports, never sentences
*(Class A, for the cases that matter most)*

Where subspec *N* depends on module *M*'s behaviour, express it as a **test that
imports *M***, not a paragraph describing *M*. Then a change to *M* breaks *N*'s
test instead of silently invalidating *N*'s prose.

Decisions Log item 17 rests the entire 04b/04c simplification on a description of
code neither subspec owns: that `_Peaks.peakCollectionRanges` enforces only
run-contiguity and monotonic `scan_point`, "the latter guaranteed upstream by
`SubRuns.set` (`sample_logs.py:164-166`)". Two subspecs' design weight on one
sentence. Corollary: reproduce the counterparty's logic **locally** in the test, so
it fails when that module changes rather than silently tracking it.

### 5.4 A landing trigger
*(Class A and A7)*

When a subspec lands, intersect `git diff --name-only <base>..HEAD` with the files
each document cites; the overlap is the set needing re-audit. Mechanical, seconds,
and precisely targeted at claims that just went stale.

The PyRS wrinkle, which decides a design question in section 7.2: **`git log` on a
plan document does not date its claims.** These documents were authored on an
unsquashed parent branch and re-added wholesale at `5b9dfe98`; their prose was
written against code at or before `492350bd`. The commit the claims were written
against must therefore be *declared*, not derived.

### 5.5 Where findings go

A PyRS series already keeps two **human-facing** records, and audit findings belong
in neither: `open-questions/NN-*.md` (stakeholder questions, with a `Blocking?`
index and quoted answers) and `README.md`'s `## 4. Decisions Log` (decisions taken).

Audit findings go in an **append-only `## Follow-up N` section of the document
being audited**, after its `## Verification` section, so the finding sits with the
claim it corrects:

```markdown
## Follow-up 1 - 2026-09-25 (first seven-axis pass)

**F1.1** (A3) -- "### `pyrs/interface/texture_fitting/model.py`"
- Referent: `pyrs/interface/texture_fitting/`
- Verdict: no such file; it is `texture_fitting_model.py`.
- Action: heading corrected; README section 5's brace expansion likewise.
```

`N` increments and is never renumbered; an earlier Follow-up is never edited.
`README.md` gets Follow-ups too -- it holds two of the seven dead links and all
four drifted citations.

**The one in-place exception is a `path:LINE` citation.** It is a pointer, not a
belief: fix it in the body *and* log the finding. Leaving it stale so the log can
"preserve the record" preserves nothing and misdirects the next reader.

When a finding forces a design change, the *decision* goes to the Decisions Log and
the Follow-up holds the evidence.

### 5.6 "Audited" is not a terminal state

Audit eliminates Class A and Class C. It **cannot** eliminate Class B. Budget a
document-correction pass per PR as normal cost, not as an audit failure; section
5.5 is where it lands.

What is usually missing is the expectation: the frustration of finding defects
during implementation is partly a mis-set belief that step IV could have found
them. For Class B it could not -- but section 5.2 moves most of that cost from
implementation to probing, an order of magnitude cheaper.

---

## 6. Running a pass

The order is deliberately **not** the order the documents are in.

0. **Build and run the toolkit** (section 7). Seconds, and it bounds how much
   reading the rest of the pass needs. A7 falls out of this step.
1. **A4/A5 probes.** The class reading cannot reach, and the one that costs weeks.
2. **A1**, each document against itself.
3. **A2**, documents against their siblings.
4. **A3 and A6** for anything the first three disturbed.

The per-series list of probe candidates deliberately does **not** live here -- it
lives in the editable block of [`process-prompt.md`](process-prompt.md). That is
what makes this document reusable across series.

---

## 7. The toolkit: build specification

### 7.0 Status

**None of this is built.** The audit agent builds it as the first step of its pass,
before recording any finding. Everything below is normative.

### 7.1 Location

`plans/audit-process/tools/`, for two reasons. **It must not ship:**
`[tool.hatch.build.targets.wheel] packages = ["pyrs", "scripts"]` puts everything
under `scripts/` into the wheel, while `plans/` ships nothing -- so no packaging
change is needed and none should be made. **It is series-agnostic and never copied
per series:** everything series-specific lives in one manifest.
`tools/README.md` carries the index -- one row per tool, what it covers, what it
cannot.

### 7.2 The manifest: `plans/<series>/audit.toml`

One file per series; tools are never edited to retarget them. Read only through
`tools/manifest.py`, so **no tool parses the TOML itself** and validation cannot be
skipped by a tool that forgot. `tomllib` is stdlib at this repo's Python 3.13, so
no dependency is added. Worked example:
[`plans/NXstress-prod/audit.toml`](../NXstress-prod/audit.toml).

**Every key is required, and a wrong value fails at load, not at use.**

| Key | Failure mode |
|---|---|
| `series.readme` | Resolved against the manifest's directory and existence-checked at load; missing gives `SystemExit(2)` naming key, value and resolved path |
| `series.earlier_draft` | `""` is legal and makes the A6 tool report *"not applicable"*. **Absent is an error** -- explicit absence beats a key indistinguishable from a forgotten one |
| `series.subspecs` | Zero matches is an error |
| `series.ownership.heading` | Must be found verbatim in the README; not found is an error quoting both |
| `series.dependencies.marker` | Must appear in every file matched by `subspecs`; the error lists those missing it |
| `git.base_commit` | **Triple-checked:** the object exists; `git merge-base --is-ancestor <sha> HEAD` passes; and `git log -1 --format=%s` **equals `base_subject`** |
| `scope.*` | A glob matching zero files, a file in both lists, or a file matched by `subspecs` and in neither, are each errors |
| `exclude.paths` | Same exhaustiveness rule: every discovered `*.md` matches exactly one classification. A zero-match pattern is an error; so is a path matched both here and by `subspecs` |
| `series.probes`, `series.findings` | **Output paths, not inputs: created on demand, never existence-checked.** Every other path key fails loudly when absent; these two are absent on a series' first pass by definition, so applying the same rule would make the manifest unloadable exactly when it is first needed |
| `code.roots` | Each root must exist; a citation resolving outside them is reported `OUT-OF-SCOPE`, never dropped |
| any unknown key | Error. Typos are the thing this catches |

Two of those need their reasoning stated.

**The `base_subject` check is the point of the `[git]` section.** Every other error
is loud on its own -- a wrong path finds no file. A stale base commit fails
**silently**: `git diff` against an unrelated commit succeeds and reports a
plausible count over the wrong range. A hash from another branch will exist and may
even be an ancestor; what it will not do is match the subject line written beside
it. `landing_trigger.py --base=<sha>` exists so nobody edits a file to check a
different range.

**Discovery is by directory walk; classification is mandatory.** Tools do not scan
only what is declared -- they walk the series directory and require every `*.md`
found to match exactly one classification. Scanning only what is declared would
leave a document nobody classified silently unaudited, which is the precise failure
this document is about. Excluding is not hiding: `archive/` is excluded from every
scanner because a superseded draft is stale *by definition* and reporting its
staleness would bury live findings, yet `check_drift.py` still reads it through
`series.earlier_draft`. One acknowledged tension: `[scope]` is a small second copy
of where each document sits in its lifecycle and goes stale the day a subspec
lands -- the same failure mode as section 9's claims files. The exhaustiveness rule
is the mitigation, converting silent under-scope into a load error.

### 7.3 One interpreter

`pixi run python`, the `default` environment, with `pyrs` installed editable.
Verified present: `h5py` 3.16.0, `nexusformat` 1.0.8, `mantid` 6.16.20260918.1225,
`qtpy` 2.4.3, `numpy` 2.1.3, `neutrons_standard`, `pydantic`. Verified **absent:
`PyQt5`** -- this repo is PyQt6 via `qtpy`. Use `-e prod|qa|dev` only for a claim
sensitive to the Mantid channel.

> **A probe never modifies `pyproject.toml` or `pixi.lock`.** If a probe needs a
> package the environment lacks, **that absence is the A4 finding.** Record it; do
> not install it.

### 7.4 The tools

Shared conventions: **stdlib only** -- a tool adding a dependency violates the rule
above. One positional argument, the series directory. Every `main()` begins
`m = manifest.load(series_dir)`. `--format=tsv|md`. **Exit 0 = clean, 1 = findings,
2 = manifest or usage error**, so a manifest error is never mistaken for "no
findings". Note `[tool.ruff] exclude = ["pyrs/icons", "scripts"]` does **not** cover
`plans/`, so pre-commit lints and formats everything written here.

**`manifest.py`** -- the validating loader above; not a tool.
*Blind spot:* validates structure, not truth -- a `readme` pointing at the wrong
*existing* file passes.

**`check_links.py` (A1/A2)** -- highest value: finds all seven of section 2.1's
dead links in one run. Resolves Markdown inline links across every discovered,
non-excluded `*.md`: relative targets, `../../pyrs/...` into the repo, bare
`#anchor` against the containing file, `path#Lnn` against target length, and
`path#heading-anchor` against a real heading's GitHub slug -- it must resolve the
live case `open-questions/README.md:7` -> `../README.md#4-decisions-log`. The eight
section-style references series-wide fold in here.
**Must strip fenced code blocks *and* inline code spans:** section 2 of this
document quotes a dead link as inline code three times as an example, and a checker
that does not strip backticks reports those as findings -- observed while writing
this document.
*Blind spots:* cannot tell whether a resolved target **says** what the citing
sentence claims (the reading half of A1); blind to a claim citing nothing; no
reference-style `[x][y]` links, of which the series has none -- stated so a future
one is not missed silently.

**`check_citations.py` (A3/A7)** -- finds sections 2.2 and 2.3. Handles all three
citation styles (~218 of them): **backticked** -- bare (`` `_peaks.py:313` ``),
qualified, ranged (`:246-338`), comma-listed (`:106,109,116,122,127-135`), or
**prefix-inherited**, where a following bare `` `:494` `` takes the previous
citation's path (live at `README.md:445`); **parenthetical inside a heading**
(`### ... (L235-239)`), taking that heading's path; and **Markdown `#Lnn` links**.
The last overlaps `check_links.py`: **this tool owns the line-number half, that one
the existence half, and they must not double-report.** Output is a location report,
not a verdict: `doc:line | cited path:LINE | the current text of that line`.
*Blind spots:* reports **locations, not claims** -- section 2.2's four numbers were
wrong and its claims right; cannot see a *missing* citation, nor one "corrected"
only in a Follow-up while the body still cites the old number (why section 5.5
fixes pointers in place); an ambiguous bare basename is a **finding, not noise** --
`model.py` matches two packages here, so report it, never guess.

**`landing_trigger.py` (A7)** -- `git diff --name-only` over a range, intersected
with the repo paths each document cites, reusing `check_citations.py`'s extractor.
Default `base_commit..HEAD`; `--since=<sha>`, `--pr`, `--base=<sha>`.
*Blind spots:* flags citations, not claims; **flags boilerplate** -- here
`pyrs/resources/application.yml` appears in five of six ownership rows and
`pyproject.toml` in most documents, asserting nothing specific; and a rename by a
PR that never touched the citing file is invisible to it. That last matters:
**A3 and A7 are complements, not duplicates** -- A7 finds a stale citation because
a PR touched the file, A3 by asking whether the symbol is there at all.

**`check_ownership.py` (A2)** -- reads `README.md`'s `## 5. Files to be Modified`
plus each subspec's `###` children under the configured change headings, checking
that every file a subspec claims appears in its README row, every file in the table
is claimed by some subspec, and a subspec citing a file created by *M* has *M* in
its dependency closure. *Three parsing hazards, all verified present:*
`**Depends on:**` takes **three shapes** -- inline link (02, 03, 04, 04c, 10),
bullet list below (04b, 05, 09), em dash plus parenthetical (01, 06, 07, 08);
`:LINE` suffixes must be stripped before comparison; and **brace notation must be
expanded** (`{_peaks,_fit,_definitions,_instrument}.py`), where section 2.3's bad
path hides.
*Blind spots:* **covers paths, not names** -- types, config keys and vocabularies
are traced by reading; the first column's phase-to-spec mapping is prose (`1`,
`2/3 (04b)`, `3 (05)`), so the pattern is approximate and ambiguous rows must be
reported, not dropped. It should **surface rather than paper over** that the change
heading is itself non-uniform (`## NXstress / GUI Changes` in 02, 03;
`## NXstress Changes` in 04, 04b, 04c, 05, 07, 09) -- that is an A2 finding.

**`check_drift.py` (A6)** -- finds section 2.5. Pairs the README's and the earlier
draft's headings by their shared `## 1`-`## 6` numbering, reporting each pair as
identical, diverged, or present in only one.
*Blind spots:* a **shortlist generator, not an oracle** -- its value is the
~20-row list, short enough to read; cannot see a divergence that is not
heading-aligned; says nothing about whether a divergence was *recorded*, the
reading half of A6. **A6 has weaker tooling here than the other axes; this says so
rather than implying parity.**

**Recommended against building.** A **symbol-existence checker**: a series cites
the symbols its subspecs will *create* -- `save_as_nxstress`, `arm_shift_applied`,
`beam_intensity_profile`, `DiffractogramData`, `HidraWorkspace.direction` -- far
more than symbols that exist, so a containment check flags every planned API as
missing. A **claims-file runner** (section 9). A **standalone section-reference
resolver** (eight occurrences; folded into `check_links.py`).

**Axes that stay reading tasks**, stated so gaps are not mistaken for coverage: the
semantic half of A1 and A2, and all of A4 and A5, which are probes not checkers.

### 7.5 Conventions for adding a probe

1. **Name it `a4_<cluster>.py` or `a5_<cluster>.py`** -- the prefix is the axis.
   **Never `test_`**: there is no `testpaths` in `pyproject.toml`, so a bare
   `pytest` at the repo root would collect `plans/**/test_*.py`. The pixi tasks all
   pass `./tests` explicitly and are safe; do not rely on it.
2. **State the claims verbatim in the module docstring, with a link to each.** A
   probe whose purpose must be reconstructed from its output is a script, not
   evidence.
3. **Print the claim next to the real result**, so the output can be pasted into a
   document and still make sense.
4. **Paste the real output into the document at the claim** -- never the predicted
   output, never a summary.
5. **Add a row to `plans/<series>/probes/README.md`** with the verdict. That index
   is the entry point: a probe re-testing an already-covered claim is worse than
   none, because it implies the uncovered ones are covered.
6. **Supersede, do not delete.** When a conclusion is overturned, keep the probe and
   say so in both files; the record of what was believed must survive.
7. **Record a probe that produced a false finding, and how it was caught.** A
   toolkit recording only successes teaches nothing about its failure modes.
8. **Assign its disposition in the same commit** -- `promote`, `retire to record`,
   or `stays a probe` -- naming **both** the PR that makes promotion possible and
   the PR that owns the invariant. Do not write the promoted test during the audit:
   **an audit flags, the implementing PR writes.**

### 7.6 What these tools cannot do

Written *before* the tools exist, as the required contents of a section their
builder fills in from experience. Two warnings are not negotiable:

- **Every tool needs a tuning round against known-good input before any finding is
  trusted.** Expect false positives from path-qualified references, prefix
  inheritance across commas and line breaks, and wrapped `Depends on:` headers --
  each already identified as a hazard in this series.
- **A tool whose first output is wrong in a plausible-looking way is worse than no
  tool.** It buries real defects in noise or manufactures false ones, and
  discredits the approach on its first outing.

The calibration set is section 2. A link checker not reporting exactly those seven
dead links is not ready; neither is a citation checker missing section 2.2's drifts.

### 7.7 The self-check property

`check_citations.py` resolves citations inside `## Follow-up N` sections too, so the
audit's own line numbers are checked by the same tool on the next run. An audit
that cites line numbers generates exactly the artifact it exists to catch.

The caution: **a drifted line number does not imply a stale claim.** Sections 2.2
and 2.3 are the worked pair -- one where every pointer was wrong and every claim
survived, one where the name itself was wrong. Conflating them buries real defects
or manufactures false ones. Record both: where it moved, and whether the claim
survived.

---

## 8. What has migrated to `CLAUDE.md`

[`CLAUDE.md`](../../CLAUDE.md)'s *"Auditing a Plan or Subspec"* carries what an
agent needs in context on every task:

| From here | There |
|---|---|
| Section 4's seven-axis table | *"What a complete audit consists of"* |
| Section 4.1's anti-patterns | *"Anti-patterns -- things that look like auditing and are not"* |
| Section 5.2's probe rule and table | *"Probe before asserting (A4/A5)"* |
| Section 5.1 | *"Invariants belong in tests, not prose"* |
| Sections 5.5 and 5.6 | *"'Audited' is not a terminal state"* |

Deliberately **not** in `CLAUDE.md` until they have a series' worth of use: the
toolkit and manifest schema (section 7), the landing trigger (section 5.4), and the
`## Follow-up N` format (section 5.5). All specified here, none proven.

---

## 9. Mechanisms considered and rejected

**A machine-checkable claims file per subspec** -- a `claims/<subspec>.yml` pairing
each prose claim with a `check:` shell command, run by one script. Attractive, and
it should not be built.

A claims file is a **second copy of the document's assertions**, hand-maintained,
drifting from the document exactly as the document drifts from the code -- with
nothing to catch it, since nothing checks the claims file against the prose it
paraphrases. That is the original problem with one more place to keep in sync.

Section 7's tools avoid it by **deriving** what to check from the document's own
citations: no second copy, nothing to maintain. What a claims file would have
covered and the derived tools do not is the assertion that cites nothing -- and the
answer to those is section 5.1 (make it a test) or 5.2 (make it a probe).
