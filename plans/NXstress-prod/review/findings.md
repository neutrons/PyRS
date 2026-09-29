# Audit findings — `plans/NXstress-prod/`, first seven-axis pass

**Date:** 2026-09-25 (round one) / 2026-09-26 (round two, closing the
round-one gaps) · **Branch:** `EWM12484_NXstress_hookup_PR_2`
**Audited against code at:** `68639d3b` (the branch tip when the pass began; the
pass's own output was committed afterwards, so later hashes contain these findings
rather than predating them)
**Manifest:** [`../audit.toml`](../audit.toml) · **Base:** `7a5ef73f`
("Merge pull request #1000 from neutrons/qt_lock_fix"), triple-checked at load
**Toolkit:** [`../../audit-process/tools/`](../../audit-process/tools/) ·
**Probes:** [`../probes/`](../probes/)

Prior rounds — the nine corrections in the Decisions Log — covered **A2 and A3**.
This is the first pass to run A1, A4, A5, A6 and A7. Corrections live in each
document's own `## Follow-up 1`, so the finding sits with the claim it corrects;
this file is the index and the coverage record.

---

## 0. Coverage matrix

**Required section.** Without it, "which axes were actually covered on 06?" is
unanswerable, and the rule that a partial audit is reported as partial has
nothing to attach to.

Key: **✓** covered · **✓✓** covered, then re-covered against a referent that did not
exist at the first pass · **~** partial (scope stated below) · **✗** not run ·
**n/a** not applicable · **ex** exempt, with the exemption recorded

| Document | A1 self | A2 siblings | A3 codebase | A4 library | A5 runtime | A6 draft | A7 fresh |
|---|---|---|---|---|---|---|---|
| `README.md` | ✓ | ✓ | ✓ | ✓ | ~ | ✓ | ✓ |
| `01-config-and-test-infra-PR.md` | ✗ | ✗ | ✗ | ✓ | ✗ | n/a | ✓ |
| `02-peak-and-texture-nxstress.md` | ✗ | ✗ | ~ | ✗ | ✗ | n/a | ✓ |
| `03-combine-runs-nxstress.md` | ✗ | ✗ | ~ | ✗ | ✗ | n/a | ✓ |
| `04-nxstress-internal-cleanup.md` | ✓✓ | ✓✓ | ✓✓ | ✓✓ | ✓✓ | n/a | ✓ (2026-10-15) |
| `04b-multi-workspace-nxstress.md` | ✓✓ | ✓✓ | ✓✓ | ✓✓ | ✓✓ | n/a | ✓ (2026-10-15) |
| `04c-nxstress-append.md` | ✓✓ | ✓✓ | ✓✓ | ✓✓ | ✓✓ | n/a | ✓ (2026-10-15) |
| `05-strain-stress-viewer.md` | ✓ | ✓ | ✓ | ✓ | ~ | n/a | ✓ |
| `06-manual-reduction-prereqs.md` | ✓ | ✓ | ✓ | ex | ex | n/a | ✓ |
| `07-manual-reduction-nxstress.md` | ✓ | ✓ | ✓ | ✓ | ✓ | n/a | ✓ |
| `08-fit-spectrum-prereqs.md` | ✓ | ✓ | ✓ | ✓ | ~ | n/a | ✓ |
| `09-fit-spectrum-nxstress.md` | ✓ | ✓ | ✓ | ✓ | ~ | n/a | ✓ |
| `10-flip-defaults.md` | ✓ | ✓ | ✓ | ✓ | ~ | n/a | ✓ |
| `open-questions/*.md` (12) | ~ | ✗ | ✓ | ✗ | ✗ | n/a | ✓ |
| `archive/overview.md` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

### What each non-✓ mark means

- **01, 02, 03 — A1/A2/A3/A5 not run, by scope decision.** These have landed;
  `audit.toml`'s `[scope] freshness_only` gives them A7 and nothing else. 02 and
  03 show **~** on A3 because their citations were mechanically resolved and one
  pointer defect in each was fixed, not because they were read against the code.
  01 shows **✓** on A4 only because probing README §2.3 surfaced two
  `neutrons_standard` behaviours that bear directly on the test framework 01
  shipped; they are recorded in 01's Follow-up.
- **A5 "~" across 04–10.** The in-repo counterparty claims that *can* be probed
  were probed. Those that cannot — because the code does not exist yet — are
  listed in §3 and in [`../probes/README.md`](../probes/README.md). They are
  **uncovered, not verified**.
- **A4 on 08 and 07, and A5 on 04 — closed in a second round.** All three were
  marked short in the first round on the reasoning that the code consuming them
  did not exist yet. That reasoning was wrong: each claim is about an installed
  package or existing source, and all were executable throughout. Probed in
  `a4_fit_error_propagation.py`, `a4_basename_extensions.py` and
  `a5_nxstress_internals_today.py`; see each document's Follow-up 2. The general
  lesson is recorded in `probes/README.md`'s failure-modes table.
- **04 — A4/A5 marked ✓✓, and A7 re-dated.** Both axes were genuinely covered on
  2026-09-26 and both were *re-opened* by the 04 implementation pass on 2026-09-29,
  because the NXstress schema and the NeXus `validItemName` rule turned out to be
  available after the first pass recorded them as absent. A4's earlier verdict was
  not sloppy — it was scoped to the installed `nexusformat` package — but it was
  reported more broadly than it was tested, and a design decision (item 23) rested
  on it. A5 was re-opened separately: the transformations claim had been probed by
  *reading source*, which cannot see that the chain is unreachable. See 04's
  Follow-up 7 and Decisions rows 25-27. **This is the axis-coverage equivalent of
  "audited is not a terminal state": both axes were covered, and both were wrong.**
- **04b — A4/A5 raised to ✓✓, and A7 re-dated (2026-09-30).** A5 was `~` for the
  stated reason that the code did not exist; the 04b PR wrote it, and the four
  claim clusters §3 listed are now covered by tests. A4 was `✓`, scoped to the
  precedent argument in Q1, and is re-covered against the **vendored schema**,
  which did not exist at the first pass — plus a second A4 axis nothing had
  asked about, whether a Python-string sample log is writable at all. It is
  not. Four of the eleven findings in 04b's Follow-up 2 came from probing
  before implementing; two of those falsified instructions in the spec body
  that would have produced code that could not work. **The same lesson as 04:
  both axes were covered, and both were incomplete.**
- **04c — A4/A5 raised to ✓✓, and A7 re-dated (2026-10-01).** A5 was `~` for the
  stated reason that the code did not exist; the 04c PR wrote it, and the cluster
  §3 listed is now covered by `a5_append_preconditions.py` and by
  `test_append.py`. A4 was `✓`, scoped to `a4_h5py_nexusformat_append.py`'s
  verdict that the tail-append mechanism works on reopened `NXfield`s — **true,
  and narrower than it was recorded.** That probe built every fixture *with*
  `maxshape`, so it established the mechanism on a resizable dataset, not that
  the writer emits resizable datasets. Most of the scan-point family did not:
  `a5_scan_point_family_resizable.py` found 54 of 86 datasets contiguous, which
  forced Decisions row 32 and a write-path change across five modules. A second
  A4 axis nothing had asked about — whether a fixed-width `|S` string column
  survives an append — turned out to **truncate silently**. **The third
  consecutive subspec where both axes were covered and both were incomplete;
  at this point it is the expected result of implementing, not a surprise.**
- **A4/A5 "ex" on 06.** 06 asserts nothing third-party and nothing cross-module.
  Genuinely exempt — written down because afterwards "no probe was needed" and
  "no probe was written" look identical.
- **A6 on README — closed in the second round.** All nine diverged pairs have
  now been read and each traces to a Decisions Log entry; the table is in the
  README's Follow-up 2 F2.1. One divergence is justified on a wrong rationale
  (F2.2). **What A6 still does not establish:** the five byte-identical sections
  are not thereby confirmed. §1 is identical to the draft *and* is where ten
  stale pointers were found — the draft carries the same stale numbers, and
  agreeing with it is evidence of staleness, not health.
- **A6 "n/a" on subspecs.** The archived draft is a single overview document; it
  pairs with `README.md` only.
- **`open-questions/` — A1 "~".** All twelve were link- and citation-checked
  (**A3 ✓**); only `04b` Q6 and `04c` Q5 were read as prose, because Decisions
  items 17 and 18 rest on them. Per the scope decision taken for this pass, the
  manifest was left unchanged; see §4.
- **`archive/` — n/a throughout.** Excluded from every scanner: a superseded
  draft is stale by definition, and reporting its staleness would bury live
  findings. It is still read, through `series.earlier_draft`, by `check_drift.py`
  — which is the entire reason it was kept rather than deleted.

---

## 1. Findings by document

Each is written up in full, with evidence and the established fix, in that
document's `## Follow-up 1`. Decisions forced by a finding are Decisions Log
rows 19–22.

| # | Doc | Axis | Finding | Disposition |
|---|---|---|---|---|
| F1.1 | README | A3 | Ten drifted citations in §1.1/§1.2's architecture tables; every prose claim still true. Two name no file and resolve into the wrong one. | Fixed in place |
| F1.2 | README, 04 | A3/A7 | The `file_object.py` "legacy log-name FIXMEs" were removed by `492350bd` — this series' own `claims_written_against` commit. Cited lines were also wrong when written. | **Decision 19**: 04's PyRS item struck |
| F1.3 | README, 04b, 04c | A3 | Decisions 17's load-bearing `sample_logs.py:164-166` points at a *different* `RuntimeError` in the same method; the quoted guard is at `:167-168`. | Fixed in place; claim probe-confirmed |
| F1.4 | README, 10 | A4 | "Only PeakFittingViewer has a status bar" is too strong; `self.statusBar()` is called nowhere today, so the mechanism is net-new — but safe. | Reword; mechanism kept |
| F1.5 | README, 04 | A4 | `nexusformat` 1.0.8 ships no validator (confirmed). Spec 04's Verification is the only place that omits the "separate repository" hedge. | Hedge added to 04 |
| F1.6 | README, 05, 07 | A2 | §5's test inventory and the subspecs name different files; GUI assertions are scheduled into the integration tier. | **Decision 21** |
| F1.7 | README, 07 | A2 | Spec 07's `Depends on` says "none"; it requires 04b's list signature. | **Decision 20** |
| F1.8 | README | A2 | Four conventions for naming files in a change section; two spellings of the change heading. | Recorded, not changed |
| F1.9 | README | A6 | Nine of fourteen numbered sections diverge from the draft; five are identical — which is evidence of staleness, not health. | Recorded |
| F1.1 | 04 | A3/A7 | (see F1.2 above) | **Decision 19** |
| F1.2 | 04 | A1 | Tests section put sx/sy/sz in `test_sample.py`; its own Scope locates them in `_peaks.py`. | Fixed in place |
| F1.4 | 04 | A1/A2 | Rotation-order cross-check names two non-interchangeable referents (`DENEXDetectorGeometry` vs "the reduction pipeline"), which 08 says bypass each other. | Reword; open question flagged |
| F1.2 | 04b | A3 | Seven stale `workspaces.py` citations from **one** upstream insertion (+2 before ~L463, +20 after ~L500). | Fixed in place |
| F1.3 | 04b | A3 | Two citations land on the comment above the code they quote. | Fixed in place |
| F1.3 | 04c | A4 | "Isn't possible without adding a new on-disk column" — adding one *succeeds*. A scope decision, not a format limit. | **Decision 22** |
| F1.4 | 04c | A3 | `L180-181` had no resolvable antecedent in its paragraph. | Qualified in place |
| F1.1 | 05 | A1/A2 | A `QAction` assertion scheduled into `tests/integration/`, which runs without `QT_QPA_PLATFORM=offscreen`. | **Decision 21** |
| F1.2 | 05 | A3 | `workspaces.py:55-1155` excludes `sample_log_names` at 1168. | Fixed in place |
| F1.1 | 06 | A3 | `powder_pattern.py:224` → `save_diffraction_data` is at `:231`. | Fixed in place |
| F1.2 | 06 | A2 | 06's files appear in no §5 row — §5 is organised by phase and 06 is unphased. | Recorded |
| F1.3 | 07 | A1 | The two basename branches strip extensions differently; the prose says they behave alike. | Reword + extend test |
| F1.1 | 08 | A3 | The correction at `file_object.py:510` mis-describes what is there (`:510` is not a comment; the TODO is at `:541`). | Reword |
| F1.2 → F2.1-2 | 08 | A4 | `uncertainties` and Mantid claims — **now probed and confirmed**, with the refinement that the one path computing a covariance is dead *scipy* code, not Mantid. | Closed |
| F2.3 | 08 | A6 | `file_object.py:541` is dismissed as "an unrelated return-type-annotation TODO"; it records the same observation the draft pointed at. | Reword |
| F2.1-3 | 04 | A5 | All four premises confirmed. **New:** `allowed_identifier` leaves `/` untouched, and a `/` in a group name does not raise — it **silently nests**. 04's Scope lists `$` and whitespace, not `/`. | **Decision 23** |
| F3.1 | 04 | A5 | **`allowed_identifier` is many-to-one with no collision check.** Two distinct PV logs converge on one identifier; the second overwrites the first *including* the `local_name` attribute meant to preserve the original — silent data loss, **present today** with only the `:` replacement. Claimed nowhere; found by probing. | **Decision 23** (made impossible, not merely detected) |
| F4.2 | 04 | A4 | **A design decision was blocked by a missing artifact.** `.` as an escape marker required `.` to be legal in a NeXus identifier; no `validItemName` rule exists in `nexusformat` or the repo, so the only support was an unsourced code comment. Same missing schema doc spec 10 tracks. | **Decision 23** — adopt a rule needing no external spec |
| F4.3-4, F5.1-4 | 04 | A5 | Python-identifier rule; `$` is not legal, contrary to expectation. **F4.4's "the doubling cost is not avoidable" was wrong and unprobed** — an `__` introducer avoids it entirely: 116 of 185 names encode verbatim, 0 need an escaped underscore. Also corrects a miscount (37 names contain `_`, not 47 — that was the occurrence count). | **Decision 23** |
| F2.1 | 07 | A4 | The two basename branches disagree on double extensions — and 07's own Verification case is one where they agree, so it cannot detect it. | Change the test case |
| F2.1-2 | README | A6 | All nine diverged pairs read; every divergence acknowledged. Five identical sections remain unconfirmed by this axis. | Closed |
| F1.1 | 09 | A3 | **No defects.** All 21 citations land exactly. | — |
| F1.2 | 10 | A3 | The `filesSelected` citation points at the slots, not the handler. | Fixed in place |
| F1.3 | 10 | A1 | `## Tests` names no test file, for the one spec adding new UI. | **Decision 21** scope |
| F2.1-11 | 04b | A1/A2/A3/A4/A5 | **Eleven from the implementation pass.** The two config keys spec 01 was said to deliver were never shipped; `append_hidra_project` is unusable as the merge and the merged scan-point axis cannot be a `SubRuns` at all; the vendored schema answers Q1; a Python-string sample log cannot be written, and even the writable form returns `bytes`; `merge_workspaces` cannot merge two inputs sharing a compound key. | **Decisions 28-31** |
| F2.1-10 | 04c | A1/A2/A3/A4/A5 | **Ten from the implementation pass.** The scan-point family was written non-resizable, so the central mechanism was unavailable for half the groups in scope; a fixed-width `|S` log column truncates an appended value silently; `write()` was asked to mean both "add an entry" and "grow an entry"; the conflict policy enumerates three outcomes and needs seven, and two of the new checks were initially placed where they broke the no-op guarantee; `entry/start_time`/`end_time` are scan-point-family members the spec's table omits; `if name in group.NXdata` is dead code. | **Decisions 32-34** |
| F3.1-5 | 04c | A1/A3/A4/A5 | **Five from an adversarial review round, after the PR was otherwise finished.** `tail_append`'s own two refusals were never pre-flighted, so a two-theta-width mismatch left a half-grown entry that read back without error — the no-op property Follow-up 2 reported as closed; `_Sample._append_group` validated after mutating, under a comment naming that exact hazard; an appended workspace's detector masks were silently discarded; `data: NXdata = None` raises under `@validate_call_` when passed explicitly. | **Decision 35** |
| F1.1 | 01 | A4 | Two undocumented `neutrons_standard` behaviours: a write to `~/.pyrs/` on every load; and under a test env the resources root resolves **outside the repository**. | Flagged for a follow-up PR |
| F1.1 | 02 | A3 | Heading named `texture_fitting/model.py`, which does not exist. Landed with it intact. | Fixed in place |
| F1.1 | 03 | A7 | `combine_runs_model.py:17` → the cited call is at `:21`. Caught by the landing trigger. | Fixed in place |

**Not reported as consistent.** Five documents still carry an A5 gap — all of
them blocked on code that has not been written. Every A4 gap is now closed. See
§3.

---

## 2. Tool state after corrections

```console
$ pixi run python plans/audit-process/tools/check_links.py plans/NXstress-prod
# 0 finding(s) across 25 document(s)          exit 0
```

The three remaining `check_citations.py` findings are all in
`open-questions/06-manual-reduction-prereqs.md` — bare `Lnn` tokens whose
paragraph names no file. **Reported, never guessed**; left as-is because that
document is a stakeholder Q&A record, not specification.

Round one's own Follow-ups generated **22** citation findings when re-run
through `check_citations.py` — section 7.7's self-check property working on the
audit itself. All were fixed: historical line numbers quoted in correction
tables are now written as plain digits, so a superseded value is not
citation-shaped. Round two's Follow-ups generated **zero**.

The tool also gained a capability round one lacked: a citation naming a sibling
plan document (`README.md:729`) now resolves against the series directory rather
than being reported as a missing source file.

`check_ownership.py` and `check_drift.py` still exit 1: their remaining output is
the recorded-not-fixed material in F1.6, F1.8 and F1.9, plus the boilerplate
`process.md` §7.4 predicts.

**Updated 2026-09-30 (04b PR):** `check_citations.py` resolves 388 citations, still with the same three `open-questions/06` findings; `check_ownership.py` goes from 9 findings to 10, the new row being `_discriminator.py`, which 04b's own heading never path-qualified because Q5 left the module's location open. Recorded in 04b's Follow-up 2. Test tiers re-measured after the PR: `test-unit` 360 passed / 148 deselected · `test-integration` 102 passed / 28 skipped / 2 xfailed · `test-gui` 16 passed.

**Updated 2026-10-01 (04c PR):** `check_ownership.py` holds at 10 findings — 04c's
row gained `_definitions.py` and `_instrument.py` in the same pass that added them
to the code, so no new row appeared. Test tiers re-measured after the PR:
`test-unit` **412 passed** / 150 deselected · `test-integration` 104 passed /
28 skipped / 2 xfailed · `test-gui` 16 passed.

**Updated 2026-10-15 (human PR review of 04 + 04b + 04c, batch 28).** Measured,
not carried forward:

| Tool | Exit | Result |
|---|---|---|
| `check_links.py` | 0 | 0 findings across 25 documents |
| `check_citations.py` | 1 | 3 problems out of **502** resolved citations — still the same three bare-`Lnn` tokens in `open-questions/06`, and nothing else |
| `check_ownership.py` | 1 | 10 findings over 8 rows, 12 subspecs — unchanged accepted residue |
| `check_drift.py` | 1 | 10 of 14 sections diverge — unchanged |

Test tiers: `test-unit` **547 passed** / 158 deselected · `test-integration`
**112 passed** / 28 skipped / 2 xfailed · `test-gui` **16 passed**.

The citation total rose from 388 to 502 because this review's own `## Follow-up`
sections are themselves scanned — §7.7's self-check property, the same mechanism
that caught 22 findings in round one's Follow-ups. **The three remaining
problems are the same three throughout**, which is the point: a count that moves
while the problem set does not is growth, not regression.

`manifest.py` gained `[exclude] dirs` in this batch, so `prompts/` is excluded on
a directory basis rather than relying on it happening to hold no `*.md` — see
§4.

**Test tiers re-measured at `68639d3b`** under `QT_QPA_PLATFORM=offscreen`,
rather than carried forward from the `8634088a` figures — identical:
`test-unit` 298 passed / 140 deselected · `test-integration` 94 passed /
28 skipped / 2 xfailed · `test-gui` 16 passed.

---

## 3. What this pass could not cover

Stated so gaps are not mistaken for coverage.

| Claim cluster | Axis | Why |
|---|---|---|
| ~~04b's N-workspace write/read, discriminator resolution, scan-point-family split, the `≥1 PeakCollection when N>1` invariant~~ | A5 | **Closed 2026-09-30 by the 04b PR.** Listed here so the first round's gap is not lost from the record. |
| ~~04c's conflict classification, Case-A/B dispatch, `entry_number` targeting~~ | A5 | **Closed 2026-10-01 by the 04c PR.** Listed here so the first round's gap is not lost from the record. The note that "the *mechanism* beneath it **is** covered" is the part that proved misleading — it was covered for the peak-index family only. |
| 08's fit-spectrum reconstruction, `beam_intensity_profile`, `DENEXDetectorGeometry` repairs | A5 | Net-new PyRS code. 08's claims about the **current** broken state are A3 and were checked. |
| ~~`uncertainties` propagation; Mantid diagonal fit errors~~ | A4 | **Closed in round two** — see 08's Follow-up 2. Listed here only so the first round's gap is not lost from the record. |
| `STRESS_FIELD` shape (`_sample.py:272`) | A4/A5 | Externally blocked — no file in the repository carries the log. Genuinely exempt. |
| The NXstress `.xml`/`.html` schema doc | A4 | Not in the repo. The absence *is* the finding. |
| `open-questions/` prose (10 of 12) | A1 | Scope decision for this pass; mechanically checked, not read. |

**"Audited" is not a terminal state.** This pass eliminates staleness and
self-contradiction. It cannot eliminate false assumptions about library or
cross-module behaviour — only probing does, and seven clusters above are
un-probeable until their code exists. Budget a document-correction pass per PR
as normal cost.

---

## 4. Scope decisions taken during this pass

- **`open-questions/` left in the manifest as-is.** All twelve remain scanned —
  they hold 33 of the corpus's 107 links, the one live `#4-decisions-log` anchor
  case, and the hardest citation cases in the series. Only 04b Q6 and 04c Q5
  were read as prose. Worth noting for a future pass: `[exclude]` controls
  *scanning* and `[scope]` controls *reading depth*, and the manifest has no
  tier meaning "classified and scanned but not read" — which is exactly what
  `open-questions/` now is.
- **`prompts/` is now excluded by directory, not by accident (2026-10-15).**
  It was previously uncovered on the reasoning that it "holds no `*.md` (only
  `append.prompt.txt`)" — a parenthetical that was itself wrong, the directory
  holding two `.txt` files. More to the point the reasoning was **load-bearing
  on a coincidence**: the day a `*.md` landed there, the loader would fail,
  because `prompts/*.md` matches zero files today and the zero-match rule makes
  that pattern an error. A glob cannot express "and anything added here later".
  `manifest.py` gained `[exclude] dirs`, whose staleness check is that the
  directory exists — the same guarantee `paths` gets from zero-match, expressed
  for a directory. Both halves verified: a `*.md` dropped into `prompts/` leaves
  the document count at 25, and a `dirs` entry naming a missing directory fails
  at load with a message saying why. `prompts/` is the author's working
  directory — `implementation-prompt.txt` is adjusted before each subspec's
  implementation begins — so it is an input to the work, not a record of it.
- **`open-questions/` stays scanned.** Excluding it was considered in the same
  batch and **not done**, because `series.open_questions` classifies those
  documents and `[exclude]` would be a second classification — which the loader
  rejects, loudly and correctly. Making it possible would mean changing
  `manifest.py`'s classification precedence mid-review. Two reasons not to:
  the series rule is that the audit process is not altered while it is running,
  and the three `check_citations` findings that remain all live in
  `open-questions/06`, so excluding it would take the tool to **0 findings** for
  a reason that has nothing to do with the findings being fixed.
- **`manifest.py` gained an `audit_output` classification.** `series.probes` and
  `series.findings` are declared as output paths but were never wired into the
  exhaustiveness rule, so the loader failed the moment this pass wrote
  `probes/README.md`. They are now classified and deliberately **not** scanned.
  §7.7's self-check property is unaffected: `## Follow-up` sections live inside
  the plan documents, which are scanned — and re-running `landing_trigger.py`
  after this pass shows citation counts rising from 65 to 112 precisely because
  this audit's own citations are now being checked.

---

## 5. Invariants flagged, not written

An audit flags; the implementing PR writes. Most of the code these would guard
does not exist yet, so writing them now produces tests with no subject. Tiers
follow `CLAUDE.md`.

| # | Invariant | Owner PR | Possible from | Tier |
|---|---|---|---|---|
| 1 | `_peaks.py`'s splitter enforces contiguity and monotonic `scan_point` and **nothing more** — with the rule reproduced locally, so it fails when `_peaks.py` changes rather than tracking it | 04b | **WRITTEN** (04b PR, 2026-09-30) | unit |
| 2 | Tail-append grows each dataset by exactly N, leaves existing rows unchanged, and a pre-resize abort is byte-neutral | 04c | **WRITTEN** (04c PR, 2026-10-01) | unit — *not* the integration tier recorded here; see Decisions row 34 |
| 3 | `neutrons_standard.config` is never imported before `init("pyrs")` — a source scan, since the failure is at import time | follow-up to 01 | now | unit |
| 4 | `setEnabled(False)` leaves a `QAction` visible — a **third-party API-surface pin**, a net-new shape for this repo | 10 | 02 | gui + integration |
| 5 | No viewer uses `setVisible` for a format-gated action — a **convention scan**, also net-new here | 10 | 02 | unit |
| 6 | `pyrs/projectfile/file_object.py` carries no legacy-log-name `FIXME` | 04 | **WRITTEN** (04 PR, 2026-09-29) | unit |
| 7 | `encode(name)` matches `VALID_ITEM_NAME` and fits the 63-char cap, for every name — *restated: was `isidentifier()`, superseded by Decisions row 25* | 04 | **WRITTEN** (04 PR, 2026-09-29) | unit |
| 8 | `decode(encode(name)) == name` over adversarial inputs (`a_3Ab`, `__`, `_3A`) and the real `tests/data` log names | 04 | **WRITTEN** (04 PR, 2026-09-29) | unit |
| 9 | `encode` is injective — no two distinct inputs share an output (regression for F3.1) | 04 | **WRITTEN** (04 PR, 2026-09-29) | unit |
| 10 | Every transformation written is reachable by following `depends_on` from the detector | 04 | **WRITTEN** (04 PR, 2026-09-29) | unit |
| 11 | Default and named masks round-trip, and `DEFAULT_TAG` never leaks into the workspace as a mask name | 04 | **WRITTEN** (04 PR, 2026-09-29) | unit |
| 12 | Every per-scan-point dataset the writer emits is resizable — a **sweep** of the written entry, not a list of field names | 04c | **WRITTEN** (04c PR, 2026-10-01) | unit |
| 13 | Every refusal an append can make is reachable from `NXstress.write` and leaves the file byte-for-byte unchanged — asserted by `digest()`, not by `pytest.raises` | 04c | **WRITTEN** (04c PR, 2026-10-01) | unit |

Item 13 is the one that matters most and the one nothing asked for. Its absence
is why F3.1 shipped: the two-theta refusal *was* tested, at the `tail_append`
level and at the `_Diffractogram` level, and neither test went through
`NXstress.write`, so neither could see the 27 datasets that had already grown.
`pytest.raises` passed throughout.

Item 12 was not on the flagged list, and is the one this audit most wishes had
been: F2.1 is exactly the defect it catches, and nothing in the series was
looking for it. Its companion test adds a fixed-size field *locally* rather than
through the writer, per `process.md` §5.3, so the detector is checked against
something other than the code under test.

Items 10 and 11 were not on the flagged list: 10 came out of probing the chain
during implementation (Decisions row 26), and 11 is the test spec 04's `_fit.py`
item asked for, which the existing round-trip fixture did not cover — it sets
`with_masks=True` but no `mask_names`, so only the default mask was ever exercised.

**A tier note on 7-9.** They are recorded as unit, and the unit tests do exhaust
the rule — every byte value at every position, plus escape-shaped inputs. The
*real-namespace* sweep those rows describe reads `tests/data`, which `CLAUDE.md`
classifies as integration; it stays in the probe rather than being re-tiered, so
the guarantee is checked on every commit and the measurement stays where it was.

`process.md` §5.1 is explicit that nothing in PyRS yet pins a third-party API
surface or scans source for a convention. Items 3, 4 and 5 are all net-new
shapes; the idiom to copy for iteration-based guarantees is
[`tests/unit/pyrs/utilities/NXstress/test_definitions.py`](../../../tests/unit/pyrs/utilities/NXstress/test_definitions.py).
