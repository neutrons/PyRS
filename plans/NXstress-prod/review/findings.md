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

Key: **✓** covered · **~** partial (scope stated below) · **✗** not run ·
**n/a** not applicable · **ex** exempt, with the exemption recorded

| Document | A1 self | A2 siblings | A3 codebase | A4 library | A5 runtime | A6 draft | A7 fresh |
|---|---|---|---|---|---|---|---|
| `README.md` | ✓ | ✓ | ✓ | ✓ | ~ | ✓ | ✓ |
| `01-config-and-test-infra-PR.md` | ✗ | ✗ | ✗ | ✓ | ✗ | n/a | ✓ |
| `02-peak-and-texture-nxstress.md` | ✗ | ✗ | ~ | ✗ | ✗ | n/a | ✓ |
| `03-combine-runs-nxstress.md` | ✗ | ✗ | ~ | ✗ | ✗ | n/a | ✓ |
| `04-nxstress-internal-cleanup.md` | ✓ | ✓ | ✓ | ✓ | ✓ | n/a | ✓ |
| `04b-multi-workspace-nxstress.md` | ✓ | ✓ | ✓ | ✓ | ~ | n/a | ✓ |
| `04c-nxstress-append.md` | ✓ | ✓ | ✓ | ✓ | ~ | n/a | ✓ |
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
| F1.1 | 01 | A4 | Two undocumented `neutrons_standard` behaviours: a write to `~/.pyrs/` on every load; and under a test env the resources root resolves **outside the repository**. | Flagged for a follow-up PR |
| F1.1 | 02 | A3 | Heading named `texture_fitting/model.py`, which does not exist. Landed with it intact. | Fixed in place |
| F1.1 | 03 | A7 | `combine_runs_model.py:17` → the cited call is at `:21`. Caught by the landing trigger. | Fixed in place |

**Not reported as consistent.** Six documents still carry an A5 gap — all of
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

**Test tiers re-measured at `68639d3b`** under `QT_QPA_PLATFORM=offscreen`,
rather than carried forward from the `8634088a` figures — identical:
`test-unit` 298 passed / 140 deselected · `test-integration` 94 passed /
28 skipped / 2 xfailed · `test-gui` 16 passed.

---

## 3. What this pass could not cover

Stated so gaps are not mistaken for coverage.

| Claim cluster | Axis | Why |
|---|---|---|
| 04b's N-workspace write/read, discriminator resolution, scan-point-family split, the `≥1 PeakCollection when N>1` invariant | A5 | The code does not exist. `write` takes a single `HidraWorkspace` today — pinned by `a5_nxstress_roundtrip.py`. |
| 04c's conflict classification, Case-A/B dispatch, `entry_number` targeting | A5 | Unimplemented. The *mechanism* beneath it **is** covered (`a4_h5py_nexusformat_append.py`); the dispatch is not. |
| 08's fit-spectrum reconstruction, `beam_intensity_profile`, `DENEXDetectorGeometry` repairs | A5 | Net-new PyRS code. 08's claims about the **current** broken state are A3 and were checked. |
| ~~`uncertainties` propagation; Mantid diagonal fit errors~~ | A4 | **Closed in round two** — see 08's Follow-up 2. Listed here only so the first round's gap is not lost from the record. |
| `STRESS_FIELD` shape (`_sample.py:107`) | A4/A5 | Externally blocked — no file in the repository carries the log. Genuinely exempt. |
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
| 1 | `_peaks.py`'s splitter enforces contiguity and monotonic `scan_point` and **nothing more** — with the rule reproduced locally, so it fails when `_peaks.py` changes rather than tracking it | 04b | now | unit |
| 2 | Tail-append grows each dataset by exactly N, leaves existing rows unchanged, and a pre-resize abort is byte-neutral | 04c | 04c | integration |
| 3 | `neutrons_standard.config` is never imported before `init("pyrs")` — a source scan, since the failure is at import time | follow-up to 01 | now | unit |
| 4 | `setEnabled(False)` leaves a `QAction` visible — a **third-party API-surface pin**, a net-new shape for this repo | 10 | 02 | gui + integration |
| 5 | No viewer uses `setVisible` for a format-gated action — a **convention scan**, also net-new here | 10 | 02 | unit |
| 6 | `pyrs/projectfile/file_object.py` carries no legacy-log-name `FIXME` | 04 | now | unit |
| 7 | `encode(name).isidentifier()` for every name, including leading-digit and non-ASCII | 04 | now | unit |
| 8 | `decode(encode(name)) == name` over adversarial inputs (`a_3Ab`, `__`, `_3A`) and the real `tests/data` log names | 04 | now | unit |
| 9 | `encode` is injective — no two distinct inputs share an output (regression for F3.1) | 04 | now | unit |

`process.md` §5.1 is explicit that nothing in PyRS yet pins a third-party API
surface or scans source for a convention. Items 3, 4 and 5 are all net-new
shapes; the idiom to copy for iteration-based guarantees is
[`tests/unit/pyrs/utilities/NXstress/test_definitions.py`](../../../tests/unit/pyrs/utilities/NXstress/test_definitions.py).
