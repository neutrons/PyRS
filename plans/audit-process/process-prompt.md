# Agent prompt -- audit pass over a PyRS plan series

> **This header is for you, not the agent. Do not send it.**
> Send everything from the `RETARGET` marker below to the end of the file.

### Pre-launch checklist

Work down the RETARGET block. Three fields go stale on their own; the rest only
change when you point this at a different series.

| | Field | Check before every launch |
|---|---|---|
| [ ] | `CURRENT TEST STATE` | **Goes stale.** Re-run the three tiers if the branch moved. Never send an assumed value -- an audit that starts from "tests pass" and is wrong wastes the whole pass. |
| [ ] | `KNOWN FINDINGS` | **Goes stale.** These are the calibration set; the line numbers behind them live in process.md section 2 and drift whenever the cited files change. The prompt tells the agent to re-verify, but confirm section 2 has been re-checked since the last landing. |
| [ ] | `FULL PASS` / `FRESHNESS ONLY` | **Goes stale as subspecs land.** A landed subspec moves from the first list to the second. Every subspec must appear in exactly one. |
| [ ] | `SERIES` / `WHAT IT IS` | Only when retargeting. `SERIES` must contain an `audit.toml`. |
| [ ] | `PRIOR AUDIT EVIDENCE` | Only when retargeting -- it sets the agent's expectation of what earlier passes already covered. |
| [ ] | `PROBE CANDIDATES` | Only when retargeting. A starting list, never a checklist. |
| [ ] | `HIGHEST A7 EXPOSURE` | Only when retargeting -- the subspec whose referents have actually moved. |

**To retarget at a different series:** edit the RETARGET block and nothing else.
Everything after it is series-agnostic by construction; if you find yourself
editing below the marker, that is a bug in this file.

---

<!-- ===== RETARGET: edit these nine fields ===== -->

    SERIES                plans/NXstress-prod

    WHAT IT IS            wiring the already-built pyrs/utilities/NXstress/ library
                          (a NeXus-compliant residual-stress schema) into the PyRS
                          Qt GUI's existing I/O endpoints

    FULL PASS             README.md, 04, 04b, 04c, 05, 06, 07, 08, 09, 10

    FRESHNESS ONLY        01, 02, 03 -- landed; their claims are settled by shipped
                          code and passing tests, so they get A7 and nothing else

    PRIOR AUDIT EVIDENCE  18 entries in README.md's "## 4. Decisions Log", 9 of them
                          recording a correction found by re-reading against the
                          codebase. Those rounds covered A2 and A3.

    CURRENT TEST STATE    All three tiers green, measured at 8634088a on 2026-09-25
                          (QT_QPA_PLATFORM=offscreen):
                            pixi run test-unit         298 passed, 140 deselected
                            pixi run test-integration   94 passed, 28 skipped, 2 xfailed
                            pixi run test-gui           16 passed
                          Re-measure if the branch has moved; do not carry this
                          forward as an assumption.

    KNOWN FINDINGS        The five in process.md section 2. They are the CALIBRATION
                          SET for the tools you build: 7 dead links across 5
                          documents, 4 drifted line citations in one README table
                          cell plus 2 more at README.md:58, a cited file that does
                          not exist (02:83), and a superseded draft specifying a
                          different architecture under the same section number.

    PROBE CANDIDATES      Starting list, NOT exhaustive:
                          - neutrons_standard.Config -- init()-before-import
                            ordering, the claimed init() race, env= deep-merge, and
                            the required pyrs/resources/application.yml location
                            (README section 2.3)
                          - h5py resize-then-assign on a zero-sized resizable
                            dataset -- 04c's entire tail-append design rests on it
                          - nexusformat 1.0.8 validator capability (Decisions 16)
                          - qtpy/PyQt6 -- lazy self.statusBar(); setEnabled leaves a
                            QAction visible (spec 10). NOTE: PyQt5 is ABSENT here.
                          - NXstress multi-workspace round trip -- this is A5, not
                            A4: the library is in-repo and already landed

    HIGHEST A7 EXPOSURE   04 -- it depends on 02 and 03, both landed, so its
                          referents have actually moved. 05 through 10 depend on
                          work that has not shipped yet.

<!-- ===== end RETARGET ===== -->

---

We are auditing the plan series named in SERIES above, which specifies WHAT IT IS.
The directory holds a grounded `README.md`, numbered subspecs each corresponding to
one planned PR, a parallel `open-questions/` directory, an `audit.toml` manifest,
and an `archive/` holding the pre-codebase draft the README superseded.

**Do not implement any subspec.** The deliverable is corrected specification
documents plus the evidence behind the corrections.

Read, in this order:

1. `CLAUDE.md`, section **"Auditing a Plan or Subspec"** -- the seven axes an audit
   must cover and the anti-patterns to avoid.
2. `plans/audit-process/process.md` -- why that definition exists, the failure
   taxonomy behind it, and **section 7, the toolkit build specification**.

**Build the toolkit first; it does not exist yet.** Implement
`plans/audit-process/tools/` per process.md section 7, against the manifest at
`<SERIES>/audit.toml`. Then **tune it before you record anything**: KNOWN FINDINGS
above are the calibration set. **Re-verify them first** -- process.md section 2
gives the command for each, and its line numbers are themselves claims about a
moving branch. Tuning a checker against a stale expected value is the defect this
whole process exists to prevent. A link checker that does not report exactly those
seven dead links is not ready to be trusted, and neither is a citation checker that
misses those drifts. Every tool in this class needs a tuning round against
known-good input -- a tool whose first output is wrong in a plausible-looking way
is worse than no tool, because it buries real defects in noise.

Calibration for your expectations: see PRIOR AUDIT EVIDENCE. Those passes were
conscientious and they covered two axes. **Assume A1, A4, A5, A6 and A7 have never
been run, and that re-reading these documents the way they have already been
re-read will find nothing.**

Then work in this order, which is deliberately *not* the order the documents are in:

1. **A4/A5 probes.** The defects that cost weeks, and the ones reading cannot find.
   Write and run a probe for every claim about third-party library behaviour or
   about how another module will treat our output. Paste the real observed output
   next to the claim. Treat PROBE CANDIDATES as a starting list, not a checklist.
2. **A1 -- each document against itself.** Every prose claim against every code
   block, table and link in the *same* document.
3. **A2 -- documents against their siblings.** Shared paths, config keys and
   dependency ordering.
4. **A3 and A6** for anything the first three passes disturbed.

Documents under FRESHNESS ONLY get an A7 check and nothing more.

Deliverables:

- **Corrections appended as `## Follow-up N` sections** to the document being
  audited, per process.md section 5.5 -- never rewritten in place, so the record of
  what was believed survives. The one exception is a `path:LINE` citation, which is
  a pointer rather than a belief: fix it in the body **and** log the finding. Do
  **not** put findings in `open-questions/` or the README Decisions Log; both are
  human-facing records with different audiences.
- **Probes written** under `<SERIES>/probes/`, each with its real output pasted
  into the document at the claim it supports, plus the index at
  `probes/README.md`.
- **A findings document** at `<SERIES>/review/findings.md`, whose **section 0 is a
  required coverage matrix**: every document against all seven axes, marked
  covered / partial / not run. Without it, "which axes were actually covered on
  06?" is unanswerable and the rule that a partial audit is reported as partial has
  nothing to attach to. Never report a partial audit as "consistent".
- **Invariants that should become tests: flagged, not written.** An audit flags;
  the implementing PR writes. Most of the code they would guard does not exist yet,
  so writing them now produces tests with no subject.

**Do not commit anything, and do not stage anything.** Leave the working tree for
the user to review; they write their own commit messages. This holds for every
artifact above -- probes, Follow-up sections, corrected citations and the findings
document alike.

Start by proposing a plan for the pass before beginning it.
