# Open Questions — Index

Companion to the [NXstress GUI Hookup plan](../README.md). Each file here
lists the open questions specific to the correspondingly-numbered sub-spec —
things the spec text flags as unresolved, ambiguous, or contingent on an
audit/decision that hasn't happened yet. Resolved decisions already live in
the plan's own [Decisions Log](../README.md#4-decisions-log); this index is
for what's still outstanding.

| # | Spec | Blocking? | Question count |
|---|---|---|---|
| [02](02-peak-and-texture-nxstress.md) | NXstress I/O for PeakFitting & Texture viewers | No | 1 |
| [03](03-combine-runs-nxstress.md) | NXstress I/O for CombineRuns viewer | No | 1 |
| [04](04-nxstress-internal-cleanup.md) | NXstress internal cleanup (Phase 2 TODOs) | No | 4 |
| [04b](04b-multi-workspace-nxstress.md) | Multi-workspace NXstress I/O | No | 7 |
| [04c](04c-nxstress-append.md) | NXstress append mode (library only) | No | 6 |
| [05](05-strain-stress-viewer.md) | StrainStressViewer NXstress hookup | No | 5 |
| [06](06-manual-reduction-prereqs.md) | Manual reduction PyRS prerequisites (not required for NXstress) | No | 2 |
| [07](07-manual-reduction-nxstress.md) | ManualReductionViewer NXstress hookup | No | 3 |
| [08](08-fit-spectrum-prereqs.md) | Reconstructed fit spectrum & calibration fidelity (PyRS) | Partially | 3 |
| [09](09-fit-spectrum-nxstress.md) | Fit spectrum & calibration fidelity (NXstress) | No | 4 |
| [10](10-flip-defaults.md) | Flip defaults — NXstress becomes primary | Partially | 3 |

**No hard implementation gates remain in specs 01–07 or 09.**
[04b, Q1](04b-multi-workspace-nxstress.md#q1) — whether `NXreflections`
permits additional index columns at all — was the plan's one hard
implementation gate (generalizing what was previously tracked as 05-Q1) but
is now downgraded to a tracked follow-up: `_peaks.py` already writes
non-required columns (`mask`, `scan_point`, etc.) onto `NXreflections`, the
same category of extension a discriminator column would be. Verify against
`NXstress.html` and the `nexusformat`-org validator once both are added to
the repo — tracked as a concrete reminder in
[10-flip-defaults.md](../10-flip-defaults.md)'s Overview (see also
[10, Q3](10-flip-defaults.md)), since neither artifact exists yet — and
record the outcome; implementation of 04b, 04c, and Phase 3 proceeds in the
meantime.

**Two genuine blockers remain:**
- [08, Q3](08-fit-spectrum-prereqs.md) — `STRESS_FIELD`'s shape cannot be
  verified from anything currently in the repository; investigated
  directly, not merely unstarted. Blocks only that one item within spec 08
  (and its pass-through in spec 09) — not the rest of Phase 5.
- [10, Q1](10-flip-defaults.md) — the Phase-6 trigger criteria (all
  round-trip tests passing, a scientist-signed-off real-data smoke test, no
  open correctness bugs) have no concrete owner or tracked checklist yet.
  This gates *entering* Phase 6, not anything in Phases 1–5.
