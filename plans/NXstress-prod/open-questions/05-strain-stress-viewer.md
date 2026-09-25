# Open Questions — 05 StrainStressViewer NXstress Hookup

**Spec:** [05-strain-stress-viewer.md](../05-strain-stress-viewer.md)
**Blocking:** No — Q1 and Q2 below are resolved by
[04b — Multi-workspace NXstress I/O](../04b-multi-workspace-nxstress.md).
04b's own schema question (see
`open-questions/04b-multi-workspace-nxstress.md` Q1) is itself downgraded
from a blocking gate to a tracked follow-up, so nothing here blocks this
spec either.

---

<a id="q1"></a>

## Q1 — RESOLVED by 04b: is a `direction` axis on `NXreflections` schema-conformant?
Originally framed as specific to this spec's `direction` axis. Chris's
response below (workspace-per-direction, provenance on output) is exactly
the general N-workspace, discriminator-indexed mechanism now scoped as its
own spec, [04b](../04b-multi-workspace-nxstress.md), with `direction` as one
discriminator field rather than a StrainStress-specific extension.

The general form of this question — does `NXreflections` permit additional
index columns at all — moved to 04b (see
`open-questions/04b-multi-workspace-nxstress.md` Q1), where it is now
downgraded from a blocking gate to a tracked follow-up: `_peaks.py` already
writes non-required columns (`mask`, `scan_point`, etc.) onto
`NXreflections`, which is the same category of extension a discriminator
column would be. Cross-check against `NXstress.html` and the
`nexusformat`-org validator once both are added to the repo. This spec's
remaining, narrower question is only whether `direction` (values
`"11"`/`"22"`/`"33"`) is the right name and semantics for one such column —
not the schema-conformance question itself.

**Response from Chris:** The `direction` can be defined based on the description logs. But the stress-strain viewer does not automatically search for this type of log. Instead, a User defines the `direction` in the GUI or API. We can auto populate anything that is then udpated when the stress-strain calculation is completed.

---

## Q2 — RESOLVED by 04b: can `HidraWorkspace` represent direction-merged sample-log content, or does it need a new container?

Chris's response settles this: a user provides one `HidraWorkspace` per
direction, and NXstress combines them, with provenance, on write — no
direction-merged `HidraWorkspace` container is needed. That is exactly
[04b](../04b-multi-workspace-nxstress.md)'s `list[HidraWorkspace]` signature.
This spec's PyRS Changes section no longer carries a direction-aware
container item; it depends on 04b instead.

**Response from Chris:** The current approach is that a user would provide 1 `HidraWorkspace` per direction. A final NXstress output should contain the 2/3 directions with provenance about the calculation.

Could we define the nexusformat to have the following approach? A user provides 3 nexus formated files with a single direction and with a header note that states `NXstrain`. The stress calulator would offer the option to output a NXstress file with 2 or 3 directions with provenance about the calculation.

**Note:** the second paragraph — N single-direction `NXstrain` files in, one
`NXstress` file out — is a file-level concern one layer above 04b's
workspace-level merge, and is **not** folded into 04b. It remains an open
question for this spec to pick up at its own kickoff, carrying its own
`definition`-field and provenance questions (not yet written up as a
numbered question here).

---

## Q3 — Which specific `NotImplementedError` methods in `fields.py` will the round-trip test actually exercise?

The spec deliberately scopes this down: *"resolve any `NotImplementedError`
in `StrainField` / `StressField` that is exercised by the read-back path ...
Fix only the methods the round-trip test actually calls"* — but the actual
set of methods among the 11 flagged in the README (`fields.py` L239, 611,
816, 927, 956, 960, 964, 983, 986, 994, 1184) isn't known until the test is
written.

**Why it matters:** the spec's own scope and effort estimate depend on this
— if the read-back path for a direction-indexed `StressField` happens to
exercise most of those 11 methods, this becomes a much larger PyRS change
than "fix only what's hit" implies at a glance.

**Next step:** write the round-trip test's assertions first (against a stub
`StressField` reconstruction), run it, and let the resulting
`NotImplementedError` tracebacks enumerate the actual method list.

**Chris:** Do existing tests define the `NotImplementedError`?

---

## Q4 — RESOLVED: how does `direction` actually travel between the viewer and NXstress?

04b resolves discriminator values *from* the workspace object itself
(a matching `@property`, else a `SampleLogs` fallback) — it does not accept
them as a separate `write()` argument. But `HidraWorkspace` had no notion of
direction at all, and the viewer tracked it purely at the model level
(`filenames_11/22/33` slots), not on the workspace.

**Decided:** this spec adds a real, settable `direction` `@property` to
`HidraWorkspace` (get and set), per 04b's own forward-looking convention
note. This also required a small correction to 04b itself: its discriminator
resolver is now explicitly **bidirectional** — a "get" half at write time
(unchanged) and a symmetric "set" half at read time, so each workspace
`NXstress.read()` reconstructs has `.direction` already populated, letting
the viewer select `next(ws for ws in wss if ws.direction == direction)`
with no extra plumbing. See 04b's "Discriminator value resolution" section
for the updated bidirectional sketch.

---

## Q5 — RESOLVED: config precondition and default

04b raises if `discriminator_fields` is empty and more than one workspace is
passed (unless `merge_workspaces: true`), but says nothing about the case
where `discriminator_fields` is *non-empty but doesn't include `"direction"`*
— which would most likely surface as an opaque duplicate-index error several
layers removed from the real cause (all three per-direction workspaces
resolving to the same discriminator tuple).

**Decided:** `save_as_nxstress` checks
`"direction" in Config["nxstress.discriminator_fields"]` before calling
`write()`, raising a clear, StrainStress-specific error if absent. The
shipped default in `pyrs/resources/application.yml` (spec 01) is updated to
`nxstress.discriminator_fields: ["direction"]` (was `[]`), so this works
out of the box for a fresh install rather than requiring manual opt-in.
