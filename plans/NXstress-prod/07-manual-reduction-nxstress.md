# 07 — ManualReductionViewer NXstress Hookup

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 4
**Depends on:** — (none within this plan; see Overview)

---

## Overview

Wire NXstress into `reduce_hidra_workflow`'s existing, automatic save step,
so a reduction can produce a `.nxs` file in addition to (or instead of) the
current `.h5` output.

**Corrected hookup point.** Earlier drafts of this spec targeted
`HB2BReductionManager.save_project` (via spec 06). Both parts of that were
wrong: the class is `ReductionController`
(`pyrs/interface/manual_reduction/pyrs_api.py`), not `HB2BReductionManager`
(an unrelated class elsewhere); and `save_project` has **zero callers
anywhere in the codebase** — it isn't where saving actually happens. The
real, currently-functional save path is `reduce_hidra_workflow`'s automatic
`reducer.save_diffraction_data(project_file_name)` call
(`pyrs_api.py:348`), which runs as an inherent side effect of every
reduction — there is no separate "Save"/"Save as…" dialog or menu action in
this viewer at all. **This spec no longer depends on spec 06** — neither of
spec 06's items (the orphaned `save_project` stub; two unrelated
`nexus_conversion.py` input-parsing gaps) touches this save step.

**No GUI action, therefore no auto-write-both-formats concern.** The other
NXstress-wired viewers gate independent, manually-clicked actions by
`legacy_io.enable`/`nxstress.enable`, and a single click never writes both
formats — that rule exists to protect a user's expectation of what *their
click* does. Precisely: PeakFitting/Texture (spec 02) and CombineRuns
(spec 03) each gate two such actions ("Save" for `.h5`, "Save as
NXstress…" for `.nxs`); StrainStress (spec 05) gates only one
(`nxstress.enable` → "Save as NXstress…") since it never had a `.h5` Save
action to begin with — its existing save paths are CSV/JSON, unaffected by
either config flag. `reduce_hidra_workflow` has no click to protect the
meaning of: it writes automatically, so it simply **writes once per
currently-enabled format** — one file if only one format is enabled, both
files (same basename) if both are. This is not an ambiguity requiring a
tie-break; it's the direct, correct consequence of "both formats are
currently enabled" applied to a path with no per-click choice.

**Extension is never caller-controlled**, exactly as in the GUI viewers —
only the *basename* varies with `project_file_name` (auto-derived from the
input NeXus filename when `None`, or whatever base a caller supplies when
given explicitly). Any extension on a caller-supplied `project_file_name`
is stripped and ignored, never validated against and never a reason to
raise — the extension(s) actually written are always exactly whichever of
`nxstress.extension`/`legacy_io.extension` are currently enabled.

---

## Scope

**In scope:**
- `reduce_hidra_workflow` (`pyrs/interface/manual_reduction/pyrs_api.py`):
  replace the hardcoded `+ ".h5"` default-extension logic and the single
  `reducer.save_diffraction_data(project_file_name)` call with: derive the
  basename (from `project_file_name` if given, stripping any extension; or
  from the input NeXus filename if `None`), then write once per
  currently-enabled format (`legacy_io.enable` → `.h5` via
  `reducer.save_diffraction_data`; `nxstress.enable` → `.nxs` via
  `NXstress(path, "w").write([hidra_ws], [])` — a length-1 list, per 04b's
  signature, which lands in the Phase 2/3 bridge, before this Phase 4
  spec).
- Config validation reuse: `pyrs.utilities.config.validate_config()`'s
  existing "at least one format enabled" rule (spec 01) already guarantees
  this function always writes at least one file.
- Round-trip test: reduce a run with only `nxstress.enable: true`, confirm
  `.nxs` is written and readable; with both enabled, confirm both files are
  written with the same basename.

**Out of scope:**
- Any change to `ReductionController.save_project` (spec 06 — independent,
  not required here).
- Any new GUI dialog, menu action, or file-dialog filter — none exists to
  extend; this hookup is entirely inside `reduce_hidra_workflow`.
- Append mode (spec 04c) — not used by this pathway.
- Multi-workspace input (spec 04b) — manual reduction produces one
  workspace per save.
- Fit-spectrum data (spec 09).
- Detector-calibration fidelity fixes (spec 09).

---

## PyRS Changes

_None._ `reduce_hidra_workflow` is itself a PyRS/GUI-boundary function
(`pyrs/interface/manual_reduction/pyrs_api.py`), not a PyRS-core data-object
change — the edit lives entirely in the "NXstress Changes" section below.

---

## NXstress Changes

### `pyrs/interface/manual_reduction/pyrs_api.py`

- `reduce_hidra_workflow(nexus, output_dir, progressbar, ..., project_file_name=None)`:

  ```python
  from pyrs.utilities.config import Config

  if project_file_name is None:
      basename = os.path.basename(nexus).split(".")[0]
  else:
      basename = os.path.splitext(os.path.basename(project_file_name))[0]
  base_path = os.path.join(output_dir, basename)

  # ... existing NeXus conversion + reduction unchanged ...

  if Config["legacy_io.enable"]:
      reducer.save_diffraction_data(base_path + Config["legacy_io.extension"])
  if Config["nxstress.enable"]:
      with NXstress(base_path + Config["nxstress.extension"], "w") as nxs:
          nxs.write([hidra_ws], [])  # length-1 list, per 04b's signature
  ```

  (Illustrative — the existing file-exists/overwrite-permission checks at
  `pyrs_api.py:292-311` apply per resolved path, once per enabled format.)
- `peakss` is always `[]`: `ManualReductionModel`/`ReductionController` has
  no `PeakCollection` concept at all — reduction produces a `HidraWorkspace`
  only. This is a fully valid, already-supported input to `NXstress.write`
  (see spec 03's identical finding for `CombineRunsModel`).

---

## Tests

`tests/integration/test_nxstress_viewer_roundtrip.py` (extend):
- Manual-reduction save-as-NXstress round-trip: `nxstress.enable: true`,
  `legacy_io.enable: false` — confirm only a `.nxs` file is written and is
  readable via `NXstress.read()`.
- Both-enabled case: `nxstress.enable: true`, `legacy_io.enable: true` —
  confirm both a `.h5` and a `.nxs` file are written, same basename, both
  independently valid/readable.
- Legacy-only regression: `nxstress.enable: false`, `legacy_io.enable: true`
  (today's default) — confirm behavior is unchanged from the current
  hardcoded-`.h5` path.
- Extension-stripping: pass an explicit `project_file_name` with a
  mismatched extension (e.g. `"foo.nxs"` while only `legacy_io.enable` is
  true); confirm the extension is ignored and a correctly-suffixed `.h5`
  file is written under basename `foo` — no error raised.

---

## Delivered Feature

> **For end users and instrument scientists:**
> Reduction can now produce NeXus-compliant NXstress (`.nxs`) output, in
> addition to or instead of the existing `.h5` output — controlled purely
> by config (`nxstress.enable`, `legacy_io.enable`), not by any new button
> or menu action. With both formats enabled, every reduction writes both
> files automatically, under the same base name — useful for validating
> NXstress output against the established `.h5` format during the
> transition period.

---

## Verification

- Reduce a run with `nxstress.enable: true`, `legacy_io.enable: false`;
  confirm a `.nxs` file is written and readable.
- Reduce a run with both enabled; confirm both `.h5` and `.nxs` files are
  written, same basename.
- Reduce a run with the current default (`legacy_io.enable: true`,
  `nxstress.enable: false`); confirm behavior is unchanged from today.
- `pytest tests/integration/test_nxstress_viewer_roundtrip.py` — all pass
  (no regression in earlier specs).

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass)

**F1.1** (A1/A2) — `**Depends on:** — (none within this plan; see Overview)`.
- Referent: this document's own `## NXstress Changes` code block, and
  `README.md`'s `## Sub-specifications` table, whose row for 07 also reads `—`.
- Verdict: **wrong — this spec depends on 04b.** Its own code block calls
  `NXstress(path, "w").write([hidra_ws], [])` and annotates it "a length-1 list,
  **per 04b's signature**, which lands in the Phase 2/3 bridge, before this
  Phase 4 spec". That call cannot be written until 04b changes `write` to take a
  list — confirmed against the landed library, which today takes a single
  `HidraWorkspace`
  ([`probes/a5_nxstress_roundtrip.py`](probes/a5_nxstress_roundtrip.py)):

  ```console
    CLAIM   write takes a single HidraWorkspace today; 04b changes it to a list
    RESULT  NXstress.write(self, ws: pyrs.core.workspaces.HidraWorkspace,
                           peakss: list[pyrs.peaks.peak_collection.PeakCollection])
  ```

  The Overview's argument establishes only that 07 does not depend on **06**;
  the header generalised that to "nothing". `## Scope`'s "Out of scope:
  Multi-workspace input (spec 04b)" is about not using 04b's *features*, which is
  compatible with needing its *signature*.
- Action (for the implementing PR): set `**Depends on:** [04b](04b-multi-workspace-nxstress.md)`
  here, and change `README.md`'s §Sub-specifications row for 07 from `—` to
  `04b`. Phase ordering already satisfies the dependency (04b is Phase 2/3, this
  is Phase 4), so **this is a record defect, not a scheduling break** — but the
  `Depends on` column is the explicit ordering record and is what a reader
  trusts.

**F1.2** (A2) — `## Tests` and `## Verification` name
`tests/integration/test_nxstress_viewer_roundtrip.py`.
- Referent: `README.md:748-753`'s "Tests to extend/add" list.
- Verdict: the two disagree. README schedules
  `tests/integration/test_nxstress_reduction.py` ("new in Phase 4 — manual
  reduction, fresh-write only") for exactly this work, and
  `test_nxstress_viewer_roundtrip.py` appears nowhere in README §5. Specs 04b and
  05 name the same viewer-roundtrip file, so three specs share a file the plan's
  own inventory does not list. See the README's Follow-up 1 for the full picture.
- Action (for the implementing PR): pick one name and use it in both places.
  README's `test_nxstress_reduction.py` is the better fit here — this pathway has
  no viewer and no GUI action at all (this spec's own Overview says so), so
  filing it under "viewer roundtrip" is misleading.

**F1.3** (A1) — "Any extension on a caller-supplied `project_file_name` is
stripped and ignored".
- Referent: this document's own code block.
- Verdict: the two branches strip differently. `os.path.basename(nexus).split(".")[0]`
  strips **every** extension; `os.path.splitext(os.path.basename(project_file_name))[0]`
  strips only the **last**. For `foo.nxs.h5` these give `foo` and `foo.nxs`. HB2B
  NeXus inputs are commonly double-extensioned, so the aggressive first branch is
  presumably deliberate — which makes the unqualified prose, not the code, the
  defect. `## Verification`'s extension-stripping check uses a single-extension
  name (`"foo.nxs"`) and so cannot catch the difference.
- Action (for the implementing PR): state the rule as "the basename is taken up
  to the **first** dot for an auto-derived name, and with the **final**
  extension removed for a caller-supplied one" — or make both branches use
  `.split(".")[0]` and say so once. Either way, extend the Verification case to a
  double-extension name (`foo.nxs.h5`), which is the only one that distinguishes
  them.

**Checked and accurate — no action.** `pyrs_api.py:348` and `pyrs_api.py:292-311`
both land exactly on their claimed targets.

---

## Follow-up 2 — 2026-09-26 (closing the A4 gap left open by Follow-up 1)

**F2.1** (A4) — Follow-up 1 F1.3 reasoned that the two basename branches strip
extensions differently, but did not execute them. Probed now —
[`probes/a4_basename_extensions.py`](probes/a4_basename_extensions.py) — and
the reasoning holds, with one consequence F1.3 did not draw:

```console
  input                    auto-derived         caller-supplied      agree?
  ------------------------ -------------------- -------------------- ------
  HB2B_1234.nxs.h5         HB2B_1234            HB2B_1234.nxs        NO
  HB2B_1234.h5             HB2B_1234            HB2B_1234            yes
  foo.nxs                  foo                  foo                  yes
  foo                      foo                  foo                  yes
  run.2024.03.nxs.h5       run                  run.2024.03.nxs      NO

  CLAIM   spec 07's Verification case ('foo.nxs') detects the difference (claim 3)
  RESULT  auto-derived='foo', caller-supplied='foo' -- identical, so the proposed
          test CANNOT detect it. A double-extension name can: 'HB2B_1234.nxs.h5'
          -> 'HB2B_1234' vs 'HB2B_1234.nxs'.
```

The consequence: **the Verification case this spec proposes is the one case in
which the two branches agree.** A test written exactly as specified would pass
while the inconsistency remained. Note also that the third row shows
`.split(".")[0]` truncating at the *first* dot, so a run name containing dots
(`run.2024.03.nxs.h5`) collapses to `run` — an aggressive strip that is
presumably deliberate for HB2B NeXus inputs but is not what "the extension is
stripped" describes.
- Action (for the implementing PR), unchanged from F1.3 and now evidenced: state
  the two rules separately rather than as one, and change the Verification case
  to a double-extension name.
