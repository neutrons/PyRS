# 03 — NXstress I/O for CombineRuns Viewer

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 1
**Depends on:** [01 — Config infrastructure & test framework](01-config-and-test-infra-PR.md)

---

## Overview

Wire `NXstress` into the CombineRunsViewer's export path. After combining
multiple `HidraProjectFile`s into a merged workspace, users can now export the
result as a `.nxs` file in addition to the existing `.h5` export.

Because NXstress does not yet support appending to an existing NXentry (that
arrives in spec 04c, as a library-only capability with no GUI wiring), the
CombineRuns integration is **fresh-write only**: the user must combine
first, then export as a single NXstress file.

### Pre-merge stays — resolved, not a follow-on simplification

`open-questions/04b-multi-workspace-nxstress.md` Q3 floated dropping the
PyRS-level pre-merge (`combine_project_files`, via repeated
`HidraWorkspace.append_hidra_project` calls) once 04b's `list[HidraWorkspace]`
signature landed, on the theory that NXstress could do the same merge
itself. Now that 04b is fully specified, that's **not** worth doing:
`append_hidra_project` already discards per-run boundaries exactly the way
04b's `nxstress.merge_workspaces: true` opt-out would — same semantics,
just already implemented at the PyRS-workspace layer. And `export_project_files`'s
existing `.h5` branch still needs the single merged workspace regardless
(unaffected by this spec), so skipping the pre-merge for `.nxs` only would
mean maintaining two separate data paths through `CombineRunsModel` for an
identical resulting file. **Decided:** keep `combine_project_files`
unchanged; the `.nxs` export branch passes the already-merged workspace as
a **length-1 list** — `NXstress.write([self._hidra_ws], [])` — once 04b's
signature lands. No `nxstress.discriminator_fields` or `merge_workspaces`
config is touched by this spec at all (`N == 1` bypasses that mechanism
entirely, per 04b's own empty-config policy).

Mechanically, this spec ships in Phase 1 written against the
**pre-04b** signature — `NXstress.write(self._hidra_ws, [])`, a bare
workspace, not a list — since 04b lands later (Phase 2/3 bridge). The
trivial call-site update (wrapping `self._hidra_ws` in `[...]`) is 04b's own
responsibility, already listed in its Scope section ("Update the spec-02/03
call sites... to pass/receive a length-1... list"); this spec does not need
to depend on 04b or wait for it.

### `peakss` is always `[]` for this viewer

`CombineRunsModel` has no concept of `PeakCollection` at all — it only
merges and exports raw/reduced diffraction data
(`save_experimental_data`/`save_reduced_diffraction_data`), never peak-fit
results. `NXstress.write`'s `peakss` argument is a hardcoded empty list in
this spec's `.nxs` branch, not threaded through from anywhere. This is a
fully valid, already-supported input:
`_Peaks.validateNoDuplicatePeaks([])`/`_Fit.validateWorkspaceAndPeaksData(ws, [])`
both no-op on an empty list, and `_Peaks.init_group([]  , ...)` /
`_Fit.init_group(ws, [], ...)` both produce a valid, empty `PEAKS`/`FIT`
group rather than raising.

---

## Scope

**In scope:**
- New `Export as NXstress…` action in `combine_runs_viewer.py` with a
  `NXstress (*.nxs)` file-dialog filter.
- New export path in `combine_runs_model.py` that calls
  `NXstress.write(self._hidra_ws, [])` when the destination is `*.nxs`.
- Round-trip integration test.

**Out of scope:**
- Append-to-existing-NXentry (spec 04c — library-only, no GUI wiring in
  any viewer, including this one).
- Multi-workspace write (spec 04b) — this spec's export always passes a
  single (pre-merged) workspace; see "Pre-merge stays" above.
- Any change to the existing `.h5` export path.
- Loading a `.nxs` file into CombineRunsViewer (not a supported workflow —
  the viewer only combines inputs, not reads back a combined result).

---

## PyRS Changes

_None._

---

## NXstress / GUI Changes

### `pyrs/interface/combine_runs/combine_runs_model.py`

- `export_project_files(self, fileout)` (real current signature — uses
  `self._hidra_ws`, no `merged_ws`/`peakss` parameters exist today): add a
  suffix-dispatch branch — if `Path(fileout).suffix == ".nxs"`, write via
  `NXstress(fileout, "w").write(self._hidra_ws, [])` instead of the
  `HidraProjectFile` path (becomes `write([self._hidra_ws], [])` once 04b's
  list-based signature lands — see above).
- `combine_project_files` is unchanged. Note: it merges with
  `load_raw_counts=False` — `self._hidra_ws` never carries raw counts, so
  the `.nxs` export never populates the optional `input_data` group. This
  is unaffected by this spec; see `open-questions/03-combine-runs-nxstress.md`
  Q1 for why the round-trip test's assertion scope is narrowed accordingly.

### `pyrs/interface/combine_runs/combine_runs_viewer.py`

- Add `Export as NXstress…` button / `QAction`, wired to a
  `export_as_nxstress` slot that opens a `QFileDialog` with filter
  `"NXstress (*.nxs)"` and calls `model.export_project_files`.
- **Config-driven enablement (both actions, always visible, never hidden):**
  the new NXstress export button/action gets
  `.setEnabled(Config["nxstress.enable"])`; the existing `.h5` `Export`
  action gets `.setEnabled(Config["legacy_io.enable"])`. Qt
  `setEnabled`, not `setVisible` — grayed out, not removed, when disabled.
- **Extension is imposed, not user-chosen:** each export slot enforces its
  own section's `extension` (`nxstress.extension` / `legacy_io.extension`)
  on the `QFileDialog`'s returned filename, regardless of what a user types.

---

## Tests

`tests/integration/test_nxstress_viewer_roundtrip.py` (extend):

- **CombineRuns round-trip:** create two workspaces using
  `minimal_HidraWorkspace` (spec 01); merge via `CombineRunsModel.combine_project_files`; export to a
  `.nxs` path; read back with `NXstress.read()`; assert sub-run counts and
  sample-log arrays match the merged workspace. Deliberately does **not**
  assert raw-counts equality — `combine_project_files` merges with
  `load_raw_counts=False`, so `input_data` is never populated by this
  viewer's own data flow; raw-counts round-trip (with `input_data` present)
  is a general NXstress capability, tested elsewhere, not specific to this
  spec. See `open-questions/03-combine-runs-nxstress.md` Q1.
- **Suffix routing:** assert `.h5` export still goes through `HidraProjectFile`
  (no regression).
- **Enablement wiring:** with `nxstress.enable: false`, assert the NXstress
  export action is disabled but still visible; with `legacy_io.enable: false`,
  assert the `.h5` export action is disabled the same way.

---

## Delivered Feature

> **For end users:**
> Combined runs can now be exported as a NXstress (`.nxs`) file:
>
> *Combine Runs → Export as NXstress…*
>
> This allows combined datasets from multiple `.h5` project files to be
> archived in a single NeXus-compliant NXstress file. Existing `.h5` export
> is unaffected.
>
> **Note:** the NXstress format currently requires a complete combined
> workspace to be exported in one step. Incrementally appending additional
> runs to an existing `.nxs` file is a library-only capability as of spec 04c
> — it is not exposed as a GUI action in this viewer (or any viewer) yet.

---

## Verification

- GUI smoke test: load two `.h5` files in CombineRunsViewer, combine,
  **Export as NXstress…**, confirm a `.nxs` file is written.
- Confirm existing **Export** (`.h5`) works without regression.
- `pytest tests/integration/test_nxstress_viewer_roundtrip.py` — all pass.

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass, freshness check only)

This spec has landed. Per the pass's scope it received an **A7 freshness check
and nothing else**.

**F1.1** (A7) — `open-questions/03-combine-runs-nxstress.md:21` cited
`combine_runs_model.py:17` for the call
`self._hidra_ws.load_hidra_project(_project, load_raw_counts=False, …)`.
- Verdict: drifted. `:17` is `self._hidra_ws = None`; the cited call is at
  **`:21`**. Caught by `landing_trigger.py`, which flagged the citation because
  `combine_runs_model.py` changed over `7a5ef73f..HEAD` — the case that A3 alone
  would not have prioritised. The claim is true; only the pointer moved.
- Action: corrected in place.

**F1.2** (A7) — No other stale citations in this document or its open-questions
companion.
