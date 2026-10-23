# 02 — NXstress I/O for PeakFitting & Texture Viewers

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 1
**Depends on:** [01 — Config infrastructure & test framework](01-config-and-test-infra-PR.md)

---

## Overview

Wire the `NXstress` writer and reader into the two GUI viewers whose data
model already matches `NXstress.write(ws, peakss)` exactly:

- **PeakFittingViewer** (`pyrs/interface/peak_fitting/`)
- **TextureFittingViewer** (`pyrs/interface/texture_fitting/`)

Both viewers gain a new `Save as NXstress…` menu action and an added
`NXstress (*.nxs)` filter in their load dialogs. The existing
`HidraProjectFile` (`.h5`) paths are untouched. Any data NXstress requires
but PyRS does not yet produce (sx/sy/sz, diffractogram fit/fit_errors) is
written as `NaN` and documented as a Phase-1 limitation.

---

## Scope

**In scope:**
- New save path in `PeakFittingModel.save_fit_result` that calls
  `NXstress.write(ws, peakss)` when the destination is `*.nxs`.
- New load path in `PeakFittingModel.load_hidra_project` that calls
  `NXstress.read()` when the source is `*.nxs`.
- Matching save/load additions in `TextureFittingModel`.
- New `Save as NXstress…` `QAction` and `NXstress (*.nxs)` file-dialog
  filter in `peak_fitting_viewer.py` and `texture_fitting_viewer.py`.
- Round-trip integration tests for both viewers.
- User-facing note (release notes / viewer tooltip) documenting the NaN
  placeholders.

**Out of scope:**
- Fixing the NaN placeholders (specs 04, 09).
- StrainStressViewer or CombineRunsViewer (specs 03, 05).
- Any change to the existing `HidraProjectFile` code paths.

---

## PyRS Changes

_None._ The `HidraWorkspace` and `PeakCollection` objects are passed
through as-is to `NXstress.write`.

---

## NXstress / GUI Changes

### `pyrs/interface/peak_fitting/peak_fitting_model.py`

- `save_fit_result(out_file_name)`: if `Path(out_file_name).suffix == ".nxs"`,
  write via `NXstress(out_file_name, "w").write(ws, peakss)` instead of the
  existing `HidraProjectFile` path.
- `load_hidra_project(project_files)`: if the first file ends in `.nxs`,
  call `NXstress(project_files[0], "r").read()` to obtain
  `(HidraWorkspace, list[PeakCollection])` and populate `self.hidra_workspace`
  and `self.fit_result` accordingly.

### `pyrs/interface/peak_fitting/peak_fitting_viewer.py`

- Add `QAction` `actionSaveAsNXstress` wired to a new `save_as_nxstress`
  slot that calls `model.save_fit_result` with a `QFileDialog` using
  filter `"NXstress (*.nxs)"`.
- Extend the existing load filter to include `"NXstress (*.nxs)"`.
- **Config-driven enablement (both actions, always visible, never hidden):**
  `actionSaveAsNXstress.setEnabled(Config["nxstress.enable"])`; the
  existing `Save` (`.h5`) action gets
  `.setEnabled(Config["legacy_io.enable"])`. Uses Qt `setEnabled`, not
  `setVisible` — an action stays in the menu, just grayed out, when its
  format is disabled.
- **Extension is imposed, not user-chosen:** each save slot enforces its
  own section's `extension` (`nxstress.extension` / `legacy_io.extension`)
  on whatever filename the `QFileDialog` returns — a user typing a
  different extension does not change which writer runs or what the file
  is named on disk.

### `pyrs/interface/texture_fitting/texture_fitting_model.py`

- Same pattern as `PeakFittingModel` — suffix-dispatch in `save_fit_result`
  and `load_hidra_project_file`.

### `pyrs/interface/texture_fitting/texture_fitting_viewer.py`

- Same filter and action additions as for PeakFittingViewer, including the
  config-driven enablement and extension-imposition rules above.

---

## Tests

`tests/unit/pyrs/utilities/NXstress/test_NXstress.py` (extend) and new
`tests/integration/test_nxstress_viewer_roundtrip.py`:

- **PeakFitting round-trip:** construct a `HidraWorkspace` + a
  `list[PeakCollection]` using `minimal_HidraWorkspace`/`minimal_PeakCollection`
  (spec 01); call
  `PeakFittingModel.save_fit_result` with a `.nxs` path; call
  `PeakFittingModel.load_hidra_project` on the same path; assert workspace
  sub-run counts and peak-parameter arrays match.
- **Texture round-trip:** same pattern with a multi-mask `PeakCollection`.
- **Suffix routing:** assert that a `.h5` path still goes through the
  `HidraProjectFile` code path (no regression).
- **Enablement wiring:** with `nxstress.enable: false`, assert
  `actionSaveAsNXstress.isEnabled()` is `False` while the action remains
  visible; with `legacy_io.enable: false`, assert the existing `Save`
  action is disabled the same way.

---

## Delivered Feature

> **For end users:**
> Peak-fitting and texture results can now be saved in the NXstress (`.nxs`)
> format directly from the **File** menu:
>
> - *Peak Fitting:* **File → Save as NXstress…**
> - *Texture Fitting:* **File → Save as NXstress…**
>
> The resulting `.nxs` file is a NeXus-compliant NXstress file that can be
> read by other NXstress-aware tools. Existing `.h5` project files are
> unaffected and continue to work exactly as before.
>
> **Known Phase-1 limitation:** sample-position fields (sx, sy, sz) and the
> reconstructed fit spectrum are stored as `NaN` in the current release.
> These will be populated in later phases.

---

## Verification

- GUI smoke test: open a `.h5` file in PeakFittingViewer, fit peaks,
  **File → Save as NXstress…**, confirm a `.nxs` file is written.
- Load the `.nxs` back into PeakFittingViewer; confirm peaks are displayed.
- Repeat for TextureFittingViewer.
- Confirm existing `Save` (`.h5`) still works without regression.
- `pytest tests/integration/test_nxstress_viewer_roundtrip.py` — all pass.

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass, freshness check only)

This spec has landed. Per the pass's scope it received an **A7 freshness check
and nothing else**; the one item below is a pointer correction, which
`process.md` §5.5 fixes in place regardless of tier.

**F1.1** (A3) — Section heading: "`### pyrs/interface/texture_fitting/model.py`".
- Referent: `pyrs/interface/texture_fitting/`.
- Verdict: **no such file.** It is `texture_fitting_model.py`. This spec was
  implemented and landed with the heading intact — the implementer read it, went
  to the right file anyway, and nobody noticed. Harmless to a human; fatal to
  anything automated.
- Action: heading corrected in place. The same bad path appeared inside a brace
  expansion in `README.md` §5 (`{texture_fitting_viewer,model}.py`) and is
  corrected there too.

**F1.2** (A7) — No stale citations otherwise: `landing_trigger.py` over
`7a5ef73f..HEAD` finds none in this document.
