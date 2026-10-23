> **SUPERSEDED -- do not implement from this document.**
>
> This is the pre-codebase draft (step I of the planning process). It was
> superseded by [`../README.md`](../README.md) when the subspecs were split out,
> and has not been maintained since. It is retained **only** as the A6 referent,
> so an audit can check what the grounded plan changed without saying so -- see
> [`../../audit-process/process.md`](../../audit-process/process.md) section 2.5.
>
> It disagrees with `README.md` in ways that matter. Its `### 2.3` specifies a
> `pyyaml` + `config/pyrs.default.yml` + pydantic + `argparse --config` design;
> README's `### 2.3` specifies `neutrons_standard.Config` +
> `pyrs/resources/application.yml` and states that this work adds no `--config`
> wiring at all. **Subspec 01 landed implementing README's design.**
>
> Both documents number their sections `## 1` through `## 6`, so a bare section
> reference such as "section 2.3" is ambiguous between them. Line citations here
> are stale and are deliberately **not** corrected: this file records what was
> believed, not what is true.
>
> Archived 2026-09-25.

---

# NXstress GUI Hookup — Design Overview

## Context

An NXstress-compliant (NeXus schema) I/O implementation was recently completed
under [`pyrs/utilities/NXstress/`](../../pyrs/utilities/NXstress/). The next
step is to attach it to the existing PyRS GUI I/O endpoints so that project
data can be persisted in NXstress-native form (either alongside, or eventually
in place of, the current `HidraProjectFile` HDF5 layout).

This document maps the current I/O terrain and flags gaps that must be
resolved (or explicitly scheduled) before and during wiring NXstress into the
GUI. The phased schedule in Section 3 is the primary deliverable.

---

## 1. Current I/O Architecture

### 1.1 Core layers

| Layer | Module | Role |
|---|---|---|
| Low-level HDF5 | `pyrs/projectfile/file_object.py` — `HidraProjectFile` | Read/write every field in a `.h5` project file (raw counts, reduced diffraction, sample logs, peak parameters, masks, instrument geometry, wavelengths). |
| Workspace | `pyrs/core/workspaces.py` — `HidraWorkspace` | High-level in-memory model; wraps `HidraProjectFile` load/save. Methods: `load_hidra_project` (L463), `append_hidra_project` (L497), `save_experimental_data` (L1025), `save_reduced_diffraction_data` (L1096). |
| Peaks | `pyrs/peaks/peak_collection.py` — `PeakCollection` / `PeakCollectionLite` | The unit persisted by `HidraProjectFile.write_peak_parameters` (L778) / `read_peak_parameters` (L712). |
| Fields | `pyrs/dataobjects/fields.py` — `StrainField`, `StressField`, `ScalarFieldSample` | Built _from_ `HidraProjectFile` reads; persisted via CSV (`ScalarFieldSample.to_csv`, L610) and the CSV summary generators. Not directly serialized to HDF5 today. |
| Summary writers | `pyrs/core/summary_generator.py`, `summary_generator_stress.py` | CSV exports for peaks + logs and for stress fields. |
| Calibration | `pyrs/utilities/calibration_file_io.py` | JSON/ASCII calibration recipes (independent of project-file layout). |

### 1.2 GUI endpoints that touch I/O

All GUI dialogs live under `pyrs/interface/` and are launched via
`pyrs/interface/pyrs_main.py::PyRSLauncher` (L36). The six top-level windows
and their I/O touch points are:

| Viewer | Load path (I/O method) | Save path (I/O method) | File filter |
|---|---|---|---|
| **PeakFittingViewer** (`peak_fitting/`) | `PeakFittingModel.load_hidra_project` → `HidraProjectFile(...)` (`peak_fitting_model.py:100`) | `save_fit_result` → copyfile + `HidraProjectFile(..., READWRITE)` + `write_peak_parameters` per `PeakCollection` + `save(False)` (`peak_fitting_model.py:189-199`) | `HDF (*.hdf);H5 (*.h5)` load; `H5 (*.h5);;HDF (*.hdf5)` save |
| **TextureFittingViewer** (`texture_fitting/`) | `TextureFittingModel.load_hidra_project_file` → `HidraProjectFile(READONLY)` (`model.py:37`) | `save_fit_result` → `HidraProjectFile(READWRITE)` + `write_peak_parameters` per masked `PeakCollection` (`model.py:141-165`) | `HidraProjectFile (*.h5)` |
| **StrainStressViewer** (`strainstressviewer/`) | `Model.load_hidra_project_file` per direction → `HidraProjectFile(READONLY)` + `read_peak_tags` + `read_peak_parameters` (`model.py:280-293`) | CSV via `SummaryGeneratorStress` (`model.py:258-264`); state via JSON (`to_json`/`from_json`, `model.py:316/344`); d0 grid via `np.savetxt`/`np.loadtxt` | `HidraProjectFile (*.h5)` load; `CSV (*.csv)` / `JSON (*.json)` save |
| **CombineRunsViewer** (`combine_runs/`) | `Model.combine_project_files` → multiple `HidraProjectFile(...)` reads (`combine_runs_model.py:16-21`) | `Model.export_project_files` → new `HidraProjectFile("w")` + `save_experimental_data` + `save_reduced_diffraction_data` + `save` (`combine_runs_model.py:25-31`) | `HidraProjectFile (*.h5)` |
| **ManualReductionViewer** (`manual_reduction/`) | Browse-only line edits for NeXus/mask/vanadium inputs | `manual_reduction/pyrs_api.py::HB2BReductionManager.save_project` — **currently `NotImplementedError`** (L211); the actual reduction pathway writes via `ReductionApp.save_diffraction_data` (`pyrs_api.py:348`) | `NeXus(*.nxs.h5)`, `Mantid Mask(*.xml)`, `HiDRA project(*.h5)` |
| **DetectorCalibrationViewer** (`detector_calibration/`) | `Nexus (*.nxs.h5)` load, `json (*.json)` recipe load | `write_calibration` via calibration object; `json.dumps` recipe (`detector_calibration_viewer.py:558`) | `Nexus (*.nxs.h5)`, `JSON (*.json)` |

Common file filters in use today: `*.h5` (dominant), `*.hdf5`/`*.hdf`
(peak-fit compatibility), `*.nxs.h5` (NeXus event files), `*.csv`, `*.json`,
`*.xml` (masks). No bare `*.nxs` is currently registered anywhere in the GUI.

### 1.3 Non-GUI entry points

`pyproject.toml` registers CLI scripts under `scripts/`:

- `pyrsplot` — GUI launcher, no direct I/O.
- `pyrs-calibration` (`scripts/pyrs_calibration.py:144`) — CLI-only, writes
  calibration JSON via `write_calibration` (L212–229). Not a project-file path.
- `pyrs-calibration-correlation`, `create-mask` — supporting scripts.

Per-viewer standalone `start.py` scripts route all I/O through the GUI model.
None of the installed scripts currently accept `--config` or any equivalent
flag (see Section 2.3).

### 1.4 What NXstress currently expects

The public surface is one class,
[`pyrs.utilities.NXstress.NXstress`](../../pyrs/utilities/NXstress/NXstress.py):

```python
with NXstress(path, "w") as nxS:
    nxS.write(ws: HidraWorkspace, peakss: list[PeakCollection])

with NXstress(path, "r") as nxS:
    ws, peakss = nxS.read(entry_number=1)
```

Its input/output types are **exactly** the pair
`(HidraWorkspace, list[PeakCollection])` that PeakFittingViewer and
TextureFittingViewer already produce and consume, making those two viewers
the natural first integration point.

Multiple `NXentry` groups per file are supported (`NXstress.py:149`). These
are reserved for "distinct data-reduction or sample conditions" — not for
multi-direction stress data, which is handled via the compound `PeakIndex`
(see Section 2.4).

---

## 2. Issues to Resolve Before / During Hookup

### 2.1 NXstress-side gaps (already noted in the code)

These are things the NXstress implementation itself cannot yet do. Some are
self-contained fixes; others require upstream PyRS changes (scheduled in the
phases below).

- **Appending to an existing entry / workspace** — `_input_data.py:44,63` and
  `NXstress.py:151-152` raise on any operation that would extend an existing
  `NXentry`. Each write must currently be a fresh entry (save-as). Scheduled:
  Phase 4.
- **NXbeam intensity profile empty** — `_instrument.py:127`. Fine for Phase 1;
  readers must tolerate absence. Scheduled: Phase 5 (depends on reduction
  pipeline producing a profile).
- **Detector L2 not extractable from `DENEXDetectorGeometry`** —
  `_instrument.py:138-140`. Round-tripping a calibrated geometry may lose the
  arm-shift information. Requires a new accessor on `DENEXDetectorGeometry`.
  Scheduled: Phase 5.
- **Detector rotation-order TODO** — `_instrument.py:165`. A wrong order
  silently mis-orients the instrument in the file. Must be cross-checked
  against the reduction code's convention. Scheduled: Phase 2.
- **Instrument name hardcoded to `"HB2B"`** — `_instrument.py:70`. Fine for
  single-beamline use; parameterize before any HB2A/other-instrument use.
  Scheduled: Phase 2.
- **Scan-point positions `sx`/`sy`/`sz` hardcoded to NaN** —
  `_peaks.py:235-239`. The commented-out `logs['sx'/'sy'/'sz']` block was
  disabled pending log-key naming confirmation. Needs to be restored and the
  key names reconciled against what the reduction pipeline emits. Scheduled:
  Phase 2.
- **Fit diffractogram `fit`/`fit_errors` initialized to NaN** —
  `_fit.py:426-429,486-487`. PyRS does not currently reconstruct the fitted
  model spectrum after peak-fitting. Blocked on a PyRS fit-engine change.
  Scheduled: Phase 5.
- **Mask naming/storage inconsistency** — `_fit.py:546-549`. Default vs. named
  masks are addressed differently in the current writer; needs a single
  convention before external readers can rely on the layout. Scheduled:
  Phase 2.
- **Sample `STRESS_FIELD` shape unverified** — `_sample.py:107`. No example
  data existed when the writer was drafted; verify against a real dataset.
  Scheduled: Phase 5.
- **Group-name validator workarounds** — several `GROUP_NAME` values
  (`_definitions.py:106,109,116,122,127-135`) and
  `DGRAM_TWO_THETA_NAME = "XAXIS"` (`_fit.py:414`) were set to upper-case /
  generic forms to work around a `nexusformat` validator bug. The generic names
  are likely acceptable on-disk schema-wise, but the production intent is
  meaningful lowercase names. **Resolution:** expose as a `Config` toggle
  (Section 2.3) so the writer can emit either form without a code change.
  Whether uppercase names are strictly allowed by NXstress requires a
  schema-spec check. Scheduled: Phase 1 (Config infrastructure) + Phase 2
  (flip default once validator bug is resolved upstream).
- **`allowed_identifier` disallowed-character coverage incomplete** —
  `_definitions.py:229`. Scheduled: Phase 2.

### 2.2 Upstream PyRS data-object gaps

Missing capabilities in PyRS proper that NXstress cannot fill in on its own:

- **No `StrainField` / `StressField` serialization** — NXstress today
  serializes `HidraWorkspace` + `list[PeakCollection]` but does not consume or
  produce the field-level objects (`pyrs/dataobjects/fields.py::StrainField`,
  `StressField`). Wiring StrainStressViewer means building the fields on-the-
  fly at read time from a direction-indexed NXstress file. Scheduled: Phase 3.
- **No reconstructed model spectrum** — PyRS's fit engine returns parameter
  values but no full-length reconstructed profile on the two-theta grid.
  This blocks populating `diffractogram/fit` and `diffractogram/fit_errors`
  with real data. Scheduled: Phase 5.
- **`HidraProjectFile.read_instrument_geometry` returns `DENEXDetectorGeometry`
  even for calibrated instruments** (`file_object.py:510` FIXME) —
  round-tripping through NXstress inherits this inaccuracy. Scheduled: Phase 5.
- **`HidraProjectFile` legacy log-name patches** (`file_object.py:404`, `:494`
  FIXME) — audit to ensure NXstress does not codify the legacy names into the
  schema-conformant file. Scheduled: Phase 2.
- **`ScalarFieldSample` / `StrainField` `NotImplementedError` methods** in
  `fields.py` (L239, 611, 816, 927, 956, 960, 964, 983, 986, 994, 1184) — not
  directly on the write path, but any read-back that reconstructs these fields
  for the StrainStressViewer will hit them. Scheduled: Phase 3 (as needed).
- **`HB2BReductionManager.save_project`** in `manual_reduction/pyrs_api.py:211`
  is `NotImplementedError`. Attaching NXstress there requires this to be
  implemented first. Scheduled: Phase 4.
- **`nexus_conversion.py:118, 374`** — two unimplemented branches; confirm
  they are or are not on the manual-reduction save path. Scheduled: Phase 4.

### 2.3 New capability: YAML Config

PyRS has no runtime configuration file. The installed scripts (`pyrsplot`,
`pyrs-calibration`, `create-mask`) use either no CLI flags or positional
arguments only; no `argparse` + `--config` plumbing exists. This work adds:

- A YAML config file parsed with `pyyaml` (verify it is available at
  implementation time). Minimum v1 fields:
  - `nxstress.use_production_names: bool` — toggle between the current
    validator-safe generic group names and the production lowercase forms.
  - `nxstress.default_extension: ".nxs" | ".h5"` — default save extension
    for NXstress-aware viewers.
  - Reserved namespace for future flags (`nxstress.write_raw_counts`,
    `nxstress.strict_schema_validation`, etc.).
- Default config location: `config/pyrs.default.yml` (checked into the repo,
  ships with the package). The loader also honors
  `~/.config/pyrs/config.yml` if present and merges it over the default.
- A shared loader at `pyrs/utilities/config.py` returning a validated
  (pydantic) `Config` object, injected at NXstress callsites that read the
  flags.
- `argparse` wiring on each installed script — `pyrsplot --config <path>`
  (analogously for `pyrs-calibration`, `create-mask`). When `--config` is
  absent, fall back to the default location.

This is a prerequisite for closing the Section 2.1 group-name validator
workaround cleanly. Scheduled: Phase 1.

### 2.4 Beyond-code decisions

- **File-extension policy** — **decided:** distinct `*.nxs` extension across
  every viewer; coexists with existing `*.h5` `HidraProjectFile` paths. No
  auto-detection needed; extension is authoritative.
- **Coexistence vs. replacement** — **decided (phased):** Phase 1 ships
  additive (both paths coexist; NXstress offered as new menu entries and file-
  dialog filters). Phase 6 flips defaults (NXstress becomes primary;
  `HidraProjectFile` becomes legacy-read-only). Config flag steers the default
  during the transition. Concrete acceptance criteria for the Phase 6 flip to
  be defined at the end of Phase 1.
- **Multi-direction semantics** — StrainStressViewer today loads N project
  files per direction into three parallel slots (`filenames_11/22/33`).
  **Provisional decision:** extend `_peaks.py::PeakIndex` with a `direction`
  axis so a single NXstress file (single NXentry) carries all three directions,
  distinguished by that axis. NOT one-NXentry-per-direction. **Must be
  re-evaluated at Phase 3 kickoff** against the canonical NXstress.xml schema —
  whether a compound-index extension of `NXreflections` is schema-conformant is
  currently unknown.
- **Multi-NXentry semantics (unchanged)** — multiple NXentry within a single
  `.nxs` file remain reserved for "distinct data-reduction or sample
  conditions" (`NXstress.py:107-125`). The compound peak index is the axis for
  finer variation (masks, phases, directions) within a single condition.
- **Read-back completeness** — `NXstress.read` reconstructs wavelengths, sample
  logs, masks, reduced diffraction data, and peak collections. It does NOT
  reconstruct raw counts unless the optional `input_data` group was written.
  Verify each viewer's load path is satisfied by this subset during Phase 1
  wiring.

---

## 3. Phased Implementation Schedule

The work is deliberately phased so Phase 1 can ship using existing PyRS
capabilities, with `NaN` as a documented placeholder for data PyRS does not
yet produce. Subsequent phases progressively close those placeholders —
first by adding capability on the PyRS side, then by wiring the NXstress
writer/reader to use it.

**Every TODO listed in Sections 2.1 and 2.2 must be scheduled — none are
deferred indefinitely.**

Each phase is broken into two strictly-separate sub-lists:

- **PyRS changes** — modifications to PyRS proper (data objects, reduction,
  fit engine, workspace, etc.), independent of NXstress.
- **NXstress extensions** — modifications inside
  `pyrs/utilities/NXstress/` and the GUI wiring in `pyrs/interface/`.

---

### Phase 1 — Baseline hookup with `NaN` placeholders

**Goal:** Every viewer whose data model already matches
`NXstress.write(ws, peakss)` gains a working NXstress save/load path,
alongside (not replacing) the existing `HidraProjectFile` I/O. Any data
NXstress requires but PyRS does not yet produce is written as `NaN` and
documented as a Phase-1 limitation.

**Required PyRS changes:** _none._ Phase 1 is entirely additive on the
consumer side.

**Required NXstress extensions / GUI wiring:**
- Introduce the `Config` infrastructure (Section 2.3): `config/pyrs.default.yml`,
  `pyrs/utilities/config.py`, `--config` on all installed scripts. Ship with
  `nxstress.use_production_names = false` (validator-safe generic names) as
  the default.
- Wire `NXstress` into `PeakFittingModel.save_fit_result` and
  `PeakFittingModel.load_hidra_project` as a new save/load path, triggered by
  a new `Save as NXstress…` menu action and an added `NXstress (*.nxs)` filter
  in the load dialog (`peak_fitting_viewer.py`, `peak_fitting_model.py`).
- Same wiring in `TextureFittingModel` (`save_fit_result` /
  `load_hidra_project_file`, `texture_fitting_viewer.py`, `model.py`).
- Same wiring in `CombineRunsModel.export_project_files` /
  `combine_project_files` — fresh-write only; document that append is not yet
  supported.
- Round-trip tests for each of the three viewers: write via NXstress, read
  back, assert equality of `HidraWorkspace` and `list[PeakCollection]`.
- User-facing docs: note the `NaN` placeholders (sx/sy/sz,
  diffractogram fit/fit_errors) as documented Phase-1 limitations.

**Deferred to later phases (all scheduled):**
StrainStressViewer (Phase 3), ManualReductionViewer (Phase 4), all Section 2.1
and 2.2 TODOs.

---

### Phase 2 — Close the low-cost NXstress-side TODOs

**Goal:** Resolve NXstress-internal issues that don't require any PyRS
data-model change.

**Required PyRS changes:**
- Audit `HidraProjectFile` legacy log-name patches (`file_object.py:404`,
  `:494` FIXME) and confirm the correct canonical names to emit from NXstress.

**Required NXstress extensions:**
- Restore `sx`/`sy`/`sz` reading from `SampleLogs` in `_peaks.py:235-239`
  (the commented-out block); reconcile log-key names against what the
  reduction pipeline emits.
- Reconcile default vs. named mask storage into a single convention
  (`_fit.py:546-549`).
- Complete disallowed-character coverage in
  `_definitions.py::allowed_identifier` (L229).
- Flip `nxstress.use_production_names` default to `true` in
  `config/pyrs.default.yml` once the upstream `nexusformat` validator bug
  is resolved.
- Parameterize the hardcoded instrument name in `_instrument.py:70`
  (currently `"HB2B"`) — draw from `Config` or from `HidraWorkspace`.
- Detector rotation-order cross-check + fix in `_instrument.py:165`
  against the reduction code's convention; add a regression test.

---

### Phase 3 — StrainStressViewer + compound-index extension

**Goal:** Hook up StrainStressViewer. Extend the compound `PeakIndex` with a
direction axis so a single NXentry carries all three directions.

> **Schema check required first:** Before modifying `_peaks.py::PeakIndex`,
> verify against the canonical NXstress.xml whether adding a `direction` axis
> to `NXreflections` is schema-conformant, or whether a different mechanism
> (multi-NXentry, dedicated stress-field subgroup, etc.) is the intended
> pattern. Update the decision in Section 4 accordingly and adjust the
> sub-lists below if needed.

**Required PyRS changes:**
- Confirm `HidraWorkspace` can represent (or cleanly merge) sample-log content
  from multiple direction measurements, or design a direction-aware container.
- Resolve any `NotImplementedError` in `pyrs/dataobjects/fields.py`
  (L239, 611, 816, 927, 956, 960, 964, 983, 986, 994, 1184) that falls on the
  StressField reconstruction read path.

**Required NXstress extensions:**
- Extend `_peaks.py::PeakIndex` with the chosen direction axis; update
  `sort_key`, `validateNoDuplicatePeaks`, and the `NXreflections` layout in
  `_init` / `init_group` / `peakCollectionsFromNexus`.
- Wire NXstress into `strainstressviewer/model.py::load_hidra_project_file` /
  `load_hidra_project_files` — read a direction-indexed NXstress file and
  reconstruct `StrainField` / `StressField` in the viewer.
- Add a `Save as NXstress…` action on the StrainStressViewer.
- Round-trip test: construct a `StressField` from a single NXstress file,
  confirm it matches the CSV-summary output for the same inputs.

---

### Phase 4 — ManualReductionViewer + append support

**Goal:** Close the capability gaps that block manual-reduction and incremental
workflows.

**Required PyRS changes:**
- Implement `HB2BReductionManager.save_project` in
  `manual_reduction/pyrs_api.py:211` (currently `NotImplementedError`).
- Confirm or resolve the two `NotImplementedError` branches in
  `pyrs/core/nexus_conversion.py:118, 374` — are they on the
  manual-reduction save path?

**Required NXstress extensions:**
- Implement append-to-existing-NXentry:
  - `_input_data.py:44-46` — append `detector_counts` on write.
  - `_input_data.py:63-72` — append `detector_counts` to workspace on read.
  - `NXstress.py:151-152` — remove the "not implemented" guard once
    subgroup-level appends work end-to-end.
  - Peaks compound-index append path in `_peaks.py:180-181`.
- Wire NXstress into `ManualReductionModel.save_project` and the reduction
  save flow, using append where appropriate.
- Round-trip test: reduce a NeXus event file, write NXstress, then verify a
  second reduction can append into the same file cleanly.

---

### Phase 5 — Reconstructed fit spectrum + detector-calibration fidelity

**Goal:** Replace the remaining `NaN` placeholders with real data.

**Required PyRS changes:**
- Fit engine (`pyrs/peaks/…`, `pyrs/core/peak_profile_utility.py`): add a
  method that returns the reconstructed model spectrum on the original
  two-theta grid, indexed by `PeakCollection` × mask × scan_point. This is
  the largest single PyRS-side item in the plan.
- Add an accessor to `DENEXDetectorGeometry`
  (`pyrs/core/instrument_geometry.py`) that reports whether the arm shift has
  been applied. Fix `pyrs/projectfile/file_object.py:510`
  (`HidraProjectFile.read_instrument_geometry` returning
  `DENEXDetectorGeometry` even for calibrated instruments).
- Beam-intensity profile: add a data path from the reduction pipeline that
  carries the profile if measured (mark optional otherwise).
- Verify the `SampleLogs` `STRESS_FIELD` shape against a real example dataset
  (`_sample.py:107`).

**Required NXstress extensions:**
- Populate `_fit.py:426-429,486-487` diffractogram `fit` / `fit_errors` from
  the new fit-engine method.
- Populate NXbeam intensity profile in `_instrument.py:127` when the
  reduction data carries one.
- Fix the L2 arm-shift round-trip in `_instrument.py:138-140` using the new
  `DENEXDetectorGeometry` accessor.
- Fix `_sample.py:107` `STRESS_FIELD` dimensions once the shape is confirmed.

---

### Phase 6 — Flip defaults (NXstress becomes primary)

**Goal:** Complete the phased-replacement decision. Acceptance criteria for
triggering this phase to be defined at the end of Phase 1.

**Required PyRS changes:** _none._

**Required NXstress extensions:**
- Set `nxstress.default_extension = ".nxs"` and
  `nxstress.use_production_names = true` as the defaults in
  `config/pyrs.default.yml`.
- Demote `HidraProjectFile` save paths to legacy-read-only (still loadable,
  no longer offered in save dialogs).
- Write deprecation-warning docs and a migration note for existing users with
  `.h5` project files.

---

### Skipped

**DetectorCalibrationViewer** — its outputs (calibration JSON, Mantid NeXus
event files) are independent of the NXstress project-file layout. No NXstress
hookup is planned unless a downstream requirement emerges.

---

## 4. Decisions Log

| # | Question | Decision | Notes |
|---|---|---|---|
| 1 | File-extension policy | **New `*.nxs` filter** across every viewer; NXstress writes to `.nxs`, coexisting with existing `*.h5` `HidraProjectFile` paths. | No auto-detection needed; extension is authoritative. Config key: `nxstress.default_extension`. |
| 2 | Coexistence vs. replacement | **Additive now, replace later (phased).** Keep existing `HidraProjectFile` save/load unchanged in Phase 1; add NXstress as new menu entries and file-dialog filters. Phase 6 flips defaults so NXstress is primary and `HidraProjectFile` becomes legacy-read-only. | Phase-6 trigger criteria to be defined at the end of Phase 1. |
| 3 | StressField / multi-direction persistence | **Provisional:** extend the compound `PeakIndex` with a `direction` axis so a single NXentry carries all three stress directions. `StressField` is reconstructed in the viewer from the indexed peaks at read time. | **Re-evaluate at Phase 3 kickoff** against the canonical NXstress.xml schema. If the schema's intended mechanism differs (multi-NXentry, dedicated subgroup, etc.), update this decision and revise the Phase 3 sub-lists accordingly. |
| 4 | NaN placeholder tolerance | **`NaN` is acceptable for Phase 1.** No NXstress TODO is a Phase-1 blocker. Every TODO in Sections 2.1 and 2.2 is scheduled across Phases 2–5. | The phased schedule (Section 3) is the deliverable of this decision. |
| 5 | Priority ordering | **Keep drafted order.** Phase 1 hooks up PeakFitting, Texture, and CombineRuns (data model matches `NXstress.write` 1:1, zero upstream changes). StrainStressViewer comes in Phase 3, after Phase 2 stabilizes NXstress internals and after the schema check Q3 requires. | Ship something quickly, learn from real use, then tackle the harder integration. |

---

## 5. Files to be Modified

| Phase | PyRS files | NXstress / GUI files |
|---|---|---|
| 1 | _(none)_ | `pyrs/utilities/config.py` (new), `config/pyrs.default.yml` (new), `scripts/pyrsplot.py`, `scripts/pyrs_calibration.py`, `scripts/create_mask.py`, `pyrs/interface/peak_fitting/{peak_fitting_viewer,peak_fitting_model}.py`, `pyrs/interface/texture_fitting/{texture_fitting_viewer,model}.py`, `pyrs/interface/combine_runs/{combine_runs_viewer,combine_runs_model}.py` |
| 2 | `pyrs/projectfile/file_object.py` (audit legacy log names) | `pyrs/utilities/NXstress/{_peaks,_fit,_definitions,_instrument}.py`, `config/pyrs.default.yml` |
| 3 | `pyrs/core/workspaces.py`, `pyrs/dataobjects/fields.py` | `pyrs/utilities/NXstress/{_peaks,NXstress}.py`, `pyrs/interface/strainstressviewer/{strain_stress_view,model}.py` |
| 4 | `pyrs/interface/manual_reduction/pyrs_api.py`, `pyrs/core/nexus_conversion.py` | `pyrs/utilities/NXstress/{NXstress,_input_data,_peaks}.py`, `pyrs/interface/manual_reduction/…` |
| 5 | `pyrs/peaks/…`, `pyrs/core/{peak_profile_utility,instrument_geometry}.py`, `pyrs/projectfile/file_object.py` | `pyrs/utilities/NXstress/{_fit,_instrument,_sample}.py` |
| 6 | _(none)_ | `config/pyrs.default.yml`, viewer save-dialog filters |

Tests to extend/add:
- `tests/unit/pyrs/utilities/NXstress/*` (existing suite — extend each phase).
- `tests/ui/test_nxstress_roundtrip.py` (new — GUI-level round-trips).
- `tests/integration/test_nxstress_reduction.py` (new in Phase 4 — manual
  reduction + append).

---

## 6. Verification

- **Round-trip pytest** — for each viewer connected, load an existing
  `HidraProjectFile`, do the workflow, save as NXstress, reload from NXstress,
  assert equality of `HidraWorkspace` and `list[PeakCollection]`.
- **Schema validation** — run each written file through the NXstress validator
  (via `nexusformat`) with `nxstress.use_production_names = true` once the
  upstream validator bug is resolved (Phase 2).
- **GUI smoke test** — launch each affected viewer via `pyrsplot`, exercise the
  new save/load actions, confirm no regressions in existing `HidraProjectFile`
  workflows.
- **Demo script** — keep `tests/scripts/cis_tests/NXstress_demo_script.py`
  passing and update it to reflect current usage after each phase.
