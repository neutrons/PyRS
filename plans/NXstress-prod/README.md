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

This document is the primary reference for the NXstress production hookup work. Sub-specifications for each PR-sized chunk of work are listed below.

## Sub-specifications

| Spec | Title | Phase | Depends on |
|---|---|---|---|
| [01](01-config-and-test-infra-PR.md) | Config infrastructure & test framework | 1 | — |
| [02](02-peak-and-texture-nxstress.md) | NXstress I/O for PeakFitting & Texture viewers | 1 | 01 |
| [03](03-combine-runs-nxstress.md) | NXstress I/O for CombineRuns viewer | 1 | 01 |
| [04](04-nxstress-internal-cleanup.md) | NXstress internal cleanup (Phase 2 TODOs) | 2 | 02, 03 |
| [04b](04b-multi-workspace-nxstress.md) | Multi-workspace NXstress I/O | 2/3 | 01, 04 |
| [04c](04c-nxstress-append.md) | NXstress append mode (library only) | 3 | 04b |
| [05](05-strain-stress-viewer.md) | StrainStressViewer NXstress hookup | 3 | 01, 04, 04b |
| [06](06-manual-reduction-prereqs.md) | Manual reduction PyRS prerequisites (not required for NXstress) | — (independent PyRS cleanup) | — |
| [07](07-manual-reduction-nxstress.md) | ManualReductionViewer NXstress hookup | 4 | — |
| [08](08-fit-spectrum-prereqs.md) | Reconstructed fit spectrum & calibration fidelity (PyRS) | 5 | — |
| [09](09-fit-spectrum-nxstress.md) | Fit spectrum & calibration fidelity (NXstress) | 5 | 07, 08 |
| [10](10-flip-defaults.md) | Flip defaults — NXstress becomes primary | 6 | 09 |

---

## 1. Current I/O Architecture

### 1.1 Core layers

| Layer | Module | Role |
|---|---|---|
| Low-level HDF5 | `pyrs/projectfile/file_object.py` — `HidraProjectFile` | Read/write every field in a `.h5` project file (raw counts, reduced diffraction, sample logs, peak parameters, masks, instrument geometry, wavelengths). |
| Workspace | `pyrs/core/workspaces.py` — `HidraWorkspace` | High-level in-memory model; wraps `HidraProjectFile` load/save. Methods: `load_hidra_project` (L465), `append_hidra_project` (L517), `save_experimental_data` (L1045), `save_reduced_diffraction_data` (L1116). |
| Peaks | `pyrs/peaks/peak_collection.py` — `PeakCollection` / `PeakCollectionLite` | The unit persisted by `HidraProjectFile.write_peak_parameters` (`file_object.py:854`) / `read_peak_parameters` (`file_object.py:788`). |
| Fields | `pyrs/dataobjects/fields.py` — `StrainField`, `StressField`, `ScalarFieldSample` | Built _from_ `HidraProjectFile` reads; persisted via CSV (`ScalarFieldSample.to_csv`, L610) and the CSV summary generators. Not directly serialized to HDF5 today. |
| Summary writers | `pyrs/core/summary_generator.py`, `summary_generator_stress.py` | CSV exports for peaks + logs and for stress fields. |
| Calibration | `pyrs/utilities/calibration_file_io.py` | JSON/ASCII calibration recipes (independent of project-file layout). |

### 1.2 GUI endpoints that touch I/O

All GUI dialogs live under `pyrs/interface/` and are launched via
`pyrs/interface/pyrs_main.py::PyRSLauncher` (L36). The six top-level windows
and their I/O touch points are:

| Viewer | Load path (I/O method) | Save path (I/O method) | File filter |
|---|---|---|---|
| **PeakFittingViewer** (`peak_fitting/`) | `PeakFittingModel.load_hidra_project` → `HidraProjectFile(...)` (`peak_fitting_model.py:74-123`) | `save_fit_result` → copyfile + `HidraProjectFile(..., READWRITE)` + `write_peak_parameters` per `PeakCollection` + `save(False)` (`peak_fitting_model.py:211-247`) | `HDF (*.hdf);H5 (*.h5)` load; `H5 (*.h5);;HDF (*.hdf5)` save |
| **TextureFittingViewer** (`texture_fitting/`) | `TextureFittingModel.load_hidra_project_file` → `HidraProjectFile(READONLY)` (`texture_fitting_model.py:42-87`) | `save_fit_result` → `HidraProjectFile(READWRITE)` + `write_peak_parameters` per masked `PeakCollection` (`texture_fitting_model.py:175-221`) | `HidraProjectFile (*.h5)` |
| **StrainStressViewer** (`strainstressviewer/`) | `Model.load_hidra_project_file` per direction → `HidraProjectFile(READONLY)` + `read_peak_tags` + `read_peak_parameters` (`model.py:280-293`) | CSV via `SummaryGeneratorStress` (`model.py:258-264`); state via JSON (`to_json`/`from_json`, `model.py:316/344`); d0 grid via `np.savetxt`/`np.loadtxt` | `HidraProjectFile (*.h5)` load; `CSV (*.csv)` / `JSON (*.json)` save |
| **CombineRunsViewer** (`combine_runs/`) | `Model.combine_project_files` → multiple `HidraProjectFile(...)` reads (`combine_runs_model.py:18-28`) | `Model.export_project_files` → new `HidraProjectFile("w")` + `save_experimental_data` + `save_reduced_diffraction_data` + `save` (`combine_runs_model.py:29-46`) | `HidraProjectFile (*.h5)` |
| **ManualReductionViewer** (`manual_reduction/`) | Browse-only line edits for NeXus/mask/vanadium inputs | `reduce_hidra_workflow` (module-level function, `pyrs_api.py:251`) automatically calls `ReductionApp.save_diffraction_data` (`pyrs_api.py:348`) as part of every reduction — this is the actual, currently-functional save path. `ReductionController.save_project` (`pyrs_api.py:190`, not `HB2BReductionManager` — that name belongs to an unrelated class in `pyrs/core/reduction_manager.py`) is a separate, `NotImplementedError` stub with **zero callers anywhere in the codebase** | `NeXus(*.nxs.h5)`, `Mantid Mask(*.xml)`, `HiDRA project(*.h5)` |
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

Multiple `NXentry` groups per file are supported (`NXstress.py:156-164`). These
are reserved for "distinct data-reduction or sample conditions" — not for
multi-direction stress data, which is handled via the compound `PeakIndex`
(see Section 2.4).

---

## 2. Issues to Resolve Before / During Hookup

### 2.1 NXstress-side gaps (already noted in the code)

These are things the NXstress implementation itself cannot yet do. Some are
self-contained fixes; others require upstream PyRS changes (scheduled in the
phases below).

**This section is now a closed-out record, not a live work list.** Its only
consumer was spec 04, whose Overview opens *"Close the NXstress-internal TODOs
from Section 2.1 of the README"* — and specs 04, 04b and 04c have landed. A list
whose only reader is finished cannot be where open work waits, so **every entry
still open has been carried forward into the subspec that owns it**. Nothing
here is scheduled by being here.

Six of the eleven entries have since closed (one only in part). They are kept,
marked **CLOSED**, with what closed them and the evidence: a gap list that
silently drops its resolved entries cannot be checked afterwards against what
was actually delivered.

| Still open | Carried forward to |
|---|---|
| NXbeam intensity profile empty | spec 08 (`HidraWorkspace.beam_intensity_profile`), then spec 09 |
| Detector L2 / arm-shift not recoverable | spec 08 (`DENEXDetectorGeometry.arm_shift_applied`), spec 09 |
| Fit `fit`/`fit_errors` are NaN placeholders | spec 08 (fit-engine method), spec 09 (populate) |
| `STRESS_FIELD` shape unverified | spec 08, explicitly as **blocked**; `open-questions/08` Q3 |
| Group-name validator workarounds | spec 10 (`nxstress.use_production_names`) |
| **`two_theta_zero`'s rotation axis** | **spec 08** (resolve the contradiction), **spec 09** (emit it) — new, found by this review |
| Extending an existing key, and overwriting an entry | **unscheduled, and deliberately so** — 04c delivers tail-append of a *new* key only |

Audited in full, re-pointed and carried forward 2026-10-15 — see
`plans/PR_review/no-subspec-comments.md`, batch 27.

- **Appending to an existing entry** — **CLOSED in part (Phase 3, spec 04c).**
  Append is implemented as a library capability: `write()` dispatches on
  whether the resolved entry already exists
  ([NXstress.py:156-164](../../pyrs/utilities/NXstress/NXstress.py#L156)), and
  tail-appends a genuinely *new* workspace/compound key. **Still open**, and
  unchanged by Phase 3: extending a key *already in the file* with more scan
  points raises `NotImplementedError`, and overwriting an existing entry's
  contents raises
  ([NXstress.py:278](../../pyrs/utilities/NXstress/NXstress.py#L278)) — a real,
  currently-unscheduled gap. No GUI wiring is scheduled for append in this
  pass, since spec 07's reduction pathway is decided to always write a fresh
  file.
- **NXbeam intensity profile empty** — **OPEN.** `_instrument.py:250`
  (`# TODO: fill in the beam-intensity profile.`); the `NXbeam` group is
  written but unpopulated (`_instrument.py:348`). Fine for Phase 1; readers
  must tolerate absence. Scheduled: Phase 5 (depends on the reduction pipeline
  producing a profile).
- **Detector L2 not extractable from `DENEXDetectorGeometry`** — **OPEN.**
  `_instrument.py:261`. Round-tripping a calibrated geometry may lose the
  arm-shift information. Requires a new accessor on `DENEXDetectorGeometry`.
  Scheduled: Phase 5.
- **Detector rotation-order TODO** — **CLOSED (Phase 2, spec 04).** The written
  order was already correct: traversed from the last link, the rotation
  sub-chain composes to `Rx @ Ry @ Rz`, matching
  `reduce_hb2b_pyrs.py::generate_rotation_matrix`. Proven by
  [`probes/a5_transformations_chain.py`](probes/a5_transformations_chain.py),
  which also found the real defect here — the chain's `depends_on` named the
  *first* link, so 1 of 8 transformations was reachable (Decisions item 26,
  spec 04 Follow-up 7 F7.5). The
  `# TODO: check order of rotations here!!!` comment that stood in the source
  has been removed, and the order is now asserted on every commit rather than
  only during an audit: `test_instrument.py::TestDetectorRotationOrder`
  composes the written rotation sub-chain and compares it against
  `generate_rotation_matrix` itself. **Rotation *order* is closed.** The
  separate question of `two_theta_zero`'s *axis* is open and carried forward —
  see the table above.
- **Instrument name hardcoded to `"HB2B"`** — **CLOSED (Phase 2, spec 04).**
  Now read from configuration by `_Instrument._instrument_names`
  ([_instrument.py:69-81](../../pyrs/utilities/NXstress/_instrument.py#L69)),
  via `nxstress.instrument_name` / `nxstress.instrument_short_name`, with
  `"HB2B"` as the fallback rather than the only option.
- **Scan-point positions `sx`/`sy`/`sz` hardcoded to NaN** — **CLOSED
  (Phase 2, spec 04).** Written from the sample coordinate logs
  ([_peaks.py:237-262](../../pyrs/utilities/NXstress/_peaks.py#L237)), each
  column carrying a `local_name` recording the PyRS log it came from; the key
  names were reconciled in `open-questions/04` Q3. They still read NaN when the
  source coordinates are non-finite, which is now detected by asking rather
  than by catching an assertion message — see spec 04's
  `TestNonFiniteCoordinatesAreAsked`.
- **Fit diffractogram `fit`/`fit_errors` initialized to NaN** — **OPEN.**
  `_fit.py:527-530` (placeholder comment), `:531-536` (field creation). PyRS
  does not currently reconstruct the fitted model spectrum after peak-fitting.
  Blocked on a PyRS fit-engine change. Scheduled: Phase 5.
- **Mask naming/storage inconsistency** — **CLOSED (spec 04 Follow-up 8).** The
  root cause was one namespace being asked to hold two different kinds of
  string. `_definitions.py` now separates them: `nxstress_mask_names`
  ([:491](../../pyrs/utilities/NXstress/_definitions.py#L491)) reads
  `_mask_dict` and injects `DEFAULT_TAG`; `nxstress_diffractogram_keys`
  ([:509](../../pyrs/utilities/NXstress/_definitions.py#L509)) reads
  `_diff_data_set` and does not.
- **Sample `STRESS_FIELD` shape unverified** — `_sample.py:272` (actual
  comment: *"we don't have an example of these entries, so the dimensions
  may not be correct!"*). Confirmed no real dataset with a genuine
  `STRESS_FIELD` log exists anywhere in the repo — blocked, see
  `open-questions/08-fit-spectrum-prereqs.md` Q3. Scheduled: Phase 5.
- **Group-name validator workarounds** — several `GROUP_NAME` values
  (`_definitions.py:270`, `:273`, `:280`, `:286`, `:291`, `:297` — each marked
  `# *** DEBUG *** validator bug`) and `DGRAM_TWO_THETA_NAME = "XAXIS"`
  (`_definitions.py:299`) were set to upper-case /
  generic forms to work around a `nexusformat` validator bug. The generic names
  are likely acceptable on-disk schema-wise, but the production intent is
  meaningful lowercase names. **Resolution:** expose as a `Config` toggle
  (Section 2.3) so the writer can emit either form without a code change.
  Whether uppercase names are strictly allowed by NXstress requires a
  schema-spec check. Scheduled: Phase 1 (Config infrastructure) + Phase 2
  (flip default once validator bug is resolved upstream).
- **`allowed_identifier` disallowed-character coverage incomplete** —
  **CLOSED (Phase 2, spec 04).** The encoding is now defined over an explicit
  alphabet with an underscore-doubling escape, injective and round-tripping
  (`_definitions.py:574-640`), proven by
  [`probes/a5_identifier_policy.py`](probes/a5_identifier_policy.py) and
  [`probes/a4_validitemname_rule.py`](probes/a4_validitemname_rule.py) and
  pinned by `test_definitions.py`, which iterates `GROUP_NAME` rather than
  restating it. Decisions items 23 and 25.

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
- **`DENEXDetectorGeometry` accepts but discards its `calibrated` flag,
  and its arm-shift-application code path is dead/broken** — corrected
  location: not `file_object.py:510` (that FIXME doesn't exist; the only
  comment there is an unrelated return-type-annotation TODO). The real gap
  is entirely within `instrument_geometry.py::DENEXDetectorGeometry`
  itself. Scheduled: Phase 5 (spec 08).
- **`HidraProjectFile` legacy log-name patches** (`file_object.py:404`, `:494`
  FIXME) — audit to ensure NXstress does not codify the legacy names into the
  schema-conformant file. Scheduled: Phase 2.
- **`ScalarFieldSample` / `StrainField` `NotImplementedError` methods** in
  `fields.py` (L239, 611, 816, 927, 956, 960, 964, 983, 986, 994, 1184) — not
  directly on the write path, but any read-back that reconstructs these fields
  for the StrainStressViewer will hit them. Scheduled: Phase 3 (as needed).
- **`ReductionController.save_project`** (`manual_reduction/pyrs_api.py:190`
  — not `HB2BReductionManager`, an unrelated class elsewhere) is
  `NotImplementedError`, but has zero callers anywhere in the codebase.
  **Resolved: NXstress does not attach there.** The real, currently-functional
  save path is `reduce_hidra_workflow`'s automatic
  `ReductionApp.save_diffraction_data` call — that's where spec 07 hooks in
  instead. `save_project` remains an independent, optional PyRS cleanup item
  (spec 06), not required for NXstress. Not scheduled in this plan's phases.
- **`nexus_conversion.py:118, 374`** — two unimplemented branches, confirmed
  unrelated to any save/output-format path (an unsupported log-property
  type and a non-`.xml` mask-file restriction, both reachable only from
  NeXus-*conversion*, not from saving). Not on the NXstress critical path;
  tracked as an independent PyRS item (spec 06), not scheduled in this
  plan's phases.

### 2.3 New capability: Config (via `neutrons_standard`)

PyRS has no runtime configuration file. The installed scripts (`pyrsplot`,
`pyrs-calibration`, `create-mask`) use either no CLI flags or positional
arguments only; no `argparse` + `--config` plumbing exists, and this work
does not add any — a plan reviewer asked that PyRS instead build on the
shared `Config` singleton from `neutrons_standard`
(`github.com/neutrons/PythonCommons`, `neutrons` pixi channel), which is
`env`-var driven, not CLI-driven. This work adds:

- The `neutrons` pixi channel and `neutrons_standard` dependency
  (`[tool.pixi.dependencies]` / run-dependencies) — no `pyyaml` dependency
  needed; `neutrons_standard.Config` handles its own YAML I/O internally.
- A YAML config file at `pyrs/resources/application.yml` — this exact
  filename and location (a genuine `pyrs.resources` subpackage) is a hard
  requirement of `neutrons_standard`, not a PyRS convention. Two fully
  parallel, self-contained top-level sections, one per output format — each
  owns its own `enable` flag and its own `extension`; nothing is shared or
  ambiguous between them:
  ```yaml
  nxstress:
    enable: true
    extension: ".nxs"
    use_production_names: false
    discriminator_fields: []
  legacy_io:
    enable: true
    extension: ".h5"
  ```
  - `nxstress.enable` / `legacy_io.enable: bool` — whether each format is
    writable at all. Every NXstress-wired viewer's save action reads its
    own format's `enable` flag directly to decide whether that action is
    clickable (`QAction.setEnabled`, not `setVisible` — the action stays
    visible, just grayed out, when disabled).
  - `nxstress.extension` / `legacy_io.extension: str` — each format's file
    extension. **Imposed, never user-chosen:** every save action enforces
    its own section's extension on whatever filename a `QFileDialog`
    returns; a user cannot type a different extension and have it honored.
  - `pyrs.utilities.config.validate_config()` raises if
    `not (nxstress.enable or legacy_io.enable)` — at least one format must
    be writable. `neutrons_standard.Config` provides no schema validation
    of its own; this is PyRS's own rule, run eagerly at import time.
  - `nxstress.use_production_names: bool` — toggle between the current
    validator-safe generic group names and the production lowercase forms.
  - `nxstress.discriminator_fields: list[str]` (default `[]`) — names of the
    fields that discriminate one input `HidraWorkspace` from another when
    `NXstress.write` is given more than one (spec 04b).
  - ~~`nxstress.merge_workspaces: bool`~~ — **removed** (Decisions Log row
    45): more than one workspace with `discriminator_fields` empty is always
    refused; merging is done in PyRS before NXstress is called.
  - Reserved namespace for future flags (`nxstress.write_raw_counts`,
    `nxstress.strict_schema_validation`, etc.).
- Packaging: no change needed — `pyproject.toml`'s existing
  `[tool.hatch.build.targets.wheel] artifacts` glob `"pyrs/**/*.yml"`
  already covers the new `pyrs/resources/` location.
- `pyrs/utilities/config.py` — registers PyRS with `neutrons_standard`
  (`neutrons_standard.init("pyrs")`, which must run before
  `neutrons_standard.config` is ever imported) and re-exports the raw
  `Config` singleton plus `validate_config()`. `Config` must always be
  imported from *this* module, never `from neutrons_standard...import
  Config` directly elsewhere in the codebase — a stray direct import would
  race `init()` and silently corrupt `package_name` for the whole process.
  Callsites use dot-string key access directly (`Config["nxstress.enable"]`)
  — no typed wrapper layer.
- No `argparse`/`--config` wiring on any installed script. Configuration is
  overridden by setting the `env` OS environment variable to the name or
  path of a `.yml` file, whose contents are deep-merged on top of the
  shipped default (`neutrons_standard`'s own idiom) — e.g.
  `env=/path/to/override.yml pyrsplot`. `neutrons_standard` also
  auto-loads a `~/.pyrs/pyrs-user.yml` override when one exists and no
  explicit `env` is set.

This is a prerequisite for closing the Section 2.1 group-name validator
workaround cleanly. Scheduled: Phase 1.

### 2.4 Beyond-code decisions

- **File-extension policy** — **decided:** distinct `*.nxs` extension across
  every viewer; coexists with existing `*.h5` `HidraProjectFile` paths. Each
  format's extension (`nxstress.extension`/`legacy_io.extension`) is
  imposed by the action a user clicks — never typed/chosen by the user, and
  never auto-detected from an existing file's suffix.
- **Coexistence vs. replacement** — **decided (phased):** Phase 1 ships
  additive (both paths coexist; NXstress offered as new menu entries and file-
  dialog filters, each independently enabled per `nxstress.enable`/
  `legacy_io.enable`). Phase 6 flips defaults (NXstress becomes primary;
  `HidraProjectFile` becomes legacy-read-only) — **purely by flipping the
  shipped config default** (`legacy_io.enable: true` → `false`); no viewer
  code changes, since every wired viewer's save actions were already built
  in their own spec to read these flags directly. Concrete acceptance
  criteria for the Phase 6 flip to be defined at the end of Phase 1
  (tracked, still open — see `open-questions/10-flip-defaults.md` Q1).
- **Multi-direction semantics** — StrainStressViewer today loads N project
  files per direction into three parallel slots (`filenames_11/22/33`).
  **Decision:** this is a special case of the general multi-workspace
  mechanism built in spec 04b — one `HidraWorkspace` per direction in, one
  `.nxs` file (single NXentry) out, with `direction` as one of 04b's
  explicit discriminator fields on the compound peak index. NOT
  one-NXentry-per-direction. The schema question this previously gated on —
  whether `NXreflections` permits additional index columns at all — is
  downgraded from a blocking gate to a tracked follow-up: `_peaks.py`
  already writes `mask`/`scan_point`/etc. as non-required columns, which is
  the same category of extension; see
  `open-questions/04b-multi-workspace-nxstress.md` Q1 for the precedent and
  the pending cross-check against `NXstress.html` and the `nexusformat`-org
  validator once both are added to the repo — tracked as a concrete reminder
  in [10-flip-defaults.md](10-flip-defaults.md)'s Overview, not just noted
  here.
- **Multi-workspace I/O (new)** — beyond StrainStressViewer's specific case,
  `NXstress.write` / `NXstress.read` are generalized to
  `list[HidraWorkspace]`, round-trip symmetric (spec 04b). Workspace
  boundaries are recovered from discriminator field(s) named by the new
  `nxstress.discriminator_fields` config key (§2.3), resolved per workspace
  by preferring a matching `HidraWorkspace` `@property` and otherwise
  falling back to `SampleLogs`, and carried internally name-keyed rather
  than positionally so config reordering between write and read can't cause
  silent misattribution. The specific field names any deployment configures
  are decided at 04b implementation kickoff, not in this document. The
  existing no-overlap invariant (each input workspace contributes only
  unique scan points and/or other index fields) is assumed, not newly
  validated by this decision — 04b does add an explicit write-time check for
  it, and raises if asked to combine more than one workspace with no
  discriminator fields configured (the `nxstress.merge_workspaces` opt-out
  this once allowed is removed, Decisions Log row 45). Discriminator values are the most slowly
  varying coordinate of the combined index (`sort_key` prepends them) — not
  because the schema requires global sorting (it doesn't; the reader only
  requires each compound key's run to be contiguous, `_peaks.py:246-338`),
  but because it makes each workspace one contiguous super-block, which is
  what keeps the read-side split and the scan-point family's exact-match
  reader (`_input_data.py:70-72`) simple. See
  `open-questions/04b-multi-workspace-nxstress.md` Q6.
- **Append mode, library-only, tail-append-only (new)** — append (spec 04c)
  ships as a tested `NXstress(path, "a")` capability with **no GUI entry
  point** in this pass. The one pathway that might have used it
  (ManualReductionViewer, spec 07) is decided to always write a fresh file
  instead, so append and the ManualReduction hookup are no longer bundled
  (they were, in an earlier draft of spec 07). **Sortedness is relaxed for
  append**: a single-step write still produces a fully sorted index, but
  append sorts only its own incoming batch and does not re-sort the file —
  "locally sorted, globally segmented." This is safe because the reader
  never required global order (only run-contiguity and monotonic
  `scan_point`, both still enforced). It also narrows what 04c needs to
  implement to a plain tail-append for the common case (a new workspace);
  extending scan points on a workspace/key already in the file is a
  different, insertion-shaped operation this pass explicitly does not
  implement (`NotImplementedError`). See
  `open-questions/04c-nxstress-append.md` Q5.
- **Multi-NXentry semantics (unchanged)** — multiple NXentry within a single
  `.nxs` file remain reserved for "distinct data-reduction or sample
  conditions" (`NXstress.py:107-125`). The compound peak index is the axis for
  finer variation (masks, phases, directions) within a single condition.
- **Read-back completeness** — `NXstress.read` reconstructs wavelengths, sample
  logs, masks, reduced diffraction data, and peak collections. It does NOT
  reconstruct raw counts unless the optional `input_data` group was written.
  Verify each viewer's load path is satisfied by this subset during Phase 1
  wiring. **Post-04b, `read()` returns `list[HidraWorkspace]`, not a single
  `HidraWorkspace`** — each reconstructed workspace's sample logs, raw
  counts, and diffraction data are recovered by splitting the concatenated
  scan-point family via value-set membership against that workspace's own
  scan points (recovered, in turn, from the peak-index split) — see 04b's
  Q7 for the mechanism and the invariant it depends on (every input
  workspace needs at least one `PeakCollection` when N>1).

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
- Introduce the `Config` infrastructure (Section 2.3):
  `pyrs/resources/application.yml`, `pyrs/utilities/config.py` (no CLI
  wiring — `env`-var driven, per `neutrons_standard`'s own idiom). Ship
  with `nxstress.use_production_names = false`
  (validator-safe generic names), `nxstress.enable = true`,
  `legacy_io.enable = true` as the defaults (both formats available,
  matching today's `.h5`-only behavior plus the new NXstress option).
- Wire `NXstress` into `PeakFittingModel.save_fit_result` and
  `PeakFittingModel.load_hidra_project` as a new save/load path, triggered by
  a new `Save as NXstress…` menu action and an added `NXstress (*.nxs)` filter
  in the load dialog (`peak_fitting_viewer.py`, `peak_fitting_model.py`).
  Both this action and the existing `Save` (`.h5`) action are wired to read
  `nxstress.enable`/`legacy_io.enable` for their own enabled/disabled state.
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
  `pyrs/resources/application.yml` once the upstream `nexusformat` validator
  bug is resolved.
- Parameterize the hardcoded instrument name in `_instrument.py:70`
  (currently `"HB2B"`) — draw from `Config` or from `HidraWorkspace`.
- Detector rotation-order cross-check + fix in `_instrument.py:165`
  against the reduction code's convention; add a regression test.

---

### Phase 2/3 bridge — Multi-workspace NXstress I/O and append mode

**Goal:** Generalize `NXstress.write` / `NXstress.read` from one
`HidraWorkspace` to `list[HidraWorkspace]` (round-trip symmetric), then add
append mode on top of that mechanism as a separate, library-only capability.
These two specs sit between Phase 2 (internal cleanup, which touches the same
`_peaks.py`/`_fit.py`/`_instrument.py` functions this generalizes) and Phase 3
(StrainStressViewer, which is a direct consumer of the new signature).

> **Schema precedent already exists — a tracked follow-up, not a blocker:**
> `_peaks.py` already writes `mask`, `scan_point`, and other non-required
> columns onto `NXreflections`; a discriminator column is the same category
> of extension. Cross-check against `NXstress.html` and the `nexusformat`-org
> validator once both are added to the repo (tracked as a concrete reminder
> in [10-flip-defaults.md](10-flip-defaults.md)'s Overview), and record the
> outcome — but that verification doesn't gate starting 04b. This supersedes
> the narrower
> `direction`-axis question this document previously scheduled as a hard gate
> for Phase 3 kickoff. See `open-questions/04b-multi-workspace-nxstress.md` Q1.

**[04b — Multi-workspace NXstress I/O](04b-multi-workspace-nxstress.md):**
- **Depends on:** spec 01 (Config infrastructure), in addition to 04 — new
  config keys `nxstress.discriminator_fields` and `nxstress.merge_workspaces`
  (§2.3; `merge_workspaces` since removed, Decisions Log row 45) land in `pyrs/resources/application.yml`, delivered by spec 01.
- **Required PyRS changes:** none. Discriminator-value resolution (property
  on `HidraWorkspace` if one exists, else `SampleLogs` fallback) lives
  entirely in NXstress's own code — no new `HidraWorkspace` method.
- **Required NXstress extensions:** change `write`/`read` to accept/return
  `list[HidraWorkspace]`; resolve discriminator field names from
  `Config["nxstress.discriminator_fields"]` (`pyrs.utilities.config.Config`);
  add a name-keyed discriminator slot to `_peaks.py::PeakIndex` (not
  positional — robust to config reordering between write and read),
  **prepended** to `sort_key` so discriminator values are the most slowly
  varying coordinate (each workspace forms one contiguous super-block —
  not a schema requirement, but what keeps the read-side split and the
  scan-point family's exact-match reader simple; see
  `open-questions/04b-multi-workspace-nxstress.md` Q6); update
  `sort_key`, `validateNoDuplicatePeaks`, `_init`, `init_group`,
  `peakCollectionsFromNexus`; merge logic across workspaces in
  `_InputData`/`_Sample`/`_Instrument`/`_Fit` `init_group` (plain
  concatenation in workspace order); raise if asked to
  combine more than one workspace with no discriminator fields configured
  (no `merge_workspaces` opt-out: Decisions Log row 45); update specs 02/03 call
  sites to the new (length-1-list-compatible) signature.
- Retires the direction-aware-container item this document previously
  scheduled under Phase 3 — see the Decisions Log update below.

**[04c — NXstress append mode, library only](04c-nxstress-append.md):**
- **Required PyRS changes:** none. Append tail-appends directly against
  on-disk NXstress arrays — it does not reconstruct the existing entry as
  `HidraWorkspace`/`PeakCollection` to avoid round-tripping potentially
  large existing raw-count data through PyRS's outer types.
- **Required NXstress extensions:** implement coordinated **tail-append**
  (not general insertion — see below) across both position-aligned
  families — peak-index family (`_peaks.py`'s compound index at
  `_peaks.py:180-181`, reusing the existing `_append_peak`
  `resize(cur+N); arr[cur:] = …` shape, plus `_fit.py::_PeakParameters`/
  `_BackgroundParameters`, which are positionally aligned with it, not
  independently keyed) and scan-point family (`_input_data.py:44-46,63-72`
  `detector_counts`, `_sample.py`'s per-scan-point logs, and
  `_fit.py::_Diffractogram`). `_instrument.py::_Masks` needs no change — it's
  name-keyed and already append-capable. `NXstress.py`: add entry targeting
  (`entry_number`, defaulting to the last entry) and remove the guard at
  `NXstress.py:151-152` for the append case. Conflict/classification
  policy: a new compound key (Case A — a new workspace) proceeds as a
  tail-append; an exact duplicate raises `RuntimeError` (invalidates the
  instance); a key already on disk that would need more scan points
  inserted mid-run (Case B) raises `NotImplementedError` — that's a
  different, insertion-shaped operation this pass deliberately doesn't
  implement. Global sortedness is not maintained across an append — files
  become "locally sorted, globally segmented" — since the reader never
  required it (only run-contiguity and monotonic `scan_point`, both still
  enforced); see `open-questions/04c-nxstress-append.md` Q5. **No GUI
  wiring** — no viewer in this plan calls append in this pass.
- Round-trip test: write, then append a new workspace, into the same entry;
  verify combined contents.

---

### Phase 3 — StrainStressViewer

**Goal:** Hook up StrainStressViewer, using 04b's multi-workspace mechanism
with `direction` as the discriminator field — one `HidraWorkspace` per
direction in, one `.nxs` `NXentry` out.

**Required PyRS changes:**
- Add a settable `direction` `@property` to `HidraWorkspace`
  (`pyrs/core/workspaces.py`) — needed because 04b resolves discriminator
  values from the workspace object itself (a matching property, else a
  `SampleLogs` fallback), and `HidraWorkspace` had no notion of direction;
  the viewer previously tracked it only at the model level
  (`filenames_11/22/33`). This also required a correction to 04b's own
  resolver — see the Decisions Log update below.
- Resolve any `NotImplementedError` in `pyrs/dataobjects/fields.py`
  (L239, 611, 816, 927, 956, 960, 964, 983, 986, 994, 1184) that falls on the
  StressField reconstruction read path.

**Required NXstress extensions:**
- Confirm `direction: str` (`"11"`/`"22"`/`"33"`) composes correctly with
  04b's discriminator mechanism; no new `PeakIndex` machinery is built here.
- Update the default `pyrs/resources/application.yml`'s
  `nxstress.discriminator_fields` from `[]` to `["direction"]`;
  `save_as_nxstress` validates this key is set correctly before calling
  `write()`, raising a clear error otherwise.
- Wire NXstress into `strainstressviewer/model.py::load_hidra_project_file` /
  `load_hidra_project_files` — read a direction-indexed NXstress file and
  reconstruct `StrainField` / `StressField` in the viewer.
- Add a `Save as NXstress…` action on the StrainStressViewer.
- Round-trip test: construct a `StressField` from a single NXstress file,
  confirm it matches the CSV-summary output for the same inputs.

---

### Phase 4 — ManualReductionViewer hookup

**Goal:** Wire NXstress into `reduce_hidra_workflow`'s existing, automatic
save step.

**Required PyRS changes:** _none._ Earlier drafts of this phase listed
implementing `HB2BReductionManager.save_project` as a prerequisite — that
was based on the wrong class name (the real class is `ReductionController`)
and the wrong hookup point: `save_project` has zero callers anywhere in the
codebase, so it was never blocking anything. The real, currently-functional
save path is `reduce_hidra_workflow`'s automatic
`ReductionApp.save_diffraction_data` call, which already works today and
needs no PyRS-side prerequisite. (`save_project`, and the two unrelated
`nexus_conversion.py:118,374` input-parsing `NotImplementedError`s, remain
an independent, optional PyRS cleanup item — spec 06 — not scheduled in
this plan.)

**Required NXstress extensions:**
- Wire NXstress into `reduce_hidra_workflow`
  (`pyrs/interface/manual_reduction/pyrs_api.py`): write once per
  currently-enabled format (`legacy_io.enable` → `.h5` via the existing
  `reducer.save_diffraction_data` call; `nxstress.enable` → `.nxs` via
  `NXstress(path, "w").write([hidra_ws], [])` — a length-1 list, per 04b's
  signature, which lands before this phase), same basename for both when
  both are enabled. No new GUI dialog or menu action — this viewer has none
  to extend. An explicit `project_file_name`'s extension, if given, is
  stripped and ignored; only its basename is used.
- Round-trip test: reduce a run with each enablement combination
  (NXstress-only, both, legacy-only) and confirm the expected file(s) are
  written and readable.

---

### Phase 5 — Reconstructed fit spectrum + detector-calibration fidelity

**Goal:** Replace the remaining `NaN` placeholders with real data — except
`STRESS_FIELD`, which stays a documented, investigated-but-blocked gap
(see Decisions Log item 14; no dataset anywhere in the repo contains what's
needed, and this cannot be resolved without new information from the
instrument-science team).

**Required PyRS changes:**
- Fit engine (`pyrs/peaks/peak_collection.py`,
  `pyrs/core/peak_profile_utility.py`): add a method that returns the
  reconstructed model spectrum on the original two-theta grid, indexed by
  `PeakCollection` × mask × scan_point, built on the underlying peak-shape/
  background evaluator functions directly — the one existing wrapper
  (`calculate_profile`) is dead, buggy code, not a usable starting point.
  Propagated variance is a documented approximation (no covariance matrix
  exists anywhere in PyRS). This is the largest single PyRS-side item in
  the plan.
- Extend `PeakCollection.__set_fit_status` so an unrepresentable parameter
  error also sets `_exclude_list`, unifying "no representable variance"
  with the existing exclusion mechanism.
- Fix `DENEXDetectorGeometry` (`pyrs/core/instrument_geometry.py`): store
  the `calibrated` flag it already accepts but discards; add an
  `arm_shift_applied` accessor; fix `apply_shift`'s existing
  class-vs-instance `AttributeError` bug; retain the pre-shift arm-length
  value so it's recoverable after a shift is applied. (Not
  `file_object.py:510` — that FIXME doesn't exist; the gap is entirely
  within this class.)
- Add a `beam_intensity_profile` property (get/set) to `HidraWorkspace`,
  defaulting to a documented uniform/constant value — no real
  beam-intensity data path exists anywhere in PyRS today.
- `STRESS_FIELD` shape verification: **blocked**, not scheduled to
  complete in this phase. Investigated directly; no usable example dataset
  exists in the repo.

**Required NXstress extensions:**
- Populate `_fit.py:425-429,430-438` diffractogram `fit` / `fit_errors` from
  the new fit-engine method — these fields are zero-sized resizable
  datasets, so the writer must resize them to the real shape first, not
  index-assign into a pre-shaped array. By this phase, `_Diffractogram.init_group`
  already takes `list[HidraWorkspace]` (04b) — resize to the *total*
  concatenated scan-point count across all input workspaces, and place
  each `PeakCollection`'s reconstructed spectrum by scan_point-*value*
  lookup against the concatenated `scan_point` array, not by position
  (see 04b's Q7).
- Change `_Diffractogram.diffractogramFromNexus`'s return from a plain
  4-tuple to a `NamedTuple` (`DiffractogramData`, six fields including
  `fit`/`fit_errors`); update its one caller (`NXstress.py:218`) to match.
- Populate `NXbeam` intensity profile in `_instrument.py:126` (construction;
  TODO at `:127`) by reading `HidraWorkspace.beam_intensity_profile`
  unconditionally — no "if measured" branch; the property always has a
  value. Per-scan-point, like wavelength (04b's Q7 clarification) — this
  spec concatenates it across `wss` alongside the rest of the scan-point
  family, it does not validate it for cross-workspace equality.
- Fix the L2 arm-shift round-trip in `_instrument.py:132-150` (the
  calibrated/uncalibrated branch; TODO + `distance = arm_length` at
  `:138-140`) using the new `DENEXDetectorGeometry` accessors — the
  double-counting risk this item previously worried about is already
  resolved at the source (PyRS-side), so this is a plain consumer update.
  Reader side (`instrumentFromNexus`): must read the new
  `arm_shift_applied` flag and use it to decide whether to call
  `apply_shift` or use the raw arm-length value directly, or a round-trip
  will double-shift or under-shift.
- `_sample.py:171` `STRESS_FIELD` dimensions: **not touched** — remains
  blocked, carried forward from the PyRS-side item above.

---

### Phase 6 — Flip defaults (NXstress becomes primary)

**Goal:** Complete the phased-replacement decision. Acceptance criteria for
triggering this phase to be defined at the end of Phase 1.

**Required PyRS changes:** _none._

**Required NXstress extensions:**
- Set `legacy_io.enable = false` and `nxstress.use_production_names = true`
  as the defaults in `pyrs/resources/application.yml` (`nxstress.enable`
  stays `true`, already the Phase 1 default). **No viewer code changes** —
  every wired viewer's save actions already read these flags directly
  (specs 02, 03, 05, 07), so flipping the shipped default is sufficient to
  demote `HidraProjectFile` save paths to disabled-by-default (loading
  remains unaffected, still works).
- Write deprecation-warning docs and a migration note for existing users
  with `.h5` project files, including how to re-enable `.h5` saving via an
  `env`-var-pointed override file (or an auto-loaded
  `~/.pyrs/pyrs-user.yml`) if still needed.

---

### Skipped

**DetectorCalibrationViewer** — its outputs (calibration JSON, Mantid NeXus
event files) are independent of the NXstress project-file layout. No NXstress
hookup is planned unless a downstream requirement emerges.

---

## 4. Decisions Log

| # | Question | Decision | Notes |
|---|---|---|---|
| 1 | File-extension policy | **New `*.nxs` filter** across every viewer; NXstress writes to `.nxs`, coexisting with existing `*.h5` `HidraProjectFile` paths. | No auto-detection needed; extension is imposed by the action clicked, never typed by the user. Config keys: `nxstress.extension`, `legacy_io.extension` (superseding the earlier single `nxstress.default_extension` key). |
| 2 | Coexistence vs. replacement | **Additive now, replace later (phased).** Keep existing `HidraProjectFile` save/load unchanged in Phase 1; add NXstress as new menu entries and file-dialog filters, each independently enabled per `nxstress.enable`/`legacy_io.enable`. Phase 6 flips defaults so NXstress is primary and `HidraProjectFile` becomes legacy-read-only — **purely a config-default flip** (`legacy_io.enable: true` → `false`), no viewer code changes, since the enablement wiring was already built once, per-viewer, in Phase 1/3/4. | Phase-6 trigger criteria to be defined at the end of Phase 1 (still open — `open-questions/10-flip-defaults.md` Q1). |
| 3 | StressField / multi-direction persistence | **Decided:** `direction` is one discriminator field of the general multi-workspace mechanism built in spec 04b (item 6 below) — one `HidraWorkspace` per direction in, one `.nxs` `NXentry` out. `StressField` is reconstructed in the viewer from the indexed peaks at read time. | Supersedes the earlier provisional framing (a StrainStress-specific `direction` axis). The schema question this depends on is downgraded from a blocking gate to a tracked follow-up — see `open-questions/04b-multi-workspace-nxstress.md` Q1 for the precedent (`_peaks.py` already writes non-required columns) and the pending validator cross-check. |
| 4 | NaN placeholder tolerance | **`NaN` is acceptable for Phase 1.** No NXstress TODO is a Phase-1 blocker. Every TODO in Sections 2.1 and 2.2 is scheduled across Phases 2–5. | The phased schedule (Section 3) is the deliverable of this decision. |
| 5 | Priority ordering | **Keep drafted order.** Phase 1 hooks up PeakFitting, Texture, and CombineRuns (data model matches `NXstress.write` 1:1, zero upstream changes). StrainStressViewer comes in Phase 3, after Phase 2 stabilizes NXstress internals and after the 04b multi-workspace mechanism lands. | Ship something quickly, learn from real use, then tackle the harder integration. |
| 6 | Multi-workspace I/O & round-trip symmetry | **Decided:** `NXstress.write`/`read` generalize to `list[HidraWorkspace]`, round-trip symmetric — `read` recovers the same N workspaces `write` was given. Workspace boundaries are recovered from discriminator field(s) named by the new `nxstress.discriminator_fields` config key, resolved per workspace via a property-or-`SampleLogs`-fallback resolver and carried name-keyed (not positionally); the specific field names any deployment configures are decided at spec 04b's implementation kickoff, not here. | New capability, not in the original plan. Spec: [04b](04b-multi-workspace-nxstress.md). Assumes the existing no-overlap invariant (each input workspace contributes only unique scan points and/or other index fields); 04b adds an explicit write-time check for it, and raises on N>1 with no discriminator fields configured unless `nxstress.merge_workspaces` opts into merging instead. No new `HidraWorkspace` method — resolution logic is NXstress-internal. |
| 7 | Append mode scope | **Decided:** append ships as a tested library capability (`NXstress(path, "a")`) with **no GUI entry point** in this pass. The ManualReductionViewer hookup (spec 07) always writes a fresh file instead, so it no longer depends on append. | New capability, not in the original plan; also un-bundles what was previously one spec (append + ManualReduction). Spec: [04c](04c-nxstress-append.md). |
| 8 | Append architecture & scope | **Decided (superseded in part by item 17 — see there for the current scope):** append operates directly against on-disk arrays (no round-trip through `HidraWorkspace`/`PeakCollection` for the existing entry); covers both position-aligned families — peak-index (peaks + peak_parameters + background_parameters) and scan-point (detector_counts + sample logs + diffractogram) — not just the originally-scoped `_input_data.py`/`_peaks.py` pair. `NXstress(path, "a")` targets the last entry by default, with `entry_number` as an explicit override. **As of item 17, this is tail-append only, not general insertion**, and the conflict outcome is three-way, not binary: a new compound key (Case A) proceeds; a key already on disk needing more scan points inserted mid-run (Case B) raises `NotImplementedError` without invalidating the instance; only an exact duplicate raises `RuntimeError` and invalidates it. | Spec: [04c](04c-nxstress-append.md). Corrects an under-scoped original draft: `_fit.py::_PeakParameters`/`_BackgroundParameters` build rows in the same sort order as the peaks index (`_fit.py:87`), so a partial append would desynchronize the entry. |
| 9 | `direction` storage & config default | **PARTLY SUPERSEDED BY ROW 42 — read that first.** The property is read-only and *restorable*, not settable; the config half below still stands. Original text: **Decided:** `HidraWorkspace` gains a settable `direction` `@property` (get/set) — `HidraWorkspace` previously had no notion of direction at all; the viewer tracked it only at the model level (`filenames_11/22/33`), which doesn't fit 04b's workspace-resolved discriminator mechanism. 04b's discriminator resolver is corrected to be bidirectional: "get" from the workspace at write time (as before), "set" onto each reconstructed workspace at read time (new) — so any property-backed discriminator round-trips with no NXstress-side special-casing. `pyrs/resources/application.yml`'s `nxstress.discriminator_fields` default changes from `[]` to `["direction"]`; `save_as_nxstress` validates this precondition and raises a clear error if unmet. | Spec: [05](05-strain-stress-viewer.md); the resolver correction lands in [04b](04b-multi-workspace-nxstress.md). |
| 10 | CombineRuns keeps its pre-merge | **Decided:** spec 03's `.nxs` export path does **not** switch to 04b's N-workspace mechanism. `HidraWorkspace.append_hidra_project` (used by `combine_project_files`) already discards per-run boundaries the same way `nxstress.merge_workspaces: true` would — same resulting semantics, already implemented at the PyRS layer. The existing `.h5` export path still needs the single merged workspace regardless, so switching only `.nxs` would mean two data paths through `CombineRunsModel` for an identical output file. The `.nxs` branch passes the already-merged workspace as a length-1 list: `NXstress.write([self._hidra_ws], [])`. No `discriminator_fields`/`merge_workspaces` config is touched by this spec. | Spec: [03](03-combine-runs-nxstress.md). Retires `open-questions/04b-multi-workspace-nxstress.md` Q3, which had floated this as a possible simplification. |
| 11 | Config schema: two independent format sections | **Decided:** replaces the single `nxstress.default_extension` key with two fully parallel, self-contained top-level sections — `nxstress` and `legacy_io` — each owning its own `enable` flag and its own `extension`. Each viewer's save action reads its own format's `enable` to decide whether it's clickable (`setEnabled`, never `setVisible` — the action stays visible, grayed out when disabled), and each action *imposes* its own format's `extension` on whatever a user types, rather than accepting a user-chosen extension. `validate_config()` raises if neither format is enabled. No single GUI action ever auto-writes both formats — a user with both enabled invokes the two independent actions manually, one at a time. | Spec: [01](01-config-and-test-infra-PR.md). Two earlier framings were tried and corrected first: an `output_mode` enum where "both" meant one action auto-writing two formats (rejected — bad UI design); a single `mode` enum plus one shared `default_extension` (rejected — doesn't make sense once two independently-enabled formats each need their own extension). |
| 12 | ManualReductionViewer: automatic write, not a GUI action | **Decided:** `reduce_hidra_workflow` has no button click to protect the meaning of (it saves automatically, as an inherent side effect of every reduction) — so the "never auto-write both formats" rule from item 11 does not apply to it. It simply writes once per currently-enabled format: one file if only one is enabled, both files (same basename) if both are. No tie-break, no ambiguity. An explicit caller-supplied `project_file_name`'s extension is stripped and ignored — never validated against, never a reason to raise; only its basename is used, exactly as for the auto-derived case. | Spec: [07](07-manual-reduction-nxstress.md). Considered and explicitly deferred: giving this viewer a real Save/Save-As action, and a shared `Viewer` ABC across all five viewers — legitimate future architecture work, but a workflow change unrelated to NXstress, not entangled with this rollout. |
| 13 | Specs 06/07 decoupled; spec 07 hookup point corrected | **Decided:** spec 07 does not depend on spec 06 at all. Tracing the actual code found two errors in the original drafts: the class is `ReductionController`, not `HB2BReductionManager` (an unrelated class in `pyrs/core/reduction_manager.py`); and `ReductionController.save_project` — spec 06's target — has **zero callers anywhere in the codebase**, so implementing it unblocks nothing. The real, currently-functional save path is `reduce_hidra_workflow`'s automatic `ReductionApp.save_diffraction_data` call, which spec 07 now hooks into directly. Spec 06's other item (`nexus_conversion.py:118,374`) is also confirmed unrelated — both branches are on the NeXus-*conversion* step, not the save path. | Specs: [06](06-manual-reduction-prereqs.md) (re-scoped to an independent, optional PyRS cleanup item, not scheduled in this plan's phases), [07](07-manual-reduction-nxstress.md). |
| 14 | Spec 08 corrected: calibration fix scope, beam-profile mechanism, STRESS_FIELD blocked | **Decided, three parts.** (a) The `file_object.py:510` FIXME cited in the original draft doesn't exist there — the real gap is entirely within `DENEXDetectorGeometry` (discards its `calibrated` arg; `apply_shift` is dead code that would raise `AttributeError` if called; destructively overwrites `arm_length` with no way to recover the pre-shift value). Fix stays scoped to that class + NXstress; **no `.h5` format change** in this pass — explicitly flagged as a strong future need, not dropped. (b) No beam-intensity data path exists anywhere in PyRS (confirmed); rather than NXstress hardcoding a uniform placeholder, `HidraWorkspace` gains a `beam_intensity_profile` property (`direction`-style convention) that NXstress reads unconditionally, per-scan-point, concatenated across `wss` like wavelength (not validated for cross-workspace equality — see item 18) — forward-compatible with a future real beam-monitor path for free. (c) `STRESS_FIELD` shape verification is **blocked, not resolved** — the three example files a stakeholder named are real but contain no `STRESS_FIELD` log, only an unrelated `StrainDirection` label; strain and stress are physically distinct quantities, so that label is not a valid substitute. Documented as an explicit, tracked blocker rather than guessed at. | Specs: [08](08-fit-spectrum-prereqs.md), [09](09-fit-spectrum-nxstress.md) (carries the `STRESS_FIELD` block forward; the other three items proceed normally). |
| 15 | Spec 09 corrected: line numbers, `NamedTuple` return, concrete reader scope | **Decided, three parts.** (a) Cited line numbers in the original draft were imprecise (off by a few; one citation pointed at an unrelated docstring note) — corrected against the current code. (b) `diffractogramFromNexus`'s return type changes from a plain (and now-growing) positional tuple to a `NamedTuple` (`DiffractogramData`) — self-documenting, avoids positional ambiguity as more fields are added; its one existing caller (`NXstress.py:218`) is updated to match. (c) `instrumentFromNexus`'s reader-side scope, previously vague ("update... correspondingly"), is now concrete: it must read the new `arm_shift_applied` flag and use it to decide whether to call `apply_shift` or use the raw arm-length value, or a round-trip double-shifts. Also confirmed as a simplification: `_Diffractogram.init_group`'s `peakss` parameter already exists, unused — no new parameter needed. | Spec: [09](09-fit-spectrum-nxstress.md). |
| 16 | Spec 10 corrected: status-bar UI gap, validator/schema-doc unavailability | **Decided, two parts.** (a) Only `PeakFittingViewer` has a status bar today — `TextureFittingViewer`, `CombineRunsViewer`, and `StrainStressViewer` have none, only modal dialogs. Spec 10 now explicitly adds a status bar to the three that lack one (cheap — `self.statusBar()` creates it lazily), for consistent non-blocking deprecation-hint UX across all four; this is the one piece of genuinely new UI in an otherwise pure config-flip spec. `StrainStressViewer`'s one-time hint flag must live at the window level, since its three direction slots share one load handler. (b) The `nexusformat`-org NXstress validator and the `NXstress.xml`/`.html` schema doc — referenced in specs 04's, 09's, and 10's Verification sections since spec 04b's open questions (corrected from an earlier draft's "02" — spec 02 has no validator reference at all; spec 04's is the one that does) — are confirmed **not yet added to the repo**; the installed `nexusformat` 1.0.8 package has no validator capability at all. A reminder is added to spec 10's Overview to add both (the validator as a link to its separate repository; the schema doc under `docs/developer/source/design/nexus/`, linked from that directory's `IO_prototype.rst`) — not fabricated in this planning pass. | Specs: [04](04-nxstress-internal-cleanup.md), [09](09-fit-spectrum-nxstress.md), [10](10-flip-defaults.md). |
| 17 | Specs 04b/04c simplified: discriminators most-slowly-varying, sortedness relaxed for append | **Decided, following direct verification that global sortedness was never a reader requirement.** `_Peaks.peakCollectionRanges` (`_peaks.py:246-338`) — the only reader-side splitter — enforces only that each compound key's run is contiguous and `scan_point` increases within a run; the latter is guaranteed upstream by `SubRuns.set` (`sample_logs.py:167-168`) regardless of NXstress's own sort. Three changes follow: (a) `sort_key` now **prepends** discriminator values (most slowly varying), so each input workspace forms one contiguous super-block — this turns 04b's read-side workspace split into a `groupby` over already-contiguous ranges and keeps the scan-point family's exact-match reader (`_input_data.py:70-72`) working via plain concatenation, no new indexing machinery. (b) A single-step write still produces a fully sorted index, as today. (c) Append no longer re-sorts the file — "locally sorted, globally segmented" — which, combined with (a), reduces the common append case (adding a new workspace) to a plain tail-append using code (`_append_peak`'s `resize(cur+N); arr[cur:] = …` shape) that already exists. **04c's scope is cut accordingly:** it now covers only appending a genuinely new compound key; extending scan points on a key already in the file raises `NotImplementedError` rather than being implemented via mid-array insertion — removing the insertion-position computation, cross-group position-sharing, and associated tests the original draft required. The non-overlap invariant is unchanged and still fully enforced throughout. | Specs: [04b](04b-multi-workspace-nxstress.md), [04c](04c-nxstress-append.md). See `open-questions/04b-multi-workspace-nxstress.md` Q6 and `open-questions/04c-nxstress-append.md` Q5 for the full verification writeup. |
| 18 | 04b architectural gap closed: scan-point-family read-split mechanism, write-time invariant, wavelength/geometry clarification | **A full-tree consistency scan (triggered by a stale single-workspace signature spotted in spec 09) found that 04b never specified how `read()` splits the *scan-point family* (raw counts, sample logs, diffractogram, wavelength) back into N workspaces** — only the peak-index family's split (via PEAKS/`NXreflections`) was specified. **Decided, per direct discussion:** (a) recover each workspace's scan-point set from the already-split peak-index ranges, then slice the scan-point family by value-set membership, not position; (b) this requires every input workspace to contribute at least one `PeakCollection` whenever N>1 — spec 05 already satisfies this, spec 03 is exempt because it's always `N == 1`; (c) **per explicit follow-up, this is enforced, not just documented** — `write()` raises via `_validateWorkspaceAndPeaksData` if the invariant is violated, rather than silently producing an unsplittable file. **Separately, re-reading `_instrument.py` directly surfaced a related imprecision:** 04b's original "validate geometry/shift/wavelength consistency across N inputs" bullet incorrectly grouped wavelength with geometry — wavelength (and, later, spec 08/09's `beam_intensity_profile`) is stored per-scan-point and belongs to the scan-point family's concatenation pattern, not a cross-workspace equality check; only geometry/shift/calibration-state are genuinely single, entry-wide values requiring that check. Both corrections are documentation-and-validation-only — no new on-disk schema. | Spec: [04b](04b-multi-workspace-nxstress.md). See `open-questions/04b-multi-workspace-nxstress.md` Q7. Also propagated to README §2.1/§2.4/§6, Decisions Log item 8, and specs 05/09/10 — see those files' own corrections. |
| 19 | Spec 04's PyRS-side item is already done upstream | **Decided:** spec 04's only `## PyRS Changes` item — auditing `HidraProjectFile`'s legacy log-name patches — is **struck and replaced by a confirmation step**. The audit found the two FIXMEs it targets were removed by `492350bd`, this series' own `claims_written_against` commit: the silent fallback to the legacy capitalized `"2Theta"` key is now an explicit `RuntimeError` instructing the user to re-reduce (`file_object.py:428-445`), and the file contains no FIXMEs at all. The cited lines (`:404`, `:494`) were additionally wrong when written — the FIXMEs were at L423 and L513. | Evidence: [04](04-nxstress-internal-cleanup.md)'s Follow-up 1 F1.1 and this README's Follow-up 1 F1.2. Spec 04 becomes NXstress-only. |
| 20 | Spec 07 depends on 04b | **Decided:** spec 07's `Depends on` is corrected from "none within this plan" to **04b**, in both the subspec header and the `## Sub-specifications` table. Spec 07's own code block calls `NXstress(path, "w").write([hidra_ws], [])` "per 04b's signature", which cannot be written until 04b generalizes `write` to a list — confirmed against the landed library, which today takes a single `HidraWorkspace`. The Overview's reasoning was sound but established only that 07 does not depend on **06**. | Evidence: [07](07-manual-reduction-nxstress.md)'s Follow-up 1 F1.1; probe `probes/a5_nxstress_roundtrip.py`. Phase ordering already satisfied this, so it is a record correction, not a reschedule. |
| 21 | GUI assertions move to the `tests/ui/` tier, and the test inventory is reconciled | **Decided:** the enablement-wiring assertion spec 05 schedules into `tests/integration/test_nxstress_viewer_roundtrip.py` — that a disabled **Save as NXstress…** action stays visible — **moves to `tests/ui/test_nxstress_roundtrip.py`** and carries **both** `@pytest.mark.gui` and `@pytest.mark.integration`, per `CLAUDE.md`'s tier rules. Driving a `QAction` is the `gui` tier; as scheduled it would have run under `pixi run test-integration`, which does not set `QT_QPA_PLATFORM=offscreen` — the hang already recorded in `docs/ground_truths.md`. Model-level round-trips stay in the integration tier. §5's test inventory and the subspecs are reconciled at the same time: three specs (04b, 05, 07) name a file §5 never lists, while §5's `tests/ui/` entry is named by no spec. | Evidence: [05](05-strain-stress-viewer.md)'s Follow-up 1 F1.1 and this README's Follow-up 1 F1.6. The assertion itself is probe-confirmed under PyQt6 (`probes/a4_qtpy_qaction_statusbar.py`): `setEnabled(False)` leaves `isVisible()` true. |
| 22 | 04c's "no new on-disk column" is a scope decision, not a format limit | **Decided:** 04c's restriction stands, but its **rationale is corrected**. The spec reads "isn't possible without adding a new on-disk column, which this spec's tail-append design does not do", which a reader reasonably takes as a technical impossibility. Probing shows adding a column to an existing group **succeeds**. The restriction is therefore a deliberate exclusion of schema restructuring — cheap for a later spec to lift — not something the format forbids. Recorded because it changes what a future spec may assume. | Evidence: [04c](04c-nxstress-append.md)'s Follow-up 1 F1.3; probe `probes/a4_h5py_nexusformat_append.py`. The same probe confirms 04c's core mechanism works on reopened, file-backed `NXfield`s, vlen strings included. |
| 23 | Identifier policy for `allowed_identifier`: Python-identifier alphabet, `__` escape introducer, reversible | **Decided.** `allowed_identifier` today replaces only `:`, and is **many-to-one with no collision check** — two distinct PV logs can converge on one identifier, the second silently overwriting the first *including* the `local_name` attribute meant to preserve the original. That data loss exists today. The policy replacing it: **allow only what `str.isidentifier()` permits**, with `__` as the escape introducer. Forms: `__XX` (byte — `__3A` is `:`), `__uXXXX` (codepoint above U+00FF), `__5F` (a literal `_` that would otherwise be ambiguous), and a lone `_` meaning itself. Encoding needs one character of lookahead; decoding none. Being **injective**, it makes the collision impossible rather than merely detected. Measured on the real namespace (185 log names in `tests/data`): **116 encode verbatim, 69 need encoding (almost all for `:`), 0 need an escaped underscore.** A literal *double* underscore renders non-verbatim but round-trips exactly — nothing is unrepresentable. <br><br>**Three alternatives were tried and rejected**, and the rejections are the load-bearing part. *(a) Convert disallowed characters to `_`* — many-to-one, so it needs a collision check and still loses the original. *(b) Use `.` as the marker so `_` survives unescaped* — **unverifiable**: it needs `.` to be legal in a NeXus identifier, and no `validItemName` rule exists in `nexusformat` or in this repo; the only support is an unsourced comment in `_definitions.py`. This is the same missing `NXstress.xml`/`.html` schema doc spec 10 tracks, and a design must not rest on it — hence a rule that needs no external specification and is strictly narrower than any plausible NeXus rule. *(c) Single-`_` introducer with hex* — injective, but doubles every incoming underscore (`my_log_value` → `my__log__value`, 37 of 185 names). Making the **introducer** `__` avoids that entirely and removes the hex lookahead with it. Note `$` is **not** legal in a Python identifier, contrary to a reasonable expectation; `_` is the only punctuation that is. | Evidence: [04](04-nxstress-internal-cleanup.md)'s Follow-ups 3, 4 and 5; probe `probes/a5_identifier_policy.py`, verified over all 185 real log names (encode → valid identifier, exact round-trip) and injective over 1364 synthetic inputs. `tests/unit/pyrs/utilities/NXstress/test_definitions.py:100-105` pins the behaviour being replaced and must be rewritten in the same PR. <br><br>**This row consolidates what were briefly separate items 23, 24 and 25.** Follow-ups 3-5 are append-only and still cite those numbers; all three refer to the decision recorded here. |
| 24 | `peaks/sx,sy,sz` carry the sample coordinates `vx,vy,vz`, tagged with `local_name` | **Decided.** The NXstress schema requires `peaks/sx`, `sy`, `sz` (`minOccurs="1"`, dimension `n_Peaks`) and documents each as *"component of the sample position in the sample reference frame"*. In PyRS that quantity is `PointList.(vx, vy, vz)`; the identically-spelled `sx`/`sy`/`sz` sample logs are stage positions the stress/strain workflow does not use. The fields are therefore filled from `SampleLogs.get_pointlist(scan_point)`, subrun-subset indexed per `PeakCollection`, falling back to `NaN` with a logged warning when the coordinate logs are absent or non-finite — matching `_sample.py`'s existing handling rather than failing a save. **Because the field name and the value's source disagree by design, each field carries `local_name` = `'vx'`/`'vy'`/`'vz'`**, reusing the convention `_sample.py` already applies to retained logs, and the reason is stated as a comment at the code. The schema field names are *not* renamed: `sx`/`sy`/`sz` is what NXstress mandates. | Evidence: [04](04-nxstress-internal-cleanup.md)'s Follow-up 7 F7.8; vendored schema at `docs/developer/source/design/nexus/NXstress.nxdl.xml`. This also answers two doubts already recorded in `docs/developer/source/design/nexus/IO_prototype.rst` — "included from the logs, but mostly just because the logs had the same variable names -- **this is probably incorrect**!" and "Using `PointList.(vx, vy, vz)` as the sample positions? Is this correct?" — both now resolved in place. Note the indexing differs: `peaks/*` is per peak-row (`n_Peaks`), `SAMPLE_DESCRIPTION/vx` per scan-point (`nP`). |
| 25 | Identifier policy adopts the real NeXus `validItemName` rule — **supersedes item 23** | **Decided.** Item 23 adopted the Python-identifier alphabet *specifically because* no authoritative NeXus rule could be found, and rejected `.` as an escape marker on that basis (its alternative (b)). **That premise was wrong.** The rule is published as `validItemName` in `nxdl.xsd` of the NeXus definitions repository: `[a-zA-Z0-9_]([a-zA-Z0-9_.]*[a-zA-Z0-9_])?`, maximum 63 characters. It was not found in round one because only the installed `nexusformat` *package* was searched; the rule belongs to the NXDL schema, which the package does not ship. Consequences: `.` **is** legal in the interior (so `_definitions.py`'s unsourced comment was correct all along), and a **leading digit is legal**, so the real log names `2theta`/`2thetaSetpoint` need no encoding — item 23's rule emitted `__32theta`. The `__` escape introducer and its forms (`__XX`, `__uXXXX`, `__5F`, lone `_` meaning itself) are **unchanged**; only the alphabet widens. Measured on the same 185 real names: **118 encode verbatim** (was 116), 67 need encoding, 0 need an escaped underscore, injective over 3905 tested inputs. A **63-character cap is newly enforced** — nothing in the series had accounted for it, and encoding lengthens names (worst case 96 for the longest real name; no real name reaches it). `/` is encoded as `__2F`, **not** rejected: under a total encoding it round-trips and creates one group, where rejecting it would discard a whole reduction over one odd PV name — the failure mode item 23's own reasoning rejected. Empty input and the length overflow still raise. <br><br>**What item 23 got right, and keeps:** its stated virtue was that the Python rule is "strictly narrower than any plausible NeXus rule, so it stays valid whatever the schema doc eventually says". That held exactly — its output was always conformant, merely not minimal. | Evidence: [04](04-nxstress-internal-cleanup.md)'s Follow-up 7 F7.1-F7.4; probes [`a4_validitemname_rule.py`](probes/a4_validitemname_rule.py) (reads the rule live and fails if it drifts) and [`a5_identifier_policy.py`](probes/a5_identifier_policy.py). The rule itself lives as a sourced comment beside `allowed_identifier` in `_definitions.py` — a rule is not a document. |
| 26 | The detector's `depends_on` names the **last** transformation, not the first | **Decided — this repairs a defect, not a preference.** `_instrument.py` writes eight transformations, each depending on the previous, then set the detector's `depends_on` to `translation_x`: the **first** link, whose own `depends_on` is `"."`. `NXtransformations` defines the object's `depends_on` as the chain's *entry point*, resolved by following each field's own `depends_on` until `"."`, with `T_f = T₃T₂T₁`. Traversal therefore terminated immediately — **1 of 8 transformations reachable**, the composed transform a bare x-translation with identity rotation, and all three rotations plus `two_theta_zero` written to the file but excluded from the geometry. The entry point now names the last link, making all eight reachable. **The written order was already correct**: traversed from the last link the rotation sub-chain composes to `Rx @ Ry @ Rz`, matching `reduce_hb2b_pyrs.py:257::generate_rotation_matrix` — so spec 04's `# TODO: check order of rotations here!!!` is answered "the order is right, the chain was not connected". | Evidence: [04](04-nxstress-internal-cleanup.md)'s Follow-up 7 F7.5-F7.6; probe [`a5_transformations_chain.py`](probes/a5_transformations_chain.py). **Note the detection gap:** `instrumentFromNexus` reads transformations by name and never walks the chain, so spec 04's own specified test (write a geometry, read it back, compare rotation components) would have passed throughout. The replacement test walks the chain. This also gives item 14's open F1.4 question a concrete referent — `reduce_hb2b_pyrs.py`, not `DENEXDetectorGeometry`; `open-questions/04` Q2's citation of `file_object.py:generate_rotation_matrix` names a symbol that file does not contain. |
| 27 | Instrument identity is configuration; the NXstress schema is vendored into the repo | **Decided, two parts.** (a) Spec 04 asks for the instrument name to be drawn "from the `HidraWorkspace` (e.g., from a sample-log entry or a new workspace attribute)". **Neither exists** — the Hidra project format's `instrument` group holds only calibration, efficiency calibration, geometry setup and monochromator settings, and no sample log carries a beamline name; a new workspace attribute would be a PyRS data-model change, which spec 04 puts out of scope. Resolved through the config mechanism spec 01 shipped: `nxstress.instrument_name` and `nxstress.instrument_short_name` in `pyrs/resources/application.yml`, read by `_Instrument._instrument_names()` with a logged fallback to `"HB2B"` when unset or blank. This meets the stated goal — usable at another beamline without a code change — and touches no data model. (b) `NXstress.nxdl.xml` is **vendored** at `docs/developer/source/design/nexus/NXstress.nxdl.xml`, byte-identical to `nexusformat/definitions` at NXDL v2026.01 (commit `004da96e`, sha256 `bee46c07…`), linked from `IO_prototype.rst` with its provenance. It is excluded from pre-commit's whitespace hooks so it stays diffable against upstream — those hooks modified it on first commit, which is how the requirement was found. This closes the schema-doc half of item 16(b) and of spec 10's Overview reminder. **The validator is a separate matter and remains outstanding**: it lives in its own repository and the installed `nexusformat` 1.0.8 still has none, so spec 04's Verification hedge (Follow-up 1 F1.3) stands. | Evidence: [04](04-nxstress-internal-cleanup.md)'s Follow-up 7 F7.7; [10](10-flip-defaults.md)'s Follow-up. |
| 28 | `NXstress.write`/`read` take one `PeakCollection` list **per workspace** | **Decided.** 04b left `peakss`' shape "TBD in implementation". It is `write(wss: list[HidraWorkspace], peakss: list[list[PeakCollection]])`, with `len(peakss) == len(wss)` enforced, and `read()` returns the same two parallel lists. The alternative — a flat list, with each collection attributed to whichever workspace's sub-runs contain its scan points — is well defined under the no-overlap invariant, but it makes attribution an inference rather than an input, and it cannot express "this workspace contributed none" at all, which is exactly the condition item 18(b) requires `write()` to *reject*. Under the per-workspace shape that check is `if not peakss[i]`. **Consequence:** a call with no peak collections is now `write([ws], [[]])`, not `write([ws], [])`. That supersedes the spelling in rows 10 and 20 and in specs 03 and 07, whose own Follow-ups record it. | Spec: [04b](04b-multi-workspace-nxstress.md), Follow-up 2 F2.7. Propagated to [03](03-combine-runs-nxstress.md) and [07](07-manual-reduction-nxstress.md). |
| 29 | Configuration is the read-time authority for discriminator fields, and a mismatch is **loud** | **Decided.** `read()` takes the field set from `Config["nxstress.discriminator_fields"]`, encodes each name, and cross-checks it against the peaks group's non-reserved columns. Exactly one disagreement is tolerated: a file carrying *no* discriminator column reads back as a single workspace, because that is what a pre-04b file, a single-workspace file written before any field was configured, and a `merge_workspaces` file all look like. **Any other disagreement raises**, naming both sets. The alternative considered — deriving the field set from the file and ignoring configuration — would make a file readable under any configuration, but at the cost of accepting a file written under a different deployment policy without saying so. This codebase does not degrade silently. | Spec: [04b](04b-multi-workspace-nxstress.md). Implemented as `_discriminator.names_for_read`; covered by `test_multi_workspace.py::TestBackCompat`. |
| 30 | Discriminator columns are written whenever a field is configured, `N == 1` included | **Decided.** 04b's "`N == 1` is unaffected by either key" is read as scoped to the empty-config *raise* policy — a single workspace never raises — not to column emission. Writing the column at `N == 1` keeps row 29's cross-check a plain equality test with no special case, and lets spec 05 round-trip a single direction the same way it round-trips three. | Spec: [04b](04b-multi-workspace-nxstress.md); relevant to [05](05-strain-stress-viewer.md). Covered by `test_multi_workspace.py::TestDiscriminatorRoundtrip::test_single_workspace_also_carries_the_column`. |
| 31 | Merging N workspaces is raw-array concatenation inside NXstress, not a merged `HidraWorkspace` | **Decided — this repairs a false premise, not a preference.** 04b tells the implementer to "reuse the merge logic `HidraWorkspace.append_hidra_project` already implements in-memory". That method takes a `HidraProjectFile`, not a workspace, and **renumbers** appended subruns to 1..N of the merged workspace — destroying the very scan points the exact-match reader at `_input_data.py:95-97` compares. Worse, no merged workspace could exist anyway: the concatenated scan-point axis need not be monotonic (04b's own specified test has `[1,3,5]` and `[2,4,6]`), and `SubRuns.set` rejects a non-monotonic array outright. So each `init_group` takes `list[HidraWorkspace]` and concatenates `NXfield`s directly, and the read side splits the axis **before** constructing any `SampleLogs`. | Evidence: [04b](04b-multi-workspace-nxstress.md)'s Follow-up 2 F2.2-F2.3; probe [`a5_subruns_nonmonotonic.py`](probes/a5_subruns_nonmonotonic.py). |
| 32 | Per-scan-point datasets are written resizable; a pre-04c file is not appendable | **Decided — this repairs a false premise, not a preference.** 04c's design rests on `resize(cur+N); arr[cur:] = …` working against the reopened groups the writer emits. It does, for the **peak-index** family, every field of which carries `maxshape`/`chunks`. It does not for the **scan-point** family: `entry/start_time`, `entry/end_time`, `monochromator/wavelength`, `input_data/scan_point`, `SAMPLE_DESCRIPTION/{vx,vy,vz,temperature,stress_field,logs/*}` and all four scan-point-aligned `DIFFRACTOGRAM` datasets were written at final size, and an HDF5 dataset without `maxshape` is contiguous and cannot be extended by anything. A new `_definitions.growable(rank)` is applied at every such site, with `tail_append`/`appendable` carrying the growth rule and the pre-flight predicate. The alternative — read back, concatenate, delete, rewrite — was rejected: it round-trips existing data through memory, which 04c's architecture decision (2) excludes, and it interleaves deletes with writes, costing the byte-neutrality the conflict policy depends on. **Compatibility:** a `.nxs` written by an earlier PyRS cannot be appended to; it is refused up front by the pre-flight sweep (row 35), not by `tail_append` mid-write as first implemented. Nothing in the GUI is affected — no viewer writes through an append path, and the hookup is phases 4-6. A second, independent defect fixed with it: `_Sample._writable` passed fixed-width `|S` bytes columns through, and such a column **silently truncates** a longer value appended later; it now coerces to the variable-length dtype. | Evidence: [04c](04c-nxstress-append.md)'s Follow-up 2 F2.1-F2.2; probes [`a5_scan_point_family_resizable.py`](probes/a5_scan_point_family_resizable.py) and [`a4_growable_string_fields.py`](probes/a4_growable_string_fields.py). Note what the earlier probe could not see: `a4_h5py_nexusformat_append.py` built its own fixtures **with** `maxshape`, so its verdict was true and narrower than it read. |
| 33 | `write()` targets an entry by number, and grows it if it exists | **Decided.** 04c says to "dispatch `write()` to the tail-append path when opened with mode `"a"`", which would remove the "add another `NXentry` to an existing file" idiom — documented in `NXstress.py`'s usage comment (lines 152-156 before that pass) and relied on by 04c's own Overview, which cites it as the capability append is *not* about. One method cannot mean both. Instead **the dispatch is on whether the resolved entry already exists**: `entry_number` omitted, or naming an existing entry, appends to it (omitted = the highest-numbered); naming no existing entry writes a fresh one there. Mode governs file access, `entry_number` governs targeting, so it is accepted for `"w"` and `"a"` and refused for `"r"`. **Overwriting** an existing entry's contents is a third operation, neither append nor create, and stays unimplemented — and under this dispatch no input selects it, so the former collision guard is unreachable by construction rather than a live backstop. It is kept regardless, since the cost of that reasoning being wrong is an entry silently replaced instead of grown. A number beyond `max + 1` raises `ValueError`: a gap is far likelier a typo than an intent, and its silent outcome — a stray entry instead of the append that was meant — is not one a caller would notice. `entry_number` is **also** a `write()` argument, which neither the spec nor this decision anticipated: under the rule above a bare second `write()` appends, so a constructor-only kwarg would make "two entries in one session" unexpressible, and `test_NXentry_multiple` already covered it. | Stakeholder decision, 2026-10-01. Spec: [04c](04c-nxstress-append.md), Follow-up 2 F2.4. |
| 34 | 04c's append tests are unit, not integration | **Decided.** 04c's Tests section, this README's §5 inventory and `probes/README.md`'s Disposition column all name `tests/integration/test_nxstress_append.py`. All three predate 04b, whose structurally identical round trip landed as the **unit** test `test_multi_workspace.py` — `minimal_HidraWorkspace`, `tmp_path`, no real data — and `CLAUDE.md`'s dividing line agrees: a synthetic `tmp_path` round trip inside one component's own public API is unit. The precedent and the tier rule win over three stale documents, which are corrected. Shipped as `tests/unit/pyrs/utilities/NXstress/test_append.py`. | Spec: [04c](04c-nxstress-append.md), Follow-up 2 F2.10. Relevant to row 21, which moved GUI assertions the other way for the same reason — the tier follows what a test touches, not which document named it. |
| 35 | Every append refusal is pre-flighted, and any mutation-phase escape invalidates the instance | **Decided — this repairs a shipped defect.** Row 32 and 04c's Follow-up 2 reported that a rejected append leaves the entry byte-for-byte unchanged. It did not: `tail_append`'s own two refusals — a non-resizable dataset and a trailing-axis disagreement — were never pre-flighted, so they fired *during* mutation, after earlier groups had grown. An entry written at 20 two-theta bins, appended with a batch reduced onto 25 — the ordinary difference between two reduction passes — was left with 27 datasets at the new length and three at the old one, **and read back without error**. The tell was that `_definitions.appendable` existed precisely for this check and had no production caller. Now: a sweep of every per-scan-point dataset plus per-group two-theta, pixel-count and mask-set checks all run in the pre-flight; and the mutation phase is wrapped so that **any** escape sets `self._invalid`, inverting a policy that had invalidated the instance only in the case where the file was untouched. | Evidence: [04c](04c-nxstress-append.md)'s Follow-up 3 F3.1-F3.3; probe [`a5_append_preconditions.py`](probes/a5_append_preconditions.py) claim 6. Found by adversarial review after the PR passed its own Verification, all three tiers and the full toolkit — none of which could see it, because the defect was in a path no test took. |
| 36 | A read-only property that participates in I/O round trips is marked `@restorable()`, not made settable | **Decided.** `_discriminator.resolve` prefers a `HidraWorkspace` property over a sample log, while `apply` could only write a property that had a setter — so a field naming any of the six read-only properties was read from the property and written back to a log. The value reached the file correctly and came back as the constructor's default, silently; `write(read(f))` stored something other than `f`. Live, not hypothetical: `strainstressviewer/model.py` builds `HidraWorkspace(direction)`, so the viewer spec 05 targets already carries direction in a read-only property. **Rejected:** making the properties read/write (puts a setter on the public API for a deserializer's benefit), a `set_<name>()` method (the same thing in different spelling), and a registry mapping property to backing attribute (a second place to edit, which drifts silently). **Adopted:** a `restorable_property` descriptor — a `property` subclass naming its own backing attribute, so the declaration sits on the property. Assignment still raises `AttributeError`; `isinstance(…, property)` still holds, leaving the read path untouched; and a sweep over `vars(cls)` partitions properties exhaustively, which is what lets one test cover every property without naming any. `field_names()` additionally refuses an unrestorable field **at configuration time**, making the asymmetry unreachable rather than merely fixed. `direction` is backed by `_direction` and deliberately **not** aliased to `_name` — aliasing would make `ws.direction` answer "Combined Project Files" for `CombineRunsModel`. | Evidence: [04b](04b-multi-workspace-nxstress.md)'s Follow-up 3 F3.1; [`plans/PR_review/04b-comments.md`](../PR_review/04b-comments.md) D4. Delivers early what [05](05-strain-stress-viewer.md) scoped as "add a settable `direction` property" — see its Follow-up 2. |
| 37 | Production reads `Config` the documented way; runtime override is a test-only concern | **Decided.** `_discriminator` was the only module bypassing `config.py`'s rule (*"Import `Config` from this module"*), reaching through the module instead, with a comment explaining that a bound name goes stale when the `default_config` fixture reloads. The comment's premise is **true** and its conclusion wrong: the override capability exists for tests, so the accommodation belongs in a fixture, not in the shape of production code. Two things were found on the way. There is **no config-based property decorator** in the installed `neutrons_standard` — the absence is the finding. And **`default_config` was leaking**: its teardown reloaded to yet another new singleton, so any test using it orphaned every consumer module for the rest of the session — which is why the first override fixture passed in isolation and failed in a full run. Now: `_discriminator` uses the documented import; `default_config` restores the pre-test singleton; and a new `config_override` fixture deep-merges into the live singleton's `_config` and restores a deep copy, never swapping the instance. Lookups are live (`__getitem__` reads `_config` on every access), so no rescan is needed — but a helper must deep-merge, since a shallow assignment drops sibling keys. | Evidence: [04b](04b-multi-workspace-nxstress.md)'s Follow-up 3 F3.2-F3.3. **Side effect:** `_instrument.py` was unoverridable in tests, so any test asserting a configured instrument name was passing vacuously. Fixed by the same change, without touching `_instrument.py`. |
| 38 | Detector-mask names and diffractogram keys are two namespaces, and `masks/names` holds only the first | **Decided — this repairs a shipped defect.** `_Masks.mask_keys` returned the union of `_mask_dict` and `_diff_data_set`, but a `_diff_data_set` key is built by `reduction_manager.py:613-615` as `"{mask_id}_eta_{eta_cent}"` with either part dropped when absent — so `eta_-5.0` and `mask_a_eta_-5.0` are *reduction* identifiers that own no array. Three consequences, all measured: `masks/names` listed names resolving to nothing; a texture workspace could not be written at all, rejected once for a missing `DIFFRACTOGRAM` under the injected `_DEFAULT_` key and once for a `PeakCollection.mask` checked against the wrong namespace; and the `NXlink`-to-default branch meant to cover the first never serialised. **Making that link work was rejected as the worse defect** — it would hand the default array back under each composite key, and `set_masks_from_dict` would install `_mask_dict` entries the source workspace never had. Now: `nxstress_mask_names` (injects `DEFAULT_TAG`, reads `_mask_dict`) and a new `nxstress_diffractogram_keys` (injects nothing, reads `_diff_data_set`) are separate, and `diffractogram_detector_mask(key, known_masks)` resolves a key to its mask by matching registered names longest-first rather than splitting on `_eta_` — parsing would misread both a mask named `rotated_eta_bar` and any `__`-escaped identifier (row 20). **The load-bearing point**, supplied by the reviewer: the default detector mask is always in use, so there is no reduction "with no mask"; a bare `_DEFAULT_` key means *the default mask with no eta ROI*, which an eta-resolved reduction never produced. That is why the keyspace must not be defaulted into. | Evidence: [04](04-nxstress-internal-cleanup.md)'s Follow-up 8 F8.1-F8.3; [`plans/PR_review/04-comments.md`](../PR_review/04-comments.md) D11, D13. Pinned by `test_definitions.py::TestDiffractogramKeysAreNotMaskNames` and `test_instrument.py::TestMaskGroupHoldsOnlyRealMasks`, the latter iterating the written group so a future mask kind is covered without an edit. |
| 39 | Solid-angle masks raise `NotImplementedError` rather than being guessed at | **Decided.** `_Masks` distinguished solid-angle from detector masks by dtype and length parity, which misclassifies in both directions — a float detector mask of even length reads as solid-angle, a 3-element solid-angle mask reads as a detector mask. The branch has never run: nothing in PyRS loads a solid-angle mask into a `HidraWorkspace` (`HidraProjectFile.read_mask_solid_angle` has one caller, a test), and `_load_masks` reads detector masks only. Rather than refine a heuristic with no subject, the write path raises, and the same reasoning retires `_generate_default_mask`'s dead `detector_mask=False` branch, which returned a plausible-looking `[-180.0, 180.0]`. The `solid_angle` group stays as the schema placeholder and `masksFromNexus` still reads it, so adding the writer later needs no reader change. A second defect fixed alongside: `_generate_default_mask` returned a 2-D array that `set_detector_mask` refuses, so a workspace with no masks wrote a file PyRS could not read — invisible because all 42 round-trip tests passed `with_masks=True`. | Evidence: [04](04-nxstress-internal-cleanup.md)'s Follow-up 8 F8.4-F8.5; [`plans/PR_review/04-comments.md`](../PR_review/04-comments.md) D10, D12. |
| 40 | String sample-log values are `bytes` on disk and `str` in memory, enforced at both boundaries | **Decided — this reaches outside the NXstress package, deliberately.** `SampleLogs` stored whatever numpy dtype it was handed and nothing normalised anywhere, so a string log's type was an accident of provenance: `bytes` from `HidraProjectFile.read_sample_logs` and from NXstress's own reader, `str` from a workspace built in memory. Consumers each invented a policy — four tolerant, two not — and `_entryTimes` was one of the intolerant ones, calling `t.decode()` unconditionally inside a `try` that catches only `ValueError`, so a `str` log raised `AttributeError` past the `NO_LOG` fallback. On the append path that is the **first** mutation step, so the instance was invalidated for a type mismatch. **Both directions were built and measured across all three tiers before choosing.** Toward `bytes`: 2 unit + 1 integration failure, and the integration one was a real user-visible regression — the CSV exporter writing `# string1 = b'a constant string'` into an exported header. Toward `str`: 83 failures that were all the *same* defect, and every tier green once the two unconditional decodes were fixed. The asymmetry decided it — writers that must encode are few and already do it, readers that must decode are many and mostly untested. Enforced by **one** conversion -- `to_text`, `to_text_array` and the `is_text_array` predicate in `pyrs/utilities/convertdatatypes.py`, beside the existing `to_int`/`to_float` -- called at the two boundaries (`SampleLogs.__setitem__` on store, so **both** file readers normalise without either decoding for itself; `HidraProjectFile.add_sample_log` before `create_dataset`, which is not optional since h5py has no conversion path for `<U`) and at the **nine** sites across five modules that previously hand-rolled it into four different behaviours. **Two defects found by shipping them first, both now pinned:** the normalisation originally tested `dtype.kind == "S"`, which catches only fixed-width arrays -- HDF5 yields an *object* array of `bytes` (`kind == "O"`), so it passed all three tiers while doing nothing for any log off a real file, its unit fixture having been built with `np.array([b"11", ...])`; and the write boundary originally used `numpy.char.encode`, producing fixed-width `|S`, which truncates a longer value silently and would convert an existing project file's variable-length columns on every re-save -- it now targets `h5py.string_dtype(encoding="utf-8")`. `to_text_array` **raises** on a non-string array rather than passing it through -- a caller that may hold either (`SampleLogs`, `_Peaks._decoded`) asks `is_text_array` first, so a misdirected call cannot look like a working one. **Scope note:** `pyrs/dataobjects/sample_logs.py` and `pyrs/projectfile/file_object.py` are outside this plan series; taken here on the stakeholder's judgement that no separate PR for the core change would ever happen, and called out explicitly for review. | Evidence: [04c](04c-nxstress-append.md)'s Follow-up 5 F5.4; [`plans/PR_review/04c-comments.md`](../PR_review/04c-comments.md) D19. Ground truth: [docs/ground_truths.md](../../docs/ground_truths.md). Pinned by `test_sample_logs.py::TestStringLogsAreTextInMemory` (dataobjects) and `::TestStringLogsAreBytesOnDisk` (projectfile), which cross-reference so neither half reads as redundant. **Incidental finding, not fixed:** `summary_generator.py:169` calls `value.decode()` and discards the result, which is why bytes reached that CSV header at all. |
| 41 | Are citation line numbers subject to the append-only `## Follow-up` rule? | **No — decided by the reviewer, 2026-10-15.** A citation *value* is a pointer, not a belief, so it is updated wherever it appears, Follow-ups included. A stale value is worthless: it resolves to something that makes no sense, and a reader who checks it is misled rather than informed. The previous value is recoverable from git history, which is the only place it is worth anything. | Narrows, and does not overturn, "an earlier Follow-up is never edited": the *claims* and *verdicts* in a Follow-up remain frozen, because those are the record of what was believed. Two exceptions, both because the old number is the content: a correction record of the form `was X → now Y` keeps `X`, and a finding that diagnoses a wrong pointer keeps the pointer it diagnoses. Superseded `plans/NXstress-prod/04b-multi-workspace-nxstress.md` Follow-up 2's closing paragraph. Line numbers are expected to be replaced by semantic-id XREF keys in future, which removes the problem at its source. |
| 42 | `HidraWorkspace.direction` is read-only and *restorable*, not settable — **supersedes row 9's "settable `@property` (get/set)"** | **Decided — this repairs a shipped contradiction, not a preference.** Row 9 specified a get/set property so a reader could put `direction` back. What landed instead is a **read-only** property whose getter carries `@restores("_direction")`; a reader restores it with `pyrs.utilities.restorable_property.restore(ws, "direction", value)`. Measured: `direction.fset is None`, assignment raises `AttributeError`, and mypy reports *"Property 'direction' defined in 'HidraWorkspace' is read-only"*. The reason is that a setter exists only for the deserializer's benefit, and putting one on the public API invites the casual assignment the class does not intend to allow — while the restorable mark gives the reader a way in without widening the API. | Spec: [05](05-strain-stress-viewer.md), which row 9 also points at — so a session starting 05 would otherwise read a decision the code refutes. Row 9's **other** half stands and is still 05's work: `nxstress.discriminator_fields` flips from `[]` to `["direction"]`, which is only safe because `direction` is marked restorable — `_discriminator.field_names()` refuses a read-only property that is not. Mechanism and evidence in `plans/PR_review/04b-comments.md`'s `RE-REVIEW RECOMMENDED` section; the mark is on the getter rather than a `property` subclass because mypy does not recognise a subclass as a property. |
| 43 | Discriminator columns are named by configuration, recorded in the file, and never inferred from it — **refines row 29** | **Decided — this repairs a shipped defect (review D8).** Row 29 cross-checked configuration against *every* non-reserved `peaks` column, so a reflection field from another NXstress producer was taken for a discriminator and the file could not be read at all. Configuration stays the authority, but the file now says what it means: `write` records its fields in the `peaks` group's `discriminator_fields` attribute (an empty list records no workspace boundary). On read, an entry with a record must agree with configuration, and an empty record reads as one workspace; an entry without one is split by the configured fields if every one has a column, read whole if none does, and refused if only some do. Any other column is ignored on read and **refused on append**, which could not grow it. Cost, accepted: for an entry without the record, a configuration that omits a discriminator is not detected, and in one shape silently merges two workspaces (measured: one 6-row collection where two 3-row ones were written). Every entry PyRS writes carries the record, so this affects only foreign and pre-attribute files. | Rules in full: `plans/PRs/phase-2.md`, "Discriminator columns on read and append". Implemented in `_discriminator.names_for_read`/`recorded_names`, `_Peaks._validateDiscriminatorNames`, `NXstress._rejectUnconfiguredColumns`; covered by `test_recorded_discriminators.py`. |
| 44 | A write or append whose read-back workspace would hold out-of-order scan points is refused, not re-sorted | **Decided — this repairs a shipped defect (review D7).** A merged write, or workspaces sharing one discriminator value, are read back as one workspace in on-disk order, and `SubRuns` refuses a non-increasing axis, so such an entry was written and then could not be read. Re-sorting on write was rejected: the on-disk order is the inputs' order, and sorting it breaks the round trip. The refusal (`NotImplementedError`, nothing written, instance usable) is an interim: the ordering requirement itself is arbitrary, and dropping it is recommended. | DD-2 in [`deferred-defects.md`](../PR_review/deferred-defects.md). Implemented as `NXstress._refuseUnreadableScanPointOrder`; covered by `TestReadBackScanPointOrder` in `test_multi_workspace.py` and `test_append.py`. |
| 45 | NXstress never merges workspaces: `nxstress.merge_workspaces` is removed — **supersedes row 6's `merge_workspaces` clause, narrows row 44** | **Decided.** With the key set, N workspaces were written with no boundary recorded, and `read` could only return one workspace built from all their rows. So NXstress *did* build a merged `HidraWorkspace`, on read, and 04b's Follow-up 4 F4.3 ("NXstress performs no merge") was true of the write only. That broke write/read symmetry (row 6), and its merge was weaker than PyRS's own: two workspaces measuring the same peak were refused (04b F2.11), so the ordinary multi-direction case could not use it, and a merged write in decreasing scan-point order was unreadable (row 44). No production caller set it: CombineRuns merges in PyRS and passes a length-1 list (row 10). Now, more than one workspace with no discriminator field configured is always refused, and the message says to merge in PyRS first. A workspace merged in PyRS, whose `direction` log varies, is written as one workspace with no discriminator configured. Under a `direction` discriminator it is refused with `ValueError`, since it has no single value; it used to fail with a bare `AssertionError` from `get_sample_log_value`, as did `HidraWorkspace.direction`. Row 44's refusal now covers only workspaces sharing a discriminator value. Files written with the key still read, as one workspace. A leftover key in an override file was at first ignored, because `validate_config` checked only the keys `ConfigSchema` declares; row 46 makes it a start-up error. | Evidence: 04b's Follow-up 13. Removed from `config.py`, `application.yml`, `_discriminator.py` and `NXstress._validateMultiWorkspace`. The `ValueError`s are in `_discriminator.resolve` and `HidraWorkspace.direction`. Covered by `TestEmptyConfigPolicy` and `TestAWorkspaceMergedInPyRS` (`test_multi_workspace.py`), `TestResolve` (`test_discriminator.py`) and `TestDirection` (`test_workspaces.py`). |
| 46 | Configuration keys `ConfigSchema` does not declare are refused at start-up | **Decided.** `validate_config` used to read only the keys the schema declares, so any other key in `application.yml` or an `env` override file was never looked at. A misspelled key (`discriminator_field:`), one put in the wrong section, or one left over from a removed setting (row 45) changed nothing, and the setting it was meant to change kept its default with no sign that anything was wrong. Now every schema section forbids extra keys, and an unknown top-level section is refused too. Each unknown key gets a line in the same `Invalid PyRS configuration` report as a malformed value, so PyRS does not start. The exception is the sections `neutrons_standard` adds to the document itself (`environment`, `module`, `version`, `application`), listed in `NEUTRONS_STANDARD_SECTIONS` and left unchecked; their contents belong to the library. Finding unknown keys needs the whole merged document, which `neutrons_standard` exposes only as the private `Config._config`, so `validate_config` reads it there; `config.py` is already the one module that touches `neutrons_standard`. | `pyrs/utilities/config.py` (`extra="forbid"` on `_Section`, `NEUTRONS_STANDARD_SECTIONS`). Covered by `TestUnknownKeysAreRefused` (`test_config.py`), whose `test_a_load_adds_no_unlisted_section` fails if a `neutrons_standard` upgrade adds a section of its own, and by `tests/integration/test_config_startup.py`. |

---

## 5. Files to be Modified

| Phase | PyRS files | NXstress / GUI files |
|---|---|---|
| 1 | _(none)_ | `pyrs/utilities/config.py` (new), `pyrs/resources/{__init__.py,application.yml}` (new), `pyproject.toml` (add `neutrons` channel + `neutrons_standard`), `pyrs/interface/peak_fitting/{peak_fitting_viewer,peak_fitting_model}.py`, `pyrs/interface/texture_fitting/{texture_fitting_viewer,texture_fitting_model}.py`, `pyrs/interface/combine_runs/{combine_runs_viewer,combine_runs_model}.py` |
| 2 | `pyrs/projectfile/file_object.py` (audit legacy log names) | `pyrs/utilities/NXstress/{_peaks,_fit,_definitions,_instrument}.py`, `pyrs/resources/application.yml` |
| 2/3 (04b) | _(none)_ | `pyrs/utilities/NXstress/{NXstress,_peaks,_input_data,_sample,_instrument,_fit}.py`, `pyrs/utilities/NXstress/_discriminator.py` (new), `pyrs/utilities/config.py` (type-check the new keys), `pyrs/resources/application.yml` (new keys: `discriminator_fields`, `merge_workspaces` — **not** delivered by spec 01 as 04b's Scope claimed; see its Follow-up 2 F2.1); call-site updates in `pyrs/interface/{peak_fitting,texture_fitting,combine_runs}/…`. **Added by the PR review (Follow-up 3):** `pyrs/utilities/restorable_property.py` (new), `pyrs/core/workspaces.py` (`direction`, `hidra_project_file`), `pyrs/interface/strainstressviewer/model.py`, `tests/unit/pyrs/utilities/conftest.py` |
| 3 (04c) | _(none)_ | `pyrs/utilities/NXstress/{NXstress,_definitions,_input_data,_instrument,_sample,_fit,_peaks}.py` — library only, no GUI files. `_definitions.py` gains `growable`/`tail_append`/`appendable`; `_instrument.py` is in scope because `monochromator/wavelength` is per-scan-point, which 04c's family table missed (Follow-up 2 F2.8) |
| 3 (05) | `pyrs/core/workspaces.py` (new `direction` property), `pyrs/dataobjects/fields.py` | `pyrs/utilities/NXstress/_peaks.py` (StrainStress-specific tests only), `pyrs/resources/application.yml` (discriminator_fields default), `pyrs/interface/strainstressviewer/{strain_stress_view,model}.py` |
| 4 | _(none — see item 13 in the Decisions Log)_ | `pyrs/interface/manual_reduction/pyrs_api.py` (`reduce_hidra_workflow`) |
| 5 | `pyrs/peaks/peak_collection.py`, `pyrs/core/{peak_profile_utility,instrument_geometry,workspaces}.py` (`STRESS_FIELD`/`file_object.py`: blocked, not touched) | `pyrs/utilities/NXstress/{_fit,_instrument}.py` (`_sample.py`: blocked, not touched) |
| 6 | _(none)_ | `pyrs/resources/application.yml` only — no viewer files; every wired viewer already reads `nxstress.enable`/`legacy_io.enable` directly |

Tests to extend/add:
- `tests/unit/pyrs/utilities/NXstress/*` (existing suite — extend each phase).
- `tests/ui/test_nxstress_roundtrip.py` (new — GUI-level round-trips).
- `tests/unit/pyrs/utilities/NXstress/test_append.py` (new in spec 04c —
  library-only append round-trip; no GUI involved, no real data, so **unit** per
  `CLAUDE.md`'s tier rule and 04b's precedent. Supersedes the
  `tests/integration/test_nxstress_append.py` path named here before — Decisions
  row 34).
- `tests/integration/test_nxstress_reduction.py` (new in Phase 4 — manual
  reduction, fresh-write only).
- `tests/unit/pyrs/utilities/NXstress/{test_discriminator,test_multi_workspace}.py`
  (new in spec 04b — the resolver and its configuration, and the N-workspace
  round trips and the policies gating them).

---

## 6. Verification

- **Round-trip pytest** — for each viewer connected, load an existing
  `HidraProjectFile`, do the workflow, save as NXstress, reload from NXstress,
  assert equality of `list[HidraWorkspace]` (post-04b — a single workspace
  is just the `N == 1` case) and `list[PeakCollection]`.
- **Schema validation** — run each written file through the NXstress validator
  (via `nexusformat`) with `nxstress.use_production_names = true` once the
  upstream validator bug is resolved (Phase 2).
- **GUI smoke test** — launch each affected viewer via `pyrsplot`, exercise the
  new save/load actions, confirm no regressions in existing `HidraProjectFile`
  workflows.
- **Demo script** — keep `tests/scripts/cis_tests/NXstress_demo_script.py`
  passing and update it to reflect current usage after each phase.

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass)

The first pass to cover all seven axes. Earlier rounds — the nine corrections in
the Decisions Log — covered A2 and A3; every finding below sits on an axis that
had never been run. Tooling and evidence:
[`plans/audit-process/tools/`](../audit-process/tools/) and
[`probes/`](probes/); coverage matrix and the full list in
[`review/findings.md`](review/findings.md).

**F1.1** (A3) — §1.1 and §1.2's architecture tables carry ten drifted citations.
- Referent: the cited files.
- Verdict: **every prose claim in both tables is still true; ten pointers were
  not.** The drift is confined to §1 — the "current architecture" description,
  written first and never revisited — while §2 onward, which the Decisions
  rounds actively edited, is accurate throughout.

  | Where | Cited | Claimed symbol | Actually at |
  |---|---|---|---|
  | §1.1 | 463, 497, 1025, 1096 | the four `HidraWorkspace` methods | **465, 517, 1045, 1116** |
  | §1.1 | 778 / 712 | `write_peak_parameters` / `read_peak_parameters` | **`file_object.py`:854 / :788** |
  | §1.2 | `peak_fitting_model.py:100` | `load_hidra_project` | **74-123** |
  | §1.2 | `peak_fitting_model.py:189-199` | `save_fit_result` | **211-247** |
  | §1.2 | `model.py:37` | `TextureFittingModel.load_hidra_project_file` | **`texture_fitting_model.py`:42-87** |
  | §1.2 | `model.py:141-165` | `TextureFittingModel.save_fit_result` | **`texture_fitting_model.py`:175-221** |
  | §1.2 | `combine_runs_model.py:25-31` | `export_project_files` | **29-46** |

  Two of these are worse than a number. The `L778`/`L712` pair named no file at
  all, so it inherited `peak_collection.py` from earlier in the same table cell
  — a file of 578 lines, in which neither line exists. And `model.py` is used in
  §1.2 for **two different files** in **adjacent rows**: the TextureFitting row
  meant `texture_fitting_model.py`, the StrainStress rows meant
  `strainstressviewer/model.py`. The StrainStress ones were correct all along;
  the Texture ones pointed into the wrong file entirely and still resolved,
  because a bare basename resolves to whatever exists.
- Action: all corrected in place, with the ambiguous ones now path-qualified.

**F1.2** (A3/A7) — §2.2 and §3: "**`HidraProjectFile` legacy log-name patches**
(`file_object.py:404`, `:494` FIXME) — audit to ensure NXstress does not codify
the legacy names into the schema-conformant file."
- Referent: `pyrs/projectfile/file_object.py`.
- Verdict: **the work is already done.** At `492350bd^` the two FIXMEs were at
  L423 and L513, never 404/494 — so the citation was wrong when written. More
  importantly, `492350bd` — this series' own declared `claims_written_against`
  commit — **removed both**. The file now has zero FIXMEs: the silent fallback
  to the legacy capitalized `"2Theta"` key became an explicit `RuntimeError`
  instructing the user to re-reduce (`file_object.py:428-445`).
- Action: this empties spec 04's only `## PyRS Changes` item; see that
  document's Follow-up 1 F1.1, and Decisions item 19 below. The §2.2 bullet and
  the §3 Phase-2 bullet should be restated as a confirmation step, not an audit.

**F1.3** (A3) — Decisions item 17: "`SubRuns.set` (`sample_logs.py:164-166`)
already raises `"subruns are not sorted in increasing order"`".
- Verdict: claim true, pointer wrong — `:164-166` is a *different* `RuntimeError`
  ("Cannot change subruns when non-empty") in the same method; the quoted guard
  is at `:167-168`. Probe-confirmed in
  [`probes/a5_peakcollection_ranges.py`](probes/a5_peakcollection_ranges.py).
  This is the sentence two subspecs' design rests on.
- Action: corrected in place, here and in 04b and 04c.

**F1.4** (A4) — Decisions item 16a: "Only `PeakFittingViewer` has a status bar
today … (cheap — `self.statusBar()` creates it lazily)".
- Verdict: true of the four viewers listed, **too strong unqualified** — three
  `.ui` files declare a `QStatusBar` — and `self.statusBar()` is called nowhere
  in `pyrs/` today, so the mechanism is net-new rather than an existing pattern.
  It is nonetheless **safe**: probed, `statusBar()` returns a `.ui`-declared bar
  rather than creating a second. Full evidence in spec 10's Follow-up 1 F1.1.
- Action: qualify the sentence; keep the mechanism.

**F1.5** (A4) — Decisions item 16b: "the installed `nexusformat` 1.0.8 package
has no validator capability at all".
- Verdict: **confirmed of the installed package** — no `validate` module, no
  `nxvalidate` script, no `valid*` name in the public API
  ([`probes/a4_nexusformat_validator.py`](probes/a4_nexusformat_validator.py)).
  The NeXus-org validator itself is real; it lives in a separate repository,
  which is what this item and specs 09 and 10 say. **Spec 04's Verification is
  the outlier**: it instructs an implementer to run the validator with no such
  hedge. See spec 04's Follow-up 1 F1.3.
- Action: no change here; the correction belongs to spec 04.

**F1.6** (A2) — §5's test inventory and the subspecs name different files.
- Referent: §5's "Tests to extend/add" list against each subspec's `## Tests`.
- Verdict: only `tests/integration/test_nxstress_append.py` (04c) matches on both
  sides.
  - `tests/integration/test_nxstress_viewer_roundtrip.py` is named by **04b, 05
    and 07** and appears **nowhere** in §5.
  - `tests/ui/test_nxstress_roundtrip.py` is in §5 and named by **no subspec**.
  - `tests/integration/test_nxstress_reduction.py` is in §5 for Phase 4; spec 07
    uses the viewer-roundtrip name instead.
  Worse than a naming mismatch: §5's `tests/ui/` entry is the *correctly tiered*
  home for the GUI assertions that 05 schedules into `tests/integration/`. Under
  `CLAUDE.md`'s tier rules a test that drives a `QAction` belongs in `tests/ui/`
  with **both** `gui` and `integration` markers; as scheduled it would run under
  `pixi run test-integration`, which does not set `QT_QPA_PLATFORM=offscreen` —
  the hang `docs/ground_truths.md` already records. See spec 05's Follow-up 1
  F1.1 and Decisions item 21 below.

**F1.7** (A2) — §Sub-specifications lists spec 07's dependencies as `—`.
- Verdict: **wrong; 07 requires 04b.** Spec 07's own code block calls
  `write([hidra_ws], [])` "per 04b's signature". Phase ordering already satisfies
  it, so this is a record defect rather than a scheduling break. See spec 07's
  Follow-up 1 F1.1 and Decisions item 20.

**F1.8** (A2) — The series uses **four** different conventions for naming files
in a change section, and **two** spellings of the change heading itself.
- Verdict: headings — `## NXstress Changes` (04, 04b, 04c, 05, 07, 09) versus
  `## NXstress / GUI Changes` (01, 02, 03, 06, 08, 10), six each. File naming —
  (a) a `###` titled with a file path; (b) top-level bullets naming the file
  (05's `## PyRS Changes`, 07); (c) a `###` with a *prose* title, files named
  only in body bullets (08 throughout); (d) one `###` listing several modules
  where only the first is path-qualified (04b). Each is readable; together they
  make any mechanical ownership check approximate, which is why
  `check_ownership.py` reports ambiguous rows rather than dropping them.
- Action: recorded, not changed. Normalising twelve documents is a larger edit
  than this pass should make unilaterally.

**F1.9** (A6) — §2.3 diverges from the archived draft by design, and the rest of
the pairing was never checked until now.
- Referent: [`archive/overview.md`](archive/overview.md), via `check_drift.py`.
- Verdict: fourteen numbered sections pair up; **nine now diverge and five are
  byte-identical.** §2.3 is by far the most diverged (11% similar) — the
  known `pyyaml`-versus-`neutrons_standard` split, already behind an archive
  banner. Nothing newly alarming turned up, but note the asymmetry `process.md`
  §4.1 insists on: the five *identical* sections are **not** thereby confirmed.
  The draft is older, so agreement with it is evidence of staleness, not health —
  and §1.1, §1.3 and §1.4 are among the identical ones, while §1.1 is precisely
  where F1.1 found ten stale pointers. The draft carries the same stale numbers.
- Action: none for the archive, which is superseded by definition. Recorded so
  the identical-section list is not mistaken for a clean bill.

**A4/A5 coverage, stated rather than implied.** Six probes were written; five
claim clusters could **not** be probed because the code does not exist yet (04b's
N-workspace mechanism, 04c's dispatch logic, 08's new PyRS APIs), and one is
externally blocked (`STRESS_FIELD`). Those remain **uncovered**, not verified.
[`probes/README.md`](probes/README.md) lists every one, with reasons.

---

## Follow-up 2 — 2026-09-26 (completing A6, and the axes Follow-up 1 left partial)

**F2.1** (A6) — Follow-up 1 F1.9 reported the divergence shortlist but reviewed
it only at heading level, leaving A6 partial. Each diverged pair has now been
read. **Every divergence is acknowledged and justified**, and each traces to a
Decisions Log entry:

| § | Divergence | Acknowledged in |
|---|---|---|
| 1.1, 1.2 | Citation corrections | This pass — Follow-up 1 F1.1 |
| 2.1 | Append narrowed from "Phase 4" to Phase 3, library-only, tail-append only, with key-extension explicitly left unsupported | Decisions 7, 8, 17 |
| 2.2 | `file_object.py:510` FIXME relocated into `DENEXDetectorGeometry`; `ReductionController.save_project` found to have zero callers | Decisions 13, 14a |
| 2.3 | `pyyaml` → `neutrons_standard.Config`; the whole config architecture | Decisions 11; draft behind an archive banner |
| 2.4 | Extension imposed-not-chosen; Phase 6 as a pure config flip; multi-direction via 04b | Decisions 1, 2, 3, 6 |
| 4 | 10 rows → 22 | growth, by construction |
| 5 | Ownership table grew with 04b/04c/05 | Decisions 6, 7 |
| 6 | Round-trip asserts `list[HidraWorkspace]` post-04b | Decisions 6 |

No unacknowledged divergence was found. **A6 is now covered rather than partial**
— but note what that does and does not mean: it establishes that each *difference*
is justified, not that the agreements are sound. The five byte-identical sections
(§1, §1.3, §1.4, §2, §3) remain unconfirmed by this axis, and §1 is precisely
where Follow-up 1 found ten stale pointers — which the draft still carries,
unchanged, because it agrees.

**F2.2** (A6) — One divergence is justified but its *rationale* is wrong; see
spec 08's Follow-up 2 F2.3. The README (and Decisions item 14a) dismiss the
comment at `file_object.py:541` as "an unrelated return-type-annotation TODO".
It is a return-type-annotation TODO, and it is **not unrelated** — it records the
same `read_instrument_geometry`-returns-`DENEXDetectorGeometry` observation the
draft was pointing at. The relocation into `DENEXDetectorGeometry` is still
correct; only the word "unrelated" is not.

**F2.3** (A4) — Two gaps Follow-up 1 recorded as uncovered are now closed, both
in spec 08's and spec 07's Follow-up 2:
- `uncertainties` first-order propagation and "no covariance matrix is retained
  anywhere in PyRS" — **confirmed**, with the refinement that the one path which
  *computes* a covariance is dead scipy code, not Mantid, and discards it into a
  `# TODO` stub.
- Spec 07's two basename branches — **confirmed to disagree** on double-extension
  names, and the Verification case that spec proposes is the one case where they
  agree.

Follow-up 1 justified deferring these on the grounds that the consuming code did
not exist. That reasoning was wrong: both claims are about installed packages
and existing source, and were executable throughout. Recorded because the
mistake is a reusable one — **"the code that uses it does not exist yet" is not
a reason to leave a third-party claim unprobed.**

**F2.4** (A5) — Spec 04's A5 coverage is closed the same way: its counterparty is
the landed `pyrs/utilities/NXstress/`, so every premise was executable. All four
confirmed, plus a new finding — `allowed_identifier` leaves `/` untouched, and a
`/` in a group name does not raise but **silently creates a nested group**. See
spec 04's Follow-up 2 F2.2. The remaining A5 gaps (04b, 04c, 05, 08, 09, 10)
genuinely depend on unwritten code and stay uncovered.

---

## Follow-up 3 — 2026-09-29 (subspec 04 implementation pass)

**F3.1** — **Decisions rows 24-27 are new rows, not a revival of the numbers
Follow-up 2 F6.1 collapsed.** That consolidation folded three briefly-separate
rows (23, 24, 25) into a single row 23, leaving the log contiguous at 1-23. The
rows added here continue from 24 because 24 and 25 no longer name anything;
specs 04's Follow-ups 3-5 still cite "item 23"/"item 24"/"item 25" and all three
still resolve to row 23, exactly as F6.1 states. Nothing cites the new numbers
except spec 04's Follow-up 7, written alongside them.

**F3.2** — **Row 25 supersedes row 23**, and row 23 is deliberately left standing
rather than edited. Row 23's decision was sound given what was known; it was
wrong only in a premise — that no authoritative NeXus identifier rule was
available — and the trail from that premise to its correction is the useful
record. §4's convention (a consolidated row records *what was decided*, the
Follow-ups record *what was believed along the way*) applies unchanged.

**F3.3** — §2.1's NXstress-side gap list and §3's Phase 2 entry describe spec 04's
scope as it was understood before the seven-axis audit. Two items there are now
resolved differently than written: the instrument name comes from configuration
rather than from the workspace (there is nothing in the workspace to read), and
the rotation-order TODO turned out to be a chain-connectivity defect rather than
an ordering one. Both are recorded in rows 26-27 and in spec 04's Follow-up 7;
the section bodies are left as the record of what was believed, per this series'
convention.

**F3.4** — **Two instructions disagree about where a finding lives, and this pass
resolved it in favour of the Follow-up. Recorded so the next session does not
re-derive it the hard way.**

`CLAUDE.md:357-363` is unambiguous: findings go in an append-only `## Follow-up N`
section of the document being audited; decisions go to the Decisions Log and the
Follow-up holds the evidence. `plans/audit-process/process.md` — the audit
definition — specifies **no** ground-truths destination at all. Its single mention
of `docs/ground_truths.md` is at `process.md:109`, and that line is **a row in the
anti-patterns table**, about a plan asserting a PyQt5 API after the repository
moved to PyQt6; it cites the file as a place where such a fact happens to be
recorded. An incidental example, in other words, and not a routing rule — but a
reader who opens line 109 expecting routing finds a Qt anti-pattern, so what is
there is spelled out here rather than left to be checked.

Pulling the other way, `CLAUDE.md:12` states as a core principle: *"ALWAYS record
key findings in `docs/ground_truths.md`"*, and `CLAUDE.md:330-333` says a probe
result that is true of PyRS generally *"**also** belongs"* there — note **also**,
which makes it a secondary copy and never the home for a finding.

The 04 implementation pass initially wrote two findings (the `validItemName` rule,
the `depends_on` chain defect) into `docs/ground_truths.md` in full, duplicating
what `04`'s Follow-up 7 F7.1 and F7.5 already held. **That was reverted**: a second
full copy of a finding is precisely the drift mechanism `process.md` §9 rejects,
and it is the same shape as the "fourth copy" warning the implementation prompt
gives about the invariants list.

**The tension is in the instructions, not in this series, and it is not ours alone
to settle** — `CLAUDE.md` is shared, and the instrument scientist maintains a
separate one. So this Follow-up does not propose editing it. What it records is
the rule this series follows, and why:

- a finding about **a document in this plan** → that document's `## Follow-up N`,
  always, and nowhere else in full;
- a finding that is **series-wide** → this README's `## Follow-up N`;
- `docs/ground_truths.md` → at most a short durable fact *pointing at* the
  Follow-up that holds the evidence, never a copy of the finding. This pass wrote
  nothing there.

A future session reading `CLAUDE.md:12` in isolation will reach for
`ground_truths.md` first. It should read `CLAUDE.md:357-363` alongside it, and
this entry.

**F3.5** — **`audit.toml`'s `[code] roots` gained `plans/` and `CLAUDE.md`.**
F3.4 above needed to cite `CLAUDE.md:357-363` and `plans/audit-process/process.md`
— the two documents whose disagreement it records. Neither was in `roots`, so
`check_citations.py` reported all six pointers as `out-of-scope` or
`path-not-found`: correctly, since it is built never to guess at a path it cannot
resolve.

The alternative was to write them as prose so they stopped being citation-shaped,
which the round-one Follow-ups did for *historical* line numbers. That is right
for a superseded value and wrong here: these are live pointers into files that
will move, and the whole purpose of F3.4 is to be found and re-read by a later
session. Suppressing the check would have made them exactly the kind of quietly
drifting pointer this toolkit exists to catch — ten of which it found in this
README's own architecture tables.

Widening the roots resolved all six and introduced **zero** new findings across
the corpus: 364 citations, the same 3 accepted `open-questions/06` residue as
before. `CLAUDE.md`'s line numbers are now checked on every run, which is the
point — that file is shared and edited outside this series, so its numbers are
more likely to drift than most, not less.
