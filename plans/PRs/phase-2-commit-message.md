NXstress hookup: phase 2

This commit includes the following changes:

I. Changes to the PyRS codebase:

  - Loading a project file (`HB2BReductionManager.load_hidra_project`) always opens it read-only, and closes it even when the load fails. It used to open any writable file read-write.

  - String sample logs are `str` in memory, whatever they were read from: `SampleLogs` converts them on the way in, and `HidraProjectFile.add_sample_log` writes them as variable-length UTF-8. Whether a log came back as `bytes` or `str` used to depend on where the workspace came from, and consumers handled the two inconsistently.

  - Added `to_text`, `is_text_array` and `to_text_array` to `pyrs.utilities.convertdatatypes`, replacing nine hand-written `bytes`/`str` conversions that behaved four different ways.

  - Added `pyrs/utilities/restorable_property.py`. `@restores("_attr")` marks a read-only property whose value a file reader may put back, through `restore()`. The property stays read-only to everything else, including mypy.

  - Added the restorable `HidraWorkspace.direction` (`'11'`, `'22'` or `'33'`). It is the value the workspace was given or restored with, and falls back to the `direction` sample log. A workspace whose `direction` log varies across sub-runs, such as one merged in PyRS, has no single direction, and `direction` raises `ValueError` saying so. `hidra_project_file` is restorable too.

  - `HidraWorkspace.set_reduced_diffraction_data` refuses data whose shape disagrees with the 2θ matrix or the sub-runs. `set_sample_log` no longer sorts the sub-run axis, which silently paired each value with the wrong sub-run when they arrived out of order.

  - Configuration is validated when PyRS starts, against a pydantic schema (`ConfigSchema` in `pyrs/utilities/config.py`). A malformed value or an unknown key stops start-up with a message naming each one. Extensions are configured without their dot (`nxs`, `h5`), and `gui_helper.impose_extension` adds it. New keys: `nxstress.instrument_name`, `nxstress.instrument_short_name` and `nxstress.discriminator_fields`.

  - Strain/stress viewer: each loaded workspace is given the direction it was selected as, and a project file whose `direction` log records a different one is refused with an error naming both.

  - `.pre-commit-config.yaml` excludes the vendored `docs/developer/source/design/nexus/NXstress.nxdl.xml`, which must stay byte-identical to upstream.

  - Tests:
    - the `config_override` fixture changes configuration in place, so modules that imported `Config` see the change, and it validates what it installs;
    - `tests/util/workspace_helpers.py` builds the synthetic `HidraWorkspace` used by the NXstress tests and the demo scripts;
    - `tests/data/HB2B_1327_with_instrument.h5`, a real run with instrument geometry, is made by `tests/scripts/make_nxstress_sample_position_fixture.py`;
    - `tests/scripts/NXstress/` holds the by-hand demo scripts listed under the How to test section of plans/PRs/phase-2.md.

II. Phase 2 implementation:

  - `NXstress.write` takes a list of workspaces and one list of peak collections per workspace, and `read` returns the same shape: `write([ws], [peaks])`, `(wss, peakss) = read()`. It replaces `write(ws, peaks)` / `(ws, peaks) = read()`; the viewers and CombineRuns are updated.

  - A file can hold several `NXentry` groups, numbered from 1. `write` and `read` take an `entry_number`, and `entry_numbers()` lists the entries a file holds. With no number, a write grows the last entry; a number one past the last creates a new entry; any other unused number is refused, and so is a number below 1, by both `write` and `read`.

  - Several workspaces can share one entry, told apart by the fields in `nxstress.discriminator_fields` (such as `direction`). Each field becomes a `peaks` column, and `read` splits the entry back into the workspaces written. A field is taken from a `HidraWorkspace` property if there is one, which must be restorable, and otherwise from a sample log. Field names that are not valid NeXus names are encoded reversibly. The rules for which columns count are in VI below.

  - NXstress never merges workspaces (Decisions Log row 45 in plans/NXstress-prod/README.md). With no discriminator field configured, writing more than one workspace is refused: merge them in PyRS first, as CombineRuns does. A workspace with more than one value of a configured field, such as a merged workspace under a `direction` discriminator, is refused with `ValueError`; written with no discriminator, it round-trips as one workspace.

  - Workspaces sharing an entry must agree on instrument geometry, on their detector masks (compared by contents, not only by name), and on sample name and chemical formula, and must not share a scan point. Raw detector counts must be present on all of them or none.

  - Workspaces that share one discriminator value, and would read back as one workspace with scan points out of increasing order (such as `[4, 5, 6]` + `[1, 2, 3]`), are refused with `NotImplementedError` rather than re-sorted, since re-sorting would break the round trip. The ordering requirement itself is arbitrary, and DD-2 in `plans/PR_review/deferred-defects.md` recommends dropping it.

  - Mode `"a"` appends to an entry: a workspace with a discriminator value the entry does not hold yet is added after the existing rows. More scan points for a value already there are refused with `NotImplementedError`. An append must satisfy the same agreement rules as a write, and is also refused for a sample log or discriminator value whose type the entry's existing column cannot hold, or for an entry with a `peaks` column that is neither PyRS's own nor a configured discriminator. Every check runs before anything is written, so a refused append leaves the file byte-for-byte unchanged, and the `NXstress` instance can no longer be used.

  - Every per-scan-point and per-peak dataset is written resizable, which is what makes an append possible. A file written before phase 2 is refused for append, with a message naming its fixed-size datasets.

  - `pyrs.interface.nxstress_load.read_single_workspace`: PeakFitting and TextureFitting refuse a file holding more than one entry, or more than one workspace in its entry, with an error naming what it holds. Showing only the first workspace would silently drop the rest, and a save would then lose them.

  - `peaks/sx`, `sy` and `sz` hold each scan point's `vx`/`vy`/`vz` sample coordinates, with a `local_name` attribute naming the log. They were always NaN.

  - Transformation chains: the detector's `depends_on` names the last link of its chain, so all eight transformations take effect; it named `translation_x`, so only that one did. The sample has a chain too: translations from `vx`/`vy`/`vz`, a fixed frame offset, then the `omega`/`chi`/`phi` rotations.

  - Raw detector counts, diffractograms and sample logs from several workspaces are concatenated. A workspace without one of the diffractograms the others have gets NaN rows for it.

  - Detector masks: a mask that looks like a solid-angle mask is refused with `NotImplementedError`, where it used to be written as `[-180, 180]`. The broken `NXlink` fallback is removed.

  - The instrument name and short name are read from `nxstress.instrument_name` and `nxstress.instrument_short_name`, and were hard-coded to `HB2B`.

  - Tests: unit tests for each group's multi-workspace, append and read paths (`test_append.py`, `test_multi_workspace.py`, `test_discriminator.py` and the per-group files), integration tests for the viewers' loads and the strain/stress direction check, and `test_nxstress_sample_position.py` against a real run with geometry.

III. PyRS defects fixed:

  Found and fixed while assembling phase 2. Each behaviour fix has a regression test that fails without it.

  - TextureFitting: loading a file that cannot be read (not HDF5, or an NXstress file with more than one entry) now shows an error and keeps the previously loaded workspace. The model used to swallow the error into a signal nothing was connected to, so nothing was reported, and on a first load the plotting that followed aborted the application.

  - TextureFitting: choosing a sample-log plot axis (`vx`, `phi`, …) before fitting, or loading a file while a fit-parameter axis is selected, aborted the application with a `TypeError`. Before a fit, sample-log axes are now plotted from the workspace, and an axis with nothing to show clears the plot.

  - PeakFitting: loading a non-HDF5 `.h5`, a malformed `.nxs`, or a multi-file selection that includes a `.nxs` now shows an error. These raised exceptions the load handler did not catch, which aborted the application. After a failed load the viewer also stops, where it carried on against the previously loaded workspace.

  - Strain/stress viewer: after a refused load, the file field is left empty. It showed the refused file's name as if it had loaded, because `Controller.filesSelected` checked for `None` and a refused load leaves an empty list.

  - Strain/stress viewer: the project file each direction is loaded from is now closed explicitly. It was left to `HidraProjectFile.__del__`, so it stayed open for as long as anything else referenced it.

  - Configuration: a key that PyRS does not define, in `application.yml` or an `env` override file, now stops PyRS at start-up with a message naming the key. It used to be ignored, so a misspelled key, or one in the wrong section, changed nothing and the setting kept its default (Decisions Log row 46 in plans/NXstress-prod/README.md). The sections `neutrons_standard` adds itself are exempt.

IV. Test-suite defects fixed:

  - Every GUI test module now builds its window per test, on pytest-qt's `qtbot`, instead of sharing one session-scoped window, and the shared `my_qtbot` fixture is gone. Shared windows made tests order-dependent: two TextureFitting tests failed only when run after one that left plot axes selected. The segfault cited as the reason for sharing cannot be reproduced (see `tests/ui/conftest.py`).

V. Defects recorded, not fixed:

  - DD-2 in `plans/PR_review/deferred-defects.md`: the sub-run ordering requirement, and about ten sites that use a sub-run number as an array position. These break on any workspace whose sub-runs do not run 1..N. Each has a strict `xfail` test.

  - DD-3: the strain/stress viewer raises `IndexError` when a project file with no peak collections is loaded as e22 or e33. Predates phase 2.

  - DD-4: the strain/stress `Controller.fileSelected` is never called, and calls a `Model` method that does not exist.

VI. Discriminator columns on read and append:

  When several workspaces share one NXstress entry, each `peaks` row carries one discriminator column per field in `nxstress.discriminator_fields` (e.g. `direction`), and `read` uses those columns to split the entry back into workspaces. Which columns are discriminators is decided as follows (Decisions Log row 43 in plans/NXstress-prod/README.md).

  Configuration is the authority. The fields in `nxstress.discriminator_fields` are the discriminators; nothing is inferred from the file's columns. To read a file written with other fields, configure those fields.

  The file records its fields. Every write sets a `discriminator_fields` attribute on the `peaks` group, listing the fields it was written with. An empty list means the entry holds no workspace boundary: it was written from a single workspace with no discriminator field configured, or with no peak collections.

  Reading an entry that has the record:

    - empty: Read as one workspace, whatever is configured.
    - same fields as configured: Split by those fields.
    - different fields: Refused; the message names the fields to configure.
    - a recorded field has no column: Refused as damaged.

  Reading an entry without the record (a file from another producer, or one written by phase 1), by which configured fields have a column:

    - none: Read as one workspace.
    - all: Split by those fields.
    - some, not all: Refused.

  Any other column is ignored on read, with a log message. It may be another producer's reflection field, and cannot be told apart from an unconfigured discriminator.

  An append refuses an entry that has any column which is neither PyRS's own nor a configured discriminator, before anything is written. An append grows only the columns it knows, so such a column would be left short. The append's other rules are unchanged: the target must already record a workspace boundary, and its fields must match configuration.

  What this costs. For an entry without the record, a configuration that omits a discriminator is not detected. Usually the read is then refused anyway (repeated or out-of-order peak blocks), but when each workspace has one peak and their scan points increase across workspaces, two workspaces silently read back as one. Every file PyRS writes carries the record, so this affects only files from other producers and files written by phase 1.

  Covered by `tests/unit/pyrs/utilities/NXstress/test_recorded_discriminators.py`, one test per rule above, the silent-merge case included.
