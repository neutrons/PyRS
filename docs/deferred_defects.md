# Deferred defects

Defects in the PyRS codebase that were found and reproduced, but deliberately not fixed in the change
that found them. Each PR's commit message lists the ones it found, under "Defects recorded, not fixed".

One line per defect: its number, the date it was found, and enough of a description to identify it.
Numbers are never reused. When a defect is fixed, mark its line fixed rather than deleting it.

- **DD-1**, 2026-10-11: `validate_config()` runs only at import, so a configuration installed later with
  `Config.loadEnv` is never validated. The gap is in `neutrons_standard.Config`, which has no validation hook.
- **DD-2**, 2026-10-20: a workspace's sub-runs must be in increasing order, which nothing needs, and about ten
  sites use a sub-run number as an array position, so they break on any workspace whose sub-runs do not run
  1..N. Each site has a strict `xfail` test.
- **DD-3**, 2026-10-21: strain/stress viewer: loading a project file with no peak collections as e22 or e33
  raises `IndexError`.
- **DD-4**, 2026-10-21: strain/stress viewer: `Controller.fileSelected` is never called, and calls
  `Model.set_workspace`, which does not exist.
- **DD-5**, 2026-11-05: detector calibration viewer: a recipe saved before any data were loaded cannot be
  loaded back. `PeakLinesSetupView.save_json` writes `nexus_file` only when a file is loaded, but `load_json`
  reads it unconditionally and raises `KeyError`.
