# 01 — Config Infrastructure & Test Framework

**Plan:** [NXstress GUI Hookup](README.md)
**Phase:** 1
**Depends on:** —

> **Notes**:
>
>  * For the moment, the contents of `plans/` is not included in the PRs.
>  * To completely follow what's going on, see the non-squashed parent branch:
>    `git+ssh//git@github.com/ekapadi/PyRS.git@EWM12484_NXstress_hookup`.\
>    (Use `git remote add ekapadi git@github.com:ekapadi/PyRS.git`.)
---

## Overview

This is the first and foundational sub-spec. It introduces two pieces of
infrastructure that every subsequent spec relies on:

1. **Config** — a general runtime configuration file for PyRS, loaded via the shared
   `neutrons_standard.Config` singleton.  Initially this allows
   NXstress behaviour (field naming, default file extension, etc.) to be
   controlled without code changes.  Longer term, this allows control of any PyRS
   feature or behavior that _opts-in_ to using this config.
   To override any section of this config, we create an appropriate `pyrs/resources/dev.yml` file
   including the required overrides, and then use `env=dev pyrsplot` (e.g.).
2. **Test-framework fixups** — shared pytest fixtures and conventions for the
   NXstress test suite, so every subsequent spec can write clean, consistent
   tests without repeating boilerplate. **This also includes restructuring
   of the PyRS test framework so that we can run _unit_, _integration_, and
   _integration_ + _UI_ tests as separate groups.**

Neither of these items add user-visible NXstress I/O to the GUI yet (that
starts in specs 02–03), but both are prerequisites for all later work.

---

## Scope

**In scope:**
- Add the `neutrons` pixi channel and the `neutrons_standard` dependency
  (`[tool.pixi.dependencies]` and run-dependencies)
- `pyrs/utilities/config.py` — thin wrapper that registers PyRS with
  `neutrons_standard` and re-exports its `Config` singleton, plus PyRS's own
  "at least one format enabled" validation rule
- `pyrs/resources/application.yml` (+ `pyrs/resources/__init__.py`) —
  default config file, at the exact location/name `neutrons_standard`
  requires
- The two-section `nxstress`/`legacy_io` config schema (see below)
- Shared pytest fixtures in `tests/unit/pyrs/utilities/NXstress/conftest.py`
  and `tests/unit/pyrs/utilities/conftest.py`
- Test-data management conventions (temp-file helpers, fixture cleanup)

**Out of scope:**
- Any NXstress writer/reader changes
- Any GUI viewer changes

---

## PyRS Changes

_None to existing modules_ — the config infrastructure is new code. A plan
reviewer (dev team) asked that it build on the shared `neutrons_standard.Config`
singleton (`github.com/neutrons/PythonCommons`, `neutrons` pixi channel)
rather than a from-scratch pydantic loader; this changed the design from
what this section originally specified.

Pixi wiring (`pyproject.toml`):
- `"neutrons"` added to `[tool.pixi.workspace] channels`.
- `neutrons_standard = "*"` added to `[tool.pixi.dependencies]` and
  `[tool.pixi.package.run-dependencies]`. Verified empirically that this is
  sufficient for every pixi environment (`default`, `dev`, `qa`, `prod`) to
  resolve and import `neutrons_standard` — pixi's environments compose base
  deps with feature deps additively, so no per-environment duplicate entry
  is needed.

New files:
- `pyrs/utilities/config.py` — registers PyRS with `neutrons_standard`
  (`neutrons_standard.init("pyrs")`) and re-exports its `Config` singleton;
  `Config` must always be imported from *this* module, never
  `from neutrons_standard...import Config` directly elsewhere, because
  `init()` must run before `neutrons_standard.config` is first imported (a
  stray direct import elsewhere would race `init()` and silently pin
  `neutrons_standard`'s internal `package_name` to `None` for the rest of
  the process). Also owns PyRS's own "at least one format enabled"
  validation rule — `neutrons_standard.Config` provides no schema
  validation of its own — raised eagerly at import time via
  `validate_config()`, rather than at the first NXstress-I/O callsite that
  needs a valid config.
- `pyrs/resources/__init__.py` + `pyrs/resources/application.yml` — the
  exact filename and location `neutrons_standard` requires (a genuine
  `pyrs.resources` subpackage, found via `importlib.resources`), not a
  PyRS-chosen path. Content:
  ```yaml
  nxstress:
    enable: true
    extension: ".nxs"
    use_production_names: false   # true once nexusformat validator bug resolved
  legacy_io:
    enable: true
    extension: ".h5"
  ```
  Two fully parallel, self-contained top-level sections — one per format.
  Each owns its own `enable` flag and its own `extension`; nothing is
  shared or ambiguous between them. `validate_config()` raises if
  `not (nxstress.enable or legacy_io.enable)`.
  Packaging note: no change needed — `pyproject.toml`'s existing
  `[tool.hatch.build.targets.wheel] artifacts` glob `"pyrs/**/*.yml"`
  already covers this new location.

No CLI wiring: `neutrons_standard.Config` is driven entirely by the `env`
OS environment variable (e.g. `env=/path/to/override.yml`, deep-merged on
top of the shipped default), not by an argparse flag.

`Config` (from `pyrs.utilities.config`) is importable and usable via
dot-string keys (e.g. `Config["nxstress.enable"]`) at NXstress callsites
that read the flags (spec 02 onwards).

---

## NXstress / GUI Changes

_None_ — no changes to `pyrs/utilities/NXstress/` or `pyrs/interface/` in
this spec.

---

## Test-framework Fixups

The following items clean up and extend both the PyRS test framework in general,
and the NXstress test suite specifically so that specs 02–10 can write tests
without boilerplate.

### Shared fixtures (`tests/unit/pyrs/utilities/NXstress/conftest.py`)

- To be as explicit as possible, these fixtures use a modified _snake_case_
  naming convention including the associated class name specified in its normal _CamelCase_ form.
- `minimal_HidraWorkspace` — a factory fixture, since real NXstress tests
  need several structural combinations. Returns a small, entirely
  synthetic `HidraWorkspace` — no project file is read from disk — built
  from minimal `SampleLogs` (`vx`/`vy`/`vz`, `start_time`/`end_time` as
  ISO-8601 bytes matching real HDF5 string datasets, `mrot`, `Filename`),
  one or more sub-runs, and a wavelength, with instrument geometry, masks,
  raw counts, and reduced-diffraction data each toggled on via keyword
  flags (`with_instrument`, `with_masks`, `with_raw_counts`,
  `with_reduced_diffraction`).
- `minimal_PeakCollection` — a thin, defaults-filling wrapper around the
  pre-existing `createPeakCollection` fixture
  (`tests/util/peak_collection_helpers.py`).
- `load_HidraWorkspace` (the original real-file loader) is kept as an
  explicit *legacy* fixture, documented as such, for use by any _integraton_ test that
  genuinely needs real project-file content.
- `default_config` (`tests/unit/pyrs/utilities/conftest.py` — one directory
  up from `NXstress/`, since `pyrs/utilities/config.py` isn't itself
  NXstress-specific, and a fixture there is visible down into `NXstress/`
  too) — a test-isolated `neutrons_standard.Config` singleton. Every test
  that requests it gets a genuinely fresh instance: `HOME` monkeypatched to
  `tmp_path`, `env` cleared, `reset_Singletons()` called, then
  `pyrs.utilities.config` (and `neutrons_standard.config`) reloaded so the
  already-bound `Config` name actually picks up the reset.

### Test-data hygiene

- Tests that actually require real-file I/O now have the _integration_ pytest marker.
  Where such tests did not actually depend on the _content_ of the loaded
  files, they were reworked to use synthetic fixtures and are now classified
  as _unit_ tests (with _no_ pytest marker).


### Markers & test tiers (repo-wide)

`integration`/`gui` pytest markers registered in `pyproject.toml` and
applied across the whole test suite, with `test-unit` / `test-integration`
/ `test-gui` pixi tasks. This is the mechanism specs 02–10 should use to
tag their own new tests.  **If nothing else, this now allows us to run
PyRS _unit_ tests without constantly being interrupted by GUI popups!**

### Test directory cleanup

Several pre-existing test files were moved to the `unit`/`integration`
location matching their actual behavior.  In addition, now that markers make that
classification possible; two overly-mixed files were split to allow a separate
integration-test section.

### RNG determinism & synthetic-data realism

* `createPeakCollection`'s RNG was a shared, session-global instance — a
test's outcome depended on how many random draws every other test that
happened to run earlier in the session had already consumed. It now
re-seeds fresh per test.
* Synthetic parameter uncertainties were also
bounded to realistic proportional fractions (0.5%–5% of each parameter's
own value) instead of independent absolute ranges — this fixes an
unbounded catastrophic-cancellation amplification that one PseudoVoigt
round-trip test's tolerance triggered when a randomly generated uncertainty
exceeded any realistic value.

---

## Delivered Feature

> **For end users and contributors:**
> PyRS's runtime configuration is now backed by the shared
> `neutrons_standard.Config` singleton. A default configuration file
> (`pyrs/resources/application.yml`) ships with the package and documents
> all available options. NXstress output and legacy `.h5` output are each
> independently controlled — `nxstress.enable` and `legacy_io.enable`.
>
> Override any value by setting the `env` OS environment variable to the
> name or path of a `.yml` file, whose contents are deep-merged on top of
> the shipped default — e.g. `env=/path/to/override.yml pyrsplot`.
> This matches `neutrons_standard`'s own idiom.
>
> Internally, the test suite for NXstress (and for `pyrs/utilities/`
> generally) gains shared fixtures that make it easier to write and
> maintain tests, including a `default_config` fixture that gives each test
> a fully isolated configuration singleton.

---

## Verification

- `pixi install` succeeds with `neutrons_standard` resolved from the
  `neutrons` channel, in every pixi environment (`default`, `dev`, `qa`,
  `prod`).
- `python -c "import pyrs.utilities.config as c; print(c.Config['nxstress.enable'])"`
  → `True`, confirming the resource file resolves and loads correctly from
  an installed (not just source-tree) layout.
- A config with both `nxstress.enable: false` and `legacy_io.enable: false`
  raises at `pyrs.utilities.config` import time (via `validate_config()`),
  not later.
- `env=<path>` pointing at a `.yml` file with an override merges correctly
  on top of the shipped default.  With our current very _sparse_
  config, I recommend testing this just using a config including only the two lines
  `nxstress:\ ^^enable: false`.
- `pixi run test` — full suite passes (383 passed, 29 skipped as of
  this writing), and `~/.pyrs/` does not exist afterward in a clean
  environment (`rm -rf ~/.pyrs` before the run, `ls ~/.pyrs` fails after) —
  confirming `default_config` fully isolates every config-touching test
  from the real home directory.  **Also try the new`pixi run test-unit`,
  `pixi run test-integration`, and `pixi run test-gui`**.
- `pytest tests/unit/pyrs/utilities/test_config.py` — dedicated
  config-loader unit tests pass: shipped defaults load correctly, `env`
  override merges correctly, `validate_config()` passes with defaults, and
  raises when both formats are disabled.

---

## Follow-up 1 — 2026-09-25 (first seven-axis pass, freshness check only)

This spec has landed. Per the pass's scope it received an **A7 freshness check
and nothing else** — its claims are settled by shipped code and passing tests,
not by re-reading. Recorded so that "A7 only" is never mistaken for "all seven
axes were run".

**F1.1** (A7) — No stale citations. `landing_trigger.py` over
`7a5ef73f..HEAD` finds no citation in this document pointing into a file that
changed in the range, and this document carries no `path:LINE` citations to
drift.

**F1.2** (A4) — Two `neutrons_standard` behaviours this spec's shipped code
depends on, neither documented anywhere in the series. Found while probing
README §2.3; recorded here because this is the spec that owns
`pyrs/utilities/config.py` and the test framework. Probe:
[`probes/a4_neutrons_standard_config.py`](probes/a4_neutrons_standard_config.py).

1. **`Config` writes to the user's home directory on every load.**
   `_Config.__init__` calls `persistBackup()`, which writes
   `~/.pyrs/application.yml.bak`:

   ```console
     CLAIM   UNDOCUMENTED: _Config.__init__ -> persistBackup() writes to the user's
             HOME on every load
     RESULT  /home/…/.pyrs/application.yml.bak existed_before=True; mtime changed by
             a bare Config import: True
   ```

   Benign in normal use, but it means importing `pyrs.utilities.config` is not
   side-effect-free — worth knowing before anything imports it in a context
   where the home directory is read-only or shared.

2. **Under a test environment the config resources root moves outside the
   repository.** `_find_root_dir()` redirects to `MODULE_ROOT.parent.parent /
   "tests"` whenever `isTestEnv()` — that is, when the `env` variable contains
   `"test"` **and** `conftest` is in `sys.modules`. For this repo's layout that
   resolves one level too high:

   ```console
     CLAIM   UNDOCUMENTED: with env containing 'test' AND conftest imported, the
             resources root moves to <repo>/tests/
     RESULT  FileNotFoundError: [Errno 2] No such file or directory:
             '/home/ux0/workspaces/tests/resources/application.yml'
   ```

   Note the path: `workspaces/tests`, **not** `PyRS/tests` — outside the
   repository altogether. Any test that sets `env` to a name containing "test"
   (the documented idiom for per-tier config, e.g. `integration_test.yml`) will
   fail to load config. This is a live constraint on the test framework this
   spec shipped.
- Action (for a follow-up PR, not this audit): decide deliberately whether PyRS
  ever sets `env` to a "test"-containing value. If it should, a
  `tests/resources/application.yml` is needed at the location
  `neutrons_standard` actually computes — which is not inside this repo, so the
  honest options are to avoid the idiom or to raise it upstream. **Flagged, not
  fixed**: it is a design decision, not a documentation defect.

**F1.3** (A4) — The other six README §2.3 claims about `neutrons_standard` were
probed and **confirmed**: the `init()`-before-import ordering requirement, the
`pyrs/resources/application.yml` location requirement, `env=<file>` deep merge
(siblings survive), the `~/.pyrs/pyrs-user.yml` auto-load, the absence of any
schema validation, and dot-string key access. One characterisation is wrong —
a stray import does not "silently corrupt" anything — see the README's
Follow-up 1 and `probes/README.md`.
