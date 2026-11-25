# Test conventions

Loaded when working under `tests/`; the root [CLAUDE.md](../CLAUDE.md) keeps the test naming rule
and the "Pytest markers" block.

## Test tiers, markers, and locations

Tests are split into four tiers so the fast ones can be run without waiting on
real data, a full ring calibration, or GUI pop-ups. **Put every new test in the tier
matching what it actually touches, and apply the marker that tier requires.**

| Tier | Location | Marker | Run with |
|---|---|---|---|
| Unit | `tests/unit/<module-path>/`, mirroring the `pyrs/` package layout | *none* | `pixi run test-unit` |
| Ring calibration | `tests/unit/<module-path>/`, beside the unit tests | `@pytest.mark.ring_calibration` | `pixi run test-ring-calibration` |
| Integration | `tests/integration/` (flat) | `@pytest.mark.integration` | `pixi run test-integration` |
| GUI | `tests/ui/` | `@pytest.mark.gui` **and** `@pytest.mark.integration` | `pixi run test-gui` |
| By-hand scripts | `tests/scripts/` | *n/a — never collected* | run the file directly |

Marker definitions, registered in `pyproject.toml`:

- **`integration`** — exercises real file I/O (`tests/data`, the `/HFIR`
  archive) or a multi-component workflow.
- **`ring_calibration`** — runs a full ring calibration of the synthetic run, or
  uses the shared one (`tests/util/ring_calibration_helpers.py`:
  `converged_synthetic_calibration`, `use_cached_calibrate_rings`). That
  calibration takes ~30 s and is cached per process, so the tier runs serially
  and computes it once; in the parallel unit tier every worker would compute it
  again (#1101).
- **`gui`** — requires a display (xvfb or offscreen). Constructing or driving a
  Qt widget is the usual reason, but it is not the only one: taking the `qapp`
  fixture is enough on its own (it builds a real `QApplication`, which aborts
  with no display), as is importing any module that pulls in matplotlib's
  `QtAgg` backend at module scope. If the module cannot be *imported* headless,
  it is `gui` — `python -c 'import <module>'` with `DISPLAY` and
  `QT_QPA_PLATFORM` unset is the check.

Notes:

- Markers are enforced by `addopts = "--strict-markers"`: an unregistered marker
  is an error, not a silent no-op. Register new markers in `pyproject.toml`.
- `test-gui` selects `-m gui`; `test-integration` selects
  `-m 'integration and not gui'`. GUI tests therefore carry **both** markers —
  `gui` alone would drop them from the integration tier.
- A test is a *unit* test only if it needs no real file and no widget. If it
  loads a fixture file purely for convenience, prefer rewriting it against a
  synthetic fixture and keeping it in the unit tier.
- Apply a whole-module marker with `pytestmark = pytest.mark.integration` rather
  than decorating every function.
- `tests/util/` holds shared helper modules and fixtures, not tests of its own
  (beyond tests *for* those helpers); `tests/scripts/` is excluded from
  collection via `norecursedirs`.
- `test-gui` sets `QT_QPA_PLATFORM=offscreen` itself, so no window ever appears.
  The full `pixi run test` (all tiers, serial, with coverage) does **not**. Export
  `QT_QPA_PLATFORM=offscreen` yourself before running it on a workstation with a
  real desktop session, or it will stall on the GUI tier until the timeout below
  fires — see [docs/ground_truths.md](../docs/ground_truths.md).
- **The unit and integration tiers run in parallel** (`pytest-xdist`, `-n auto
  --dist loadscope`: one worker per core, at most 8 locally; a module's top-level test functions, or a
  test class, all go to one worker, so a module-scoped fixture used by functions is
  built once). The GUI tier stays serial: its tests share one window per module.
  So a test must not depend on another test's order or write a fixed file name
  into the working folder; use `tmp_path`. To debug with `--pdb` or `-s`, run
  serially: `pixi run test-unit -n 0`. The ring-calibration tier is serial (see its marker). CI runs
  each tier as its own job (`test-ci-unit`, `test-ci-ring-calibration`, `test-ci-integration`,
  `test-ci-gui`, under `xvfb-run`, #1100, #1101).
- **Every test has a 300-second timeout** (`timeout`/`timeout_method` in
  `pyproject.toml`, via `pytest-timeout`). A hang is a failure, not an infinite
  wait. `timeout_method = "thread"` is deliberate: a test blocked inside Qt's C++
  event loop never returns to the interpreter, so the default `signal` method
  cannot interrupt it — verified. The watchdog dumps every thread's stack, which
  names the blocking line. Override per-test with `@pytest.mark.timeout(N)` for a
  genuinely long-running case rather than raising the global limit.
