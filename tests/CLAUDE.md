# Test conventions

Loaded when working under `tests/`; the root [CLAUDE.md](../CLAUDE.md) keeps the test naming rule.

## Test tiers, markers, and locations

Tests are split into three tiers so the fast ones can be run without waiting on
real data or being interrupted by GUI pop-ups. **Put every new test in the tier
matching what it actually touches, and apply the marker that tier requires.**

| Tier | Location | Marker | Run with |
|---|---|---|---|
| Unit | `tests/unit/<module-path>/`, mirroring the `pyrs/` package layout | *none* | `pixi run test-unit` |
| Integration | `tests/integration/` (flat) | `@pytest.mark.integration` | `pixi run test-integration` |
| GUI | `tests/ui/` | `@pytest.mark.gui` **and** `@pytest.mark.integration` | `pixi run test-gui` |
| By-hand scripts | `tests/scripts/` | *n/a — never collected* | run the file directly |

Marker definitions, registered in `pyproject.toml`:

- **`integration`** — exercises real file I/O (`tests/data`, the `/HFIR`
  archive) or a multi-component workflow.
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
  The full `pixi run test` does **not** — it is the task CI drives under
  `xvfb-run`, and a pixi task `env` would override that. Export
  `QT_QPA_PLATFORM=offscreen` yourself before running the full suite on a
  workstation with a real desktop session, or it will stall on the GUI tier
  until the timeout below fires — see [docs/ground_truths.md](../docs/ground_truths.md).
- **Every test has a 300-second timeout** (`timeout`/`timeout_method` in
  `pyproject.toml`, via `pytest-timeout`). A hang is a failure, not an infinite
  wait. `timeout_method = "thread"` is deliberate: a test blocked inside Qt's C++
  event loop never returns to the interpreter, so the default `signal` method
  cannot interrupt it — verified. The watchdog dumps every thread's stack, which
  names the blocking line. Override per-test with `@pytest.mark.timeout(N)` for a
  genuinely long-running case rather than raising the global limit.
