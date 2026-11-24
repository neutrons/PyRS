# Claude Code Instructions

This file configures how Claude Code should assist with development in this repository. This project is designed for scientists who may be new to software engineering, so all interactions should be educational, clear, and follow best practices. These instructions mirror [.github/copilot-instructions.md](.github/copilot-instructions.md) — keep both in sync when you change one.

## 🎯 Core Principles

1. **Always assess before acting** — Understand the current state before proposing changes.
2. **Always provide itemized plans** — Break work into clear, testable steps tracked with `TodoWrite`.
3. **Focus on progress tracking** — The user should always know what's happening and what's next.
4. **Test incrementally** — Each step should be testable on its own.
5. **Review after major changes** — Delegate review to a sub-agent via the `Agent` tool.
6. **Ground truths** — ALWAYS record key findings in [docs/ground_truths.md](docs/ground_truths.md) for future reference. Update the file with new findings, and link to it from related documentation and code comments.

## 🔄 Standard Workflow for Every Request

### Step 1: Assess
Before responding, examine:
- The current implementation (read the relevant files; parallelize reads when independent).
- Existing tests and documentation.
- Related code that may be affected.
- Project structure and conventions.

**Output:** a brief summary of what exists today and what needs to change. Reference files with markdown links — `[core.py:42](src/package_name/core.py#L42)` — so the user can click through.

### Step 2: Plan
Use `TodoWrite` to capture an itemized plan:
- Numbered, actionable steps.
- Each step independently testable.
- Dependencies between steps noted.
- Expected test coverage outlined.

Mark each task `in_progress` when you start it and `completed` the moment it's done — do not batch completions.

### Step 3: Implement Incrementally
- Complete ONE todo at a time.
- Announce the step in one short sentence before the tool calls: "Implementing step 1 — input validation."
- Pause after significant steps when it makes sense to confirm direction.

### Step 4: Test
After implementation:
- Run the relevant tests via `Bash` (`pytest`, targeted file/test where possible).
- Show the result.
- Fix failures before proceeding.
- For UI work, actually exercise the feature in a browser — type checking and tests verify code correctness, not feature correctness.

### Step 5: Review (For Major Changes)
After a significant feature or refactor, delegate review using the `Agent` tool. Prefer the project's purpose-built reviewers in [.github/agents/](.github/agents/):
- [security-reviewer](.github/agents/security-reviewer.md) — security-focused review.
- [design-reviewer](.github/agents/design-reviewer.md) — architecture and design review.
- [test-reviewer](.github/agents/test-reviewer.md) — test-coverage review.

Brief the sub-agent with what changed, why, the files touched, and what to focus on. Address findings, then update documentation.

## 📦 Environment & Launching

All environment setup, dependency management, and program launching in this repo goes through **pixi**. Use `pixi run <task>` (e.g. `pixi run pyrs`) to start the application, and `pixi install`/`pixi shell` for environment work — never hand-roll a `python setup.py build`, a bare `pip install`, or a standalone launcher shell script that bypasses pixi's environment. Stale, pixi-external artifacts (e.g. an old `build/` directory or ad hoc launcher scripts) are a common source of "it still does the old thing" bugs because they don't get updated when the source changes.

## 🛠️ Technology Stack Preferences

When the user needs to choose, prefer these well-integrated options:

### Web
- **Web framework:** Flask (simple web apps).
- **API framework:** FastAPI (APIs, MCP servers).
- **CSS:** Bootstrap.
- **Templates:** Jinja2.

### CLI
- **Framework:** Click.
- **Progress bars:** tqdm.
- **Config:** click-config or python-dotenv.

### Data
- **Manipulation:** pandas, numpy.
- **Plotting:** matplotlib, plotly.
- **Scientific computing:** scipy.

### Dev tooling
- **Testing:** pytest (+ pytest-cov).
- **Linting:** ruff.
- **Formatting:** black.
- **Type checking:** mypy, run by a pre-commit hook as `pixi run mypy pyrs scripts tests`, the same command as CI's lint job. Run that command, not mypy on single files, before pushing.
- **Docs:** Sphinx or MkDocs.

### Agent Skills
- Follow the official spec: https://agentskills.io/specification.

## 📝 Code Quality Standards

Always include:
1. **Type hints** on parameters and return values.
2. **Docstrings** — Google-style for public functions and classes.
3. **Specific exceptions** with clear messages.
4. **Input validation** at boundaries.
5. **Comments only when the WHY is non-obvious** — never narrate what the code does.

Example:

```python
from typing import Optional


def example_function(
    data: list[float],
    threshold: float = 0.5,
    normalize: bool = True,
) -> list[float]:
    """
    Process data with filtering and optional normalization.

    Args:
        data: Raw measurement values from sensor.
        threshold: Minimum value to keep (default: 0.5).
        normalize: Whether to normalize to [0, 1] range (default: True).

    Returns:
        Processed data values.

    Raises:
        ValueError: If data is empty or threshold is negative.

    Example:
        >>> example_function([0.1, 0.7, 1.2], threshold=0.5)
        [0.58, 1.0]
    """
    if not data:
        raise ValueError("Data cannot be empty")
    if threshold < 0:
        raise ValueError(f"Threshold must be non-negative, got {threshold}")

    filtered = [x for x in data if x >= threshold]
    if not filtered:
        return []

    if normalize:
        max_val = max(filtered)
        return [x / max_val for x in filtered]
    return filtered
```

## `None` must name a value, not a gap

**Absence must be distinguishable from a falsey value.** `""`, `[]`, `0` and
`None` each mean something real somewhere, so none of them can also mean "not
supplied". `arg: Optional[T] = None` is where that collapse usually enters: the
default makes "the caller decided `None`" and "the caller passed nothing" the same
observation, and nothing at the call site recovers the difference. An argument
whose absence would be a wiring bug therefore carries **no default at all** —
required and keyword-only. `None` is then free to mean what it says.

Two corollaries:

- `None` must name a value in the argument's **domain**, never a gap. A `None`
  standing in for "not computed yet" makes a half-built object indistinguishable
  from a finished one; whatever the consumer actually wanted — usually an empty
  collection — was the right value all along.
- **Count the fillers.** One filler means absence is a bug, so no default. When
  two parties fill a field — a model plus a prep hook, or a writer plus a reader
  deserialising stored rows — absence from *one* of them is the normal case, so
  the default is correct and the guard moves to whichever filler can tell that it
  failed. (A row written by an *older version* of the code is not a filler: that is
  a migration, done with the change, not an ongoing absence to guard against.)
  For example:

```python
class ReductionRequest(BaseModel):
    calibration_file: Optional[str] = None  # the caller never supplies this; a later step fills it


def prepare(request: ReductionRequest) -> None:
    ...
    raise RuntimeError("calibration_file could not be resolved")  # never a quiet return
```

Why: requiring a field that a *second* party fills publishes it to the first one, which
may have no way to supply it correctly. And a guard needs the information to be right: a
step that sees only the payload cannot tell "this field does not exist here" from "this
field was lost", so that check belongs in a test that reads the schema instead.

## Keyword-only arguments

Any function or method with more than 3 arguments, or with any argument
that has a default value, keeps only its first one or two "primary
subject" arguments positional — every other argument, whether it has a
default or is required, goes after a bare `*`. This applies even to
required arguments once the threshold is crossed, so that a function's
argument list can grow later without breaking a positional call site.

```python
def reduce_sub_run(
    run_number: int,
    *,
    detector_name: str,
    calibration_file: str,
    sub_runs: list[int],
    output_path: str,
) -> None: ...
```

Why: adopted as a general convention rather than from a specific PyRS incident. Many
existing functions already cross the 3-arg/any-default threshold with positional
signatures (e.g. `mask_util.save_mantid_mask(mask_vec, h5_name, two_theta, note=None)`);
they are not swept up retroactively. Keeping required
arguments positional past that threshold is what causes silent breakage later — a
caller relying on positional order has no warning when a new required argument is
inserted in the middle. Apply this when writing or extending a function that crosses
the threshold, not as a retroactive sweep of existing ones.

## Import placement

**All imports go at the top of the file.** A function-local (deferred)
import is allowed only when a top-level import would actually break — and
when it is, the reason goes in a comment on the import line or immediately
above it, naming the specific cycle or condition:

```python
def get_handle(self, name: str) -> "Handle":
    from other_module import Handle  # deferred: other_module imports this module

    ...
```

The three cases that genuinely justify deferring, and nothing else:

1. **A real circular import.** Name the other module and the direction, as
   above. "Might be circular" is not a justification — if you have not seen
   the `ImportError`, move it to the top.
2. **An optional dependency** not installed in every environment
   (`nltk`, `sounddevice`, ...). Say which environment lacks it.
3. **A genuinely expensive module-load** deferred off a hot import path, with
   a note on what it costs.

Do **not** defer an import to make `unittest.mock.patch` work. `from x import
f` at module level binds `f` at import time, so patching `x.f` does not reach
it — but the fix is to import the *module* at the top and call through it,
which stays patchable:

```python
import some_module  # top-level


def get_resource():
    return some_module.get_settings().resource_root
```

Why: a deferred import hides a module's real dependency graph, moves
`ImportError`s from startup to whenever that branch first runs, and costs a
`sys.modules` lookup on every call. PyRS already has instances of the shape this
rule targets — e.g. `pyrs/utilities/NXstress/_peaks.py`'s `from ._fit import
_PeakParameters, _BackgroundParameters`, `from ._definitions import
GROUP_NAME, UNDEFINED_PEAK_TAG` and `from pyrs.core.peak_profile_utility import
PeakShape, BackgroundFunction`, all inside method bodies with no comment
justifying the deferral. This rule applies to code you touch; existing
instances are not to be swept up unless asked.

## Private names do not cross module boundaries

A leading underscore means *this module owns this name*. Another module
importing it is a contradiction in terms, and it is always a signal about
placement rather than a shortcut worth taking:

```python
from some_package.internals import PublicThing, _private_helper  # do NOT do this
```

Two sanctioned fixes, and which one applies is a question about ownership:

1. **The name is genuinely part of that module's contract** and was
   underscore-prefixed by habit. Drop the underscore, so the module exports it
   deliberately.
2. **Two modules both need it and neither owns it.** Extract it to a leaf
   module they both import. This is the case whenever the import also closes an
   import cycle — the cycle is the evidence that the name is in the wrong place.

Three things this does **not** forbid:

- `import public_name as _alias`, which is the opposite move: taking a public
  name and keeping it out of the importing module's own surface. The rule is
  about the name at its *definition*, not at the call site.
- A **test** importing a private from the module it is testing —
  `tests/unit/pyrs/utilities/NXstress/test_peaks_read.py` reaching directly for
  `_peaks._Peaks` and `_fit._PeakParameters`. That is ordinary white-box
  testing, and the test breaking on a rename is the intended behaviour rather
  than collateral damage.
- A package's own orchestrating module assembling its public surface from its
  sibling private modules — `pyrs/utilities/NXstress/NXstress.py` importing
  `_Instrument`/`_Sample`/`_Fit`/`_Peaks` from `_instrument.py`/`_sample.py`/
  `_fit.py`/`_peaks.py`. These are the package's own internals talking to each
  other, not one package reaching into another's; see *Private modules are not
  the same thing as private names* below.

Why: a private name carries no stability contract, so renaming one inside its
owning module is an unannounced breaking change to code its author never looked
at — and the underscore is precisely what told that author they were free to
rename it. This rule applies to code you touch; existing violations are not to
be swept up unless asked.

### Private modules are not the same thing as private names

`pyrs/utilities/NXstress/` is built from underscore-prefixed *files* —
`_peaks.py`, `_fit.py`, `_definitions.py`, `_sample.py`, `_instrument.py`,
`_discriminator.py`, `_input_data.py`, ... — each an implementation detail of the
`NXstress` package as a whole, not of any one file inside it. That is a
**layout** choice, answered by *where a file sits*, and this rule does not
touch it: `NXstress.py` importing `_Peaks` from `_peaks.py` is the package
assembling its own public surface, the same as the test exception above.

The rule is violated when a module **outside** `pyrs/utilities/NXstress/`
imports one of these names directly — `pyrs.core` reaching for
`pyrs.utilities.NXstress._peaks._Peaks` instead of whatever `NXstress`
re-exports publicly. The leading underscore on a *module* still means
something; it just answers a narrower question — "is this file part of the
package's own internals" — than the leading underscore on a *name* crossing a
boundary the package did not draw.

## Named states are literals, not free text

**A value that names one of a fixed set of states is defined once, as a str-valued
`Enum` (or a `Literal` type), and every module that writes, reads or compares it
imports that one definition.** A state spelled out as a bare string at each use
(`"none"`, `"file"`, `"workspace"`, ...) lets a typo silently become a new state that
no reader handles, and leaves no single place that lists the states there are.

```python
class CalibrationSource(str, Enum):
    NONE = "none"  # nominal calibration: no calibration file applied
    FILE = "file"  # a calibration file
    WORKSPACE = "workspace"  # the workspace's own shift
```

- **Keep the stored strings as the members' values**, so files already written with
  those strings read back unchanged. A file holding a string that is not a member
  (e.g. a project file stamped `"unknown"`) is rejected, as the next point says.
- **A stored string that is not a member is rejected**, with an error naming the
  file and the value: never passed through as a new state, never turned into
  `None`, and never mapped to a catch-all member.
- This pairs with *`None` must name a value, not a gap*: a state is never
  expressed by borrowing `None`. A value that cannot be stated ("unknown",
  "mixed", "lost") is not a state at all: it is refused (e.g. a calibration that
  cannot be identified is rejected, #1141).

Why: the review of PR #1132 found the calibration's source written as free text in
several modules: `"none"`/`"file"`/`"workspace"` in `HidraSetup`, `"unknown"` in the
project-file writer, and a `record.get("source", "none")` default in `HidraSetup`'s reader.
Nothing ties these together, so a misspelling in any one of them is a new,
unhandled state rather than an error. This rule applies to code you touch; existing
instances are not to be swept up unless asked.

## 🧪 Testing Guidelines

Every test follows Arrange-Act-Assert:

```python
def test_example_function_filters_correctly():
    """Test that values below threshold are removed."""
    # Arrange
    input_data = [0.1, 0.5, 0.7, 1.0]
    threshold = 0.6

    # Act
    result = example_function(input_data, threshold=threshold)

    # Assert
    assert len(result) == 2
    assert all(x >= threshold for x in result)
```

Always cover:
1. **Normal cases** — typical inputs.
2. **Edge cases** — empty inputs, single items, max values.
3. **Error cases** — invalid inputs, type errors.
4. **Integration** — components working together.

Naming: `test_<function_name>_<scenario>_<expected_outcome>`, **kept under 60
characters**. Omit any part the module or class name already carries — a test in
`test_config.py` needs no `config_` prefix — and drop `_returns_*` tails when the
scenario already implies the outcome. Keep `_raises` for error cases; the
exception type is usually redundant with it.

```
# too long — the module already says `summary_generator_stress`, and
# `_raises_runtime_error` repeats what `pytest.raises` in the body states
test_summary_generator_stress_init_strain_without_filenames_raises_runtime_error

# good
test_init_strain_without_filenames_raises
```

### Test tiers, markers, and locations

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
  until the timeout below fires — see [docs/ground_truths.md](docs/ground_truths.md).
- **Every test has a 300-second timeout** (`timeout`/`timeout_method` in
  `pyproject.toml`, via `pytest-timeout`). A hang is a failure, not an infinite
  wait. `timeout_method = "thread"` is deliberate: a test blocked inside Qt's C++
  event loop never returns to the interpreter, so the default `signal` method
  cannot interrupt it — verified. The watchdog dumps every thread's stack, which
  names the blocking line. Override per-test with `@pytest.mark.timeout(N)` for a
  genuinely long-running case rather than raising the global limit.

## 🔍 Code Review Process

Trigger a review when:
- Implementing a new feature (3+ functions).
- Completing a significant refactor.
- Marking major work as complete.
- The user asks.

When delegating to a sub-agent via the `Agent` tool, hand over a self-contained brief: what changed, why, the files touched, and the specific concerns to weigh (style, tests, docs, bugs, perf, security). Sub-agents do not see your conversation — give them the context they need to make judgment calls.

## 📚 Documentation Standards

Module docstring:

```python
"""
Module for data preprocessing operations.

Provides functions for cleaning and normalizing experimental data from
the XYZ instrument. Typical workflow:

1. Load raw data with load_data().
2. Clean with remove_outliers().
3. Normalize with normalize_values().
4. Export with save_processed_data().
"""
```

Class docstring: describe purpose, key attributes, and a usage example. See [.github/copilot-instructions.md](.github/copilot-instructions.md) for the full template.

## 🚨 Common Scenarios

### Adding a feature
1. Read the relevant files in parallel.
2. Summarize current state and what needs to change.
3. Create a `TodoWrite` plan.
4. Implement step-by-step, marking todos completed as you go.
5. Run tests.
6. Offer a sub-agent review for non-trivial changes.

### Reporting/fixing a bug
1. Read the code; identify root cause (don't paper over symptoms).
2. State the location, cause, and impact in one short paragraph.
3. Plan: fix → regression test → scan for similar issues.
4. Implement; run tests; confirm the regression test fails without the fix.

### Choosing an approach
For exploratory questions, respond in 2–3 sentences with a recommendation and the main tradeoff. Present it as something the user can redirect — don't implement until they agree.

## 🎓 Educational Approach

Users may be scientists new to software engineering. Always:
- Explain **why**, not just what.
- Use clear, non-jargon language where possible.
- Provide context for decisions.
- Offer to explain concepts if the user looks new to them.
- Encourage good practices gently — don't lecture.

## ⚡ Efficiency Guidelines

1. **Parallelize independent tool calls** — multiple `Read`/`Grep`/`Bash` calls go in one message when there are no dependencies.
2. **Use `Agent` with `subagent_type="Explore"`** for broad codebase exploration spanning more than ~3 queries; for narrower lookups, just use `Grep`/`Bash` directly.
3. **Prefer dedicated tools** — `Read`/`Edit`/`Write` over `cat`/`sed`/`echo` via `Bash`.
4. **Don't re-read files you just edited** — `Edit` errors if it failed.
5. **Give brief progress updates** at key moments — finding something, changing direction, hitting a blocker.

---

**Remember:** every interaction should leave the user with working, tested, documented code — and a clear understanding of what changed and why.
