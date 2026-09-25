# Claude Code Instructions

This file configures how Claude Code should assist with development in this repository. This project is designed for scientists who may be new to software engineering, so all interactions should be educational, clear, and follow best practices.
<!-- cherry-pick 30231c18: removed trailing sentence "These instructions mirror [.github/copilot-instructions.md](.github/copilot-instructions.md) — keep both in sync when you change one." -->

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
<!-- cherry-pick 30231c18: reviewer links below changed from .github/agents/ to .claude/agents/ -->
After a significant feature or refactor, delegate review using the `Agent` tool. Prefer the project's purpose-built reviewers in [.claude/agents/](.claude/agents/):
- [security-reviewer](.claude/agents/security-reviewer.md) — security-focused review.
- [design-reviewer](.claude/agents/design-reviewer.md) — architecture and design review.
- [test-reviewer](.claude/agents/test-reviewer.md) — test-coverage review.

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
- **Type checking:** mypy.
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

<!-- cherry-pick 30231c18: BEGIN added block "Pytest markers" -->
**Pytest markers** — apply the correct marker(s) when a test is *written*,
not as a later audit pass. See `pyproject.toml`'s
`[tool.pytest.ini_options] markers` for the authoritative definitions.
- `@pytest.mark.integration` (or a module-level `pytestmark`) for a test
  that reads/writes real data (`tests/data`, `/HFIR` archive) **or**
  exercises a multi-component workflow.
- The dividing line for "multi-component": a synthetic in-memory round trip
  through `tmp_path` that stays inside one component's own public API
  (e.g. a library's own `write`/`read`) is unit, unmarked — it's testing
  that component's own correctness. The same round trip called *through* a
  second component (e.g. a GUI-model class driving that library, or a
  session-registry hookup between two separate classes) is `integration` —
  it's testing that the two integrate correctly, regardless of whether the
  underlying data is real or synthetic.
- Marker (not file location) is what actually selects the fast tier
  (`-m "not integration and not gui"`) — a correctly-marked test doesn't
  need to move directories to be classified correctly.
<!-- cherry-pick 30231c18: END added block "Pytest markers" -->

## 🔍 Code Review Process

Trigger a review when:
- Implementing a new feature (3+ functions).
- Completing a significant refactor.
- Marking major work as complete.
- The user asks.

When delegating to a sub-agent via the `Agent` tool, hand over a self-contained brief: what changed, why, the files touched, and the specific concerns to weigh (style, tests, docs, bugs, perf, security). Sub-agents do not see your conversation — give them the context they need to make judgment calls.

<!-- cherry-pick 30231c18: BEGIN added section "Auditing a Plan or Subspec" -->
## 🔬 Auditing a Plan or Subspec

This is **not** the Code Review Process above. That reviews code that was written;
this verifies a *plan document* before the code exists. A plan series lives in
`plans/<series>/` as a grounded `README.md` plus numbered PR-sized subspecs. The
rationale, the evidence, and the toolkit specification are in
[plans/audit-process/process.md](plans/audit-process/process.md).

### What a complete audit consists of

An audit is complete when **all seven axes** are covered with the stated evidence.

| # | Axis | Referent |
|---|---|---|
| **A1** | Doc vs self | Every prose claim against every code block, table and link in the same document; **every Markdown link resolved to an existing file, every anchor to an existing heading** |
| **A2** | Doc vs siblings | `README.md`'s `## Sub-specifications` table (the `Depends on` column) and `## 5. Files to be Modified`; each shared name traced to exactly one owning document |
| **A3** | Doc vs codebase | Every cited file, symbol and line range opened and read, not recalled |
| **A4** | Doc vs library | Every claim about third-party behaviour true *of the installed version* — proven by a committed probe |
| **A5** | Doc vs runtime contract | Every claim about how another module treats our output — proven by a probe through that module's real code path |
| **A6** | Doc vs earlier draft | `README.md` against `plans/<series>/archive/overview.md`; each divergence listed with rationale |
| **A7** | Freshness | Claims re-verified later than the most recent landed subspec the document depends on |

**Anything less is a partial audit and is recorded as partial — never reported as
"consistent".** A subspec asserting nothing third-party and nothing cross-module is
genuinely A4/A5-exempt, but write the exemption down: afterwards, "no probes were
needed" and "no probes were written" are indistinguishable.

### Anti-patterns — things that look like auditing and are not

- **Re-reading a claim and finding it plausible.** Plausibility is what a stale
  claim has in abundance; a dead relative link renders exactly like a live one. A
  claim is verified when checked *against its referent*, not when re-read.
- **Checking the grounded document against the earlier draft.** That confirms
  transcription, not correctness — and since the draft is *older*, agreeing with it
  is evidence of staleness.
- **Treating an exemption label as exemption from everything.** "Blocked",
  "tracked follow-up", "not scheduled in this plan's phases" and `open-questions/`
  are *scheduling* statements. Work that proceeds on an assumption is not exempt
  from A1.
- **Assessing "self-consistent" as "does not conflict with the codebase."** A
  document can match the code perfectly and contradict itself on the next page.
- **Counting audit passes.** Five passes over three axes is three axes covered.
- **Recording the correction in the log and not in the document.** A log preserves
  what was believed, which is right for a claim. A line number is a pointer, not a
  belief — log the finding *and* apply the number.

### Probe before asserting (A4/A5)

**A document may not be marked audited if it asserts third-party library behaviour,
or how another module treats its output, without a committed probe under
`plans/<series>/probes/` whose real output is pasted into the document at the
claim.** Run probes with `pixi run python plans/<series>/probes/<name>.py`.

A probe is **not** a `## Verification` section, and this is not an extension of that
convention. They share only an evidentiary standard: run it, paste the *real*
output, never the predicted output.

| | `## Verification` | Probe |
|---|---|---|
| Audience | the PR reviewer | the spec author / auditor |
| When | after implementation | before it, during the audit |
| Purpose | show the shipped thing works end to end | test whether a design assumption is true |
| If it fails | the PR is not finished | the **document** is wrong and the design must change |

**A probe never modifies `pyproject.toml` or `pixi.lock`.** If a probe needs a
package the environment lacks, **that absence is the finding** — record it, do not
install it. Name probes `a4_*.py` / `a5_*.py`, never `test_*`: there is no
`testpaths` setting, so a bare `pytest` at the repo root would collect them.

When a probe turns up something true of PyRS generally rather than of one document
— an installed library's real behaviour, an environment constraint — **it is a
ground truth and also belongs in [docs/ground_truths.md](docs/ground_truths.md)**,
in that file's `**Why this matters going forward:**` form, with the probe linked.
Findings about a *document* stay in that document (below); findings about *the code
or the environment* go to ground truths.

### Invariants belong in tests, not prose

An audit finding that can be expressed as a test must be, and the prose becomes a
comment on the test. A paragraph is re-read once per audit; a test is re-checked on
every commit. The idiom to copy is
[tests/unit/pyrs/utilities/NXstress/test_definitions.py](tests/unit/pyrs/utilities/NXstress/test_definitions.py),
which iterates `GROUP_NAME` rather than restating it, so adding a member extends
the guarantee automatically.

Be accurate about the starting point: **nothing in PyRS yet pins a third-party API
surface or scans source for a convention.** Both shapes are net-new here. Place any
new test by the tier rules in *Testing Guidelines* above.

**An audit flags; the implementing PR writes.** Writing the test during the audit
produces a test with no subject.

### "Audited" is not a terminal state

Auditing eliminates staleness and self-contradiction. It **cannot** eliminate false
assumptions about library or cross-module behaviour — only probing does. Budget a
document-correction pass per PR as normal cost, not as an audit failure.

Findings go in an **append-only `## Follow-up N` section of the document being
audited**, after its `## Verification` section, so the finding sits with the claim
it corrects. `N` increments and is never renumbered; an earlier Follow-up is never
edited. Findings do **not** go in `open-questions/NN-*.md` (stakeholder Q&A) or
`README.md`'s `## 4. Decisions Log` (decisions taken) — both are human-facing
records with different audiences. When a finding forces a design change, the
*decision* goes to the Decisions Log and the Follow-up holds the evidence.
<!-- cherry-pick 30231c18: END added section "Auditing a Plan or Subspec" -->

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

Class docstring: describe purpose, key attributes, and a usage example.
<!-- cherry-pick 30231c18: removed trailing sentence "See [.github/copilot-instructions.md](.github/copilot-instructions.md) for the full template." -->

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
