# Claude Code Instructions

This file configures how Claude Code should assist with development in this repository. This project is designed for scientists who may be new to software engineering, so all interactions should be educational, clear, and follow best practices. These instructions mirror [.github/copilot-instructions.md](.github/copilot-instructions.md) — keep both in sync when you change one.

## ⚖️ Constitution

These two rules override every other instruction in this file, including any plan already approved. When a rule and a task conflict, stop and ask.

### 1. No change to scientific production code unless a human approves it

*Scientific production code* is any code whose output feeds a scientific result: data reduction and
instrument geometry (`pyrs/core/`), peak fitting (`pyrs/peaks/`), strain/stress and texture
calculations, calibration and project-file I/O (`pyrs/utilities/`, `pyrs/projectfile/`), the
computations behind the UI, and the established calibration (`FitCalibration`).

- Before changing it, **propose the change and wait for explicit approval of that specific change** — quantify its effect on results (e.g. "pixel 2θ moves by ≤ 4.5e-7°") so the decision is informed.
- **Approval is per change.** It does not carry over to related edits, later refactors, or "behaviour-preserving" clean-ups, and approving a plan is not approving each production edit in it.
- Work under development (e.g. a new module such as `pyrs/calibration/pixel_pos_calibration.py`) may be changed as part of its approved task; anything it needs *from* production code is a separate proposal.
- Problems found in production code are reported (issue or note), not fixed in passing.

### 2. No invented correction parameters

- **Predictions of physical quantities use the physical law, unmodified** — e.g. an expected peak position is the Bragg angle `2·asin(λ / 2d)`. Do not add offsets, scales or other terms to a prediction to make it match data.
- **Every fitted or adjustable parameter must be a physical quantity of the instrument or sample model** (detector shifts and rotations, 2θ₀, wavelength, lattice constants, …), and introducing a new one needs human approval.
- **Deviations from the model are corrected through the physical model and determined by the refinement** — e.g. detector geometry is found by refining the geometry, and better starting values come from that refinement (or a previous calibration), not from an empirical stand-in.
- Do not invent empirical "effective" quantities (fudge factors, effective offsets, stretch factors) — even for intermediate steps such as placing fit windows — without explicit approval. If the model cannot describe the data, report it; do not paper over it.

## 🎯 Core Principles

1. **Always assess before acting** — Understand the current state before proposing changes.
2. **Always provide itemized plans** — Break work into clear, testable steps tracked with `TodoWrite`.
3. **Focus on progress tracking** — The user should always know what's happening and what's next.
4. **Test incrementally** — Each step should be testable on its own.
5. **Review after major changes** — Delegate review to a sub-agent via the `Agent` tool.
6. **Ground truths** — ALWAYS record key findings in [docs/ground_truths.md](docs/ground_truths.md) for future reference. Update the file with new findings, and link to it from related documentation and code comments.
7. **Deferred defects** — A defect in the PyRS codebase that you find but do not fix gets one numbered, dated line in [docs/deferred_defects.md](docs/deferred_defects.md), and is listed in the PR's commit message.

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

### Step 5: Code Review Process (For Major Changes)
After a significant feature or refactor, delegate review using the `Agent` tool. Prefer the project's purpose-built reviewers in [.claude/agents/](.claude/agents/):
- [security-reviewer](.claude/agents/security-reviewer.md) — security-focused review.
- [design-reviewer](.claude/agents/design-reviewer.md) — architecture and design review.
- [test-reviewer](.claude/agents/test-reviewer.md) — test-coverage review.

Brief the sub-agent with what changed, why, the files touched, and what to focus on.

**After a review, bring the user the findings with your recommendation and wait for their choice before acting on them, unless a finding is an outright bug in the change itself.** Then update documentation.

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
- **Type checking:** mypy, run by a pre-commit hook as `pixi run mypy pyrs scripts tests`, the same command as CI's lint job. Run that command, not mypy on single files, before pushing.

### Agent Skills
- Follow the official spec: https://agentskills.io/specification.

## 📝 Code Quality Standards

Always include:
1. **Type hints** on parameters and return values.
2. **Docstrings** — Google-style (`Args:`, `Returns:`, `Raises:`, and `Example:` where it helps) for public functions and classes, also in modules whose older code uses NumPy-style.
3. **Specific exceptions** with clear messages that name the offending value (`f"... got {value}"`).
4. **Input validation** at boundaries.
5. **Comments only when the WHY is non-obvious** — never narrate what the code does.

Full examples (a function, a test, module and class docstrings): [docs/developer/source/code_style.rst](docs/developer/source/code_style.rst).

## 🧪 Testing Guidelines

Every test follows Arrange-Act-Assert, with `# Arrange`, `# Act` and `# Assert` comments.

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

Put every new test in the tier matching what it touches (unit, ring calibration, integration, GUI)
and apply that tier's marker. **Read [tests/CLAUDE.md](tests/CLAUDE.md) before adding a test**: it
holds the tier table, marker rules, parallel-run rules and timeout policy, and loads when you work
under `tests/`.

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

Use the reviewers in [.claude/agents/](.claude/agents/) (see Step 5 above). When delegating to a sub-agent via the `Agent` tool, hand over a self-contained brief: what changed, why, the files touched, and the specific concerns to weigh (style, tests, docs, bugs, perf, security). Sub-agents do not see your conversation — give them the context they need to make judgment calls.

## 📚 Documentation Standards

Module and class docstrings: purpose, typical workflow, a usage example ([code_style.rst](docs/developer/source/code_style.rst)).

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

---

**Remember:** every interaction should leave the user with working, tested, documented code — and a clear understanding of what changed and why.
