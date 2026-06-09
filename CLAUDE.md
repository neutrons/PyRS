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

Naming: `test_<function_name>_<scenario>_<expected_outcome>`.

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
