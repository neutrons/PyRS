"""Progress callbacks: the no-op defaults and the per-part progress of a job."""

from pyrs.pole_figures.core.types import PROGRESS_EVERY, no_count_progress, no_progress, part_progress


def test_no_progress_ignores_reports() -> None:
    # Arrange
    report = (0.5, "half way")

    # Act
    result = no_progress(*report)  # type: ignore[func-returns-value]  # checks the no-op returns nothing

    # Assert
    assert result is None


def test_no_count_progress_ignores_counts() -> None:
    # Arrange
    done = 7

    # Act
    result = no_count_progress(done)  # type: ignore[func-returns-value]  # checks the no-op returns nothing

    # Assert
    assert result is None


def test_part_progress_reports_fraction_of_whole_job() -> None:
    # Arrange
    seen: list[tuple[float, str]] = []
    tick = part_progress(lambda f, m: seen.append((f, m)), part=1, parts=2, n=20, label="{311}", unit="points")

    # Act
    tick(PROGRESS_EVERY)

    # Assert
    assert seen == [(0.75, "{311}: 10 of 20 points")]


def test_part_progress_skips_between_steps() -> None:
    # Arrange
    seen: list[float] = []
    tick = part_progress(lambda f, _m: seen.append(f), part=0, parts=1, n=25, label="run", unit="points")

    # Act
    for done in range(1, 26):
        tick(done)

    # Assert
    assert seen == [10 / 25, 20 / 25, 1.0]
