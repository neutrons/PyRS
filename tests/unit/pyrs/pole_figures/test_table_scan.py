"""Table-scan building blocks: count times, measurement order, durations and spreadsheet-safe text."""

import numpy as np
import pytest

from pyrs.pole_figures.core.table_scan import count_time, format_duration, measurement_order, safe_text


@pytest.mark.parametrize(
    ("correction", "kwargs", "want"),
    [
        (1.0, {"base_s": 10}, 10),
        (2.46, {"base_s": 10}, 25),
        (0.01, {"base_s": 10}, 1),  # at least 1 s
        (2.46, {"base_s": 10, "scaled": False}, 10),
        (2.46, {"base_s": 10, "max_s": 20}, 20),
        (2.46, {"base_s": 30, "scaled": False, "max_s": 20}, 20),
    ],
    ids=["unit_correction", "scaled", "at_least_1s", "fixed", "capped", "fixed_capped"],
)
def test_count_time(correction: float, kwargs: dict, want: float) -> None:
    # Arrange
    options = kwargs

    # Act
    t = count_time(correction, **options)

    # Assert
    assert t == want


CHI = np.array([0, 0, 0, 10, 10, 10, 20, 20, 90.0])
PHI = np.array([0, 5, 10, 0, 5, 10, 0, 10, 0.0])


def test_measurement_order_serpentine() -> None:
    # Arrange
    chi, phi = CHI, PHI

    # Act
    order = measurement_order(chi, phi)

    # Assert
    assert list(zip(chi[order], phi[order], strict=True)) == [
        (0, 0), (0, 5), (0, 10), (10, 10), (10, 5), (10, 0), (20, 0), (20, 10), (90, 0)
    ]  # fmt: skip


def test_measurement_order_plain_keeps_phi_rising() -> None:
    # Arrange
    chi, phi = CHI, PHI

    # Act
    order = measurement_order(chi, phi, serpentine=False)

    # Assert
    assert list(phi[order]) == [0, 5, 10, 0, 5, 10, 0, 10, 0]


@pytest.mark.parametrize(("seconds", "want"), [(45, "45 s"), (600, "10 min"), (3 * 3600 + 7 * 60, "3 h 07 min")])
def test_format_duration(seconds: float, want: str) -> None:
    # Arrange
    duration = seconds

    # Act
    text = format_duration(duration)

    # Assert
    assert text == want


def test_safe_text_quotes_formula() -> None:
    # Arrange
    text = "=SUM(A1)"

    # Act
    safe = safe_text(text)

    # Assert
    assert safe == "'=SUM(A1)"
