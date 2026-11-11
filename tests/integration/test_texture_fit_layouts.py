"""Integration tests for `TextureFittingCrtl.fit_peaks` on the two reduced-data
layouts it supports: the default pattern alone, and out-of-plane (eta) patterns
alone.

Each test loads a real project file from `tests/data` and fits it with Mantid,
so these are `integration` (see CLAUDE.md's marker definitions). The peak-range
table is a stand-in exposing only `rowCount()` and `item(row, col).text()`, so no
widget -- and no display -- is needed.
"""

from types import SimpleNamespace

import pytest

from pyrs.interface.texture_fitting.texture_fitting_crtl import TextureFittingCrtl
from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel

pytestmark = pytest.mark.integration


class _RangeTable:
    """The peak-range table's interface as `fit_peaks` reads it: x_left, x_right, label."""

    def __init__(self, rows: list) -> None:
        self._rows = rows

    def rowCount(self) -> int:
        return len(self._rows)

    def item(self, row: int, col: int) -> SimpleNamespace:
        value = self._rows[row][col]
        return SimpleNamespace(text=lambda: str(value))


@pytest.mark.parametrize(
    "project_file, expected_keys",
    [
        # Default pattern only. Fitting this raised `TypeError` from 2026-03
        # (`4cab5d91`), when the `_var` datasets stopped being listed as masks
        # and the old "two keys" test for this layout stopped matching it.
        ("tests/data/HB2B_1599.h5", [""]),
        ("tests/data/HB2B_1599_texture.h5", ["eta_-5.0", "eta_0.0", "eta_5.0"]),
    ],
    ids=["default-only", "eta-only"],
)
def test_fit_peaks_supported_layout(project_file: str, expected_keys: list) -> None:
    # Arrange
    model = TextureFittingModel(None)
    model.load_hidra_project_file(project_file)
    crtl = TextureFittingCrtl(model)

    # Act
    fits = crtl.fit_peaks(_RangeTable([(62.864, 66.9115, "Fe 110")]), "PseudoVoigt", "Linear")

    # Assert
    assert sorted(fits) == expected_keys
    assert all(len(fit.peakcollections) == 1 for fit in fits.values())
