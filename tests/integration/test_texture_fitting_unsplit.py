"""Texture fitting of a file not split into η-wedges

Since variance entries stopped being listed as reduction masks (4cab5d91), an unsplit file lists
its masks as ``[None]``; the controller only recognised the old ``[main, main_var]`` pair and
crashed with ``TypeError`` evaluating ``"_var" not in None``.
"""

import numpy as np
import pytest

from pyrs.interface.texture_fitting.texture_fitting_crtl import TextureFittingCrtl
from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel

pytestmark = pytest.mark.integration


class _Item:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _FitRangeTable:
    """The part of the fit-range QTableWidget the controller reads: min, max and label per row."""

    def __init__(self, rows):
        self._rows = rows

    def rowCount(self):
        return len(self._rows)

    def item(self, row, column):
        return _Item(str(self._rows[row][column]))


def _fitted(project_file):
    controller = TextureFittingCrtl(TextureFittingModel(peak_fit_core=None))
    controller.load_projectfile(project_file)
    table = _FitRangeTable([(62.864, 66.9115, "peak0")])
    return controller.fit_peaks(table, "PseudoVoigt", "Linear")


def test_fit_unsplit_file_once():
    """An unsplit file is fitted once, keyed "", instead of raising."""
    # Act
    fits = _fitted("tests/data/HB2B_1599.h5")

    # Assert
    assert list(fits) == [""]
    values, _ = fits[""].peakcollections[0].get_effective_params()
    assert np.all(np.isfinite(values["Center"]))


def test_fit_split_file_per_wedge_unchanged():
    """A file split into η-wedges is still fitted once per wedge."""
    # Act
    fits = _fitted("tests/data/HB2B_1599_texture.h5")

    # Assert
    assert sorted(fits) == ["eta_-5.0", "eta_0.0", "eta_5.0"]
