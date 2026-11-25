"""`PeakFittingModel.save_peak_range_json`: the file name it writes and the d0 it stores (#1147).

`PeakFittingModel` is a `QObject` with signals, which needs no `QApplication` and no display,
so these run in the unit tier.
"""

import json
from unittest.mock import MagicMock

import pytest

from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel


@pytest.fixture
def model():
    return PeakFittingModel(peak_fit_core=MagicMock())


@pytest.mark.parametrize(
    "typed_name, written_name",
    [("peak_range", "peak_range.json"), ("peak_range.txt", "peak_range.txt")],
    ids=["no-extension", "other-extension"],
)
def test_save_file_extension(model, tmp_path, typed_name, written_name):
    """``.json`` is appended only when the typed name has no extension."""
    # Arrange
    filename = tmp_path / typed_name

    # Act
    model.save_peak_range_json(str(filename), [[78.0, 85.0]], ["Peak0"], [1.1])

    # Assert
    assert [path.name for path in tmp_path.iterdir()] == [written_name]
    assert json.loads((tmp_path / written_name).read_text())["0"]["d0"] == 1.1


def test_save_d0_list_of_other_length_stores_placeholder(model, tmp_path):
    """A d0 list that is not one value per peak is stored as 1.0, as the reader would read it."""
    # Arrange
    filename = tmp_path / "peak_range.json"

    # Act
    model.save_peak_range_json(str(filename), [[78.0, 85.0], [90.0, 95.0]], ["A", "B"], [])

    # Assert
    saved = json.loads(filename.read_text())
    assert [entry["d0"] for entry in saved.values()] == [1.0, 1.0]
