"""The texture fitting model's fit-range files go through the shared checked reader and writer (#1147).

A bad file or table is reported through `failureMsg` and leaves the table and the disk as they
were; the model never raises on one.
"""

import json

import pytest
from qtpy.QtWidgets import QTableWidgetItem  # type:ignore

from pyrs.interface.texture_fitting.texture_fitting_model import TextureFittingModel


class _Table:
    """The few members of `QTableWidget` the model uses, without needing a display."""

    def __init__(self, rows=()):
        self.cells = {(r, c): QTableWidgetItem(str(v)) for r, row in enumerate(rows) for c, v in enumerate(row)}
        self._rows = len(rows)

    def rowCount(self):
        return self._rows

    def insertRow(self, _row):
        self._rows += 1

    def item(self, row, column):
        return self.cells.get((row, column))

    def setItem(self, row, column, item):
        self.cells[(row, column)] = item

    def texts(self):
        return [[self.cells[(r, c)].text() for c in range(4)] for r in range(self._rows)]


@pytest.fixture
def model():
    return TextureFittingModel(peak_fit_core=None)


@pytest.fixture
def failures(model):
    messages = []
    model.failureMsg.connect(lambda title, message, detail: messages.append((title, message)))
    return messages


def test_from_json_reads_peaks_into_table(model, failures, tmp_path):
    # Arrange
    filename = tmp_path / "p.json"
    filename.write_text(json.dumps({"0": {"peak_range": [78.0, 85.0], "peak_label": "Fe 211", "d0": []}}))
    table = _Table()

    # Act
    model.from_json(str(filename), table)

    # Assert
    assert table.texts() == [["78.0", "85.0", "Fe 211", "1.0"]]
    assert failures == []


@pytest.mark.parametrize("content", ['{"0": {"peak_range": [78, 85], "peak_label": "A", "d0": null}}', "{oops", "{}"])
def test_from_json_bad_file_reports_and_keeps_table(model, failures, tmp_path, content):
    # Arrange
    filename = tmp_path / "bad.json"
    filename.write_text(content)
    table = _Table([[70.0, 75.0, "Old", 1.0]])

    # Act
    model.from_json(str(filename), table)

    # Assert
    assert table.texts() == [["70.0", "75.0", "Old", "1.0"]]
    assert len(failures) == 1 and "bad.json" in failures[0][0]


def test_to_json_writes_a_file_from_json_reads_back(model, failures, tmp_path):
    # Arrange
    filename = tmp_path / "out.json"
    table = _Table([[78.0, 85.0, "Fe 211", 1.17]])

    # Act
    model.to_json(str(filename), table)
    reread = _Table()
    model.from_json(str(filename), reread)

    # Assert
    assert reread.texts() == [["78.0", "85.0", "Fe 211", "1.17"]]
    assert failures == []


@pytest.mark.parametrize(
    "row", [[85.0, 78.0, "A", 1.0], [78.0, 85.0, "A", 0.0], [78.0, 85.0, "", 1.0], ["x", 85.0, "A", 1.0]]
)
def test_to_json_bad_row_reports_and_writes_nothing(model, failures, tmp_path, row):
    # Arrange
    filename = tmp_path / "out.json"

    # Act
    model.to_json(str(filename), _Table([row]))

    # Assert
    assert not filename.exists()
    assert len(failures) == 1
