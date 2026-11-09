"""Unit tests for `PeakFittingModel`.

`PeakFittingModel` is a `QObject`, but a `QObject` with signals needs no
`QApplication` and no display -- only widgets do -- so these run in the unit
tier. See docs/ground_truths.md.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel


@pytest.fixture
def model():
    peak_fit_model = PeakFittingModel(peak_fit_core=MagicMock())
    peak_fit_model.hidra_workspace = MagicMock()
    return peak_fit_model


def test_fit_diff_peaks_normal_case_returns_result(model, monkeypatch):
    """A successful fit returns the fit result and stores it on the model."""
    # Arrange
    fake_result = MagicMock()
    fake_engine = MagicMock()
    fake_engine.fit_multiple_peaks.return_value = fake_result
    monkeypatch.setattr(
        "pyrs.interface.peak_fitting.peak_fitting_model.PeakFitEngineFactory.getInstance",
        MagicMock(return_value=fake_engine),
    )
    emitted = []
    model.failureMsg.connect(lambda *args: emitted.append(args))

    # Act
    result = model.fit_diff_peaks(["peak0"], [1.0], [2.0], "PseudoVoigt", "Linear")

    # Assert
    assert result is fake_result
    assert model.fit_result is fake_result
    assert emitted == []


def test_fit_diff_peaks_error_case_emits_failure_and_returns_none(model, monkeypatch):
    """When the fit engine raises, failureMsg is emitted instead of the exception propagating."""
    # Arrange
    monkeypatch.setattr(
        "pyrs.interface.peak_fitting.peak_fitting_model.PeakFitEngineFactory.getInstance",
        MagicMock(side_effect=RuntimeError("fit did not converge")),
    )
    emitted = []
    model.failureMsg.connect(lambda *args: emitted.append(args))

    # Act
    result = model.fit_diff_peaks(["peak0"], [1.0], [2.0], "PseudoVoigt", "Linear")

    # Assert
    assert result is None
    assert model.fit_result is None
    assert len(emitted) == 1
    title, message, detail = emitted[0]
    assert "fit did not converge" in message
    assert "RuntimeError" in detail


def test_peak_range_json_round_trip_keeps_per_peak_d0(model, tmp_path):
    """Each peak's own d0 survives save -> load (#925: the whole d0 list was saved per peak)."""
    # Arrange
    filename = tmp_path / "peak_range.json"
    ranges = [[78.0, 85.0], [90.0, 95.0]]
    labels = ["Peak0", "Peak1"]
    d0 = [1.1, 1.2]

    # Act
    model.save_peak_range_json(str(filename), ranges, labels, d0)
    loaded_ranges, loaded_labels, loaded_d0 = model.load_peak_range_json(str(filename))

    # Assert
    saved = json.loads(filename.read_text())
    assert [entry["d0"] for entry in saved.values()] == d0
    assert loaded_ranges == ranges
    assert loaded_labels == labels
    assert loaded_d0 == d0


@pytest.mark.parametrize(
    "typed_name, written_name",
    [("peak_range", "peak_range.json"), ("peak_range.txt", "peak_range.txt")],
    ids=["no-extension", "other-extension"],
)
def test_save_peak_range_json_file_extension(model, tmp_path, typed_name, written_name):
    """``.json`` is appended only when the typed name has no extension (#925)."""
    # Arrange
    filename = tmp_path / typed_name

    # Act
    model.save_peak_range_json(str(filename), [[78.0, 85.0]], ["Peak0"], [1.1])

    # Assert
    assert [path.name for path in tmp_path.iterdir()] == [written_name]
    assert json.loads((tmp_path / written_name).read_text())["0"]["d0"] == 1.1


def test_peak_range_json_d0_saved_and_loaded_as_float(model, tmp_path):
    """The UI's int default d0 and table-parsed np.float64 values round-trip as floats."""
    # Arrange
    filename = tmp_path / "peak_range.json"

    # Act
    model.save_peak_range_json(str(filename), [[78.0, 85.0], [90.0, 95.0]], ["Peak0", "Peak1"], [1, np.float64(1.2)])
    _, _, loaded_d0 = model.load_peak_range_json(str(filename))

    # Assert
    assert [entry["d0"] for entry in json.loads(filename.read_text()).values()] == [1.0, 1.2]
    assert loaded_d0 == [1.0, 1.2]
    assert all(type(value) is float for value in loaded_d0)


@pytest.mark.parametrize(
    "legacy_d0, expected_d0",
    [([1.1, 1.2], [1.1, 1.2]), ([], [1.0, 1.0])],
    ids=["full-list", "empty-list"],
)
def test_load_peak_range_json_legacy_list_d0_per_peak(model, tmp_path, legacy_d0, expected_d0):
    """Files written before #925 stored the d0 list on every peak; load them as per-peak values."""
    # Arrange
    filename = tmp_path / "peak_range.json"
    legacy = {
        str(i): {"peak_range": peak_range, "peak_label": f"Peak{i}", "d0": legacy_d0}
        for i, peak_range in enumerate([[78.0, 85.0], [90.0, 95.0]])
    }
    filename.write_text(json.dumps(legacy))

    # Act
    _, _, loaded_d0 = model.load_peak_range_json(str(filename))

    # Assert
    assert loaded_d0 == expected_d0
