"""GUI-tier tests for `PeakFittingModel`.

No `QWidget` is constructed here, but these tests still need a display and so
carry `gui`: `PeakFittingModel` is a `QObject`, and the `qapp` fixture builds a
real `QApplication`, which aborts when no display or offscreen platform is
available.

Both markers are required -- `test-gui` selects `-m gui` while
`test-integration` selects `-m 'integration and not gui'`, so `gui` alone would
drop these from the integration tier.
"""

import json
from unittest.mock import MagicMock

import pytest

from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel

pytestmark = [pytest.mark.gui, pytest.mark.integration]


@pytest.fixture
def model(qapp):  # noqa: ARG001 (qapp needed for QObject/Signal machinery)
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
    assert loaded_ranges == ranges
    assert loaded_labels == labels
    assert loaded_d0 == d0


def test_save_peak_range_json_appends_missing_extension(model, tmp_path):
    """A file name typed without an extension is saved as ``.json`` (#925)."""
    # Arrange
    filename = tmp_path / "peak_range"

    # Act
    model.save_peak_range_json(str(filename), [[78.0, 85.0]], ["Peak0"], [1.1])

    # Assert
    assert not filename.exists()
    assert json.loads((tmp_path / "peak_range.json").read_text())["0"]["d0"] == 1.1


@pytest.mark.parametrize(
    "legacy_d0, expected_d0",
    [([1.1, 1.2], [1.1, 1.2]), ([], [1.0, 1.0])],
    ids=["full-list", "empty-list"],
)
def test_load_peak_range_json_legacy_list_d0(model, tmp_path, legacy_d0, expected_d0):
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
