from unittest.mock import MagicMock

import pytest

from pyrs.interface.peak_fitting.peak_fitting_model import PeakFittingModel


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
