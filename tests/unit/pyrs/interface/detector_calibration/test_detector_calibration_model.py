"""Tests for `pyrs.interface.detector_calibration.detector_calibration_model`.

Unit tier: the model is a `QObject`, which needs no display or `QApplication`.
The calibration GUI model applies the production detector mask by default, or
none if it is missing; `RecordingFitCalibration` stands in for `FitCalibration`
to record how it was constructed. Elsewhere a plain namespace stands in for the
calibration object the model delegates to.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.interface.detector_calibration import detector_calibration_model as model_module
from pyrs.interface.detector_calibration.detector_calibration_model import DetectorCalibrationModel


class RecordingFitCalibration:
    """Stands in for FitCalibration and records how it was constructed."""

    calls: list = []

    def __init__(self, **kwargs):
        RecordingFitCalibration.calls.append(kwargs)


@pytest.fixture
def recorder(monkeypatch):
    RecordingFitCalibration.calls = []
    monkeypatch.setattr(model_module, "FitCalibration", RecordingFitCalibration)
    return RecordingFitCalibration


def test_default_mask_file_found(monkeypatch, tmp_path):
    # Arrange
    mask = tmp_path / "HB2B_Mask_Latest.xml"
    mask.write_text("<detector-masking/>")
    monkeypatch.setattr(model_module, "DEFAULT_MASK_FILE", str(mask))

    # Act / Assert
    assert model_module.default_mask_file() == str(mask)


def test_default_mask_file_missing_is_none(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.setattr(model_module, "DEFAULT_MASK_FILE", str(tmp_path / "absent.xml"))

    # Act / Assert
    assert model_module.default_mask_file() is None


def test_load_and_rebin_keep_the_default_mask(monkeypatch, tmp_path, recorder):
    # Arrange
    mask = tmp_path / "HB2B_Mask_Latest.xml"
    mask.write_text("<detector-masking/>")
    monkeypatch.setattr(model_module, "DEFAULT_MASK_FILE", str(mask))
    model = DetectorCalibrationModel(None)

    # Act
    model._init_calibration("HB2B_3510.nxs.h5", tth=512, eta=3.0)
    model.set_reduction_param(256, 5.0)

    # Assert - both the first calibration object and the re-binned one carry the mask
    assert model.mask_file == str(mask)
    assert [call["mask_file"] for call in recorder.calls] == [str(mask), str(mask)]
    assert recorder.calls[1]["bins"] == 256 and recorder.calls[1]["eta_slice"] == 5.0


def test_missing_mask_falls_back_to_none(monkeypatch, tmp_path, recorder):
    # Arrange
    monkeypatch.setattr(model_module, "DEFAULT_MASK_FILE", str(tmp_path / "absent.xml"))
    model = DetectorCalibrationModel(None)

    # Act
    model._init_calibration("HB2B_3510.nxs.h5")

    # Assert
    assert model.mask_file is None
    assert recorder.calls[0]["mask_file"] is None


@pytest.mark.xfail(
    reason="DD-2: `fitted_ws` is indexed by position, and looked up as `sub_run - 1`",
    raises=IndexError,
    strict=True,
)
def test_get_fitted_diffraction_data_sub_runs_not_starting_at_one() -> None:
    """`fit_peaks` fills `fitted_ws` by position over `sub_runs`. Recorded under
    DD-2 in `plans/PR_review/deferred-defects.md`."""
    # Arrange
    model = DetectorCalibrationModel(None)
    model._calibration_obj = SimpleNamespace(fitted_ws=["five", "six", "seven"], sub_runs=np.array([5, 6, 7]))

    # Act / Assert
    assert model.get_fitted_diffraction_data(6) == "six"
