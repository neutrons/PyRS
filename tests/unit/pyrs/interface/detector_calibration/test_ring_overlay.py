"""Predicted-ring overlay of the calibration GUI's 2D view (#1039): model and controller, no widgets

The rings are the Bragg angles of each sub-run's powder at the chosen geometry's wavelength, mapped
onto the detector with that geometry. Drawing is display only; the Qt view is replaced by a stand-in.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrs.calibration.powder_standards import HB2B_STANDARDS
from pyrs.core.detector_model import DetectorModel
from pyrs.core.instrument_geometry import DENEXDetectorShift
from pyrs.interface.detector_calibration.detector_calibration_crtl import DetectorCalibrationCrtl
from pyrs.interface.detector_calibration.detector_calibration_model import (
    RING_GEOMETRY_CURRENT,
    RING_GEOMETRY_FILE,
    RING_GEOMETRY_NOMINAL,
    DetectorCalibrationModel,
)
from pyrs.utilities.calibration_file_io import write_calibration_to_json
from tests.util.instrument_helpers import default_hb2b_geometry
from tests.util.ring_calibration_helpers import synthetic_calibration_workspace

NI, FE, _MO = HB2B_STANDARDS
TRUE_PARAMS = np.array([0.004, -0.003, 0.012, 0.3, -0.5, 1.2, 0.656, 1.5308])
MONO_WAVELENGTH = 1.540
ARMS, POWDERS = (76.0, 93.0), (NI, NI)  # sub-run 1: Ni 220; sub-run 2: Ni 311 and 222


@pytest.fixture(scope="module")
def workspace():
    ws = synthetic_calibration_workspace(default_hb2b_geometry(), TRUE_PARAMS, ARMS, POWDERS)
    ws.set_wavelength(MONO_WAVELENGTH, False)
    return ws


@pytest.fixture
def model(workspace):
    """The GUI model over a loaded run whose current calibration is the true geometry"""
    gui_model = DetectorCalibrationModel(None)
    gui_model._calibration_obj = SimpleNamespace(_hidra_ws=workspace, calibration_array=TRUE_PARAMS.copy())
    gui_model._nexus_file = "synthetic"
    return gui_model


def _bragg(standard, wavelength, hkl2):
    return 2 * np.degrees(np.arcsin(wavelength / (2 * standard.lattice_a / np.sqrt(hkl2))))


def test_nominal_rings_use_the_mono_wavelength(model, workspace):
    # Act
    two_theta, rings, mask = model.ring_overlay(2, RING_GEOMETRY_NOMINAL)

    # Assert - nominal geometry, mono wavelength, the reflections on the detector
    expected = DetectorModel(default_hb2b_geometry()).two_theta(
        *np.meshgrid(np.arange(1024), np.arange(1024), indexing="ij"), -93.0, None, None
    )
    np.testing.assert_array_equal(two_theta, expected)
    assert [r.label for r in rings] == ["Ni 311", "Ni 222"]
    assert [r.two_theta for r in rings] == pytest.approx([_bragg(NI, MONO_WAVELENGTH, n) for n in (11, 12)])


def test_current_rings_use_its_geometry_and_wavelength(model):
    # Act
    two_theta, rings, _ = model.ring_overlay(1, RING_GEOMETRY_CURRENT)

    # Assert
    shift = DENEXDetectorShift(*TRUE_PARAMS[:7])
    np.testing.assert_array_equal(
        two_theta[500], DetectorModel(default_hb2b_geometry()).two_theta(500, np.arange(1024), -76.0, None, shift)
    )
    assert [r.label for r in rings] == ["Ni 220"]
    assert rings[0].two_theta == pytest.approx(_bragg(NI, TRUE_PARAMS[7], 8))


def test_file_rings_use_the_file(model, tmp_path):
    # Arrange - a calibration file with another wavelength
    path = str(tmp_path / "cal.json")
    write_calibration_to_json(DENEXDetectorShift(*TRUE_PARAMS[:7]), DENEXDetectorShift(*[0.0] * 7), 1.52, 0.0, 1, path)

    # Act
    _, rings, _ = model.ring_overlay(1, RING_GEOMETRY_FILE, path)

    # Assert
    assert rings[0].two_theta == pytest.approx(_bragg(NI, 1.52, 8))


def test_file_geometry_without_file_raises(model):
    with pytest.raises(ValueError, match="calibration file"):
        model.ring_overlay(1, RING_GEOMETRY_FILE)


def test_unknown_geometry_raises(model):
    with pytest.raises(ValueError, match="geometry must be one of"):
        model.ring_overlay(1, "Best guess")


def test_no_run_loaded_gives_no_overlay():
    assert DetectorCalibrationModel(None).ring_overlay(1, RING_GEOMETRY_NOMINAL) is None


def test_unknown_sy_gives_no_overlay(model, workspace):
    # Arrange - sub-run 2 moved to an sy with no powder
    workspace.set_sample_log("sy", np.array([1, 2]), np.array([NI.sy, 40.0]))

    # Act / Assert
    try:
        assert model.ring_overlay(2, RING_GEOMETRY_NOMINAL) is None
    finally:
        workspace.set_sample_log("sy", np.array([1, 2]), np.array([NI.sy, NI.sy]))


class _RecordingDetectorView:
    """Stands in for DetectorView: records what the controller draws."""

    def __init__(self):
        self.calls = []

    def plot_detector_view(self, counts, info):
        self.calls.append(("image", info))

    def set_ring_overlay(self, two_theta_map=None, rings=(), mask=None):
        self.calls.append(("rings", [r.label for r in rings]))

    def set_ring_points(self, accepted=None, rejected=None):
        self.calls.append(("points", accepted is not None, rejected is not None))


def test_controller_draws_rings_only_when_asked(model):
    # Arrange
    controller, view = DetectorCalibrationCrtl(model), _RecordingDetectorView()

    # Act
    controller.update_diff_view(view, 0, 2, [True, True], ring_geometry=RING_GEOMETRY_NOMINAL)
    controller.update_diff_view(view, 0, 2, [True, True])

    # Assert - the overlay follows the image; without a geometry it is cleared, not left behind
    assert view.calls == [
        ("image", (2, None)),
        ("rings", ["Ni 311", "Ni 222"]),
        ("points", False, False),
        ("image", (2, None)),
        ("rings", []),
        ("points", False, False),
    ]
