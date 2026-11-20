"""NXstress written from real reduced data describes the calibration reduction applied (#1099)

HB2B_3510 is reduced with its reference ring calibration; the reduced project file records that calibration
(#1053). Loaded back and written as NXstress, the entry's detector chain carries the calibration's values and
errors, the run's arm angles and arm length, and places the pixels where reduction does.
"""

import numpy as np
import pytest
from nexusformat.nexus import nxload

from pyrs.core.detector_model import DetectorModel
from pyrs.core.instrument_geometry import HidraSetup
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
from pyrs.core.workspaces import HidraWorkspace
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities.calibration_file_io import read_calibration_json_file
from pyrs.utilities.NXstress.NXstress import NXstress
from tests.unit.pyrs.utilities.NXstress.test_instrument import _chain_matrix

pytestmark = pytest.mark.integration

RUN = "tests/data/calibration_tests/HB2B_3510.nxs.h5"
MASK = "tests/data/HB2B_Mask_12-18-19.xml"
REFERENCE = "tests/data/calibration_tests/HB2B_3510_ring_reference.json"
NAMES = ("shift_x", "shift_y", "shift_z", "rotation_x", "rotation_y", "rotation_z", "two_theta_0")


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """HB2B_3510 sub-runs 1-2 reduced with the reference calibration, loaded back and written as NXstress"""
    folder = tmp_path_factory.mktemp("nxstress_calibration")
    reducer = ReductionApp()
    reducer.load_hidra_workspace(NeXusConvertingApp(RUN, MASK).convert())
    reducer.reduce_data(sub_runs=[1, 2], instrument_file=None, calibration_file=REFERENCE, mask=None, num_bins=256)
    project_path = folder / "reduced.h5"
    reducer.save_diffraction_data(str(project_path))
    # the detector geometry, as a project file converted by NeXusConvertingApp.save has it: a reduction's own save
    # does not write it (DD-34), and NXstress needs it
    with HidraProjectFile(str(project_path), HidraProjectFileMode.READWRITE) as project:
        project.write_instrument_geometry(HidraSetup(reducer._hydra_ws.get_instrument_setup()))
    workspace = HidraWorkspace("reduced")
    with HidraProjectFile(str(project_path), HidraProjectFileMode.READONLY) as project:
        workspace.load_hidra_project(project, load_raw_counts=False, load_reduced_diffraction=True)
    nexus_path = folder / "reduced.nxs"
    with NXstress(nexus_path, "w") as nx:
        nx.write([workspace], [[]])
    detector = nxload(str(nexus_path), "r")["entry"]["instrument"]["DETECTOR"]
    return workspace, detector


def test_chain_holds_the_applied_calibration(written):
    # Arrange
    _, detector = written
    trans = detector["transformations"]
    shift, error = read_calibration_json_file(REFERENCE)[:2]
    values = dict(
        zip(
            NAMES,
            (
                shift.center_shift_x,
                shift.center_shift_y,
                shift.center_shift_z,
                shift.rotation_x,
                shift.rotation_y,
                shift.rotation_z,
                shift.two_theta_0,
            ),
        )
    )
    errors = dict(
        zip(
            NAMES,
            (
                error.center_shift_x,
                error.center_shift_y,
                error.center_shift_z,
                error.rotation_x,
                error.rotation_y,
                error.rotation_z,
                error.two_theta_0,
            ),
        )
    )

    # Act
    written_values = {name: float(trans[name].nxdata) for name in NAMES}
    written_errors = {name: float(trans[f"{name}_errors"].nxdata) for name in NAMES}

    # Assert
    assert bool(trans.attrs["calibrated"])
    assert written_values == pytest.approx(values)
    assert written_errors == pytest.approx(errors)


def test_arm_angles_and_length_are_the_runs(written):
    # Arrange
    workspace, detector = written
    trans = detector["transformations"]
    sub_runs = list(workspace.get_sub_runs())
    l2 = workspace.get_l2(sub_runs[0])

    # Act
    arms = np.asarray(trans["arm_rotation"].nxdata)

    # Assert - one arm angle per sub-run, -(2theta log); the arm length reduction used
    assert arms.shape == (len(sub_runs),)
    np.testing.assert_array_equal(arms, [-workspace.get_detector_2theta(sub_run) for sub_run in sub_runs])
    expected_length = workspace.get_instrument_setup().arm_length if l2 is None else l2
    assert float(trans["arm_length"].nxdata) == pytest.approx(expected_length)


def test_chain_places_real_pixels_as_reduction_does(written):
    """Applied without DetectorModel, the written chain puts real HB2B pixels where reduction does"""
    # Arrange
    workspace, detector = written
    trans = detector["transformations"]
    model = DetectorModel(workspace.get_instrument_setup())
    shift = read_calibration_json_file(REFERENCE)[0]
    index_0, index_1 = np.meshgrid(np.arange(0, 1024, 128), np.arange(0, 1024, 128), indexing="ij")
    local = np.concatenate([model.local_positions(index_0, index_1), np.ones(index_0.shape + (1,))], axis=-1)
    entry = str(detector["depends_on"]).split("/")[-1]
    arm_length = float(trans["arm_length"].nxdata)

    for scan_point, sub_run in enumerate(workspace.get_sub_runs()):
        # Act
        arm = float(trans["arm_rotation"].nxdata[scan_point])
        placed = (local @ _chain_matrix(trans, entry, arm).T)[..., :3]

        # Assert
        expected = model.lab_positions(index_0, index_1, -workspace.get_detector_2theta(sub_run), arm_length, shift)
        np.testing.assert_allclose(placed, expected, rtol=0, atol=1e-12)
