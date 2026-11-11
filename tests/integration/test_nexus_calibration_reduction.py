"""Reduction with a NeXus calibration file gives exactly what the same calibration as JSON gives (#1054)"""

import numpy as np
import pytest

from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
from pyrs.projectfile import HidraProjectFile, HidraProjectFileMode  # type: ignore
from pyrs.utilities.calibration_file_io import read_calibration_json_file
from pyrs.utilities.calibration_nexus import write_calibration_nexus

pytestmark = pytest.mark.integration

RUN = "tests/data/calibration_tests/HB2B_3510.nxs.h5"
MASK = "tests/data/HB2B_Mask_12-18-19.xml"
REFERENCE = "tests/data/calibration_tests/HB2B_3510_ring_reference.json"


@pytest.fixture(scope="module")
def workspace():
    return NeXusConvertingApp(RUN, MASK).convert()


@pytest.fixture(scope="module")
def nexus_reference(workspace, tmp_path_factory):
    """The reference calibration written as NeXus, with the run's detector geometry"""
    path = str(tmp_path_factory.mktemp("nexus") / "HB2B_3510_ring_reference.nxs")
    shift, error, wavelength, wavelength_error, status = read_calibration_json_file(REFERENCE)
    write_calibration_nexus(
        path, shift, error, float(wavelength), float(wavelength_error), int(status), workspace.get_instrument_setup()
    )
    return path


def _reduce(workspace, calibration_file, eta_step=None):
    reducer = ReductionApp()
    reducer.load_hidra_workspace(workspace)
    sub_runs = list(workspace.get_sub_runs())
    reducer.reduce_data(
        sub_runs=sub_runs,
        instrument_file=None,
        calibration_file=calibration_file,
        mask=None,
        num_bins=512,
        eta_step=eta_step,
    )
    return reducer, sub_runs


@pytest.mark.parametrize("eta_step", [None, 3.0], ids=["whole-detector", "eta-slices"])
def test_nexus_and_json_reduce_bit_identically(workspace, nexus_reference, eta_step):
    # Act
    json_reducer, sub_runs = _reduce(workspace, REFERENCE, eta_step)
    json_patterns = {s: [np.array(v) for v in json_reducer.get_diffraction_data(s)] for s in sub_runs}
    nexus_reducer, _ = _reduce(workspace, nexus_reference, eta_step)

    # Assert - every sub-run's 2theta, intensity and error, exactly
    for sub_run in sub_runs:
        for json_values, nexus_values in zip(json_patterns[sub_run], nexus_reducer.get_diffraction_data(sub_run)):
            np.testing.assert_array_equal(np.asarray(nexus_values), json_values)
    assert nexus_reducer._instrument.calibration_status == json_reducer._instrument.calibration_status


def test_reduced_file_records_the_nexus_calibration(workspace, nexus_reference, tmp_path):
    """The project file names the NeXus file actually read, with its SHA-256 (#1053)"""
    # Arrange
    reducer, _ = _reduce(workspace, nexus_reference)
    path = tmp_path / "reduced.h5"

    # Act
    reducer.save_diffraction_data(str(path))
    project = HidraProjectFile(str(path), HidraProjectFileMode.READONLY)
    record = project.read_reduction_provenance()
    project.close()

    # Assert
    assert record["source"] == "file" and record["calibration_file"] == nexus_reference
    assert len(record["calibration_file_sha256"]) == 64
    assert record["calibration"]["TTH_0"] == read_calibration_json_file(REFERENCE)[0].two_theta_0
