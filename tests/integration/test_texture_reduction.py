"""
Integration test for PyRS to calculate powder pattern from detector counts
"""

import numpy as np
import h5py
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
import pytest
import os

pytestmark = pytest.mark.integration

DATA_DIR = "tests/data/"


def parse_gold_file(file_name):
    """
    Parameters
    ----------
    file_name

    Returns
    -------
    ~dict or ~numpy.ndarray
        gold data in array or dictionary of arrays
    """
    # Init output
    data_dict = dict()

    # Parse file
    gold_file = h5py.File(file_name, "r")
    data_set_names = list(gold_file["reduced diffraction data"].keys())
    for name in data_set_names:
        if ("_var" in name) or ("2theta" in name) or ("main" in name):
            pass
        else:
            if isinstance(gold_file["reduced diffraction data"][name], h5py.Dataset):
                data_dict[name] = [
                    gold_file["reduced diffraction data"]["2theta"][()][0, :],
                    gold_file["reduced diffraction data"][name][()][0, :],
                ]
            else:
                pytest.skip("project file not supplied")

        # END-IF
    # END-FOR

    if len(data_dict) == 1 and data_dict.keys()[0] == "data":
        # only 1 array with default name
        return data_dict["data"]

    return data_dict


@pytest.mark.parametrize(
    "nexusfile, mask_file_name, gold_file",
    [
        (DATA_DIR + "HB2B_1118.nxs.h5", DATA_DIR + "HB2B_Mask_12-18-19.xml", DATA_DIR + "HB2B_1118_texture.h5"),
        (
            "/HFIR/HB2B/IPTS-22331/nexus/HB2B_1428.nxs.h5",
            DATA_DIR + "HB2B_Mask_12-18-19.xml",
            DATA_DIR + "HB2B_1428_texture.h5",
        ),
    ],
    ids=["HB2B_1118_Texture", "HB2B_1428_Texture"],
)
def test_texture_reduction(nexusfile, mask_file_name, gold_file):
    """Test the powder pattern calculator (service) with HB2B-specific reduction routine

    Parameters
    ----------
    project_file_name
    mask_file_name
    gold_file
    """
    if not os.path.exists("/HFIR/HB2B/shared"):
        pytest.skip("Unable to access HB2B archive")

    CALIBRATION_FILE = DATA_DIR + "HB2B_calib_latest.json"
    VANADIUM_FILE = "/HFIR/HB2B/IPTS-22731/nexus/HB2B_1115.nxs.h5"

    # load gold file
    gold_data_dict = parse_gold_file(gold_file)

    # Parse input file
    converter = NeXusConvertingApp(nexusfile, mask_file_name)
    hidra_ws = converter.convert()

    # Start reduction service
    reducer = ReductionApp()
    reducer.load_hidra_workspace(hidra_ws)

    # Reduce raw counts
    reducer.reduce_data(
        instrument_file=None,
        calibration_file=CALIBRATION_FILE,
        mask=None,
        sub_runs=[],
        van_file=VANADIUM_FILE,
        eta_step=3.0,
    )

    for sub_run_i in list(gold_data_dict.keys()):
        # Get gold data of pattern (i).
        gold_data_i = gold_data_dict[sub_run_i]

        # Get powder data of pattern (i).
        pattern = reducer.get_diffraction_data(1, sub_run_i)

        # validate correct two-theta reduction
        np.testing.assert_allclose(pattern[0], gold_data_dict[sub_run_i][0], rtol=1e-8)

        # remove NaN intensity arrays
        pattern[1][np.where(np.isnan(pattern[1]))] = 0.0
        gold_data_i[1][np.where(np.isnan(gold_data_i[1]))] = 0.0

        # validate correct intesnity reduction
        np.testing.assert_allclose(pattern[1], gold_data_i[1], rtol=1e-8, equal_nan=True)

    # END-FOR


def test_eta_wedges_are_binned_on_the_stored_axis():
    """Regression (#1033): every eta wedge must be histogrammed on the 2theta axis stored for its sub-run.

    Each wedge used to be binned over its own pixels' 2theta range while the workspace keeps one 2theta axis
    per sub-run, so every wedge but the last written carried another wedge's axis."""
    # Arrange
    workspace = NeXusConvertingApp(
        os.path.join(DATA_DIR, "calibration_tests/HB2B_3510.nxs.h5"), os.path.join(DATA_DIR, "HB2B_Mask_12-18-19.xml")
    ).convert()
    reducer = ReductionApp()
    reducer.load_hidra_workspace(workspace)

    # Act
    reducer.reduce_data(
        sub_runs=[1], instrument_file=None, calibration_file=None, mask=None, num_bins=512, eta_step=3.0
    )

    # Assert - re-histogram each wedge's pixels on the axis it was returned with
    engine = reducer._reduction_manager._last_reduction_engine
    pixel_2theta = engine.instrument.get_pixels_2theta(1)
    eta = engine.get_eta_value()
    counts = workspace.get_detector_counts(1).astype(float)
    keep = workspace.get_detector_mask(True)
    for mask_id in workspace.reduction_masks:
        centre = float(mask_id.split("_")[-1])
        two_theta, intensity, _ = reducer.get_diffraction_data(1, mask_id=mask_id)
        step = two_theta[1] - two_theta[0]
        edges = np.append(two_theta - step / 2, two_theta[-1] + step / 2)
        wedge = (eta >= centre - 1.5) & (eta <= centre + 1.5) & (keep != 0)
        expected = np.histogram(pixel_2theta[wedge], edges, weights=counts[wedge])[0]
        misplaced = np.abs(np.asarray(intensity) - expected).sum() / expected.sum()
        assert misplaced < 1e-3, f"{mask_id}: {misplaced:.1%} of the counts are in the wrong 2theta bins"
