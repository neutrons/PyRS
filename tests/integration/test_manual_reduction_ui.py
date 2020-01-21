"""
Integration test for workflow used in manual reduction UI
"""
from pyrs.projectfile import HidraProjectFile
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.utilities import calibration_file_io
from pyrs.core import workspaces
import numpy as np
from pyrs.split_sub_runs.load_split_sub_runs import NexusProcessor
from pyrs.core.powder_pattern import ReductionApp
import os
import pytest
from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface.manual_reduction.event_handler import EventHandler
from pyrs.core.reduce_hb2b_pyrs import PyHB2BReduction


def test_load_split():
    """

    Returns
    -------

    """
    # Init load/split service instance

    # Get list of sub runs

    # Split runs

    # Split logs

    # Save to file

    return


def test_load_split_visualization():
    """

    Returns
    -------

    """
    # Init load/split service instance

    # Get list of sub runs

    # Split one specified sub run

    # verify result with gold data


@pytest.mark.parametrize('project_file, calibration_file, mask_file, gold_file',
                         [('data/HB2B_1017.h5', None, None, 'data/gold/1017_NoMask.h5'),
                          ('data/HB2B_1017.h5', None, 'data/HB2B_Mask_12-18-19.xml', 'data/gold/1017_Mask.h5')],
                         ids=('HB2B_1017_Masked', 'HB2B_1017_NoMask'))
def test_diffraction_pattern(project_file, calibration_file, mask_file, gold_file):
    """

    Parameters
    ----------
    project_file
    calibration_file
    mask_file : str or None
        mask file name
    gold_file

    Returns
    -------

    """
    # Load gold Hidra project file for diffraction pattern (run 1017)
    hidra_project = EventHandler.load_project_file(None, project_file)
    test_workspace = HidraWorkspace()
    test_workspace.load_hidra_project(hidra_project, True, False)

    # Convert to diffraction pattern
    from pyrs.core.powder_pattern import ReductionApp
    powder_red_service = ReductionApp(use_mantid_engine=False)

    # Set workspace
    powder_red_service.load_hidra_workspace(test_workspace)

    # Reduction
    powder_red_service.reduce_data(sub_runs=None,
                                   instrument_file=None,
                                   calibration_file=calibration_file,
                                   mask=mask_file,
                                   mask_id=None,
                                   van_file=None,
                                   num_bins=1000)

    # Load gold data
    gold_data_set = load_gold_data(gold_file)
    gold_sub_runs = np.array(gold_data_set.keys())
    gold_sub_runs.sort()

    #  Test number of sub runs
    np.testing.assert_allclose(gold_sub_runs, powder_red_service.get_sub_runs())

    # Get diffraction pattern
    for sub_run_i in gold_sub_runs:
        np.testing.assert_allclose(gold_data_set[sub_run_i], powder_red_service.get_diffraction_data(sub_run_i),
                                   rtol=1e-8)


def test_diffraction_pattern_geometry_shift():
    """

    Returns
    -------

    """
