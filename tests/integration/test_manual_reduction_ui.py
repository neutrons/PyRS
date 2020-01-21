"""
Integration test for workflow used in manual reduction UI
"""
import numpy as np
import pytest
from pyrs.core.workspaces import HidraWorkspace
from pyrs.interface.manual_reduction.event_handler import EventHandler
from pyrs.core.powder_pattern import ReductionApp
import h5py


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
    gold_file = h5py.File(file_name, 'r')
    data_set_names = list(gold_file.keys())
    for name in data_set_names:
        if isinstance(gold_file[name], h5py.Dataset):
            data_dict[name] = gold_file[name].value
        else:
            data_dict[name] = gold_file[name]['x'].value, gold_file[name]['y'].value
    # END-FOR

    if len(data_dict) == 1 and data_dict.keys()[0] == 'data':
        # only 1 array with default name
        return data_dict['data']

    return data_dict


def test_load_split():
    """

    Returns
    -------

    """
    pytest.skip('Manual reduction UI classes has not been refactored yet.')

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
    pytest.skip('Manual reduction UI classes has not been refactored yet.')

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
    pytest.skip('Manual reduction UI classes has not been refactored yet.')

    # Load gold Hidra project file for diffraction pattern (run 1017)
    hidra_project = EventHandler.load_project_file(None, project_file)
    test_workspace = HidraWorkspace()
    test_workspace.load_hidra_project(hidra_project, True, False)

    # Convert to diffraction pattern
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
    gold_data_set = parse_gold_file(gold_file)
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
    pytest.skip('Manual reduction UI classes has not been refactored yet.')

    return
