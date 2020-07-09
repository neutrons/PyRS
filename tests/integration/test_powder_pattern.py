"""
Integration test for PyRS to calculate powder pattern from detector counts
"""
from pyrs.projectfile import HidraProjectFile  # type: ignore
import numpy as np
import h5py
from pyrs.core.workspaces import HidraWorkspace
from pyrs.core.reduce_hb2b_pyrs import ResidualStressInstrument
from pyrs.core.instrument_geometry import AnglerCameraDetectorGeometry
from pyrs.core.reduction_manager import HB2BReductionManager
import pytest


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


def write_gold_file(file_name, data):
    """Write value to gold file (format)

    Parameters
    ----------
    file_name : str
        output file
    data : ~tuple or ~dict
        numpy array data or dictionary of data
    Returns
    -------

    """
    gold_file = h5py.File(file_name, 'w')

    if isinstance(data, np.ndarray):
        dataset = {'data': data}
    else:
        dataset = data

    for data_name in dataset:
        if isinstance(dataset[data_name], tuple):
            # write (x, y)
            group = gold_file.create_group(data_name)
            group.create_dataset('x', data=dataset[data_name][0])
            group.create_dataset('y', data=dataset[data_name][1])
        else:
            # write value directly
            gold_file.create_dataset(data_name, data=dataset[data_name])

    gold_file.close()


def test_2theta_calculation():
    """Test the calculation of pixels' 2theta values

    Returns
    -------

    """
    # Create geometry setup
    pixel_size = 0.3 / 1024.0
    arm_length = 0.985
    test_setup = AnglerCameraDetectorGeometry(1024, 1024, pixel_size, pixel_size, arm_length, False)
    # Create instrument
    instrument = ResidualStressInstrument(test_setup)

    # Set 2theta
    instrument.build_instrument(two_theta=85., l2=None, instrument_calibration=None)

    # Calculate pixel positions
    pixel_positions = instrument.get_pixel_array()

    # Calculate 2theta values
    two_theta_arrays = instrument.get_pixels_2theta(dimension=1)

    # compare with gold file
    gold_dict = parse_gold_file('tests/data/HB2B_1017_Pixels_Gold.h5')
    np.testing.assert_allclose(pixel_positions, gold_dict['positions'], rtol=1E-8)
    np.testing.assert_allclose(two_theta_arrays, gold_dict['2theta'], rtol=1E-8)


@pytest.mark.parametrize('project_file_name, mask_file_name, gold_file',
                         [('tests/data/HB2B_1017.h5', 'tests/data/HB2B_Mask_12-18-19.xml',
                           'tests/data/HB2B_1017_NoMask_Gold.h5'),
                          ('tests/data/HB2B_1017.h5', None, 'tests/data/HB2B_1017_NoMask_Gold.h5')],
                         ids=('HB2B_1017_Masked', 'HB2B_1017_NoMask'))
def test_powder_pattern_engine(project_file_name, mask_file_name, gold_file):
    """Test the powder pattern calculator (service) with HB2B-specific reduction routine

    Parameters
    ----------
    project_file_name
    mask_file_name
    gold_file

    Returns
    -------

    """
    if mask_file_name is not None:
        pytest.skip('Masking is not implemented yet')

    # Parse input file
    test_ws = HidraWorkspace('test_powder_pattern')
    test_project = HidraProjectFile(project_file_name)
    test_ws.load_hidra_project(test_project, load_raw_counts=True, load_reduced_diffraction=False)
    test_project.close()

    # Sub runs
    sub_runs = test_ws.get_sub_runs()

    # Import gold file
    gold_pattern = parse_gold_file(gold_file)

    data_dict = dict()

    # Start reduction service
    pyrs_service = HB2BReductionManager()
    pyrs_service.init_session(session_name='test_powder', hidra_ws=test_ws)

    # Reduce raw counts
    pyrs_service.reduce_diffraction_data('test_powder', False, 1000,
                                         sub_run_list=None,
                                         mask=mask_file_name, mask_id=None,
                                         vanadium_counts=None, normalize_by_duration=False)

    for index, sub_run_i in enumerate(sub_runs):
        # Get gold data of pattern (i).
        gold_data_i = gold_pattern[str(sub_run_i)]

        # Get powder data of pattern (i).
        pattern = pyrs_service.get_reduced_diffraction_data('test_powder', sub_run_i)

        # ensure NaN are removed
        gold_data_i[1][np.where(np.isnan(gold_data_i[1]))] = 0.
        pattern[1][np.where(np.isnan(pattern[1]))] = 0.

        # Verify
        np.testing.assert_allclose(pattern[1], gold_data_i[1], rtol=1E-8)

        data_dict[str(sub_run_i)] = pattern

#    if mask_file_name:
#        name = 'tests/data/HB2B_1017_Mask_Gold.h5'
#    else:
#        name = 'tests/data/HB2B_1017_NoMask_Gold.h5'
#    write_gold_file(name, data_dict)

    return


@pytest.mark.parametrize('project_file_name, mask_file_name, gold_file',
                         [('tests/data/HB2B_1017.h5', 'tests/data/HB2B_Mask_12-18-19.xml',
                           'tests/data/HB2B_1017_NoMask_Gold.h5'),
                          ('tests/data/HB2B_1017.h5', None, 'tests/data/HB2B_1017_NoMask_Gold.h5')],
                         ids=('HB2B_1017_Masked', 'HB2B_1017_NoMask'))
def test_powder_pattern_service(project_file_name, mask_file_name, gold_file):
    """Test the powder pattern calculator (service) with HB2B-specific reduction routine

    Parameters
    ----------
    project_file_name
    mask_file_name
    gold_file

    Returns
    -------

    """
    if mask_file_name is not None:
        pytest.skip('Not Ready Yet for Masking')

    # load gold file
    gold_data_dict = parse_gold_file(gold_file)

    # Parse input file
    test_ws = HidraWorkspace('test_powder_pattern')
    test_project = HidraProjectFile(project_file_name)
    test_ws.load_hidra_project(test_project, load_raw_counts=True, load_reduced_diffraction=False)
    test_project.close()

    # Start reduction service
    pyrs_service = HB2BReductionManager()
    pyrs_service.init_session(session_name='test_powder', hidra_ws=test_ws)

    # Reduce raw counts
    pyrs_service.reduce_diffraction_data('test_powder', False, 1000,
                                         sub_run_list=None,
                                         mask=mask_file_name, mask_id=None,
                                         vanadium_counts=None, normalize_by_duration=False)

    # Get sub runs
    sub_runs = test_ws.get_sub_runs()

    for index, sub_run_i in enumerate(sub_runs):
        # Get gold data of pattern (i).
        gold_data_i = gold_data_dict[str(sub_run_i)]

        # Get powder data of pattern (i).
        pattern = pyrs_service.get_reduced_diffraction_data('test_powder', sub_run_i)
        # data_dict[str(sub_run_i)] = pattern

        # validate correct two-theta reduction
        np.testing.assert_allclose(pattern[0], gold_data_dict[str(sub_run_i)][0], rtol=1E-8)

        # remove NaN intensity arrays
        pattern[1][np.where(np.isnan(pattern[1]))] = 0.
        gold_data_i[1][np.where(np.isnan(gold_data_i[1]))] = 0.

        # validate correct intesnity reduction
        np.testing.assert_allclose(pattern[1], gold_data_i[1], rtol=1E-8, equal_nan=True)

    # END-FOR
