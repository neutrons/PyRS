"""
Integration test for PyRS to calculate powder pattern from detector counts
"""
import numpy as np
import h5py
from pyrs.core.nexus_conversion import NeXusConvertingApp
from pyrs.core.powder_pattern import ReductionApp
import pytest
import os


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
    data_set_names = list(gold_file[u'reduced diffraction data'].keys())
    for name in data_set_names:
        if ('_var' in name) or ('2theta' in name) or ('main' in name):
            pass
        else:
            if isinstance(gold_file[u'reduced diffraction data'][name], h5py.Dataset):
                data_dict[name] = [gold_file[u'reduced diffraction data']['2theta'].value[0, :],
                                   gold_file[u'reduced diffraction data'][name].value[0, :]]
            else:
                pytest.skip('project file not supplied')

        # END-IF
    # END-FOR

    if len(data_dict) == 1 and data_dict.keys()[0] == 'data':
        # only 1 array with default name
        return data_dict['data']

    return data_dict


@pytest.mark.parametrize('nexusfile, mask_file_name, gold_file',
                         [('data/HB2B_1118.nxs.h5', 'data/HB2B_Mask_12-18-19.xml', 'data/HB2B_1118_texture.h5')],
                         ids=['HB2B_1118_Texture'])
def test_texture_reduction(nexusfile, mask_file_name, gold_file):
    """Test the powder pattern calculator (service) with HB2B-specific reduction routine

    Parameters
    ----------
    project_file_name
    mask_file_name
    gold_file
    """
    if not os.path.exists('/HFIR/HB2B/shared'):
        pytest.skip('Unable to access HB2B archive')

    CALIBRATION_FILE = "data/HB2B_calib_latest.json"
    VANADIUM_FILE = "/HFIR/HB2B/IPTS-22731/nexus/HB2B_1115.nxs.h5"

    # load gold file
    gold_data_dict = parse_gold_file(gold_file)

    # Parse input file
    converter = NeXusConvertingApp(nexusfile, mask_file_name)
    hidra_ws = converter.convert()

    # Start reduction service
    reducer = ReductionApp(bool('pyrs' == 'mantid'))
    reducer.load_hidra_workspace(hidra_ws)

    # Reduce raw counts
    reducer.reduce_data(instrument_file=None,
                        calibration_file=CALIBRATION_FILE,
                        mask=None,
                        sub_runs=[],
                        van_file=VANADIUM_FILE,
                        eta_step=3.0)

    for sub_run_i in list(gold_data_dict.keys()):
        # Get gold data of pattern (i).
        gold_data_i = gold_data_dict[sub_run_i]

        # Get powder data of pattern (i).
        pattern = reducer.get_diffraction_data(1, sub_run_i)

        # validate correct two-theta reduction
        np.testing.assert_allclose(pattern[0], gold_data_dict[sub_run_i][0], rtol=1E-8)

        # remove NaN intensity arrays
        pattern[1][np.where(np.isnan(pattern[1]))] = 0.
        gold_data_i[1][np.where(np.isnan(gold_data_i[1]))] = 0.

        # validate correct intesnity reduction
        np.testing.assert_allclose(pattern[1], gold_data_i[1], rtol=1E-8, equal_nan=True)

    # END-FOR
