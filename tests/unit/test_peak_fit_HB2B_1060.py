import numpy as np
from pyrs.core import pyrscore
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory
from collections import namedtuple
import pytest
import os


# Named tuple for peak information
PeakInfo = namedtuple('PeakInfo', 'center left_bound right_bound tag')


@pytest.mark.parametrize('target_values', [{'Intensity': [0.4, 0.3], 'peak_center': [91, 95], 'FWHM': [7.76, 7.76],
                                            'background_A0': [2, -0.301], 'background_A1': [0.007, 0.003]}])
def test_pseudovoigt_HB2B_1060(target_values):
    """This is a test of Pseudovoigt peak fitting for HB2B 1060.

     Data are from the real HB2B data previously reported problematic

     Returns
     -------

     """
    # Define HiDRA project file name and skip test if it does not exist (on Travis)
    # project_file_name = 'tests/data/HB2B_1060.h5'
    project_file_name = 'tests/data/HB2B_1060.h5'

    if not os.path.exists(project_file_name):
        pytest.skip('{} does not exist on Travis'.format(project_file_name))

    # Create calibration control
    controller = pyrscore.PyRsCore()

    # Load project file to HidraWorkspace
    project_name = 'HB2B_1060 Peaks'
    hd_ws = controller.load_hidra_project(project_file_name, project_name=project_name, load_detector_counts=False,
                                          load_diffraction=True)

    peak_type = 'PseudoVoigt'

    # Set peak fitting engine
    # create a controller from factory
    fit_engine = PeakFitEngineFactory.getInstance(hd_ws, peak_function_name=peak_type,
                                                  background_function_name='Linear')

    # Fit peak @ left and right
    peak_info_left = PeakInfo(91.7, 87., 93., 'Left Peak')
    peak_info_right = PeakInfo(95.8, 93.5, 98.5, 'Right Peak')

    fit_result = fit_engine.fit_multiple_peaks(peak_tags=[peak_info_left.tag, peak_info_right.tag],
                                               x_mins=[peak_info_left.left_bound, peak_info_right.left_bound],
                                               x_maxs=[peak_info_left.right_bound, peak_info_right.right_bound])

    assert len(fit_result.peakcollections) == 2, 'two PeakCollection'
    assert fit_result.fitted
    assert fit_result.difference

    # peak 'Left'
    param_values_lp, param_errors_lp = fit_result.peakcollections[0].get_native_params()

    # peak 'Right'
    param_values_rp, param_errors_rp = fit_result.peakcollections[1].get_native_params()

    assert param_values_lp.size == 117, '117 subruns'
    assert len(param_values_lp.dtype.names) == 6, '6 effective parameters'

    assert param_values_rp.size == 117, '117 subruns'
    assert len(param_values_rp.dtype.names) == 6, '6 effective parameters'

    np.testing.assert_allclose(param_values_lp['Intensity'], target_values['Intensity'][0], rtol=20.)
    np.testing.assert_allclose(param_values_lp['PeakCentre'], target_values['peak_center'][0], rtol=50.)
    np.testing.assert_allclose(param_values_lp['FWHM'], target_values['FWHM'][0], rtol=50.)
    np.testing.assert_allclose(param_values_lp['A0'], target_values['background_A0'][0], rtol=50.)
    np.testing.assert_allclose(param_values_lp['A1'], target_values['background_A1'][0], rtol=50.)

    np.testing.assert_allclose(param_values_rp['Intensity'], target_values['Intensity'][1], rtol=20.)
    np.testing.assert_allclose(param_values_rp['PeakCentre'], target_values['peak_center'][1], rtol=50.)
    np.testing.assert_allclose(param_values_rp['FWHM'], target_values['FWHM'][1], rtol=50.)
    np.testing.assert_allclose(param_values_rp['A0'], target_values['background_A0'][1], rtol=50.)
    np.testing.assert_allclose(param_values_rp['A1'], target_values['background_A1'][1], rtol=50.)
