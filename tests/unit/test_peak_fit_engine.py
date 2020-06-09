import numpy as np
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory  # type: ignore
from pyrs.core.workspaces import HidraWorkspace
from pyrs.core.peak_profile_utility import pseudo_voigt, PeakShape, BackgroundFunction
from pyrs.core.peak_profile_utility import Gaussian, PseudoVoigt
import pytest
import os
from matplotlib import pyplot as plt
from collections import namedtuple
from pyrs.core import pyrscore
import sys
# set to True when running on build servers
ON_TRAVIS = (os.environ.get('TRAVIS', 'false').upper() == 'TRUE')


def generate_test_gaussian(vec_x, peak_center_list, peak_range_list, peak_height_list):
    """
    Generate Gaussian function for test
    Parameters
    ----------
    vec_x: ndarray (N, )
        Vector of X
    peak_center_list : List
        peak center (float) list
    peak_range_list : List
        peak range (float) list.  Peak range is equal to 6 times of FWHM
    peak_height_list : List
        list of intensities

    Returns
    -------
    numpy.ndarray
        vector of Gaussian
    """
    assert len(peak_range_list) == len(peak_range_list) == len(peak_height_list)

    # Init Y
    vec_y = np.zeros_like(vec_x, dtype=float)

    # Number of peaks
    num_peaks = len(peak_range_list)
    parameters = list()
    for ipeak in range(num_peaks):
        # get peak center and range
        peak_center = peak_center_list[ipeak]
        peak_range = peak_range_list[ipeak]

        # Set FWHM to 1/6 of peak range and then to Gaussian's Sigma
        sigma = peak_range / 6. / (2. * np.sqrt(2. * np.log(2.)))

        # generate noise with amplitude of sqrt(peak_height)
        noise = (np.random.random_sample(vec_x.shape[0]) - 0.5) * np.sqrt(peak_height_list[ipeak])

        # calculate Gaussian function based on input peak center and peak range
        vec_y += noise + peak_height_list[ipeak] * np.exp(-(vec_x - peak_center) ** 2 / sigma ** 2)

        parameters.append({'peak_center': peak_center,
                           'peak_intensity': np.sqrt(2. * np.pi) * peak_height_list[ipeak] * sigma,
                           'peak_FWHM': 2. * np.sqrt(2. * np.log(2.)) * sigma
                           })

    return {'values': vec_y, 'parameters': parameters}


def generate_test_pseudovoigt(vec_x, peak_center_list, peak_range_list, intensity_list):
    """
    Generate Gaussian function for test
    Parameters
    ----------
    vec_x: ndarray (N, )
        Vector of X
    peak_center_list: List(float)
        peak center
    peak_range_list: List(float)
        range of peak

    Returns
    -------
    numpy.ndarray
        vector of Y containing Gaussian, peak center, peak range
    """
    assert len(peak_range_list) == len(peak_range_list)

    # Init Y
    vec_y = np.zeros_like(vec_x, dtype=float)

    # Number of peaks
    num_peaks = len(peak_range_list)
    parameters = list()
    # Calculate each peak
    for ipeak in range(num_peaks):
        # get peak center and range
        peak_center = peak_center_list[ipeak]
        peak_range = peak_range_list[ipeak]

        # Set FWHM
        fwhm = peak_range / 6.
        peak_intensity = intensity_list[ipeak]
        mixing = 0.75  # more Gaussian than Lorentzian

        # calculate Gaussian
        vec_y += pseudo_voigt(vec_x, peak_intensity, fwhm, mixing, peak_center)
        parameters.append({'peak_center': peak_center,
                           'peak_intensity': intensity_list[ipeak],
                           'peak_FWHM': fwhm,
                           })
    # END-FOR

    # Add noise
    noise = (np.random.random_sample(vec_x.shape[0]) - 0.5) * 2.0

    return {'values': vec_y + noise, 'parameters': parameters}


def generate_test_background(vec_x, vec_y):
    """Generate background function and add to signal/peak

    Parameters
    ----------
    vec_x
    vec_y

    Returns
    -------
    ndarray, float
        vector of Y with background
    """
    a0 = 35
    a1 = -0.3

    background = a1 * vec_x + a0

    return {'values': vec_y + background, 'parameters': {'background_A0': a0, 'backgound_A1': a1}}


def generate_hydra_workspace_single_subrun(peak_profile_type, min_x, max_x, num_x, peak_centers, peak_ranges,
                                           peak_intensities):
    """Generate HiDRAWorkspace for peak fitting test

    Default:
        min_x = 75
        max_x = 85
        num_x = 500

    Parameters
    ----------
    peak_profile_type
    min_x
    max_x
    num_x
    peak_centers
    peak_ranges
    peak_intensities

    Returns
    -------
    pyrs.core.workspaces.HidraWorkspace
        Test Hidra workspace

    """
    # Create test workspace
    test_workspace = HidraWorkspace('test')

    # resolution
    x_step = (max_x - min_x) / num_x

    # Generate vector X
    vec_x = np.arange(num_x) * x_step + min_x  # from 75 to 85 degree

    # Add profile
    if peak_profile_type.lower() == 'gaussian':
        test_fitting_function = generate_test_gaussian(vec_x, peak_centers, peak_ranges, peak_intensities)
    elif peak_profile_type.lower() == 'pseudovoigt':
        test_fitting_function = generate_test_pseudovoigt(vec_x, peak_centers, peak_ranges, peak_intensities)
        # peak_range *= 2  # PV requires larger fitting range
    else:
        raise NotImplementedError('Peak profile {} is not supported to generate testing workspace')

    vec_y = test_fitting_function['values']
    parameters = test_fitting_function['parameters']

    # Add background
    test_background_function = generate_test_background(vec_x, vec_y)
    vec_y = test_background_function['values']
    parameters.append(test_background_function['parameters'])

    # Print out the test data
    # for i in range(vec_x.shape[0]):
    #     print('{}   {}'.format(vec_x[i], vec_y[i]))

    # Add diffraction pattern
    test_workspace.set_sub_runs([1])
    test_workspace.set_reduced_diffraction_data(1, mask_id=None,
                                                two_theta_array=vec_x,
                                                intensity_array=vec_y)

    return {'workspace': test_workspace, 'parameters': parameters}


def generate_hydra_workspace_multiple_sub_runs(ws_name, sub_run_data_dict):
    """Generate a multiple sub-run HiDRA workspace
     Parameters
     ----------
     ws_name
     sub_run_data_dict
     Returns
     -------
     pyrs.core.workspaces.HidraWorkspace
         Test Hidra workspace
     """

    # Create test workspace
    test_workspace = HidraWorkspace(ws_name)

    # Sub runs:
    sub_runs_list = list(sub_run_data_dict.keys())
    test_workspace.set_sub_runs(sub_runs_list)

    # Add diffraction pattern
    for sub_run_i in sorted(sub_runs_list):
        vec_x, vec_y = sub_run_data_dict[sub_run_i]
        test_workspace.set_reduced_diffraction_data(sub_run_i,
                                                    mask_id=None,
                                                    two_theta_array=vec_x,
                                                    intensity_array=vec_y)

    return test_workspace


def print_peak_results_and_check_positive(params, errors):
    names = params.dtype.names
    for offset, (peak_param, peak_error) in enumerate(zip(params, errors)):
        for name, value, error in zip(names, peak_param, peak_error):
            print('peak[{}], {} = {} +- {}'.format(offset, name, value, error))
            if name.startswith('A'):  # don't check the background
                continue
            assert value > 0., name


def assert_checks(fit_result, exp_params, obs_params, number_of_peakCollection,
                  peak_profile_type=PeakShape.GAUSSIAN):
    """verified the fitted parameters
         Parameters
         ----------
         fit_result: object
            results of the fitting
         exp_params: dict(float)
            parameters of original shape
         obs_params:List(float)
            parameter values of fitting
         number_of_peakCollection:Integer
            number of fitted peaks
         peak_profile_type : str
            type of peak profile
         Returns
         -------

         """
    assert len(fit_result.peakcollections) == number_of_peakCollection, 'Only one PeakCollection'
    assert fit_result.fitted
    assert fit_result.difference

    peak_profile_type = PeakShape.getShape(peak_profile_type)
    if peak_profile_type == PeakShape.GAUSSIAN:
        np.testing.assert_allclose(obs_params['PeakCentre'], exp_params['peak_center'], rtol=50.)
        # np.testing.assert_allclose(obs_params['Intensity'], exp_params['peak_intensity'], rtol=50.)
        # np.testing.assert_allclose(obs_params['FWHM'], exp_params['peak_FWHM'], rtol=50.)
    elif peak_profile_type == PeakShape.PSEUDOVOIGT:
        np.testing.assert_allclose(obs_params['Intensity'], exp_params['peak_intensity'], rtol=50.)
        np.testing.assert_allclose(obs_params['PeakCentre'], exp_params['peak_center'], rtol=50.)
        np.testing.assert_allclose(obs_params['FWHM'], exp_params['peak_FWHM'], rtol=50.)


@pytest.fixture()
def setup_1_subrun(request):
    try:
        params = request.param
    except AttributeError:
        try:
            params = request._parent_request.param
        except AttributeError:
            params = dict()

    return generate_hydra_workspace_single_subrun(params['peak_profile_type'], params['min_x'], params['max_x'],
                                                  params['num_x'], params['peak_center'], params['peak_range'],
                                                  params['peak_intensities'],)


@pytest.mark.parametrize("setup_1_subrun",
                         [{'peak_profile_type': 'Gaussian', 'min_x': 75., 'max_x': 85., 'num_x': 500,
                           'peak_center': [80.], 'peak_range': [10. * 0.25], 'peak_intensities':[10]}],
                         indirect=True)
@pytest.mark.parametrize('fit_domain',
                         [(78.75, 81.25)])
def test_1_gaussian_1_subrun(setup_1_subrun, fit_domain):
    """Test fitting single Gaussian peak on 1 spectrum with background

    Returns
    -------
    None

    """
    # initialize fit engine
    fit_engine = PeakFitEngineFactory.getInstance(setup_1_subrun['workspace'], peak_function_name='Gaussian',
                                                  background_function_name='Linear', wavelength=np.nan,
                                                  out_of_plane_angle=None)

    # Fit
    peak_tag = 'UnitTestGaussian'
    fit_result = fit_engine.fit_peaks(peak_tag=peak_tag, x_min=fit_domain[0], x_max=fit_domain[1])
    number_of_peakCollection = 1
    # get back the peak collection
    peakcollection = fit_result.peakcollections[0]
    assert peakcollection.peak_tag == peak_tag
    parameters = setup_1_subrun['parameters'][0]
    # Test the fitted parameters
    fit_costs = peakcollection.fitting_costs
    sub_runs = peakcollection.sub_runs
    eff_param_values, eff_param_errors = peakcollection.get_effective_params()
    assert eff_param_values.dtype.names[0] == 'Center'
    np.testing.assert_almost_equal(eff_param_values['Center'], parameters['peak_center'], decimal=1)
    print_peak_results_and_check_positive(eff_param_values, eff_param_errors)

    # Read data again for raw data
    native_params = PeakShape.GAUSSIAN.native_parameters
    native_params.extend(BackgroundFunction.LINEAR.native_parameters)

    # assert_checks(fit_result, native_params, number_of_peakCollection, peak_tag,
    # peak_center, peak_intensity, peak_FWHM, background_A0, baclground_A1)
    sub_runs2 = peakcollection.sub_runs
    fit_cost2 = peakcollection.fitting_costs
    param_values, param_errors = peakcollection.get_native_params()

    # Test
    assert sub_runs.size == 1 == sub_runs2.size
    np.testing.assert_equal(fit_cost2, fit_costs)

    # Effective parameter list: ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']
    assert eff_param_values['Center'] == param_values['PeakCentre']   # center
    np.testing.assert_allclose(param_values['PeakCentre'], parameters['peak_center'], rtol=3e-2,
                               err_msg='Peak center is not correct')

    # Parameters verified
    assert_checks(fit_result, parameters, param_values, number_of_peakCollection)

    # fit goodness
    assert fit_costs[0] < 2.0, 'Fit cost (chi2 = {}) is too large'.format(fit_costs[0])  # TODO was 0.5


@pytest.mark.parametrize("setup_1_subrun", [{'peak_profile_type': 'Gaussian', 'min_x': 75., 'max_x': 95.,
                                             'num_x': 1000, 'peak_center': [80., 90.],
                                             'peak_range': [10. * 0.25, 11. * 0.25],
                                             'peak_intensities': [10., 4]}], indirect=True)
@pytest.mark.parametrize('fit_domain',
                         [((76., 86.), (84., 94.))])
def test_2_gaussian_1_subrun(setup_1_subrun, fit_domain):
    """Fit 2 Gaussian peaks for 1 sub run

    Returns
    -------

    """
    fit_engine = PeakFitEngineFactory.getInstance(setup_1_subrun['workspace'], peak_function_name='Gaussian',
                                                  background_function_name='Linear', wavelength=np.nan)

    # Fit

    fit_result = fit_engine.fit_multiple_peaks(peak_tags=['Left', 'Right'], x_mins=fit_domain[0], x_maxs=fit_domain[1])
    number_of_peakCollection = 2.
    # Test the fitted parameters: effective parameters
    # Effective parameter list: ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']
    # Read data
    sub_runs = fit_result.peakcollections[0].sub_runs
    fit_costs = fit_result.peakcollections[0].fitting_costs
    eff_param_values, eff_param_errors = fit_result.peakcollections[0].get_effective_params()
    assert eff_param_values.size == 1, '1 sub run'
    assert len(eff_param_values.dtype.names) == 7, '7 effective parameters'
    '''
    if abs(eff_param_values[2][0] - expected_intensity) < 1E-03:
        plt.plot(data_x, data_y, label='Test 2 Gaussian')
        plt.plot(model_x, model_y, label='Fitted Gaussian')
        plt.legend()
        plt.show()
        raise AssertionError('Peak intensity {} shall be equal to {}.'
                             ''.format(eff_param_values[2, 0], expected_intensity))
    '''

    # Test the fitted parameters: native parameters
    native_params = PeakShape.GAUSSIAN.native_parameters
    native_params.extend(BackgroundFunction.LINEAR.native_parameters)
    sub_runs2 = fit_result.peakcollections[0].sub_runs
    fit_cost2 = fit_result.peakcollections[0].fitting_costs
    param_values, param_errors = fit_result.peakcollections[0].get_native_params()
    parameters = setup_1_subrun['parameters'][0]
    # Test
    assert sub_runs.size == 1 == sub_runs2.size
    np.testing.assert_equal(fit_cost2, fit_costs)

    np.testing.assert_equal(eff_param_values['Center'], param_values['PeakCentre'])
    np.testing.assert_allclose(eff_param_values['Center'], parameters['peak_center'],
                               rtol=2e-2, err_msg='Peak center is not correct')

    # Parameters verified
    assert_checks(fit_result, parameters, param_values, number_of_peakCollection)


@pytest.mark.parametrize('target_values', [{'peak_height': [10, 0.012], 'peak_center': [75, 77], 'sigma': [0.15, 1.5],
                                            'background_A0': [2, -0.301], 'background_A1': [0.007, 0.003]}])
def test_2_gaussian_3_subruns(target_values):
    """Testing fitting 2 Gaussian peaks among 3 sub runs.

    Some of the sub runs may not have one specific peak or the peak may be off center too much.

    This is an extreme case such that
    sub run = 1: peak @ 75 and @ 83
    sub run = 2: peak @ 75 and @ 80
    sub run = 3: peak @ 75 very low and @ 80

    Returns
    -------
    None

    """
    # Generate 3 sub runs
    vec_x = np.arange(750).astype(float) * 0.02 + 70.

    # Create dictionary for test data
    test_2g_dict = dict()

    # sub run 1
    vec_y = generate_test_gaussian(vec_x, [75., 83], [3., 3.5], [10., 5])['values']
    test_2g_dict[1] = vec_x, vec_y

    # sub run 2
    vec_y = generate_test_gaussian(vec_x, [75., 80], [3., 3.5], [15., 5])['values']
    test_2g_dict[2] = vec_x, vec_y

    # sub run 3
    vec_y = generate_test_gaussian(vec_x, [75., 80], [3.1, 3.5], [0.2, 7.5])['values']
    test_2g_dict[3] = vec_x, vec_y

    # Create a workspace based on this
    test_hd_ws = generate_hydra_workspace_multiple_sub_runs('3 G 3 S', test_2g_dict)

    # Fit
    fit_engine = PeakFitEngineFactory.getInstance(test_hd_ws, peak_function_name='Gaussian',
                                                  background_function_name='Linear', wavelength=np.nan)
    fit_result = fit_engine.fit_multiple_peaks(peak_tags=['Left', 'Right'],
                                               x_mins=(72.5, 77.5), x_maxs=(77.5, 82.5))

    # were there returns
    assert len(fit_result.peakcollections) == 2, 'Two PeakCollection'
    assert fit_result.fitted
    assert fit_result.difference

    # Verify fitting result
    # ['Height', 'PeakCentre', 'Sigma'],
    gaussian_native_params = PeakShape.GAUSSIAN.native_parameters
    gaussian_native_params.extend(BackgroundFunction.LINEAR.native_parameters)

    # peak 'Left'
    param_values_lp, param_errors_lp = fit_result.peakcollections[0].get_native_params()

    # peak 'Right'
    param_values_rp, param_errors_rp = fit_result.peakcollections[1].get_native_params()

    """
    Left
    Chi2: [0.3144778  0.32581177 0.31288937]
    Height (0)
        [ 9.75107193 15.30580711  0.10662809]
        [0.34154591 0.34128258 0.25921181]
    Center (1)
        [74.99655914 74.99921417 76.14541626]
        [0.0059319  0.00377185 0.75386631]
    Sigma (2)
        [0.14838572 0.14782989 0.28031287]
        [0.00608054 0.00388058 0.85695982]

    Right
    Chi2: [0.32870555 0.33221155 0.32712516]
    Note: chi2[0] can be infinity sometimes
    Height
        [0.41185206 4.91072798 7.52280378]
        [0.6211102  0.31933406 0.30802685]
    Center
        [78.88805389 79.9979248  80.00099945]
        [0.0767037  0.01266132 0.00856138]
    Sigma
        [0.04440744 0.17071389 0.18304431]
        [0.07783635 0.01304373 0.00897603]
    """

    # Get effective peak parameters
    effective_param_values, effective_param_errors = fit_result.peakcollections[0].get_effective_params()
    assert effective_param_values.size == 3, '3 subruns'
    assert len(effective_param_values.dtype.names) == 7, '7 effective parameters'

    # TODO it is odd that there are only two in the the setup function and 3 in the result
    np.testing.assert_allclose(param_values_lp['Height'][:2], target_values['peak_height'], atol=20.)
    np.testing.assert_allclose(param_values_lp['PeakCentre'][:2], target_values['peak_center'], rtol=50.)
    np.testing.assert_allclose(param_values_lp['Sigma'][:2], target_values['sigma'], rtol=50.)
    np.testing.assert_allclose(param_values_lp['A0'][:2], target_values['background_A0'], rtol=50.)
    np.testing.assert_allclose(param_values_lp['A1'][:2], target_values['background_A1'], rtol=50.)

    effective_param_values, effective_param_errors = fit_result.peakcollections[1].get_effective_params()
    assert effective_param_values.size == 3, '3 subruns'
    assert len(effective_param_values.dtype.names) == 7, '7 effective parameters'

    # Plot
    # model_x, model_y = fit_engine.calculate_fitted_peaks(3, None)
    # data_x, data_y = test_hd_ws.get_reduced_diffraction_data(3, None)
    # assert data_x.shape == model_x.shape
    # assert data_y.shape == model_y.shape
    # plt.plot(data_x, data_y, label='Test 2 Gaussian 3 sub runs')
    # plt.plot(model_x, model_y, label='Fitted 2 Gaussian 3 sub runs')
    # plt.legend()
    # plt.show()


@pytest.mark.parametrize('target_values', [{'peak_height': [10], 'peak_center': [75], 'sigma': [0.15],
                                            'background_A0': [2], 'background_A1': [0.007]}])
def test_3_gaussian_3_subruns(target_values):
    """Test fitting 3 Gaussian peaks may or may not on a 3 sub runs

    This is an extreme case such that
    sub run = 1: peak @ 75            X in [68, 78]
    sub run = 2: peak @ 75 and @ 80   X in [72, 82]
    sub run = 3: peak @ 80 and @ 85   X in [78, 88]

    Returns
    -------

    """
    # Create dictionary for test data for 3 sub runs
    test_2g_dict = dict()

    # sub run 1
    vec_x_0 = np.arange(500).astype(float) * 0.02 + 68.
    vec_y_0 = generate_test_gaussian(vec_x_0, [75.], [3.], [10.])['values']
    test_2g_dict[1] = vec_x_0, vec_y_0

    # sub run 2
    vec_x_1 = np.arange(500).astype(float) * 0.02 + 72.
    vec_y_1 = generate_test_gaussian(vec_x_1, [75., 80], [3., 3.5], [15., 5])['values']
    test_2g_dict[2] = vec_x_1, vec_y_1

    # sub run 3
    vec_x_2 = np.arange(500).astype(float) * 0.02 + 78.
    vec_y_2 = generate_test_gaussian(vec_x_2, [80., 85], [3.5, 3.7], [0.2, 7.5])['values']
    test_2g_dict[3] = vec_x_2, vec_y_2

    # Create a workspace based on this
    test_hd_ws = generate_hydra_workspace_multiple_sub_runs('3 G 3 S', test_2g_dict)

    # Fit
    fit_engine = PeakFitEngineFactory.getInstance(test_hd_ws, peak_function_name='Gaussian',
                                                  background_function_name='Linear', wavelength=np.nan)
    fit_result = fit_engine.fit_multiple_peaks(peak_tags=['Left', 'Middle', 'Right'],
                                               x_mins=(72.5, 77.5, 82.5),
                                               x_maxs=(77.5, 82.5, 87.5))

    # Verify fitting result
    # ['Height', 'PeakCentre', 'Sigma'],
    gaussian_native_params = PeakShape.GAUSSIAN.native_parameters
    gaussian_native_params.extend(BackgroundFunction.LINEAR.native_parameters)

    # peak 'Left'
    fit_cost2_lp = fit_result.peakcollections[0].fitting_costs
    param_values_lp, param_errors_lp = fit_result.peakcollections[0].get_native_params()

    # Parameters verified
    np.testing.assert_allclose(param_values_lp['Height'][0], target_values['peak_center'][0], rtol=50.0)
    np.testing.assert_allclose(param_values_lp['Sigma'][0], target_values['sigma'][0], rtol=50.0)
    np.testing.assert_allclose(param_values_lp['A0'][0], target_values['background_A0'][0], rtol=50.0)
    np.testing.assert_allclose(param_values_lp['A1'][0], target_values['background_A1'][0], rtol=50.0)

    assert np.isinf(fit_cost2_lp[2]), 'Sub run 3 does not have peak @ 75 (Peak-Left).  Chi2 shall be infinity but' \
                                      ' not {}'.format(fit_cost2_lp[2])


# pseudo-Voigt peak fitting only works on mantid versions with https://github.com/mantidproject/mantid/pull/27809
@pytest.mark.skipif(ON_TRAVIS and sys.version_info.major == 2, reason='Need mantid version > 4.2.20200128')
@pytest.mark.parametrize("setup_1_subrun", [{'peak_profile_type': 'PseudoVoigt', 'min_x': 75., 'max_x': 85.,
                                             'num_x': 500, 'peak_center': [80.], 'peak_range': [10. * 0.25],
                                             'peak_intensities':[100.]}], indirect=True)
@pytest.mark.parametrize('fit_domain',
                         [(77.5, 82.5)])
def test_1_pv_1_subrun(setup_1_subrun, fit_domain):
    """
    Test fitting single Pseudo-voigt peak with background
    Returns
    -------
    None
    """
    # Generate test workspace and initialize fit engine

    fit_engine = PeakFitEngineFactory.getInstance(setup_1_subrun['workspace'], peak_function_name='PseudoVoigt',
                                                  background_function_name='Linear', wavelength=np.nan)

    # Fit
    peak_tag = 'UnitTestPseudoVoigt'

    fit_result = fit_engine.fit_peaks(peak_tag=peak_tag, x_min=fit_domain[0], x_max=fit_domain[1])
    number_of_peakCollection = 1.
    # get back the peak collection
    peakcollection = fit_result.peakcollections[0]
    assert peakcollection.peak_tag == peak_tag
    parameters = setup_1_subrun['parameters'][0]

    # Test the fitted parameters
    sub_runs = peakcollection.sub_runs
    eff_param_values, eff_param_errors = peakcollection.get_effective_params()
    fit_costs = peakcollection.fitting_costs

    assert eff_param_values.dtype.names[0] == 'Center'
    np.testing.assert_almost_equal(eff_param_values['Center'], parameters['peak_center'], decimal=1)
    print_peak_results_and_check_positive(eff_param_values, eff_param_errors)

    # Read data again for raw data
    native_params = PeakShape.PSEUDOVOIGT.native_parameters
    native_params.extend(BackgroundFunction.LINEAR.native_parameters)
    sub_runs2 = peakcollection.sub_runs
    fit_cost2 = peakcollection.fitting_costs
    param_values, param_errors = peakcollection.get_native_params()
    print('Ordered native parameters: {}'.format(native_params))

    # Test
    assert sub_runs.shape == (1, ) == sub_runs2.shape
    assert np.allclose(fit_cost2, fit_costs, 0.0000001)

    # Effective parameter list: ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']
    # Native parameters: ['Mixing', 'Intensity', 'PeakCentre', 'FWHM', 'A0', 'A1']
    assert eff_param_values['Mixing'] == param_values['Mixing']  # mixing
    assert eff_param_values['Center'] == param_values['PeakCentre']  # center
    assert eff_param_values['Intensity'] == param_values['Intensity']  # intensity

    assert_checks(fit_result, parameters, param_values, number_of_peakCollection, peak_profile_type='PseudoVoigt')

    if fit_costs[0] > 1.0:
        # Plot
        model_x = fit_result.fitted.readX(0)
        model_y = fit_result.fitted.readY(0)
        data_x, data_y = setup_1_subrun['workspace'].get_reduced_diffraction_data(1, None)
        assert data_x.shape == model_x.shape
        assert data_y.shape == model_y.shape
        plt.clf()
        plt.plot(data_x, data_y, label='Test 2 Gaussian 3 sub runs')
        plt.plot(model_x, model_y, label='Fitted 2 Gaussian 3 sub runs')
        plt.legend()
        # plt.show()
        raise AssertionError('Fit cost (chi2 = {}) is too large (criteria = 1.)'.format(fit_costs[0]))


def test_calculate_effective_parameters_gaussian():
    """Test the effective peak parameters calculation for Gaussian

    Returns
    -------
    None

    """
    # Set raw value
    sigma = 0.20788180862724454
    height = 0.676468683375185
    # Set gold value
    exp_fwhm = 0.48952424995272315
    exp_intensity = 0.3524959381046824

    # Calculate effective parameters
    fwhm = Gaussian.cal_fwhm(sigma)
    intensity = Gaussian.cal_intensity(height, sigma)

    # assert exp_fwhm == pytest.approx(fwhm, 1E-10), 'FWHM wrong'
    # assert exp_intensity == pytest.approx(intensity, 1E-10), 'Intensity wrong'
    assert abs(exp_fwhm - fwhm) < 1E-10, 'FWHM: {} - {} = {} > 1E-10'.format(exp_fwhm, fwhm, exp_fwhm - fwhm)
    assert abs(exp_intensity - intensity) < 1E-10, 'Intensity: {} - {} = {} > 1e-10' \
                                                   ''.format(exp_intensity, intensity, exp_intensity - intensity)

    return


def test_calculate_effective_parameters_pv():
    """Test the methods to calculate effective parameters for Pseudo-Voigt

    Returns
    -------
    None

    """
    # Set raw parameter values
    intensity = 0.45705834149790703
    fwhm = 0.44181666416237664
    mixing = 0.23636114719871532

    # Set the gold value
    exp_height = 0.7326251617860263

    # Calculate effective values
    test_height = PseudoVoigt.cal_height(intensity, fwhm, mixing)

    # Verify
    assert abs(test_height - exp_height) < 1E-10, 'Peak height: {} - {} = {} > 1e-10' \
                                                  ''.format(exp_height, test_height, exp_height - test_height)

    return


# Named tuple for peak information
PeakInfo = namedtuple('PeakInfo', 'center left_bound right_bound tag')


@pytest.mark.parametrize('target_values', [{'Intensity': [0.03, 0.0], 'peak_center': [90, 96], 'FWHM': [0, 1],
                                            'background_A0': [-0.04, 0.42], 'background_A1': [0.007, -0.003]}])
def test_pseudovoigt_HB2B_1060(target_values):
    """This is a test of Pseudovoigt peak fitting for HB2B 1060.

     Data are from the real HB2B data previously reported problematic


     """
    # Define HiDRA project file name and skip test if it does not exist (on Travis)

    project_file_name = 'tests/data/HB2B_1060_first3_subruns.h5'

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
                                                  background_function_name='Linear', wavelength=np.nan)

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
    param_values_lp, _ = fit_result.peakcollections[0].get_native_params()

    # peak 'Right'
    param_values_rp, _ = fit_result.peakcollections[1].get_native_params()

    assert param_values_lp.size == 3, '3 subruns'
    assert len(param_values_lp.dtype.names) == 6, '6 native parameters'

    assert param_values_rp.size == 3, '3 subruns'
    assert len(param_values_rp.dtype.names) == 6, '6 native parameters'

    np.testing.assert_allclose(param_values_lp['Intensity'], target_values['Intensity'][0], atol=0.9)
    np.testing.assert_allclose(param_values_lp['PeakCentre'], target_values['peak_center'][0], atol=0.8)
    np.testing.assert_allclose(param_values_lp['FWHM'], target_values['FWHM'][0], atol=1.)
    np.testing.assert_allclose(param_values_lp['A0'], target_values['background_A0'][0], atol=1.)
    np.testing.assert_allclose(param_values_lp['A1'], target_values['background_A1'][0], atol=1.)

    np.testing.assert_allclose(param_values_rp['Intensity'], target_values['Intensity'][1], atol=0.01)
    np.testing.assert_allclose(param_values_rp['PeakCentre'], target_values['peak_center'][1], atol=1)
    np.testing.assert_allclose(param_values_rp['FWHM'], target_values['FWHM'][1], atol=1.2)
    np.testing.assert_allclose(param_values_rp['A0'], target_values['background_A0'][1], atol=1.)
    np.testing.assert_allclose(param_values_rp['A1'], target_values['background_A1'][1], atol=1.)


if __name__ == '__main__':
    # Init random number
    import random
    random.seed(1)
    # Test main
    pytest.main([__file__])
