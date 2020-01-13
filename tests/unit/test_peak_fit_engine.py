import numpy as np
from pyrs.peaks import FitEngineFactory as PeakFitEngineFactory
from pyrs.core.workspaces import HidraWorkspace
from pyrs.core.peak_profile_utility import pseudo_voigt, PeakShape, BackgroundFunction
from pyrs.core.peak_profile_utility import Gaussian, PseudoVoigt
import pytest
from matplotlib import pyplot as plt


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

    for ipeak in range(num_peaks):
        # get peak center and range
        peak_center = peak_center_list[ipeak]
        peak_range = peak_range_list[ipeak]

        # Set FWHM to 1/6 of peak range and then to Gaussian's Sigma
        sigma = peak_range / 6. / (2. * np.sqrt(2. * np.log(2.)))

        # calculate Gaussian function based on input peak center and peak range
        vec_y += peak_height_list[ipeak] * np.exp(-(vec_x - peak_center) ** 2 / sigma ** 2)
    # END-FOR

    # Add noise
    noise = (np.random.random_sample(vec_x.shape[0]) - 0.5) * 2.0

    return vec_y + noise


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
    # END-FOR

    # Add noise
    noise = (np.random.random_sample(vec_x.shape[0]) - 0.5) * 2.0

    return vec_y + noise


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

    return vec_y + background


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
        vec_y = generate_test_gaussian(vec_x, peak_centers, peak_ranges, peak_intensities)
    elif peak_profile_type.lower() == 'pseudovoigt':
        vec_y = generate_test_pseudovoigt(vec_x, peak_centers, peak_ranges, peak_intensities)
        # peak_range *= 2  # PV requires larger fitting range
    else:
        raise NotImplementedError('Peak profile {} is not supported to generate testing workspace')

    # Add background
    vec_y = generate_test_background(vec_x, vec_y)

    # Print out the test data
    # for i in range(vec_x.shape[0]):
    #     print('{}   {}'.format(vec_x[i], vec_y[i]))

    # Add diffraction pattern
    test_workspace.set_sub_runs([1])
    test_workspace.set_reduced_diffraction_data(1, mask_id=None,
                                                two_theta_array=vec_x,
                                                intensity_array=vec_y)

    return test_workspace


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
    sub_runs_list = sub_run_data_dict.keys()
    test_workspace.set_sub_runs(sub_runs_list)

    # Add diffraction pattern
    for sub_run_i in sorted(sub_runs_list):
        vec_x, vec_y = sub_run_data_dict[sub_run_i]
        test_workspace.set_reduced_diffraction_data(sub_run_i,
                                                    mask_id=None,
                                                    two_theta_array=vec_x,
                                                    intensity_array=vec_y)

    return test_workspace


def test_1_gaussian_1_subrun():
    """Test fitting single Gaussian peak on 1 spectrum with background

    Returns
    -------
    None

    """
    # Set testing value
    # Default value
    min_x = 75.
    max_x = 85.
    num_x = 500
    peak_center = 80.
    peak_range = 10. * 0.25  # distance from peak center to 6 sigma

    # Generate test workspace and initialize fit engine
    gaussian_workspace = generate_hydra_workspace_single_subrun('Gaussian', min_x, max_x, num_x, [peak_center],
                                                                [peak_range], [10.])
    fit_engine = PeakFitEngineFactory.getInstance('Mantid', gaussian_workspace, out_of_plane_angle=None)

    # Fit
    m_tag = 'UnitTestGaussian'
    fit_engine.fit_peaks(peak_tag=m_tag,
                         sub_run_range=(1, 1),
                         peak_function_name='Gaussian',
                         background_function_name='Linear',
                         peak_center=peak_center,
                         peak_range=(peak_center - peak_range * 0.5, peak_center + peak_range * 0.5))

    # Get model (from fitted parameters) against each other
    model_x, model_y = fit_engine.calculate_fitted_peaks(1, None)
    data_x, data_y = gaussian_workspace.get_reduced_diffraction_data(1, None)
    # plt.plot(data_x, data_y, label='Test Gaussian')
    # plt.plot(model_x, model_y, label='Fitted Gaussian')
    assert data_x.shape == model_x.shape
    assert data_y.shape == model_y.shape

    # Test the fitted parameters
    # Read data
    eff_param_list, sub_runs, fit_costs, effective_param_values, effective_param_errors =\
        fit_engine.get_peaks(m_tag).get_effective_parameters_values()

    # Read data again for raw data
    native_params = PeakShape.GAUSSIAN.native_parameters
    native_params.extend(BackgroundFunction.LINEAR.native_parameters)
    sub_runs2, fit_cost2, param_values, param_errors = fit_engine.get_peaks(m_tag).get_parameters_values(native_params)

    # Test
    assert sub_runs.shape == (1, ) == sub_runs2.shape
    assert np.allclose(fit_cost2, fit_costs, 0.0000001)

    # Effective parameter list: ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']
    assert effective_param_values[0, 0] == param_values[1, 0]   # center
    assert abs(effective_param_values[0, 0] - peak_center) < 2e-2, 'Peak center is not correct'

    # fit goodness
    assert fit_costs[0] < 0.5, 'Fit cost (chi2 = {}) is too large'.format(fit_costs[0])

    return


def test_2_gaussian_1_subrun():
    """Fit 2 Gaussian peaks for 1 sub run

    Returns
    -------

    """
    # Set testing value
    # Default value
    min_x = 75.
    max_x = 95.
    num_x = 1000
    peak_centers = [80., 90.]
    peak_ranges = [10. * 0.25, 11. * 0.25]  # distance from peak center to 6 sigma
    peak_intensities = [10., 4]

    # Generate test workspace and initialize fit engine
    gaussian_workspace = generate_hydra_workspace_single_subrun('Gaussian', min_x, max_x, num_x, peak_centers,
                                                                peak_ranges, peak_intensities)
    fit_engine = PeakFitEngineFactory.getInstance('Mantid', gaussian_workspace, out_of_plane_angle=None)

    # Fit
    fit_engine.fit_multiple_peaks(sub_run_range=(1, 1),
                                  peak_function_name='Gaussian',
                                  background_function_name='Linear',
                                  peak_tag_list=['Left', 'Right'],
                                  peak_center_list=[79.4, 90.75],
                                  peak_range_list=[(76., 84.), (86., 94.)])

    # Get model (from fitted parameters) against each other
    model_x, model_y = fit_engine.calculate_fitted_peaks(1, None)
    data_x, data_y = gaussian_workspace.get_reduced_diffraction_data(1, None)
    assert data_x.shape == model_x.shape
    assert data_y.shape == model_y.shape

    # plt.plot(data_x, data_y, label='Test 2 Gaussian')
    # plt.plot(model_x, model_y, label='Fitted Gaussian')
    # plt.legend()
    # plt.show()

    # Test the fitted parameters: effective parameters
    # Effective parameter list: ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']
    # Read data
    eff_param_list, sub_runs, fit_costs, effective_param_values, effective_param_errors =\
        fit_engine.get_peaks('Left').get_effective_parameters_values()
    assert effective_param_values.shape == (7, 1), 'Only 1 sub run and 7 parameters'

    expected_intensity = 3.04
    if abs(effective_param_values[2][0] - expected_intensity) < 1E-03:
        plt.plot(data_x, data_y, label='Test 2 Gaussian')
        plt.plot(model_x, model_y, label='Fitted Gaussian')
        plt.legend()
        plt.show()
        raise AssertionError('Peak intensity {} shall be equal to {}.'
                             ''.format(effective_param_values[2, 0], expected_intensity))

    # Test the fitted parameters: native parameters
    native_params = PeakShape.GAUSSIAN.native_parameters
    native_params.extend(BackgroundFunction.LINEAR.native_parameters)
    sub_runs2, fit_cost2, param_values, param_errors =\
        fit_engine.get_peaks('Left').get_parameters_values(native_params)

    # Test
    assert sub_runs.shape == (1,) == sub_runs2.shape
    assert np.allclose(fit_cost2, fit_costs, 0.0000001)

    assert effective_param_values[0, 0] == param_values[1, 0]   # center
    assert abs(effective_param_values[0, 0] - peak_centers[0]) < 2e-2, 'Peak center is not correct'

    # fit goodness
    assert fit_costs[0] < 0.5, 'Fit cost (chi2 = {}) is too large'.format(fit_costs[0])

    # Test the peak on the right
    sub_runs_right, fit_cost_right, param_values_right, param_errors_right =\
        fit_engine.get_peaks('Right').get_parameters_values(native_params)
    assert fit_cost_right[0] < 0.5

    return


def test_2_gaussian_3_subruns():
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
    vec_y = generate_test_gaussian(vec_x, [75., 83], [3., 3.5], [10., 5])
    test_2g_dict[1] = vec_x, vec_y

    # sub run 2
    vec_y = generate_test_gaussian(vec_x, [75., 80], [3., 3.5], [15., 5])
    test_2g_dict[2] = vec_x, vec_y

    # sub run 3
    vec_y = generate_test_gaussian(vec_x, [75., 80], [3.1, 3.5], [0.2, 7.5])
    test_2g_dict[3] = vec_x, vec_y

    # Create a workspace based on this
    test_hd_ws = generate_hydra_workspace_multiple_sub_runs('3 G 3 S', test_2g_dict)

    # Fit
    fit_engine = PeakFitEngineFactory.getInstance('Mantid', test_hd_ws, out_of_plane_angle=None)
    fit_engine.fit_multiple_peaks(sub_run_range=(1, 3),
                                  peak_function_name='Gaussian',
                                  background_function_name='Linear',
                                  peak_tag_list=['Left', 'Right'],
                                  peak_center_list=[75.0, 80.0],
                                  peak_range_list=[(72.5, 77.5), (77.5, 82.5)])

    # Verify fitting result
    # ['Height', 'PeakCentre', 'Sigma'],
    gaussian_native_params = PeakShape.GAUSSIAN.native_parameters
    gaussian_native_params.extend(BackgroundFunction.LINEAR.native_parameters)

    # peak 'Left'
    sub_runs_lp, fit_cost2_lp, param_values_lp, param_errors_lp =\
        fit_engine.get_peaks('Left').get_parameters_values(gaussian_native_params)

    # peak 'Right'
    sub_runs_rp, fit_cost2_rp, param_values_rp, param_errors_rp =\
        fit_engine.get_peaks('Right').get_parameters_values(gaussian_native_params)

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

    # verify
    assert 0.001 < fit_cost2_lp[0] < 0.4, 'Fitting cost of sub run 1 of left peak ({}) ' \
                                          'is not reasonable or too large'.format(fit_cost2_lp[0])
    assert 0.001 < fit_cost2_rp[1] < 0.4, 'Fitting cost of sub run 1 of right peak ({}) ' \
                                          'is not reasonable or too large'.format(fit_cost2_rp[0])

    # Get effective peak parameters
    eff_param_list, sub_runs, fit_costs, effective_param_values, effective_param_errors =\
        fit_engine.get_peaks('Left').get_effective_parameters_values()
    assert effective_param_values.shape == (7, 3), 'Only 1 sub run and 7 parameters'

    eff_param_list, sub_runs, fit_costs, effective_param_values, effective_param_errors =\
        fit_engine.get_peaks('Right').get_effective_parameters_values()
    assert effective_param_values.shape == (7, 3), 'Only 1 sub run and 7 parameters'

    # Plot
    # model_x, model_y = fit_engine.calculate_fitted_peaks(3, None)
    # data_x, data_y = test_hd_ws.get_reduced_diffraction_data(3, None)
    # assert data_x.shape == model_x.shape
    # assert data_y.shape == model_y.shape
    # plt.plot(data_x, data_y, label='Test 2 Gaussian 3 sub runs')
    # plt.plot(model_x, model_y, label='Fitted 2 Gaussian 3 sub runs')
    # plt.legend()
    # plt.show()

    return


def test_3_gaussian_3_subruns():
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
    vec_y_0 = generate_test_gaussian(vec_x_0, [75.], [3.], [10.])
    test_2g_dict[1] = vec_x_0, vec_y_0

    # sub run 2
    vec_x_1 = np.arange(500).astype(float) * 0.02 + 72.
    vec_y_1 = generate_test_gaussian(vec_x_1, [75., 80], [3., 3.5], [15., 5])
    test_2g_dict[2] = vec_x_1, vec_y_1

    # sub run 3
    vec_x_2 = np.arange(500).astype(float) * 0.02 + 78.
    vec_y_2 = generate_test_gaussian(vec_x_2, [80., 85], [3.5, 3.7], [0.2, 7.5])
    test_2g_dict[3] = vec_x_2, vec_y_2

    # Create a workspace based on this
    test_hd_ws = generate_hydra_workspace_multiple_sub_runs('3 G 3 S', test_2g_dict)

    # Fit
    fit_engine = PeakFitEngineFactory.getInstance('Mantid', test_hd_ws, out_of_plane_angle=None)
    fit_engine.fit_multiple_peaks(sub_run_range=(1, 3),
                                  peak_function_name='Gaussian',
                                  background_function_name='Linear',
                                  peak_tag_list=['Left', 'Middle', 'Right'],
                                  peak_center_list=[75.0, 80.0, 85.0],
                                  peak_range_list=[(72.5, 77.5), (77.5, 82.5), (82.5, 87.5)])

    # Verify fitting result
    # ['Height', 'PeakCentre', 'Sigma'],
    gaussian_native_params = PeakShape.GAUSSIAN.native_parameters
    gaussian_native_params.extend(BackgroundFunction.LINEAR.native_parameters)

    # peak 'Left'
    sub_runs_lp, fit_cost2_lp, param_values_lp, param_errors_lp =\
        fit_engine.get_peaks('Left').get_parameters_values(gaussian_native_params)

    assert np.isinf(fit_cost2_lp[2]), 'Sub run 3 does not have peak @ 75 (Peak-Left).  Chi2 shall be infinity but' \
                                      ' not {}'.format(fit_cost2_lp[2])

    # Check with visualization
    # set
    fig, axs = plt.subplots(3)
    for sb_i in range(1, 4):
        model_x_i, model_y_i = fit_engine.calculate_fitted_peaks(sb_i, None)
        data_x_i, data_y_i = test_hd_ws.get_reduced_diffraction_data(sb_i, None)
        axs[sb_i - 1].plot(data_x_i, data_y_i, label='Sub run {} Data'.format(sb_i))
        axs[sb_i - 1].plot(model_x_i, model_y_i, label='Sub run {} Model'.format(sb_i))
    # plt.show()

    return


def test_1_pv_1_subrun():
    """
    Test fitting single Pseudo-voigt peak with background
    Returns
    -------
    None
    """
    # Generate test workspace and initialize fit engine
    # Default value
    min_x = 75.
    max_x = 85.
    num_x = 500
    peak_center = 80.
    peak_range = 10. * 0.25  # distance from peak center to 6 sigma
    pv_workspace = generate_hydra_workspace_single_subrun('PseudoVoigt', min_x, max_x, num_x, [peak_center],
                                                          [peak_range], [100.])

    fit_engine = PeakFitEngineFactory.getInstance('Mantid', pv_workspace, out_of_plane_angle=None)

    # Fit
    peak_tag = 'UnitTestPseudoVoigt'
    fit_engine.fit_peaks(peak_tag=peak_tag,
                         sub_run_range=(1, 1),
                         peak_function_name='PseudoVoigt',
                         background_function_name='Linear',
                         peak_center=peak_center,
                         peak_range=(peak_center - peak_range * 1.0, peak_center + peak_range * 1.0))

    # Get model (from fitted parameters) against each other
    model_x, model_y = fit_engine.calculate_fitted_peaks(1, None)
    data_x, data_y = pv_workspace.get_reduced_diffraction_data(1, None)
    plt.plot(data_x, data_y, label='Test PseudoVoigt')
    plt.plot(model_x, model_y, label='Fitted PseudoVoigt')
    assert data_x.shape == model_x.shape
    assert data_y.shape == model_y.shape

    # Test the fitted parameters
    # Read data
    eff_param_list, sub_runs, fit_costs, effective_param_values, effective_param_errors =\
        fit_engine.get_peaks(peak_tag).get_effective_parameters_values()

    # Read data again for raw data
    native_params = PeakShape.PSEUDOVOIGT.native_parameters
    native_params.extend(BackgroundFunction.LINEAR.native_parameters)
    sub_runs2, fit_cost2, param_values, param_errors =\
        fit_engine.get_peaks(peak_tag).get_parameters_values(native_params)
    print('Ordered native parameters: {}'.format(native_params))

    # Test
    assert sub_runs.shape == (1, ) == sub_runs2.shape
    assert np.allclose(fit_cost2, fit_costs, 0.0000001)

    # Effective parameter list: ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']
    # Native parameters: ['Mixing', 'Intensity', 'PeakCentre', 'FWHM', 'A0', 'A1']
    assert effective_param_values[4, 0] == param_values[0, 0]  # mixing
    assert effective_param_values[0, 0] == param_values[2, 0]  # center
    assert effective_param_values[2, 0] == param_values[1, 0]  # intensity

    # fit goodness
    if fit_costs[0] > 1.0:
        # Plot
        model_x, model_y = fit_engine.calculate_fitted_peaks(1, None)
        data_x, data_y = pv_workspace.get_reduced_diffraction_data(1, None)
        assert data_x.shape == model_x.shape
        assert data_y.shape == model_y.shape
        plt.clf()
        plt.plot(data_x, data_y, label='Test 2 Gaussian 3 sub runs')
        plt.plot(model_x, model_y, label='Fitted 2 Gaussian 3 sub runs')
        plt.legend()
        # plt.show()
        raise AssertionError('Fit cost (chi2 = {}) is too large (criteria = 1.)'.format(fit_costs[0]))

    return


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


if __name__ == '__main__':
    # Init random number
    import random
    random.seed(1)
    # Test main
    pytest.main()
