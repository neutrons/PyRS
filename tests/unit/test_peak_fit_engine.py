import numpy as np
from pyrs.core.mantid_fit_peak import MantidPeakFitEngine
from pyrs.core.workspaces import HidraWorkspace
from pyrs.core.peak_profile_utility import pseudo_voigt, NATIVE_BACKGROUND_PARAMETERS, NATIVE_PEAK_PARAMETERS
import pytest
from matplotlib import pyplot as plt


def generate_test_gaussian(vec_x, peak_center, peak_range):
    """
    Generate Gaussian function for test
    Parameters
    ----------
    vec_x: ndarray (N, )
        Vector of X

    Returns
    -------
    ndarray, float, tuple
        vector of Y containing Gaussian, peak center, peak range
    """
    # Set sigma
    sigma = peak_range / 6. / (2. * np.sqrt(2. * np.log(2.)))

    # calculate Gaussian
    vec_y = 10. * np.exp(-(vec_x - peak_center)**2 / sigma**2)

    # Add noise
    noise = (np.random.random_sample(vec_x.shape[0]) - 0.5) * 2.0

    return vec_y + noise


def generate_test_pseudovoigt(vec_x, peak_center, peak_range):
    """
    Generate Gaussian function for test
    Parameters
    ----------
    peak_center: float
        peak center
    peak_range: float
        range of peak
    vec_x: ndarray (N, )
        Vector of X

    Returns
    -------
    ndarray, float, tuple
        vector of Y containing Gaussian, peak center, peak range
    """
    # Set FWHM
    fwhm = peak_range / 6.
    peak_intensity = 100.
    mixing = 0.75  # more Gaussian than Lorentzian

    # calculate Gaussian
    vec_y = pseudo_voigt(vec_x, peak_intensity, fwhm, mixing, peak_center)

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


def generate_hydra_workspace(peak_profile_type):
    """
    Generate HiDRAWorkspace
    Parameters
    ----------
    peak_profile_type

    Returns
    -------

    """
    # Create test workspace
    test_workspace = HidraWorkspace('test')

    # Generate vector X
    vec_x = np.arange(500) * 0.1 * 0.2 + 75.  # from 75 to 85 degree
    # Determine peak range and center
    peak_center = 0.5 * (vec_x[0] + vec_x[-1])
    data_range = vec_x[-1] - vec_x[0]
    peak_range = 0.25 * data_range  # distance from peak center to 6 sigma

    # Add profile
    if peak_profile_type.lower() == 'gaussian':
        vec_y = generate_test_gaussian(vec_x, peak_center, peak_range)
    elif peak_profile_type.lower() == 'pseudovoigt':
        vec_y = generate_test_pseudovoigt(vec_x, peak_center, peak_range)
        peak_range *= 2  # PV requires larger fitting range
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
                                                bin_edges=vec_x,
                                                hist=vec_y)

    return test_workspace, peak_center, (peak_center - peak_range, peak_center + peak_center)


def test_gaussian():
    """
    Test fitting single Gaussian peak with background
    Returns
    -------
    None
    """
    # Generate test workspace and initialize fit engine
    test_suite = generate_hydra_workspace('Gaussian')
    gaussian_workspace, peak_center, peak_range = test_suite
    fit_engine = MantidPeakFitEngine(gaussian_workspace, mask_name=None)

    # Fit
    fit_engine.fit_peaks(sub_run_range=(1, 1),
                         peak_function_name='Gaussian',
                         background_function_name='Linear',
                         peak_center=peak_center,
                         peak_range=peak_range,
                         cal_center_d=False)

    # Get model (from fitted parameters) against each other
    model_x, model_y = fit_engine.get_calculated_peak(1)
    data_x, data_y = gaussian_workspace.get_reduced_diffraction_data(1, None)
    plt.plot(data_x, data_y, label='Test Gaussian')
    plt.plot(model_x, model_y, label='Fitted Gaussian')
    assert data_x.shape == model_x.shape
    assert data_y.shape == model_y.shape

    # Test the fitted parameters
    # Read data
    eff_param_list, sub_runs, fit_costs, effective_param_values =\
        fit_engine.get_fitted_effective_params(True)

    # Read data again for raw data
    native_params = NATIVE_PEAK_PARAMETERS['Gaussian'][:]
    native_params.extend(NATIVE_BACKGROUND_PARAMETERS['Linear'])
    sub_runs2, fit_cost2, param_values = fit_engine.get_fitted_params(native_params, True)

    # Test
    assert sub_runs.shape == (1, ) == sub_runs2.shape
    assert np.allclose(fit_cost2, fit_costs, 0.0000001)

    # Effective paramter list: ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']
    assert effective_param_values[0, 0, 0] == param_values[1, 0, 0]   # center

    # fit goodness
    assert fit_costs[0] < 0.5, 'Fit cost (chi2 = {}) is too large'.format(fit_costs[0])

    # If everything is correct, optionally show the result
    # plt.show()

    return


def test_pseudo_voigt():
    """
    Test fitting single Pseudo-voigt peak with background
    Returns
    -------
    None
    """
    # Generate test workspace and initialize fit engine
    test_suite = generate_hydra_workspace('PseudoVoigt')
    gaussian_workspace, peak_center, peak_range = test_suite
    fit_engine = MantidPeakFitEngine(gaussian_workspace, mask_name=None)

    # Fit
    fit_engine.fit_peaks(sub_run_range=(1, 1),
                         peak_function_name='PseudoVoigt',
                         background_function_name='Linear',
                         peak_center=peak_center,
                         peak_range=peak_range,
                         cal_center_d=False)

    # Get model (from fitted parameters) against each other
    model_x, model_y = fit_engine.get_calculated_peak(1)
    data_x, data_y = gaussian_workspace.get_reduced_diffraction_data(1, None)
    plt.plot(data_x, data_y, label='Test PseudoVoigt')
    plt.plot(model_x, model_y, label='Fitted PseudoVoigt')
    assert data_x.shape == model_x.shape
    assert data_y.shape == model_y.shape

    # Test the fitted parameters
    # Read data
    eff_param_list, sub_runs, fit_costs, effective_param_values =\
        fit_engine.get_fitted_effective_params(True)

    # Read data again for raw data
    native_params = NATIVE_PEAK_PARAMETERS['PseudoVoigt'][:]
    native_params.extend(NATIVE_BACKGROUND_PARAMETERS['Linear'])
    sub_runs2, fit_cost2, param_values = fit_engine.get_fitted_params(native_params, True)
    print('Ordered native parameters: {}'.format(native_params))

    # Test
    assert sub_runs.shape == (1, ) == sub_runs2.shape
    assert np.allclose(fit_cost2, fit_costs, 0.0000001)

    # Effective parameter list: ['Center', 'Height', 'Intensity', 'FWHM', 'Mixing', 'A0', 'A1']
    # Native parameters: ['Mixing', 'Intensity', 'PeakCentre', 'FWHM', 'A0', 'A1']
    assert effective_param_values[4, 0, 0] == param_values[0, 0, 0]  # mixing
    assert effective_param_values[0, 0, 0] == param_values[2, 0, 0]  # center
    assert effective_param_values[2, 0, 0] == param_values[1, 0, 0]  # intensity

    # fit goodness
    assert fit_costs[0] < 0.5, 'Fit cost (chi2 = {}) is too large'.format(fit_costs[0])

    # If everything is correct, optionally show the result
    # plt.show()

    return

if __name__ == '__main__':
    pytest.main()
    plt.show()
