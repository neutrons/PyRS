import numpy as np
from pyrs.core.mantid_fit_peak import MantidPeakFitEngine
from pyrs.core.workspaces import HidraWorkspace
from pyrs.core.peak_profile_utility import EFFECTIVE_PEAK_PARAMETERS


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

    return


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
    vec_x = np.arange(100) * 0.1 + 75.  # from 75 to 85 degree
    # Determine peak range and center
    peak_center = 0.5 * (vec_x[0] + vec_x[-1])
    data_range = vec_x[-1] - vec_x[0]
    peak_range = peak_center - 0.25 * data_range, peak_center + 0.25 * data_range

    # Add profile
    if peak_profile_type.lower() == 'gaussian':
        vec_y = generate_test_gaussian(vec_x, peak_center, peak_range)
    elif peak_profile_type.lower() == 'pseudovoigt':
        vec_y = generate_test_pseudovoigt(vec_y)
    else:
        raise NotImplementedError('Peak profile {} is not supported to generate testing workspace')

    # Add background
    vec_y = generate_test_background(vec_x, vec_y)

    # Add diffraction pattern
    test_workspace.set_reduced_diffraction_data(1, mask_id=None,
                                                bin_edges=vec_x,
                                                hist=vec_y)

    return test_workspace, peak_center, peak_range


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
    fit_engine.fit_peaks(sub_run_range=[1],
                         peak_function_name='Gaussian',
                         background_function_name='Linear',
                         peak_center=peak_center,
                         peak_range=peak_range,
                         cal_center_d=False)

    # Read data
    sub_runs, fit_costs, effective_param_values =\
        fit_engine.get_fitted_effective_params(EFFECTIVE_PEAK_PARAMETERS, True)

    # Test
    assert sub_runs.shape == (1, )

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
    fit_engine.fit_peaks(sub_run_range=[1],
                         peak_function_name='PseudoVoigt',
                         background_function_name='Linear',
                         peak_center=peak_center,
                         peak_range=peak_range,
                         cal_center_d=False)

    # Read data
    sub_runs, fit_costs, effective_param_values =\
        fit_engine.get_fitted_effective_params(EFFECTIVE_PEAK_PARAMETERS, True)

    # Test
    assert sub_runs.shape == (1, )

    return


