# Peak fitting engine
import numpy
import scipy
import scipy.optimize
from pyrs.utilities import checkdatatypes


def gaussian(x, a, sigma, x0):
    """
    Gaussian with linear background
    :param x:
    :param a:
    :param sigma:
    :param x0:
    :return:
    """
    return a * numpy.exp(-((x - x0)/sigma)**2)


def loranzian(x, a, sigma, x0):
    """
    Lorentian
    :param x:
    :param a:
    :param sigma:
    :param x0:
    :return:
    """
    return


def quadratic_background(x, b0, b1, b2, b3):
    """
    up to 3rd order
    :param x:
    :param b0:
    :param b1:
    :param b2:
    :param b3:
    :return:
    """
    return b0 + b1*x + b2*x**2 + b3*x**3


def fit_peak(peak_func, vec_x, obs_vec_y, p0, p_range):
    """

    :param peak_func:
    :param vec_x:
    :param obs_vec_y:
    :param p0:
    :param p_range: example  # bounds=([a, b, c, x0], [a, b, c, x0])
    :return:
    """
    def calculate_chi2(covariance_matrix):
        """

        :param covariance_matrix:
        :return:
        """
        # TODO
        return 1.

    # check input
    checkdatatypes.check_numpy_arrays('Vec X and observed vec Y', [vec_x, obs_vec_y], 1, check_same_shape=True)

    # fit
    fit_results = scipy.optimize.curve_fit(peak_func, vec_x, obs_vec_y, p0=p0, bounds=p_range)

    fit_params = fit_results[0]
    fit_covmatrix = fit_results[1]
    cost = calculate_chi2(fit_covmatrix)

    return cost, fit_params, fit_covmatrix
