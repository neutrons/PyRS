from __future__ import (absolute_import, division, print_function)  # python3 compatibility
import numpy as np
from pyrs.core.peak_profile_utility import PeakShape, BackgroundFunction, get_parameter_dtype
from pyrs.peaks import PeakCollection
import pytest

def check_peak_shape_enum(peak_shape):
    assert PeakShape.getShape(peak_shape) == PeakShape[peak_shape.upper()]
    assert PeakShape.getShape(peak_shape.upper()) == PeakShape[peak_shape.upper()]
    with pytest.raises(KeyError):
        PeakShape.getShape('non-existant-peak-shape')

    assert len(PeakShape.getShape(peak_shape).native_parameters) == 3


def test_peak_shape_enum_gaussian():
    check_peak_shape_enum('gaussian')


def test_peak_shape_enum_pseudoVoigt():
    check_peak_shape_enum('PseudoVoigt')


def test_background_enum():
    assert BackgroundFunction.getFunction('linear') == BackgroundFunction.LINEAR
    assert BackgroundFunction.getFunction('LINEAR') == BackgroundFunction.LINEAR

    with pytest.raises(KeyError):
        BackgroundFunction.getFunction('non-existant-peak-shape')

    assert len(BackgroundFunction.getFunction('linear').native_parameters) == 2


def check_peak_collection(peak_shape, NUM_SUBRUN, target_error):
    # create test data with two peaks - both peaks will be normalized gaussians
    subruns = np.arange(NUM_SUBRUN) + 1
    chisq = np.array([42., 43.])
    raw_peaks_array = np.zeros(NUM_SUBRUN, dtype=get_parameter_dtype(peak_shape, 'Linear'))
    if peak_shape == 'PseudoVoigt':
        raw_peaks_array['Intensity'] = [1, 2]
    else:
        raw_peaks_array['Height'] = [1, 2]
    raw_peaks_array['PeakCentre'] = [3, 4]
    raw_peaks_array['Sigma'] = np.array([4, 5], dtype=float)
    # background terms are both zeros
    raw_peaks_errors = np.zeros(NUM_SUBRUN, dtype=get_parameter_dtype(peak_shape, 'Linear'))

    peaks = PeakCollection('testing', peak_shape, 'Linear')
    # uncertainties are being set to zero
    peaks.set_peak_fitting_values(subruns, raw_peaks_array,
                                  raw_peaks_errors, chisq)

    # check general features
    assert peaks
    np.testing.assert_equal(peaks.get_subruns(), subruns)
    np.testing.assert_equal(peaks.get_chisq(), chisq)
    assert len(peaks.get_fit_status()) == NUM_SUBRUN
    # check raw/native parameters
    obs_raw_peaks, obs_raw_errors = peaks.get_native_params()
    np.testing.assert_equal(obs_raw_peaks, raw_peaks_array)
    np.testing.assert_equal(obs_raw_errors, raw_peaks_errors)
    # check effective parameters
    obs_eff_peaks, obs_eff_errors = peaks.get_effective_params()
    assert obs_eff_peaks.size == NUM_SUBRUN
    np.testing.assert_equal(obs_eff_errors, target_error)
    np.testing.assert_equal(obs_eff_peaks['Center'], raw_peaks_array['PeakCentre'])
    if peak_shape == 'PseudoVoigt':
        np.testing.assert_equal(obs_eff_peaks['Intensity'], raw_peaks_array['Height'])
    else:
        np.testing.assert_equal(obs_eff_peaks['Height'], raw_peaks_array['Height'])
    np.testing.assert_equal(obs_eff_peaks['FWHM'], 2. * np.sqrt(2. * np.log(2.)) * raw_peaks_array['Sigma'])
    np.testing.assert_equal(obs_eff_peaks['A0'], NUM_SUBRUN * [0.])
    np.testing.assert_equal(obs_eff_peaks['A1'], NUM_SUBRUN * [0.])



def test_peak_collection_Gaussian():
    # create test data with two peaks - both peaks will be normalized gaussians
    NUM_SUBRUN=2
    check_peak_collection('Gaussian', NUM_SUBRUN, np.zeros(NUM_SUBRUN, dtype=get_parameter_dtype(effective=True)))


def test_peak_collection_PseudoVoigt():
    # create test data with two peaks - both peaks will be normalized gaussians
    NUM_SUBRUN = 2
    check_peak_collection('PseudoVoigt', NUM_SUBRUN, np.array([(3.0, 0.0,4.0,0.0,0.0,0.0,1.0),(4.0,0.0,5.0,0.0,0.0,0.0,2.0)],
                                                     dtype=get_parameter_dtype(effective=True)))

if __name__ == '__main__':
    pytest.main([__file__])
