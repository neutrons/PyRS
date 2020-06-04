import numpy as np
from pyrs.core.peak_profile_utility import PeakShape, BackgroundFunction, get_parameter_dtype
from pyrs.peaks import PeakCollection
import pytest


def check_peak_shape_enum(peak_shape, num_native_params):
    """check peak shape enum

    Parameters
    ----------
     peak_shape : str
            Peak shape
    num_native_params : integer
        number of native parameters

    Returns
    -------

    """
    assert PeakShape.getShape(peak_shape) == PeakShape[peak_shape.upper()]
    assert PeakShape.getShape(peak_shape.upper()) == PeakShape[peak_shape.upper()]
    with pytest.raises(KeyError):
        PeakShape.getShape('non-existant-peak-shape')

    assert len(PeakShape.getShape(peak_shape).native_parameters) == num_native_params


def test_peak_shape_enum_gaussian():
    check_peak_shape_enum('gaussian', 3)


def test_peak_shape_enum_pseudoVoigt():
    check_peak_shape_enum('PseudoVoigt', 4)


def test_background_enum():
    assert BackgroundFunction.getFunction('linear') == BackgroundFunction.LINEAR
    assert BackgroundFunction.getFunction('LINEAR') == BackgroundFunction.LINEAR

    with pytest.raises(KeyError):
        BackgroundFunction.getFunction('non-existant-peak-shape')

    assert len(BackgroundFunction.getFunction('linear').native_parameters) == 2


def test_peak_collection_init():
    assert PeakCollection('test1', 'Gaussian', 'Linear')

    assert PeakCollection('test2', 'Gaussian', 'Linear', wavelength=22.)

    d_peak_collection = PeakCollection('test3', 'Gaussian', 'Linear', d_reference=1.26)
    assert d_peak_collection
    d_ref, d_ref_err = d_peak_collection.get_d_reference()
    np.testing.assert_equal(d_ref, np.asarray((1.26,)))
    np.testing.assert_equal(d_ref_err, np.asarray((0.,)))


def check_peak_collection(peak_shape, NUM_SUBRUN, target_errors,
                          wavelength=None, d_reference=None,
                          target_d_spacing_center=np.nan,
                          target_d_spacing_center_error=np.asarray([0., 0.]),
                          target_strain=np.nan,
                          target_strain_error=np.asarray([0., 0.])):
    """check the peak collection

    Parameters
    ----------
    subruns : numpy.array
        1D numpy array for sub run numbers
    peak_shape : str
            Peak shape
    target_errors : numpy.ndarray
        numpy structured array for peak/background parameter fitted target error
    wavelength : float
        neutron wavelength
    target_d_spacing_center : list
        d spacing center target value
    target_d_spacing_center_error : list
        d spacing center target error

    Returns
    -------

    """
    subruns = np.arange(NUM_SUBRUN) + 1
    chisq = np.array([42., 43.])
    raw_peaks_array = np.zeros(NUM_SUBRUN, dtype=get_parameter_dtype(peak_shape, 'Linear'))
    if peak_shape == 'PseudoVoigt':
        raw_peaks_array['Intensity'] = [1, 2]
        raw_peaks_array['FWHM'] = np.array([4, 5], dtype=float)
        raw_peaks_array['Mixing'] = [0, 1]
    else:
        raw_peaks_array['Height'] = [1, 2]
        raw_peaks_array['Sigma'] = np.array([4, 5], dtype=float)
    raw_peaks_array['PeakCentre'] = [90., 91.]

    # background terms are both zeros
    raw_peaks_errors = np.zeros(NUM_SUBRUN, dtype=get_parameter_dtype(peak_shape, 'Linear'))
    if wavelength is None:
        peaks = PeakCollection('testing', peak_shape, 'Linear')
    else:
        peaks = PeakCollection('testing', peak_shape, 'Linear', wavelength=wavelength)

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
    np.testing.assert_equal(obs_eff_errors, target_errors)
    np.testing.assert_equal(obs_eff_peaks['Center'], raw_peaks_array['PeakCentre'])
    if peak_shape == 'PseudoVoigt':
        np.testing.assert_equal(obs_eff_peaks['Intensity'], raw_peaks_array['Intensity'])
        np.testing.assert_equal(obs_eff_peaks['FWHM'], raw_peaks_array['FWHM'])
        np.testing.assert_equal(obs_eff_peaks['Mixing'], raw_peaks_array['Mixing'])
    else:
        np.testing.assert_equal(obs_eff_peaks['Height'], raw_peaks_array['Height'])
        np.testing.assert_equal(obs_eff_peaks['FWHM'], 2. * np.sqrt(2. * np.log(2.)) * raw_peaks_array['Sigma'])
    np.testing.assert_equal(obs_eff_peaks['A0'], NUM_SUBRUN * [0.])
    np.testing.assert_equal(obs_eff_peaks['A1'], NUM_SUBRUN * [0.])

    # check d spacing
    obs_dspacing, obs_dspacing_errors = peaks.get_dspacing_center()
    np.testing.assert_allclose(obs_dspacing, target_d_spacing_center, atol=0.01)
    np.testing.assert_allclose(obs_dspacing_errors, target_d_spacing_center_error, atol=0.01)

    # check strain
    if d_reference is None:
        peaks.set_d_reference()
    else:
        peaks.set_d_reference(values=d_reference)
    strain, strain_error = peaks.get_microstrain()
    np.testing.assert_allclose(strain, target_strain, atol=0.5)
    np.testing.assert_allclose(strain_error, target_strain_error, atol=0.5)


def test_peak_collection_Gaussian():
    NUM_SUBRUN = 2
    # without wavelength
    check_peak_collection('Gaussian', NUM_SUBRUN, np.zeros(NUM_SUBRUN, dtype=get_parameter_dtype(effective=True)))
    # with wavelength
    check_peak_collection('Gaussian', NUM_SUBRUN, np.zeros(NUM_SUBRUN, dtype=get_parameter_dtype(effective=True)),
                          wavelength=1.53229, d_reference=1.08, target_d_spacing_center=[1.08, 1.07],
                          target_d_spacing_center_error=[0.0, 0.0], target_strain=[3234., -5408.],
                          target_strain_error=[0.0, 0.0])


def test_peak_collection_PseudoVoigt():
    NUM_SUBRUN = 2
    # without wavelength
    check_peak_collection('PseudoVoigt', NUM_SUBRUN,
                          np.array([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                                   dtype=get_parameter_dtype(effective=True)))
    # with wavelength
    check_peak_collection('PseudoVoigt', NUM_SUBRUN,
                          np.array([(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                                   dtype=get_parameter_dtype(effective=True)),
                          wavelength=1.53229, d_reference=1.08, target_d_spacing_center=[1.08, 1.07],
                          target_d_spacing_center_error=[0.0, 0.0], target_strain=[3234., -5408.],
                          target_strain_error=[0.0, 0.0])


if __name__ == '__main__':
    pytest.main([__file__])
