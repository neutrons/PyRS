import numpy as np
from numpy.testing import assert_allclose
from pyrs.dataobjects.fields import StrainField, StrainFieldSingle, StressField
from pyrs.peaks.peak_collection import to_microstrain

to_megapascal = StressField.to_megapascal


def test_strain_single_builder(strain_single_builder):
    # all input data
    scan = {
        'runnumber': '1234',
        'subruns': [1, 2, 3, 4, 5, 6, 7, 8],
        'peak_tag': 'test',
        'wavelength': 2.0,
        'd_reference': 1.0,
        'peak_profile': 'pseudovoigt',
        'background_type': 'linear',
        # parameters appropriate to the selected peak shape, except for parameter PeakCentre
        'native': {
            'Intensity': [100, 110, 120, 130, 140, 150, 160, 170],
            'FWHM': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            'Mixing': [1.0] * 8,
            'A0': [10., 11., 12., 13., 14., 15., 16., 17.],
            'A1': [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        },
        'fit_costs': [1.0] * 8,
        # will back-calculate PeakCentre values in order to end up with these lattice spacings
        'd_spacing': [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],
        # errors in native parameters are taken to be their values times this error fraction
        'error_fraction': 0.1,
        'vx': [0., 1., 2., 3., 4., 5., 6., 7.],
        'vy': [0.] * 8,
        'vz': [0.] * 8
    }

    # create a StrainField object
    strain = strain_single_builder(scan)

    # assert entered data
    assert_allclose(strain.point_list.vx, np.array(scan['vx']))
    assert_allclose(strain.get_d_reference().values, np.repeat(scan['d_reference'], len(scan['subruns'])))
    assert_allclose(strain.get_dspacing_center().values, np.array(scan['d_spacing']))
    strain_expected_values = to_microstrain((scan['d_spacing'] - scan['d_reference']) / scan['d_reference'])
    assert_allclose(strain.field.values, strain_expected_values, atol=1)
    for name in ['Intensity', 'FWHM', 'Mixing', 'A0', 'A1']:
        assert_allclose(strain.get_effective_peak_parameter(name).values, np.array(scan['native'][name]))


def test_strain_builder(strain_builder):
    # all input data
    scan = {
        'runnumber': '1234',
        'subruns': [1, 2, 3, 4, 5, 6, 7, 8],
        'peak_tag': 'test',
        'wavelength': 2.0,
        'd_reference': 1.0,
        'peak_profile': 'pseudovoigt',
        'background_type': 'linear',
        # parameters appropriate to the selected peak shape, except for parameter PeakCentre
        'native': {
            'Intensity': [100, 110, 120, 130, 140, 150, 160, 170],
            'FWHM': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            'Mixing': [1.0] * 8,
            'A0': [10., 11., 12., 13., 14., 15., 16., 17.],
            'A1': [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        },
        'fit_costs': [1.0] * 8,
        # will back-calculate PeakCentre values in order to end up with these lattice spacings
        'd_spacing': [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],
        # errors in native parameters are taken to be their values times this error fraction
        'error_fraction': 0.1,
        'vx': [0., 1., 2., 3., 4., 5., 6., 7.],
        'vy': [0.] * 8,
        'vz': [0.] * 8
    }

    # create a StrainField object
    strain = strain_builder(scan)

    # assert entered data
    assert_allclose(strain.point_list.vx, np.array(scan['vx']))
    assert_allclose(strain.get_d_reference().values, np.repeat(scan['d_reference'], len(scan['subruns'])))
    assert_allclose(strain.get_dspacing_center().values, np.array(scan['d_spacing']))
    strain_expected_values = to_microstrain((scan['d_spacing'] - scan['d_reference']) / scan['d_reference'])
    assert_allclose(strain.field.values, strain_expected_values, atol=1)
    for name in ['Intensity', 'FWHM', 'Mixing', 'A0', 'A1']:
        assert_allclose(strain.get_effective_peak_parameter(name).values, np.array(scan['native'][name]))


def test_strain_single_object_0(strain_single_object_0):
    assert isinstance(strain_single_object_0, StrainFieldSingle)
    expected = to_microstrain([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    assert_allclose(strain_single_object_0.values, expected, atol=1)


def test_strain_object_0(strain_object_0):
    assert isinstance(strain_object_0, StrainField)
    expected = to_microstrain([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    assert_allclose(strain_object_0.values, expected, atol=1)


def test_strain_object_1(strain_object_1):
    assert isinstance(strain_object_1, StrainField)
    expected = to_microstrain(np.arange(0.01, 0.085, 0.01))
    assert_allclose(strain_object_1.values, expected, atol=1)


def test_strain_object_2(strain_object_2):
    assert isinstance(strain_object_2, StrainField)
    expected = to_microstrain(np.arange(0.01, 0.085, 0.01))
    assert_allclose(strain_object_2.values, expected, atol=1)


def test_strain_stress_object_0(strain_stress_object_0):
    strains = strain_stress_object_0['strains']
    assert_allclose(strains['11'].values, to_microstrain(np.arange(0.0, 0.045, 0.01)), atol=1)
    assert_allclose(strains['22'].values, to_microstrain(np.arange(0.10, 0.145, 0.01)), atol=1)
    assert_allclose(strains['33'].values, to_microstrain(np.arange(0.20, 0.245, 0.01)), atol=1)

    # Check stress values.  Young's modulus and Poisson ratio values yield simple formulae
    stress = strain_stress_object_0['stresses']['diagonal']
    trace = stress.strain11.values + stress.strain22.values + stress.strain33.values
    assert_allclose(stress['11'].values, to_megapascal(stress.strain11.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['22'].values, to_megapascal(stress.strain22.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['33'].values, to_megapascal(stress.strain33.values + trace), equal_nan=True, atol=1)

    stress = strain_stress_object_0['stresses']['in-plane-strain']
    assert_allclose(stress.strain33.values, np.zeros(5))
    trace = stress.strain11.values + stress.strain22.values + stress.strain33.values
    assert_allclose(stress['11'].values, to_megapascal(stress.strain11.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['22'].values, to_megapascal(stress.strain22.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['33'].values, to_megapascal(stress.strain33.values + trace), equal_nan=True, atol=1)

    stress = strain_stress_object_0['stresses']['in-plane-stress']
    assert_allclose(stress.strain33.values,
                    -1. * (stress.strain11.values + stress.strain22.values), equal_nan=True, atol=1)
    trace = stress.strain11.values + stress.strain22.values
    assert_allclose(stress['11'].values, to_megapascal(stress.strain11.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['22'].values, to_megapascal(stress.strain22.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress.stress33.values, np.zeros(5))


def test_strain_stress_object_1(strain_stress_object_1):
    strains, stresses = strain_stress_object_1['strains'], strain_stress_object_1['stresses']

    # check strain values
    assert_allclose(strains['11'].values, to_microstrain(np.arange(0.0, 0.075, 0.01)), atol=1)
    assert_allclose(strains['22'].values, to_microstrain(np.arange(0.01, 0.085, 0.01)), atol=1)
    assert_allclose(strains['33'].values, to_microstrain(np.arange(0.02, 0.095, 0.01)), atol=1)

    # Check stress values.  Young's modulus and Poisson ratio values yield simple formulae
    stress = stresses['diagonal']
    trace = stress.strain11.values + stress.strain22.values + stress.strain33.values
    assert_allclose(stress['11'].values, to_megapascal(stress.strain11.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['22'].values, to_megapascal(stress.strain22.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['33'].values, to_megapascal(stress.strain33.values + trace), equal_nan=True, atol=1)

    stress = stresses['in-plane-strain']
    assert_allclose(stress.strain33.values, np.zeros(9))
    trace = stress.strain11.values + stress.strain22.values + stress.strain33.values
    assert_allclose(stress['11'].values, to_megapascal(stress.strain11.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['22'].values, to_megapascal(stress.strain22.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['33'].values, to_megapascal(stress.strain33.values + trace), equal_nan=True, atol=1)

    stress = stresses['in-plane-stress']
    assert_allclose(stress.strain33.values, -1. * (stress.strain11.values + stress.strain22.values))
    trace = stress.strain11.values + stress.strain22.values
    assert_allclose(stress['11'].values, to_megapascal(stress.strain11.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress['22'].values, to_megapascal(stress.strain22.values + trace), equal_nan=True, atol=1)
    assert_allclose(stress.stress33.values, np.zeros(9))
